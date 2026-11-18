# -*- coding: utf-8 -*-
r"""
hisse_tahmin.py — Tek hisse için 8 regresyon + LSTM(PyTorch) fiyat tahmini
=============================================================================
Enflasyon projesindeki (InflationForecast / enflasyon_guncel.py) model kadrosunun
hisse senedi karşılığı. Model hiperparametreleri birebir korunmuştur:

  1) SGD Regressor        : make_pipeline(StandardScaler(), SGDRegressor(max_iter=100000, tol=1e-6))
  2) Linear Regression    : LinearRegression()
  3) Lasso Regression     : Lasso(alpha=0.001)
  4) Lars Regression      : Lars(random_state=0)
  5) Kernel Regression    : KernelRidge()  (varsayılan: alpha=1, linear kernel)
  6) Gaussian Regression  : GaussianProcessRegressor(kernel=DotProduct()+WhiteKernel())
  7) Bayessian Regression : BayesianRidge()
  8) XGBoost Regression   : XGBRegressor()  (random_state=0)
  9) LSTM (PyTorch)       : LSTM(128) -> Linear(64)+SELU -> Linear(64)+ELU -> Linear(1)
                            RMSprop(lr=0.01), MSE, 100 epoch, batch=32,
                            10 tohum ortalaması (enflasyon projesindeki gibi)

Yöntem (enflasyon projesiyle aynı mantık):
  - Hedef: bir sonraki barın getiri yüzdesi (enflasyondaki "aylık % değişim" gibi)
  - Özellikler: getiri gecikmeleri (1-10), hacim değişim gecikmeleri (1-5),
    MA5/MA10/MA20/MA60 sapmaları, getiri volatilitesi (5, 20 bar)
  - Tahminler fiyat seviyesine zincirlenir: fiyat *= (1 + tahmin/100)
  - Çok adımlı tahmin özyinelemeli: ortak fiyat yolu, 8 modelin ORTALAMA getiri
    tahminiyle simüle edilir (enflasyondaki tek tahmindf mantığı); her modelin
    kendi tahmini kendi fiyat yoluna zincirlenir. Simülasyon adımlarında hacim
    değişimleri donuk (0) alınır; MA/volatilite simüle fiyattan yeniden hesaplanır.

Farklar (belgeli):
  - Enflasyonda Keras LSTM'e (n,16) girişi "zaman adımı gibi" verilmişti; burada
    PyTorch karşılığı düzgün kullanılır: son 16 bar x tüm özellikler penceresi.
  - n>2000 örnek iken GPR'da optimizer=None: L-BFGS döngüsü O(n^3) olduğundan
    ~6000 günlük barda pratik değil; kernel parametreleri başlangıçta kalır.
  - Özyinelemeli adımlarda model tahminleri iki guard ile sınırlandırılır:
    adım başına |getiri| <= tarihsel 99. yüzdelik; tüm ufukta toplam sapma
    <= 2.5*sigma*sqrt(ufuk). Ortak fiyat yolu ORTALAMA yerine MEDYAN ile
    sürülür (tek modelin uç sapması diğer modellerin özelliklerini bozmasın).
    LSTM eğitiminde gradyan kırpma (clip 1.0) vardır.

Veri kaynağı: veri_periyot.py -> VeriPeriyot sınıfı (D:\new\yeniden\ Metastock CSV).
veri_periyot.py DEĞİŞTİRİLMEZ; yalnızca salt-okuma kullanılır.

Kullanım:
    python hisse_tahmin.py                 (varsayılan: THYAO gunluk 30)
    python hisse_tahmin.py THYAO           (gunluk 30)
    python hisse_tahmin.py THYAO gunluk 60
    python hisse_tahmin.py EDATA haftalik 12
Çıktılar (betiğin yanında):
    {HISSE}_{periyot}_tahmin.csv
    grafikler/{HISSE}_{periyot}_tahmin.png
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                       # penceresiz: grafikler diske yazılır
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (SGDRegressor, LinearRegression, Lasso, Lars,
                                  BayesianRidge)
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from xgboost import XGBRegressor

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
np.random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 0) Yol ve CLI
# ---------------------------------------------------------------------------
BETIK_DIZIN = Path(__file__).resolve().parent
sys.path.insert(0, str(BETIK_DIZIN))        # veri_periyot.py aynı dizinde
from veri_periyot import VeriPeriyot         # noqa: E402  (dosya DEĞİŞTİRİLMEZ)

GRAFIK_DIZIN = BETIK_DIZIN / "grafikler"
GRAFIK_DIZIN.mkdir(exist_ok=True)

HISSE   = sys.argv[1].upper() if len(sys.argv) > 1 else "ASELS"
PERIYOT = sys.argv[2].lower() if len(sys.argv) > 2 else "gunluk"
UFUK    = int(sys.argv[3]) if len(sys.argv) > 3 else 30

if PERIYOT not in ("gunluk", "haftalik", "aylik"):
    sys.exit("Bu betik gunluk/haftalik/aylik periyotlar icindir "
             "(60dk/240dk icin ayri uzerine calisilabilir).")

print(f"HISSE={HISSE} | PERIYOT={PERIYOT} | UFUK={UFUK} | torch={torch.__version__} "
      f"| cuda={torch.cuda.is_available()}")

# ---------------------------------------------------------------------------
# 1) Veri (VeriPeriyot uzerinden — salt okuma)
# ---------------------------------------------------------------------------
vp = VeriPeriyot()
df = vp.get(HISSE, PERIYOT)
if df is None or len(df) < 120:
    sys.exit(f"{HISSE} icin yeterli {PERIYOT} veri yok "
             f"(bulunan: {'yok' if df is None else len(df)} bar). Baska hisse/periyot dene.")
df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
close = df["Close"].astype(float).values
volume = df["Volume"].astype(float).values
n = len(close)
print(f"Veri: {n} bar | {df.index[0].date()} -> {df.index[-1].date()} | son fiyat: {close[-1]:.2f}")

# ---------------------------------------------------------------------------
# 2) Özellik yapımı (t = son bilinen bar; hedef y[t] = t -> t+1 getiri %)
# ---------------------------------------------------------------------------
GERI_GETIRI = 10     # getiri gecikmeleri
GERI_HACIM  = 5      # hacim degisim gecikmeleri
MA_PENCERELER = (5, 10, 20, 60)
SEQ_LEN = 16         # LSTM bakış penceresi (enflasyondaki 16 adım gibi)

def getiriler(c):
    c = np.asarray(c, dtype=float)
    r = np.full(len(c), np.nan)
    r[1:] = (c[1:] / c[:-1] - 1.0) * 100.0
    return r

def hacim_degisimleri(v, t):
    """t. bardaki hacim degisim gecikmeleri (%). Simule adimlarda (t >= len(v)) donuk: 0."""
    if t < len(v):
        cik = []
        for i in range(1, GERI_HACIM + 1):
            if t - i >= 0 and v[t - i] > 0 and v[t - i - 1] > 0:
                cik.append((v[t - i] / v[t - i - 1] - 1.0) * 100.0)
            else:
                cik.append(0.0)
        return cik
    return [0.0] * GERI_HACIM

def ozellik_adi():
    ad = [f"getiri_lag{i}" for i in range(1, GERI_GETIRI + 1)]
    ad += [f"hacim_lag{i}" for i in range(1, GERI_HACIM + 1)]
    ad += [f"ma{k}_sapma" for k in MA_PENCERELER]
    ad += ["volatilite5", "volatilite20"]
    return ad

def ozellik_satiri(c, v, r, t):
    """t. barda bilinen bilgiyle t->t+1 getirisini tahmin etmek için satır."""
    sat = [r[t - i + 1] if t - i + 1 >= 0 else 0.0 for i in range(1, GERI_GETIRI + 1)]
    sat += hacim_degisimleri(v, t)
    for k in MA_PENCERELER:                             # MA sapmaları (bilinen bara kadar)
        bas = max(0, t - k + 1)
        ma = float(np.mean(c[bas:t + 1]))
        sat.append((c[t] / ma - 1.0) * 100.0)
    for k in (5, 20):                                   # volatilite (son k getiri std)
        bas = max(0, t - k + 1)
        sat.append(float(np.nanstd(r[bas:t + 1])))
    return sat

def egitim_matrisi(c, v):
    r = getiriler(c)
    X, y, t_index = [], [], []
    basla = max(MA_PENCERELER[-1], GERI_GETIRI, SEQ_LEN)
    for t in range(basla, len(c) - 1):
        X.append(ozellik_satiri(c, v, r, t))
        y.append((c[t + 1] / c[t] - 1.0) * 100.0)
        t_index.append(df.index[t])
    return np.array(X), np.array(y), pd.Index(t_index)

print("Ozellikler hazirlaniyor...")
X, y, t_idx = egitim_matrisi(close, volume)
FEATURES = ozellik_adi()
print(f"Egitim matrisi: X={X.shape} | ozellik sayisi={len(FEATURES)}")

# ---------------------------------------------------------------------------
# 3) 7 regresyon — enflasyon projesindeki birebir tanımlar
# ---------------------------------------------------------------------------
def modeller_yap(n_ornek):
    return {
        "SGD Regressor":        make_pipeline(StandardScaler(), SGDRegressor(max_iter=100000, tol=1e-6)),
        "Linear Regression":    LinearRegression(),
        "Lasso Regression":     Lasso(alpha=0.001),
        "Lars Regression":      Lars(random_state=0),
        "Kernel Regression":    KernelRidge(),
        "Gaussian Regression":  GaussianProcessRegressor(
                                    kernel=DotProduct() + WhiteKernel(),
                                    optimizer=None if n_ornek > 2000 else "fmin_l_bfgs_b",
                                    normalize_y=True),
        "Bayessian Regression": BayesianRidge(),
        "XGBoost Regression":   XGBRegressor(random_state=0, n_jobs=-1),
    }

# ---------------------------------------------------------------------------
# 4) LSTM — PyTorch (Keras bloğunun karşılığı, 10 tohum ortalaması)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, girdi_boyut):
        super().__init__()
        self.lstm = nn.LSTM(input_size=girdi_boyut, hidden_size=128, batch_first=True)
        self.bas = nn.Sequential(nn.Linear(128, 64), nn.SELU(),
                                 nn.Linear(64, 64), nn.ELU(),
                                 nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.bas(out[:, -1, :])

def lstm_egit(Xs, Ys, tohum, donan=100, parti=32):
    """Xs: (n, seq, feat) standardize; Ys: (n,) standardize hedef -> egitilmis model."""
    torch.manual_seed(tohum)
    np.random.seed(tohum)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMModel(Xs.shape[2]).to(dev)
    opt = torch.optim.RMSprop(model.parameters(), lr=0.01)
    kayip_f = nn.MSELoss()
    Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.tensor(Ys, dtype=torch.float32, device=dev).unsqueeze(1)
    g = torch.Generator().manual_seed(tohum)
    for _ in range(donan):
        model.train()
        karisim = torch.randperm(Xt.size(0), generator=g).to(dev)
        for i in range(0, len(karisim), parti):
            sec = karisim[i:i + parti]
            opt.zero_grad()
            kayip = kayip_f(model(Xt[sec]), yt[sec])
            kayip.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradyan patlamasını önler
            opt.step()
    model.eval()
    return model

def lstm_tahmin(model, pencere):
    """pencere: (seq, feat) standardize -> skaler standardize tahmin."""
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor(pencere[None], dtype=torch.float32, device=dev)
        return float(model(x).cpu().ravel()[0])

def lstm_dizi_yap(X_s, y_s):
    Xs, Ys = [], []
    for i in range(SEQ_LEN, len(X_s)):
        Xs.append(X_s[i - SEQ_LEN:i])
        Ys.append(y_s[i])
    return np.array(Xs), np.array(Ys)

# ---------------------------------------------------------------------------
# 5) Doğrulama (son 60 bar, tek adım ileri)
# ---------------------------------------------------------------------------
T_DOGRULAMA = 60
print(f"\n--- Dogrulama (son {T_DOGRULAMA} bar, tek adim ileri) ---")
X_e, y_e = X[:-T_DOGRULAMA], y[:-T_DOGRULAMA]
X_d, y_d = X[-T_DOGRULAMA:], y[-T_DOGRULAMA:]

for ad, model in modeller_yap(len(X_e)).items():
    model.fit(X_e, y_e)
    tah = model.predict(X_d)
    print(f"{ad:22s} MAE: {np.mean(np.abs(tah - y_d)):6.3f}%")

sx_d, sy_d = StandardScaler(), StandardScaler()
X_s_d = sx_d.fit_transform(X_e)
y_s_d = sy_d.fit_transform(y_e.reshape(-1, 1)).ravel()
Xs_e, Ys_e = lstm_dizi_yap(X_s_d, y_s_d)
X_s_tam_d = sx_d.transform(X)                       # dogrulama pencereleri icin
Xs_d = np.array([X_s_tam_d[i - SEQ_LEN:i] for i in range(len(X_e), len(X))])
m_val = lstm_egit(Xs_e, Ys_e, tohum=0)
tah_val = np.array([lstm_tahmin(m_val, p) for p in Xs_d])
tah_val = sy_d.inverse_transform(tah_val.reshape(-1, 1)).ravel()
print(f"{'LSTM (1 tohum)':22s} MAE: {np.mean(np.abs(tah_val - y_d)):6.3f}%")

# ---------------------------------------------------------------------------
# 6) Final: tüm veriyle eğitim + özyinelemeli UFUK adım tahmin
# ---------------------------------------------------------------------------
print(f"\n--- Final egitim + {UFUK} adim tahmin ---")
modeller = modeller_yap(len(X))
for ad, model in modeller.items():
    model.fit(X, y)
    print(f"{ad} egitildi")

sx_f, sy_f = StandardScaler(), StandardScaler()
X_s_f = sx_f.fit_transform(X)
y_s_f = sy_f.fit_transform(y.reshape(-1, 1)).ravel()
Xs_f, Ys_f = lstm_dizi_yap(X_s_f, y_s_f)

print("LSTM egitiliyor: 10 tohum (GPU/CPU)...")
lstm_modeller = []
for tohum in range(10):
    m = lstm_egit(Xs_f, Ys_f, tohum=tohum)
    lstm_modeller.append(m)
    print(f"  tohum {tohum} tamam")
MODEL_ADLARI = list(modeller) + ["LSTM"]

# Özyinelemeli adım sınırları (çıktı sağlamlaştırma; model tanımları değişmez):
#  - adım başına |getiri| <= tarihsel |getiri| 99. yüzdeliği (uç sapmaları kes)
#  - tüm ufuk boyunca toplam sapma <= 2.5 * sigma * sqrt(ufuk) (istatistiksel zarf)
GETIRI_SINIR = float(np.percentile(np.abs(y), 99))
YOL_ZARF = float(2.5 * np.std(y) * np.sqrt(UFUK))
_ZARF_ALT = float(close[-1] * (1 - YOL_ZARF / 100))
_ZARF_UST = float(close[-1] * (1 + YOL_ZARF / 100))
print(f"Adim siniri: +-{GETIRI_SINIR:.2f}% | ufuk zarfi: +-{YOL_ZARF:.2f}% "
      f"({_ZARF_ALT:.2f} - {_ZARF_UST:.2f})")

# --- özyinelemeli yol: ortak fiyat yolu (8 modelin ortalaması) ---
c_yol = close.copy()
r_yol = getiriler(c_yol)
getiri_satirlari = {ad: [] for ad in MODEL_ADLARI}
pencere_yolu = list(X_s_f[-(SEQ_LEN - 1):])         # son 15 standardize satır

for adim in range(UFUK):
    t = len(c_yol) - 1
    ham = np.array(ozellik_satiri(c_yol, volume, r_yol, t)).reshape(1, -1)
    olcekli = sx_f.transform(ham).ravel()
    pencere = np.array(pencere_yolu[-(SEQ_LEN - 1):] + [olcekli])

    for ad, model in modeller.items():
        tah = float(model.predict(ham)[0])
        if not np.isfinite(tah):
            tah = 0.0
        getiri_satirlari[ad].append(float(np.clip(tah, -GETIRI_SINIR, GETIRI_SINIR)))

    tah_std = np.mean([lstm_tahmin(m, pencere) for m in lstm_modeller])
    tah_lstm = float(sy_f.inverse_transform(np.array([[tah_std]])).ravel()[0])
    if not np.isfinite(tah_lstm):
        tah_lstm = 0.0
    getiri_satirlari["LSTM"].append(
        float(np.clip(tah_lstm, -GETIRI_SINIR, GETIRI_SINIR)))

    pencere_yolu.append(olcekli)                    # LSTM penceresi kaydırılır
    # Ortak yol: ORTALAMA yerine MEDYAN (tek modelin sapması yolu zehirlemesin)
    r_ort = float(np.nanmedian([getiri_satirlari[ad][-1] for ad in MODEL_ADLARI]))
    if not np.isfinite(r_ort):
        r_ort = 0.0
    yeni = c_yol[-1] * (1.0 + r_ort / 100.0)
    c_yol = np.append(c_yol, float(np.clip(yeni, _ZARF_ALT, _ZARF_UST)))
    r_yol = getiriler(c_yol)

# --- getiri yollarını fiyat seviyesine zincirle (enflasyondaki gibi) ---
if PERIYOT == "gunluk":
    tarihler = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=UFUK)
elif PERIYOT == "haftalik":
    tarihler = pd.date_range(start=df.index[-1] + pd.offsets.Week(weekday=4), periods=UFUK, freq="W-FRI")
else:
    tarihler = pd.date_range(start=df.index[-1] + pd.offsets.MonthEnd(1), periods=UFUK, freq="ME")

tahmindf = pd.DataFrame(index=tarihler)
for ad in MODEL_ADLARI:
    f = close[-1]
    yol = []
    for g in getiri_satirlari[ad]:
        f = float(np.clip(f * (1.0 + g / 100.0), _ZARF_ALT, _ZARF_UST))
        yol.append(f)
    tahmindf[ad] = yol
tahmindf["Ortalama"] = tahmindf[MODEL_ADLARI].mean(axis=1)
tahmindf["Medyan"]   = tahmindf[MODEL_ADLARI].median(axis=1)
tahmindf["Maksimum"] = tahmindf[MODEL_ADLARI].max(axis=1)
tahmindf["Minimum"]  = tahmindf[MODEL_ADLARI].min(axis=1)

csv_yol = BETIK_DIZIN / f"{HISSE}_{PERIYOT}_tahmin.csv"
tahmindf.to_csv(csv_yol, float_format="%.2f")
print(f"\nCSV yazildi: {csv_yol.name}")

# ---------------------------------------------------------------------------
# 7) Grafik: son 120 gerçek bar + tüm model yolları + sağ panelde özet
# ---------------------------------------------------------------------------
son = tahmindf.iloc[-1]
ozet_satirlar = [
    f"=== UFUK SONU TAHMINI (bar: {tarihler[-1].date()}) ===",
    f"Son gercek fiyat : {close[-1]:.2f}",
    f"Ortalama         : {son['Ortalama']:.2f}  ({(son['Ortalama']/close[-1]-1)*100:+.1f}%)",
    f"Medyan           : {son['Medyan']:.2f}",
    f"Bant (min-maks)  : {son['Minimum']:.2f} - {son['Maksimum']:.2f}",
]
for ad in MODEL_ADLARI:
    ozet_satirlar.append(
        f"  {ad:22s}: {son[ad]:.2f}  ({(son[ad]/close[-1]-1)*100:+.1f}%)")
ozet_metni = "\n".join(ozet_satirlar)
print("\n" + ozet_metni)

plt.style.use("fivethirtyeight")
fig, ax = plt.subplots(figsize=(19, 7.5))
fig.subplots_adjust(right=0.70)                     # sağda özet paneli için yer
gecmis_n = min(120, n)
ax.plot(df.index[-gecmis_n:], close[-gecmis_n:], label="Gerçek", linewidth=2.5, color="#d62728")
for ad in MODEL_ADLARI:
    ax.plot(tarihler, tahmindf[ad], linewidth=1.0, alpha=0.65)
ax.plot(tarihler, tahmindf["Ortalama"], "--", label="Ortalama Tahmin", linewidth=3, color="#1f77b4")
ax.fill_between(tarihler, tahmindf["Minimum"], tahmindf["Maksimum"], alpha=0.25, color="gray")
ax.axvline(df.index[-1], color="black", linestyle=":", linewidth=1)
ax.set_title(f"{HISSE} — {PERIYOT} — {UFUK} adım tahmin "
             f"({len(modeller)} regresyon + LSTM/PyTorch)")
ax.legend(loc=2, fontsize=9)
plt.xticks(rotation=45)

fig.text(0.715, 0.97, ozet_metni, va="top", ha="left",
         family="monospace", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                   edgecolor="#999999", alpha=0.95))

png_yol = GRAFIK_DIZIN / f"{HISSE}_{PERIYOT}_tahmin.png"
fig.savefig(png_yol, dpi=120, bbox_inches="tight")
print(f"GRAFIK KAYDEDILDI: {png_yol}")
