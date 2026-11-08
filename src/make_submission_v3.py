"""v3 submission (metric 0.2*RMSE_seen + 0.8*RMSE_new) with EXTERNAL public data:
Zillow ZHVI/ZORI (cited by the competition), GeoNames lat/lon, IRS SOI ZIP income distribution, CDC PLACES SDOH.
Seen ZIPs: June label + mechanistic adjustment for the ZIP's own Zillow home-value change Jun->Jul
          (y += beta * dlog(ZHVI_raw) * y(100-y)/100, beta fitted on May->Jun without intercept; CV RMSE 1.252 -> 1.183).
New ZIPs: NNLS-weighted GBM ensemble (weights from ens_rmse.py on the same feature set), 3 seeds."""
import os, sys, json, warnings
os.environ["EXTRA_GEO"] = "1"
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from common import *
from pipeline import fit_predict, drift_table, dedrift

warnings.filterwarnings("ignore")
NAME = sys.argv[1] if len(sys.argv) > 1 else "sub07_zillow_irs_v3"
SEEDS = [0, 1, 2]
W = {k.split("@")[0]: v for k, v in json.load(open(f"{OUT}/ens_rmse_weights.json")).items() if v > 0.005}
s = sum(W.values()); W = {k: v / s for k, v in W.items()}
print("weights:", {k: round(v, 3) for k, v in W.items()})
EXT = lambda d: add_irs(add_zillow(add_geo(d)))

tr, te, ss = load()
jun = tr[tr.Month == JUN].reset_index(drop=True); may = tr[tr.Month == MAY].reset_index(drop=True)
known = te.Zip.isin(jun.Zip)
shift = drift_table(jun, te[known])
jul_dd = dedrift(te, shift)
ref = jul_dd[known].copy(); ref[T] = ref.Zip.map(jun.set_index("Zip")[T]).values
J, R, Q = EXT(jun), EXT(ref), EXT(jul_dd[~known].copy())
preds = []
for sd in SEEDS:
    p, cols = fit_predict(J, R, Q, list(W), seed=sd)
    preds.append(sum(W[m] * p[m] for m in W)); print(f"seed {sd}: {len(cols)} features, mean {preds[-1].mean():.2f}", flush=True)
p_new = np.clip(np.mean(preds, 0), 0, 100)

# seen ZIPs: persistence + Zillow-change adjustment
z = pd.read_csv(ZILLOW_PATH).set_index(["Zip", "Month"])["zhvi_raw"]
def zr(zips, mon): return np.array([z.get((k, mon), np.nan) for k in zips])
mz = may.set_index("Zip").loc[jun.Zip]
xm = np.nan_to_num(np.log(zr(jun.Zip, JUN) / zr(jun.Zip, MAY)) * mz[T].values * (100 - mz[T].values) / 100)
beta = LinearRegression(fit_intercept=False).fit(xm[:, None], jun[T].values - mz[T].values).coef_[0]
kz = te.loc[known, "Zip"].values
y_jun = pd.Series(jun.set_index("Zip")[T]).loc[kz].values.astype(float)
xj = np.nan_to_num(np.log(zr(kz, JUL) / zr(kz, JUN)) * y_jun * (100 - y_jun) / 100)
p_seen = np.clip(y_jun + beta * xj, 0, 100)
print(f"seen adjustment beta={beta:.3f}; mean |adj|={np.abs(beta * xj).mean():.3f}")

sub = ss[["Zip", "State"]].copy(); sub["Zip"] = sub.Zip.astype(int).astype(str).str.zfill(5)
sub[T] = np.nan
sub.loc[known.values, T] = p_seen
sub.loc[~known.values, T] = p_new
assert sub[T].notna().all() and sub[T].between(0, 100).all()
sub.to_csv(f"{SUB}/{NAME}.csv", index=False)
print("wrote", NAME)
