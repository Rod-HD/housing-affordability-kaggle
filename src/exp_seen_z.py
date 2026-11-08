"""Seen ZIPs (weight 0.2): can the ZIP's OWN Zillow change (known for the target month) improve on persistence?
Honest protocol: May->June transition, ZIP-grouped CV, features use only the ZIP's own data (no same-period
neighbour/state target deltas, which would be unavailable for July)."""
import warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from common import *

warnings.filterwarnings("ignore")
tr, te, ss = load()
z = pd.read_csv(ZILLOW_PATH)
d = tr.merge(z, on=["Zip", "Month"], how="left")
m = d[d.Month == MAY].set_index("Zip").sort_index(); j = d[d.Month == JUN].set_index("Zip").loc[m.index]
y_prev, y = m[T].values.astype(float), j[T].values.astype(float)
X = pd.DataFrame(index=m.index)
X["y_prev"] = y_prev
X["dlog_raw"] = np.log(j.zhvi_raw / m.zhvi_raw)
X["dlog_sm"] = np.log(j.zhvi / m.zhvi)
X["dlog_rent"] = np.log(j.RentIndex / m.RentIndex)
X["log_pti"] = np.log(j.zhvi / j.AnnualMedianHouseholdIncome)
X["owner"] = j.OwnerPercent
X["d_sample"] = np.log(j.SampleSize / m.SampleSize)
X["d_owner"] = j.OwnerPercent - m.OwnerPercent
X["dlog_raw_x_slope"] = X.dlog_raw * (y_prev * (100 - y_prev)) / 100   # logistic-type sensitivity
X = X.fillna(0)
rows = []
def rec(n, p): r = metrics(y, p); r["strategy"] = n; rows.append(r); print(f"{n:55s} RMSE={r['RMSE']:.4f} MAE={r['MAE']:.4f}")
rec("persistence", y_prev)
kf = KFold(5, shuffle=True, random_state=0)
for name, cols in [("ridge: dlog_raw*slope", ["dlog_raw_x_slope"]),
                   ("ridge: zillow deltas", ["dlog_raw", "dlog_sm", "dlog_raw_x_slope", "dlog_rent"]),
                   ("ridge: all own features", list(X.columns))]:
    p = np.zeros(len(y))
    for a, b in kf.split(X):
        mdl = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X.iloc[a][cols], (y - y_prev)[a]); p[b] = y_prev[b] + mdl.predict(X.iloc[b][cols])
    rec(name, p)
p = np.zeros(len(y))
for a, b in kf.split(X):
    mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, verbose=-1, n_jobs=4)
    mdl.fit(X.iloc[a], (y - y_prev)[a]); p[b] = y_prev[b] + mdl.predict(X.iloc[b])
rec("lgbm: all own features (L2)", p)
pd.DataFrame(rows).to_csv(f"{OUT}/seen_zillow_results.csv", index=False)
