"""Experiments for KNOWN ZIPs (80% of test): predict month t+1 from month t.
Only one labelled transition exists (May->June), so it is the validation set; ZIP-grouped CV is used
whenever a model is fitted on the transition itself."""
import warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
from common import *

warnings.filterwarnings("ignore")
tr, te, ss = load()
tr = add_basic_features(tr)
may = tr[tr.Month == MAY].sort_values("Zip").reset_index(drop=True)
jun = tr[tr.Month == JUN].sort_values("Zip").reset_index(drop=True)
assert (may.Zip.values == jun.Zip.values).all()
y_prev, y = may[T].values.astype(float), jun[T].values.astype(float)
rows = []


def rec(name, p, note=""):
    r = metrics(y, p); r.update(strategy=name, note=note); rows.append(r)
    print(f"{name:45s} MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R2={r['R2']:.5f} {note}")


rec("K1 persistence (y_t)", y_prev)
d = (y - y_prev).mean()
rec("K2 persistence + global drift (oracle)", y_prev + d, f"drift={d:.3f}; upper bound, drift unknown in advance")
rec("K2b persistence + drift, rounded (oracle)", np.round(y_prev + d), "")
rec("K2c persistence - 1 (oracle direction)", y_prev - 1, "")

# Cross-sectional model prediction for June from June features (OOF over ZIPs) -> used for shrinkage
feat_cols = RAW_NUM + ENG_NUM
jun2 = jun.copy(); may2 = may.copy()
kf = KFold(5, shuffle=True, random_state=0)
cs = np.zeros(len(jun)); dl = np.zeros(len(jun)); dl_l1 = np.zeros(len(jun))
# delta-model features: May level, May neighbours, feature deltas
nbr = target_features(may2, may2, exclude_same=True)  # May labels of OTHER zips (neighbour context)
D = pd.DataFrame({f"d_{c}": jun[c].values - may[c].values for c in RAW_NUM if c != "AnnualMedianHouseholdIncome"})
D["y_prev"] = y_prev
D["gap_knn5"] = y_prev - nbr["knn5_mean"].values
D["gap_metro"] = y_prev - nbr["te_Metro"].values
D["gap_city"] = y_prev - nbr["te_CityKey"].values
Xd = pd.concat([D, may[feat_cols].reset_index(drop=True), nbr.reset_index(drop=True)], axis=1)
for a, b in kf.split(jun):
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)
    m.fit(Xd.iloc[a], (y - y_prev)[a]); dl[b] = m.predict(Xd.iloc[b])
    m = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)
    m.fit(Xd.iloc[a], (y - y_prev)[a]); dl_l1[b] = m.predict(Xd.iloc[b])
    # cross-sectional (level) model on May rows of other zips -> predicts level
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, verbose=-1, n_jobs=4)
    Xl = pd.concat([may[feat_cols].reset_index(drop=True), nbr.reset_index(drop=True)], axis=1)
    m.fit(Xl.iloc[a], y_prev[a]); cs[b] = m.predict(Xl.iloc[b])

rec("K3 delta model LGBM-L2 (grouped CV)", y_prev + dl, "learn y_t+1 - y_t from levels, neighbours, feature deltas")
rec("K3b delta model LGBM-L1 (grouped CV)", y_prev + dl_l1, "")
rec("K3c delta model L2, rounded", np.round(y_prev + dl), "")
rec("K3d delta model L1, rounded", np.round(y_prev + dl_l1), "")
for a in [0.02, 0.05, 0.1, 0.2]:
    rec(f"K4 shrink to cross-sectional model a={a}", (1 - a) * y_prev + a * cs, "")
print("corr(delta, gap to knn5) =", np.corrcoef(y - y_prev, D.gap_knn5)[0, 1])
print("corr(delta, y_prev) =", np.corrcoef(y - y_prev, y_prev)[0, 1])
# boundary behaviour
print(pd.crosstab(np.clip(y_prev, 0, 100) == 0, y - y_prev))
pd.DataFrame(rows).to_csv(f"{OUT}/known_zip_results.csv", index=False)

# ---- robustness: which inputs drive the delta model? (feature deltas shift strongly Jun->Jul) ----
variants = {
    "K5 delta-L1, levels+neighbours only (no feature deltas)": [c for c in Xd.columns if not c.startswith("d_")],
    "K6 delta-L1, y_prev + neighbour gaps only": ["y_prev", "gap_knn5", "gap_metro", "gap_city"],
    "K7 delta-L1, y_prev + State codes": ["y_prev"],
}
Xd["state_code"] = may.State.astype("category").cat.codes.values
variants["K7 delta-L1, y_prev + State codes"].append("state_code")
for name, cols in variants.items():
    p = np.zeros(len(y))
    for a, b in kf.split(jun):
        m = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)
        m.fit(Xd.iloc[a][cols], (y - y_prev)[a]); p[b] = m.predict(Xd.iloc[b][cols])
    rec(name + " rounded", np.round(y_prev + p), "")
    rec(name + " raw", y_prev + p, "")
m = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4).fit(Xd, y - y_prev)
imp = pd.Series(m.booster_.feature_importance("gain"), Xd.columns).sort_values(ascending=False)
print("delta model gain importance (top 15):\n", (imp / imp.sum()).head(15).round(3))
imp.to_csv(f"{OUT}/known_delta_importance.csv")
pd.DataFrame(rows).to_csv(f"{OUT}/known_zip_results.csv", index=False)
