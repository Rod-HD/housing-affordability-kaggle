"""Group ablation on the cached July-proxy fold matrices (full external feature set, 152 features):
drop one feature group at a time and re-score a fixed LightGBM (lgb_slow settings)."""
import pickle, re, time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from common import OUT

warnings.filterwarnings("ignore")
folds = pickle.load(open(f"{OUT}/fold_cache_full.pkl", "rb"))
cols = folds[0][0].columns.tolist()
GROUPS = {
    "knn_zipnum": r"^knn\d+_",
    "rel_nbr": r"^rel_",
    "target_enc": r"^(te|cnt)_(State|Metro|CityKey|zip3|zip2)$",
    "geo_knn": r"^geo\d+_|^geo_r\d+",
    "geo_sim": r"^geo_sim|^geo\d+_idw|^rel_rent_burden_geo",
    "gwr": r"^gwr",
    "struct_lognormal": r"^struct_",
    "irs": r"^irs_",
    "sdoh": r"^sdoh_",
    "zillow_tiers": r"^log_pti_(low|high|sfr|condo|b2|b3|b4)$|zhvi_tier_spread",
    "household_raw": r"^(AvgNumberof|AvgHousehold|AvgGenerations|YoungAdult|SeniorAdult|MedianHome)",
    "lat_lon_state": r"^(lat|lon|state_code)$",
}


def score(keep):
    se = n = 0
    for Xt, yt, Xv, yv in folds:
        m = lgb.LGBMRegressor(n_estimators=2500, learning_rate=0.01, num_leaves=31, min_child_samples=30, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=8, random_state=0)
        m.fit(Xt[keep], yt); p = np.clip(m.predict(Xv[keep]), 0, 100); se += ((p - yv) ** 2).sum(); n += len(yv)
    return np.sqrt(se / n)


t0 = time.time(); rows = []
base = score(cols); print(f"ALL ({len(cols)} feats): {base:.4f}", flush=True)
rows.append({"dropped": "none", "n_feats": len(cols), "RMSE": base, "delta": 0.0})
for g, rx in GROUPS.items():
    drop = [c for c in cols if re.search(rx, c)]
    if not drop: continue
    r = score([c for c in cols if c not in drop])
    rows.append({"dropped": g, "n_feats": len(cols) - len(drop), "RMSE": r, "delta": r - base})
    print(f"drop {g:18s} (-{len(drop):3d}) RMSE {r:.4f}  delta {r-base:+.4f}  {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/ablation_results.csv", index=False)
