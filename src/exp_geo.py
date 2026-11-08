"""Does EXTERNAL geographic data (ZIP lat/lon + county, GeoNames) improve unseen-ZIP prediction?
Same GroupKFold protocol (June->June) with and without geo features."""
import sys, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import *
from pipeline import fit_predict

warnings.filterwarnings("ignore")
SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0"])]
MODELS = ["lgb", "lgb_huber", "cat"]
tr, te, ss = load()
jun = tr[tr.Month == JUN].reset_index(drop=True)
jun_geo = add_geo(jun)
print("geo matched exactly:", jun.Zip.isin(pd.read_csv(GEO_PATH, dtype={"postal_code": str}).query("postal_code.str.isdigit()", engine="python").postal_code.astype(int)).mean())
zips = jun.Zip.values
rows, oof = [], {}
t0 = time.time()
for seed in SEEDS:
    for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=100 + seed).split(zips)):
        y = jun.iloc[b][T].values
        for tag, D in [("nogeo", jun), ("geo", jun_geo)]:
            preds, cols = fit_predict(D.iloc[a], D.iloc[a], D.iloc[b], MODELS, seed=seed)
            preds["ens3"] = np.mean([preds[m] for m in MODELS], 0)
            for m, p in preds.items():
                r = metrics(y, p); r.update(model=f"{m}_{tag}", seed=seed, fold=f); rows.append(r)
                oof.setdefault(f"{m}_{tag}|{seed}", np.zeros(len(zips)))[b] = p
        print(f"seed {seed} fold {f} {time.time()-t0:.0f}s geo-ens MAE={rows[-1]['MAE']:.3f}", flush=True)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/geo_cv_folds.csv", index=False)
s = res.groupby("model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).sort_values(("MAE", "mean")).round(3)
print(s.to_string()); s.to_csv(f"{OUT}/geo_cv_summary.csv")
o = pd.DataFrame(oof); o.insert(0, "Zip", zips); o.insert(1, "y", jun[T].values); o.to_csv(f"{OUT}/geo_oof.csv", index=False)
