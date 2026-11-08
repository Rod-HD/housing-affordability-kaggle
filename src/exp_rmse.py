"""RMSE-focused experiments for NEW ZIPs (metric = 0.2*RMSE_seen + 0.8*RMSE_new).
July-proxy + de-drift + geo setting (the one that matched the public LB within ~0.1).
Config A = current geo features; config B = + similarity-weighted neighbour features (EXTRA_GEO=1)."""
import os, sys, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import *
from pipeline import fit_predict, drift_table, dedrift

warnings.filterwarnings("ignore")
CFG = sys.argv[1] if len(sys.argv) > 1 else "A"
MODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["lgb", "lgb_huber", "cat", "xgb", "cat_deep", "lgb_slow",
                                                          "lgb_resid", "cat_resid"]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
if CFG.startswith("B"):
    os.environ["EXTRA_GEO"] = "1"
tr, te, ss = load()
jun = tr[tr.Month == JUN].set_index("Zip").sort_index()
jul = te[te.Zip.isin(jun.index)].set_index("Zip").loc[jun.index]; jul[T] = jun[T]
shift = drift_table(jun.reset_index(), jul.reset_index())
J = add_geo(jun.reset_index()).set_index("Zip")
Jd = add_geo(dedrift(jul.reset_index(), shift)).set_index("Zip")
zips = J.index.values; y_all = J[T].values
oof = {m: np.zeros(len(zips)) for m in MODELS}; t0 = time.time(); rows = []
for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=200 + SEED).split(zips)):
    preds, cols = fit_predict(J.iloc[a].reset_index(), Jd.iloc[a].reset_index(), Jd.iloc[b].reset_index(), MODELS, seed=SEED)
    for m, p in preds.items():
        oof[m][b] = p; r = metrics(y_all[b], p); r.update(model=m, fold=f, cfg=CFG); rows.append(r)
    print(f"cfg {CFG} fold {f} {time.time()-t0:.0f}s " + " ".join(f"{m}={metrics(y_all[b], p)['RMSE']:.3f}" for m, p in preds.items()), flush=True)
res = pd.DataFrame(rows)
print(res.groupby("model")[["RMSE", "MAE"]].agg(["mean", "std"]).sort_values(("RMSE", "mean")).round(3).to_string())
o = pd.DataFrame(oof); o.insert(0, "Zip", zips); o.insert(1, "y", y_all)
o.to_csv(f"{OUT}/rmse_oof_{CFG}_s{SEED}.csv", index=False); res.to_csv(f"{OUT}/rmse_folds_{CFG}_s{SEED}.csv", index=False)
