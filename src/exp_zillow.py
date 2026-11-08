"""Unseen-ZIP RMSE with Zillow features (July-proxy + de-drift + geo + extra geo), same folds as exp_rmse.py.
argv: tag models seed"""
import os, sys, warnings, time
os.environ["EXTRA_GEO"] = "1"
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import *
from pipeline import fit_predict, drift_table, dedrift

warnings.filterwarnings("ignore")
TAG = sys.argv[1] if len(sys.argv) > 1 else "Z"
MODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["lgb", "cat", "lgb_slow"]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
tr, te, ss = load()
jun = tr[tr.Month == JUN].set_index("Zip").sort_index()
jul = te[te.Zip.isin(jun.index)].set_index("Zip").loc[jun.index]; jul[T] = jun[T]
shift = drift_table(jun.reset_index(), jul.reset_index())
EXT = (lambda d: add_irs(add_zillow(add_geo(d)))) if os.environ.get("IRS") == "1" else (lambda d: add_zillow(add_geo(d)))
if os.environ.get("ACS") == "1":
    _ext = EXT; EXT = lambda d: add_acs(_ext(d))
J = EXT(jun.reset_index()).set_index("Zip")
Jd = EXT(dedrift(jul.reset_index(), shift)).set_index("Zip")
zips = J.index.values; y_all = J[T].values
AUG = os.environ.get("AUG") == "1"
if AUG:  # May rows (month-matched Zillow) as extra training rows
    may = tr[tr.Month == MAY].set_index("Zip").loc[J.index]
    M = EXT(may.reset_index()).set_index("Zip")
oof = {m: np.zeros(len(zips)) for m in MODELS}; t0 = time.time(); rows = []
for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=200 + SEED).split(zips)):
    preds, cols = fit_predict(J.iloc[a].reset_index(), Jd.iloc[a].reset_index(), Jd.iloc[b].reset_index(), MODELS, seed=SEED,
                              extra_train=M.iloc[a].reset_index() if AUG else None)
    for m, p in preds.items():
        oof[m][b] = p; r = metrics(y_all[b], p); r.update(model=m, fold=f, cfg=TAG); rows.append(r)
    print(f"{TAG} fold {f} {time.time()-t0:.0f}s nfeat={len(cols)} " + " ".join(f"{m}={metrics(y_all[b], p)['RMSE']:.3f}" for m, p in preds.items()), flush=True)
res = pd.DataFrame(rows)
print(res.groupby("model")[["RMSE", "MAE"]].agg(["mean", "std"]).sort_values(("RMSE", "mean")).round(3).to_string())
o = pd.DataFrame(oof); o.insert(0, "Zip", zips); o.insert(1, "y", y_all)
o.to_csv(f"{OUT}/rmse_oof_{TAG}_s{SEED}.csv", index=False); res.to_csv(f"{OUT}/rmse_folds_{TAG}_s{SEED}.csv", index=False)
