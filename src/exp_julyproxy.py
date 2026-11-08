"""July-proxy validation: known ZIPs are treated as 'new' and predicted from their JULY features
(the real test distribution), scored against their June label (July label ~ June label, persistence MAE ~0.7).
Compares feature-drift handling strategies."""
import sys, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import *
from pipeline import *

warnings.filterwarnings("ignore")
SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0"])]
MODELS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["lgb", "cat"]
tr, te, ss = load()
jun = tr[tr.Month == JUN].set_index("Zip").sort_index()
may = tr[tr.Month == MAY].set_index("Zip").sort_index()
jul = te[te.Zip.isin(jun.index)].set_index("Zip").loc[jun.index]
jul[T] = jun[T]                                  # latest known label
shift = drift_table(jun.reset_index(), jul.reset_index())
print("median June->July shift:\n", shift.round(3).to_string())
zips = jun.index.values
rows = []
t0 = time.time()
for seed in SEEDS:
    for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=100 + seed).split(zips)):
        A, B = zips[a], zips[b]
        y = jun.loc[B, T].values
        train = jun.loc[A].reset_index()
        scen = {
            "S1 June->June (standard CV)": (jun.loc[A].reset_index(), jun.loc[B].reset_index()),
            "S2 July feats, July ref (raw)": (jul.loc[A].reset_index(), jul.loc[B].reset_index()),
            "S3 July feats de-drifted": (dedrift(jul.loc[A].reset_index(), shift), dedrift(jul.loc[B].reset_index(), shift)),
            "S4 July query, June ref (naive mix)": (jun.loc[A].reset_index(), jul.loc[B].reset_index()),
        }
        for sname, (ref, q) in scen.items():
            preds, _ = fit_predict(train, ref, q, MODELS, seed=seed)
            preds["mean_" + "+".join(MODELS)] = np.mean([preds[m] for m in MODELS], axis=0)
            for m, p in preds.items():
                r = metrics(y, p); r.update(scenario=sname, model=m, seed=seed, fold=f); rows.append(r)
        # augmentation with May rows (same training ZIPs) in the July-dedrift scenario
        preds, _ = fit_predict(train, *scen["S3 July feats de-drifted"], MODELS, seed=seed,
                               extra_train=may.loc[A].reset_index())
        preds["mean_" + "+".join(MODELS)] = np.mean([preds[m] for m in MODELS], axis=0)
        for m, p in preds.items():
            r = metrics(y, p); r.update(scenario="S5 S3 + May rows augmentation", model=m, seed=seed, fold=f); rows.append(r)
        print(f"seed {seed} fold {f} {time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/julyproxy_folds.csv", index=False)
s = res.groupby(["scenario", "model"])[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(3)
print(s.to_string()); s.to_csv(f"{OUT}/julyproxy_summary.csv")
