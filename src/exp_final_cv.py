"""Final repeated validation of the chosen pipeline (3 seeds x 5 folds) in the July-proxy de-drifted setting,
plus the combined (known + new ZIP) score estimate weighted like the test set."""
import sys, warnings, time
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import *
from pipeline import fit_predict, drift_table, dedrift
from make_submission import tuned_models  # noqa (re-uses tuned params)

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2]
NAMES = ["lgb_tuned", "lgb_huber", "cat"]
tr, te, ss = load()
jun = tr[tr.Month == JUN].set_index("Zip").sort_index()
jul = te[te.Zip.isin(jun.index)].set_index("Zip").loc[jun.index]
jul[T] = jun[T]
shift = drift_table(jun.reset_index(), jul.reset_index())
juld = dedrift(jul.reset_index(), shift).set_index("Zip")
G = {False: (jun, juld), True: (add_geo(jun.reset_index()).set_index("Zip"), add_geo(juld.reset_index()).set_index("Zip"))}
zips = jun.index.values
rows = []; t0 = time.time()
for seed in SEEDS:
    for f, (a, b) in enumerate(KFold(5, shuffle=True, random_state=200 + seed).split(zips)):
        A, B = zips[a], zips[b]; y = jun.loc[B, T].values
        for geo in (False, True):
            J, Jd = G[geo]
            preds, _ = fit_predict(J.loc[A].reset_index(), Jd.loc[A].reset_index(), Jd.loc[B].reset_index(), NAMES,
                                   seed=seed, models=tuned_models(geo, seed))
            preds["ENSEMBLE"] = np.mean([preds[n] for n in NAMES], 0)
            for n, p in preds.items():
                r = metrics(y, p); r.update(model=n, geo=geo, seed=seed, fold=f); rows.append(r)
        print(f"seed {seed} fold {f} {time.time()-t0:.0f}s", flush=True)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/final_cv_folds.csv", index=False)
s = res.groupby(["geo", "model"])[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).round(3)
print(s.to_string()); s.to_csv(f"{OUT}/final_cv_summary.csv")
# combined estimate, test composition: 6830 known (persistence) + 1704 new
nk, nn = 6830, 1704
known_mae, known_rmse = 0.7117, 1.2524
for geo in (False, True):
    e = res[(res.geo == geo) & (res.model == "ENSEMBLE")]
    mae = (nk * known_mae + nn * e.MAE.mean()) / (nk + nn)
    rmse = np.sqrt((nk * known_rmse ** 2 + nn * (e.RMSE ** 2).mean()) / (nk + nn))
    print(f"COMBINED estimate geo={geo}: MAE={mae:.3f}  RMSE={rmse:.3f}")
