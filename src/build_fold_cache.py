"""Build cached July-proxy fold matrices for a feature set (same folds as tune_v2.py).
Usage: python build_fold_cache.py v4   -> outputs/fold_cache_v4.pkl (v3 + purchase-year costs)
       python build_fold_cache.py v5   -> v4 + Realtor.com listing data
       v6 = v5 + HUD SAFMR rents, v7 = v5 + Realtor hotness, v8 = v5 + both"""
import os, sys, pickle, warnings
os.environ["EXTRA_GEO"] = "1"
import numpy as np
from sklearn.model_selection import KFold
from common import *
from pipeline import build_matrices, drift_table, dedrift
from extra_sources import add_realtor, add_hud, add_hot

warnings.filterwarnings("ignore")
NAME = sys.argv[1] if len(sys.argv) > 1 else "v4"
EXT = {"v3": lambda d: add_irs(add_zillow(add_geo(d))),
       "v4": lambda d: add_hist(add_irs(add_zillow(add_geo(d)))),
       "v5": lambda d: add_realtor(add_hist(add_irs(add_zillow(add_geo(d))))),
       "v6": lambda d: add_hud(add_realtor(add_hist(add_irs(add_zillow(add_geo(d)))))),
       "v7": lambda d: add_hot(add_realtor(add_hist(add_irs(add_zillow(add_geo(d)))))),
       "v8": lambda d: add_hot(add_hud(add_realtor(add_hist(add_irs(add_zillow(add_geo(d)))))))}[NAME]
tr, te, ss = load()
jun = tr[tr.Month == JUN].set_index("Zip").sort_index()
jul = te[te.Zip.isin(jun.index)].set_index("Zip").loc[jun.index]; jul[T] = jun[T]
shift = drift_table(jun.reset_index(), jul.reset_index())
J = EXT(jun.reset_index()).set_index("Zip"); Jd = EXT(dedrift(jul.reset_index(), shift)).set_index("Zip")
folds = []
for a, b in KFold(5, shuffle=True, random_state=200).split(J.index.values):
    Xt, yt, Xv = build_matrices(J.iloc[a].reset_index(), Jd.iloc[a].reset_index(), Jd.iloc[b].reset_index())
    folds.append((Xt, yt, Xv, J.iloc[b][T].values))
pickle.dump(folds, open(f"{OUT}/fold_cache_{NAME}.pkl", "wb"))
print(NAME, "features:", folds[0][0].shape[1])
