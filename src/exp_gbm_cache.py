"""LightGBM models from pipeline.get_models on a cached fold set, for blending with TabPFN.
Usage: python exp_gbm_cache.py v5 -> outputs/rmse_oof_GBM_v5_s0.csv (lgb, lgb_slow_h_w)"""
import pickle, sys, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import OUT, load, JUN
from pipeline import get_models

warnings.filterwarnings("ignore")
NAME = sys.argv[1] if len(sys.argv) > 1 else "v5"
folds = pickle.load(open(f"{OUT}/fold_cache_{NAME}.pkl", "rb"))
tr, _, _ = load(); zips = np.sort(tr[tr.Month == JUN].Zip.values)
splits = list(KFold(5, shuffle=True, random_state=200).split(zips))
M = get_models(0); out = {"Zip": zips}; y = np.zeros(len(zips))
for col, base, weighted in (("lgb", "lgb", False), ("lgb_slow_h_w", "lgb_slow_h", True)):
    oof = np.zeros(len(zips))
    for (Xt, yt, Xv, yv), (a, b) in zip(folds, splits):
        m = M[base](); kw = {"sample_weight": np.sqrt(Xt["SampleSize"].clip(lower=100).values)} if weighted else {}
        m.fit(Xt, yt, **kw); oof[b] = np.clip(m.predict(Xv), 0, 100); y[b] = yv
    out[col] = oof; print(col, round(float(np.sqrt(((oof - y) ** 2).mean())), 4), flush=True)
out = pd.DataFrame(out); out.insert(1, "y", y)
out.to_csv(f"{OUT}/rmse_oof_GBM_{NAME}_s0.csv", index=False)
