"""Independent model families on the cached July-proxy folds (same folds/order as exp_zillow.py):
MLP (PyTorch, quantile-transformed inputs, Huber loss, sqrt(SampleSize) weights, seed bagging) and XGBoost (Huber + weights).
Writes OOF predictions for ensembling: outputs/rmse_oof_NN_s0.csv"""
import os, pickle, sys, time, warnings
import numpy as np, pandas as pd, torch, xgboost as xgb
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import KFold
from common import OUT, load, JUN, T
from nn_model import prep, mlp_fit_predict

warnings.filterwarnings("ignore"); torch.set_num_threads(8)
folds = pickle.load(open(f"{OUT}/fold_cache_full.pkl", "rb"))
tr, _, _ = load(); zips = np.sort(tr[tr.Month == JUN].Zip.values)
splits = list(KFold(5, shuffle=True, random_state=200).split(zips))


oof = {"mlp": np.zeros(len(zips)), "xgb_h_w": np.zeros(len(zips))}
y_all = np.zeros(len(zips)); t0 = time.time()
for f, ((Xt, yt, Xv, yv), (a, b)) in enumerate(zip(folds, splits)):
    y_all[b] = yv
    wt = np.sqrt(Xt["SampleSize"].clip(lower=100).values)
    Zt, Zv = prep(Xt, Xv)
    oof["mlp"][b] = np.clip(np.mean([mlp_fit_predict(Zt, yt, wt, Zv, s) for s in range(3)], 0), 0, 100)
    m = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=12.0, n_estimators=2500, learning_rate=0.01,
                         max_depth=6, min_child_weight=10, subsample=0.7, colsample_bytree=0.5, reg_lambda=5,
                         tree_method="hist", n_jobs=8, random_state=0)
    m.fit(Xt, yt, sample_weight=wt); oof["xgb_h_w"][b] = np.clip(m.predict(Xv), 0, 100)
    print(f"fold {f} {time.time()-t0:.0f}s  mlp={np.sqrt(((oof['mlp'][b]-yv)**2).mean()):.3f}  "
          f"xgb_h_w={np.sqrt(((oof['xgb_h_w'][b]-yv)**2).mean()):.3f}", flush=True)
for k, v in oof.items():
    print(k, "RMSE", round(float(np.sqrt(((v - y_all) ** 2).mean())), 4))
o = pd.DataFrame(oof); o.insert(0, "Zip", zips); o.insert(1, "y", y_all)
o.to_csv(f"{OUT}/rmse_oof_NN_s0.csv", index=False)
