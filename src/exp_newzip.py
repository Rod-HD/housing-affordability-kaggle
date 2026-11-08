"""Experiments for UNSEEN ZIPs (20% of test). Validation = repeated GroupKFold over ZIPs on June data:
held-out ZIPs play the role of new test ZIPs, remaining ZIPs are the 'labeled neighbours'.
All target-derived features for training rows are computed out-of-fold (nested) -> no leakage."""
import sys, time, warnings, json
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from common import *

warnings.filterwarnings("ignore")
SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "2"])]
AUG_MAY = "--may" in sys.argv
TAG = "_may" if AUG_MAY else ""

tr, te, ss = load()
tr = add_basic_features(tr)
tr["state_code"] = tr["State"].astype("category").cat.codes
jun = tr[tr.Month == JUN].reset_index(drop=True)
may = tr[tr.Month == MAY].reset_index(drop=True)
zips = jun.Zip.values

FS = {
    "raw": RAW_NUM + ["state_code"],
    "eng": RAW_NUM + ENG_NUM + ["state_code"],
}


def models():
    return {
        "ridge_eng": ("eng", lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(10))),
        "ridge_full": ("full", lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(10))),
        "knnreg_full": ("full", lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                      KNeighborsRegressor(25, weights="distance"))),
        "rf_full": ("full", lambda: make_pipeline(SimpleImputer(strategy="median"),
                                                  RandomForestRegressor(500, min_samples_leaf=3, max_features=0.4, n_jobs=8, random_state=0))),
        "et_full": ("full", lambda: make_pipeline(SimpleImputer(strategy="median"),
                                                  ExtraTreesRegressor(500, min_samples_leaf=3, max_features=0.5, n_jobs=8, random_state=0))),
        "hgb_full": ("full", lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=31,
                                                                   min_samples_leaf=20, l2_regularization=1.0, random_state=0)),
        "lgb_raw": ("raw", lambda: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)),
        "lgb_eng": ("eng", lambda: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)),
        "lgb_te": ("te", lambda: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)),
        "lgb_full": ("full", lambda: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4)),
        "lgb_full_l1": ("full", lambda: lgb.LGBMRegressor(objective="l1", n_estimators=1200, learning_rate=0.03, num_leaves=31,
                                                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                                                          colsample_bytree=0.8, verbose=-1, n_jobs=4)),
        "lgb_full_huber": ("full", lambda: lgb.LGBMRegressor(objective="huber", alpha=8.0, n_estimators=1000, learning_rate=0.03,
                                                             num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                                             colsample_bytree=0.8, verbose=-1, n_jobs=4)),
        "xgb_full": ("full", lambda: xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=5,
                                                      subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=8, tree_method="hist")),
        "cat_full": ("full", lambda: CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=3,
                                                       verbose=0, random_seed=0, thread_count=8)),
        "cat_full_mae": ("full", lambda: CatBoostRegressor(iterations=2000, learning_rate=0.04, depth=6, l2_leaf_reg=3,
                                                           loss_function="MAE", verbose=0, random_seed=0, thread_count=8)),
    }


def build_fold(trz, vaz, seed):
    lab = jun[jun.Zip.isin(trz)]
    val = jun[jun.Zip.isin(vaz)]
    ftr = oof_target_features(lab, seed=seed)
    fva = target_features(lab, val)
    Xtr = pd.concat([lab, ftr], axis=1)
    ytr = lab[T].values
    if AUG_MAY:  # add May rows of the same training ZIPs, with features from May labels
        labm = may[may.Zip.isin(trz)]
        fm = oof_target_features(labm, seed=seed)
        Xtr = pd.concat([Xtr, pd.concat([labm, fm], axis=1)], ignore_index=True)
        ytr = np.r_[ytr, labm[T].values]
    Xva = pd.concat([val, fva], axis=1)
    return Xtr, ytr, Xva, val[T].values, ftr.columns.tolist()


rows, oof = [], {}
t0 = time.time()
for seed in SEEDS:
    kf = KFold(5, shuffle=True, random_state=100 + seed)
    oof_seed = {}
    for f, (a, b) in enumerate(kf.split(zips)):
        Xtr, ytr, Xva, yva, tcols = build_fold(set(zips[a]), set(zips[b]), seed)
        te_cols = [c for c in tcols if c.startswith(("te_", "cnt_"))]
        FS["te"] = FS["eng"] + te_cols
        FS["full"] = FS["eng"] + tcols
        # simple baselines
        base = {"global_mean": np.full(len(yva), ytr.mean()), "state_mean": Xva["te_State"].values,
                "metro_te": Xva["te_Metro"].values, "city_te": Xva["te_CityKey"].values, "zip3_te": Xva["te_zip3"].values,
                "knn1": Xva["knn1_mean"].values, "knn5": Xva["knn5_mean"].values, "knn10_w": Xva["knn10_wmean"].values}
        preds = dict(base)
        for name, (fs, mk) in models().items():
            m = mk()
            m.fit(Xtr[FS[fs]], ytr)
            preds[name] = np.clip(m.predict(Xva[FS[fs]]), 0, 100)
        for name, p in preds.items():
            r = metrics(yva, p); r.update(model=name, seed=seed, fold=f, n_train=len(ytr))
            rows.append(r)
            oof_seed.setdefault(name, np.zeros(len(zips)))[b] = p
        print(f"seed {seed} fold {f} done {time.time()-t0:.0f}s  lgb_full MAE={metrics(yva, preds['lgb_full'])['MAE']:.3f}", flush=True)
    for k, v in oof_seed.items():
        oof[f"{k}|{seed}"] = v

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/newzip_cv_folds{TAG}.csv", index=False)
summ = res.groupby("model")[["MAE", "RMSE", "R2"]].agg(["mean", "std"]).sort_values(("MAE", "mean"))
print(summ.round(3).to_string())
summ.to_csv(f"{OUT}/newzip_cv_summary{TAG}.csv")
o = pd.DataFrame(oof); o.insert(0, "Zip", zips); o.insert(1, "y", jun[T].values)
o.to_csv(f"{OUT}/newzip_oof{TAG}.csv", index=False)
