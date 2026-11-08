"""Core fit/predict routine for unseen ZIPs, reused by the July-proxy validation and the final submission."""
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from common import *

DRIFT_COLS = ["SampleSize", "OwnerPercent", "RenterPercent", "MedianHomeLengthofResidence", "AvgNumberofChildren",
              "AvgNumberofAdults", "AvgHouseholdSize", "AvgGenerationsinHousehold", "YoungAdultinHouseholdPercent",
              "SeniorAdultinHouseholdPercent"]


def drift_table(prev_rows, cur_rows):
    """Median per-feature shift between two months, measured on the same ZIPs."""
    a = prev_rows.set_index("Zip")[DRIFT_COLS]
    b = cur_rows.set_index("Zip")[DRIFT_COLS]
    idx = a.index.intersection(b.index)
    return (b.loc[idx] - a.loc[idx]).median()


def dedrift(rows, shift):
    rows = rows.copy()
    for c, v in shift.items():
        rows[c] = rows[c] - v
    return rows


def get_models(seed=0):
    return {
        "lgb": lambda: lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4,
                                         random_state=seed),
        "lgb_l1": lambda: lgb.LGBMRegressor(objective="l1", n_estimators=1200, learning_rate=0.03, num_leaves=31,
                                            min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                            verbose=-1, n_jobs=4, random_state=seed),
        "lgb_huber": lambda: lgb.LGBMRegressor(objective="huber", alpha=8.0, n_estimators=1000, learning_rate=0.03,
                                               num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                               colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=seed),
        "xgb": lambda: xgb.XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
                                        colsample_bytree=0.7, reg_lambda=2.0, n_jobs=8, tree_method="hist", random_state=seed),
        "cat": lambda: CatBoostRegressor(iterations=1500, learning_rate=0.04, depth=6, l2_leaf_reg=3, verbose=0,
                                         random_seed=seed, thread_count=8),
        "cat_mae": lambda: CatBoostRegressor(iterations=2000, learning_rate=0.04, depth=6, l2_leaf_reg=3, loss_function="MAE",
                                             verbose=0, random_seed=seed, thread_count=8),
        "cat_deep": lambda: CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=5, verbose=0,
                                              random_seed=seed, thread_count=8),
        "lgb_slow": lambda: lgb.LGBMRegressor(n_estimators=2500, learning_rate=0.01, num_leaves=31, min_child_samples=30,
                                              subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5,
                                              verbose=-1, n_jobs=4, random_state=seed),
        "et": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                    ExtraTreesRegressor(500, min_samples_leaf=3, max_features=0.5, n_jobs=8, random_state=seed)),
    }


def feature_frame(rows, tfeat, state_map):
    X = pd.concat([add_basic_features(rows), tfeat], axis=1)
    X["state_code"] = X["State"].map(state_map).fillna(-1).astype(int)
    return X


def build_matrices(train_rows, ref_rows, query_rows, seed=0):
    """Feature matrices exactly as fit_predict builds them (for tuning / custom models)."""
    state_map = {s: i for i, s in enumerate(sorted(train_rows.State.unique()))}
    train_rows = add_basic_features(train_rows)
    ftr = oof_target_features(train_rows, seed=seed)
    Xtr = feature_frame(train_rows, ftr, state_map)
    q = add_basic_features(query_rows)
    Xq = feature_frame(query_rows, target_features(add_basic_features(ref_rows), q), state_map)
    cols = RAW_NUM + ENG_NUM + ["state_code"] + ftr.columns.tolist() + [c for c in ZFEATS if c in Xtr.columns]
    return Xtr[cols], train_rows[T].values, Xq[cols]


def fit_predict(train_rows, ref_rows, query_rows, model_names, seed=0, extra_train=None, models=None):
    """train_rows: labelled rows (one per ZIP) used to fit; their target features are OOF among themselves.
    ref_rows: labelled reference rows (features in the query month, labels = latest known) used for query features.
    query_rows: rows to predict. extra_train: optional additional labelled rows (e.g. May), OOF featurized."""
    state_map = {s: i for i, s in enumerate(sorted(train_rows.State.unique()))}
    train_rows = add_basic_features(train_rows)
    if extra_train is not None:
        extra_train = add_basic_features(extra_train)
    ftr = oof_target_features(train_rows, seed=seed)
    Xtr = feature_frame(train_rows, ftr, state_map); ytr = train_rows[T].values
    if extra_train is not None:
        fe = oof_target_features(extra_train, seed=seed)
        Xtr = pd.concat([Xtr, feature_frame(extra_train, fe, state_map)], ignore_index=True)
        ytr = np.r_[ytr, extra_train[T].values]
    ref = add_basic_features(ref_rows)
    q = add_basic_features(query_rows)
    Xq = feature_frame(query_rows, target_features(ref, q), state_map)
    cols = RAW_NUM + ENG_NUM + ["state_code"] + ftr.columns.tolist() + [c for c in ZFEATS if c in Xtr.columns]
    models = models or get_models(seed)
    out = {}
    for name in model_names:
        base = name[:-6] if name.endswith("_resid") else name
        m = models[base]()
        if name.endswith("_resid"):  # learn the residual w.r.t. the distance-weighted neighbour mean
            anchor_tr, anchor_q = Xtr["geo10_idw"].values, Xq["geo10_idw"].values
            m.fit(Xtr[cols], ytr - anchor_tr)
            out[name] = np.clip(anchor_q + m.predict(Xq[cols]), 0, 100)
        else:
            m.fit(Xtr[cols], ytr)
            out[name] = np.clip(m.predict(Xq[cols]), 0, 100)
    return out, cols
