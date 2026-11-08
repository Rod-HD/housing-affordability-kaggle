"""Optuna (RMSE) for LightGBM / CatBoost on the full external-feature set, July-proxy + de-drift folds
(same folds as exp_zillow.py). Fold matrices are built once and cached. Also tests target transforms."""
import os, sys, json, time, warnings, pickle
os.environ["EXTRA_GEO"] = "1"
import numpy as np, pandas as pd, optuna, lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from common import *
from pipeline import build_matrices, drift_table, dedrift

warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
ALGO = sys.argv[1] if len(sys.argv) > 1 else "lgb"; NT = int(sys.argv[2]) if len(sys.argv) > 2 else 40
CACHE = f"{OUT}/fold_cache_full.pkl"
if os.path.exists(CACHE):
    folds = pickle.load(open(CACHE, "rb"))
else:
    tr, te, ss = load()
    jun = tr[tr.Month == JUN].set_index("Zip").sort_index()
    jul = te[te.Zip.isin(jun.index)].set_index("Zip").loc[jun.index]; jul[T] = jun[T]
    shift = drift_table(jun.reset_index(), jul.reset_index())
    EXT = lambda d: add_irs(add_zillow(add_geo(d)))
    J = EXT(jun.reset_index()).set_index("Zip"); Jd = EXT(dedrift(jul.reset_index(), shift)).set_index("Zip")
    folds = []
    for a, b in KFold(5, shuffle=True, random_state=200).split(J.index.values):
        Xt, yt, Xv = build_matrices(J.iloc[a].reset_index(), Jd.iloc[a].reset_index(), Jd.iloc[b].reset_index())
        folds.append((Xt, yt, Xv, J.iloc[b][T].values))
    pickle.dump(folds, open(CACHE, "wb"))
print("features:", folds[0][0].shape[1], flush=True)


def cv(make, transform=None):
    se, n = 0.0, 0
    for Xt, yt, Xv, yv in folds:
        m = make()
        if transform == "logit":
            q = (yt + 0.5) / 101.0; m.fit(Xt, np.log(q / (1 - q))); p = 101.0 / (1 + np.exp(-m.predict(Xv))) - 0.5
        elif transform == "sqrt_arcsin":
            m.fit(Xt, np.arcsin(np.sqrt(yt / 100))); p = 100 * np.sin(np.clip(m.predict(Xv), 0, np.pi / 2)) ** 2
        else:
            m.fit(Xt, yt); p = m.predict(Xv)
        p = np.clip(p, 0, 100); se += ((p - yv) ** 2).sum(); n += len(yv)
    return np.sqrt(se / n)


if ALGO == "transforms":
    base = lambda: lgb.LGBMRegressor(n_estimators=2500, learning_rate=0.01, num_leaves=31, min_child_samples=30, subsample=0.7,
                                     subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=6)
    for t in (None, "logit", "sqrt_arcsin"):
        print("transform", t, "RMSE", round(cv(base, t), 4), flush=True)
    sys.exit()


def objective(trial):
    if ALGO == "lgb":
        P = dict(n_estimators=trial.suggest_int("n_estimators", 800, 5000, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.004, 0.05, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 120, log=True),
                 subsample=trial.suggest_float("subsample", 0.4, 1.0), subsample_freq=1,
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.15, 0.9),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                 min_split_gain=trial.suggest_float("min_split_gain", 0, 1.0),
                 max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]), verbose=-1, n_jobs=6)
        mk = lambda: lgb.LGBMRegressor(**P)
    else:
        P = dict(iterations=trial.suggest_int("iterations", 1000, 5000, log=True),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                 depth=trial.suggest_int("depth", 4, 9), l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
                 random_strength=trial.suggest_float("random_strength", 0.1, 5, log=True),
                 bagging_temperature=trial.suggest_float("bagging_temperature", 0, 2),
                 rsm=trial.suggest_float("rsm", 0.2, 1.0), verbose=0, thread_count=6, random_seed=0)
        mk = lambda: CatBoostRegressor(**P)
    return cv(mk)


st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=1))
t0 = time.time()
st.optimize(objective, n_trials=NT, callbacks=[lambda s, t: print(f"trial {t.number} {t.value:.4f} best {s.best_value:.4f} {time.time()-t0:.0f}s", flush=True)])
print("BEST", st.best_value, st.best_params)
json.dump({"best_value": st.best_value, "params": st.best_params}, open(f"{OUT}/best_v2_{ALGO}.json", "w"), indent=2)
st.trials_dataframe().to_csv(f"{OUT}/optuna_v2_{ALGO}.csv", index=False)
