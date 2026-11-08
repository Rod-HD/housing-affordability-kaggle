"""Optuna hyper-parameter search for the unseen-ZIP model (LightGBM / CatBoost), on cached
GroupKFold feature matrices (June->June, leak-free nested target features)."""
import sys, warnings, json, time
import numpy as np, pandas as pd, optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from common import *
from pipeline import feature_frame

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ALGO = sys.argv[1] if len(sys.argv) > 1 else "lgb"
NTRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
LOSS = sys.argv[3] if len(sys.argv) > 3 else "mae"   # metric to minimise

tr, te, ss = load()
GEO = "--geo" in sys.argv
jun = tr[tr.Month == JUN].reset_index(drop=True)
if GEO:
    jun = add_geo(jun)
jun = add_basic_features(jun)
zips = jun.Zip.values
state_map = {s: i for i, s in enumerate(sorted(jun.State.unique()))}
folds = []
for a, b in KFold(5, shuffle=True, random_state=100).split(zips):
    lab, val = jun.iloc[a], jun.iloc[b]
    ftr = oof_target_features(lab, seed=0)
    Xtr = feature_frame(lab, ftr, state_map); Xva = feature_frame(val, target_features(lab, val), state_map)
    cols = RAW_NUM + ENG_NUM + ["state_code"] + ftr.columns.tolist()
    folds.append((Xtr[cols], lab[T].values, Xva[cols], val[T].values))
print("features:", len(cols))


def score(y, p):
    return np.abs(y - p).mean() if LOSS == "mae" else np.sqrt(((y - p) ** 2).mean())


def objective(trial):
    if ALGO == "lgb":
        params = dict(objective=trial.suggest_categorical("objective", ["l2", "l1", "huber"]),
                      n_estimators=trial.suggest_int("n_estimators", 300, 2500, log=True),
                      learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                      num_leaves=trial.suggest_int("num_leaves", 7, 127, log=True),
                      min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
                      subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
                      colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
                      reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                      reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                      verbose=-1, n_jobs=4)
        if params["objective"] == "huber":
            params["alpha"] = trial.suggest_float("huber_alpha", 2, 20)
        mk = lambda: lgb.LGBMRegressor(**params)
    else:
        params = dict(loss_function=trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
                      iterations=trial.suggest_int("iterations", 500, 3000, log=True),
                      learning_rate=trial.suggest_float("learning_rate", 0.015, 0.1, log=True),
                      depth=trial.suggest_int("depth", 4, 9),
                      l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
                      random_strength=trial.suggest_float("random_strength", 0.1, 5, log=True),
                      bagging_temperature=trial.suggest_float("bagging_temperature", 0, 2),
                      verbose=0, thread_count=8, random_seed=0)
        mk = lambda: CatBoostRegressor(**params)
    s = []
    for i, (Xt, yt, Xv, yv) in enumerate(folds):
        m = mk(); m.fit(Xt, yt)
        s.append(score(yv, np.clip(m.predict(Xv), 0, 100)))
        trial.report(np.mean(s), i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return np.mean(s)


study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
t0 = time.time()
study.optimize(objective, n_trials=NTRIALS, callbacks=[lambda s, t: print(f"trial {t.number} {t.value} best {s.best_value:.4f} {time.time()-t0:.0f}s", flush=True)])
print("BEST", study.best_value, study.best_params)
study.trials_dataframe().to_csv(f"{OUT}/optuna_{ALGO}_{LOSS}{'_geo' if GEO else ''}.csv", index=False)
json.dump({"best_value": study.best_value, "params": study.best_params}, open(f"{OUT}/best_{ALGO}_{LOSS}{'_geo' if GEO else ''}.json", "w"), indent=2)
