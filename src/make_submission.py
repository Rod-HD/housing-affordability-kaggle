"""Build final submissions.
Known ZIPs (in train): latest label (June) -- validated persistence MAE 0.712 (May->June).
New ZIPs: ensemble trained on all labelled June rows; July features de-drifted to June scale;
reference neighbours = known ZIPs with July (de-drifted) features + June labels."""
import os, sys, json, warnings
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from common import *
from pipeline import fit_predict, get_models, drift_table, dedrift

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2]
tr, te, ss = load()
jun = tr[tr.Month == JUN].reset_index(drop=True)
may = tr[tr.Month == MAY].reset_index(drop=True)
jul = te.copy()
known = jul.Zip.isin(jun.Zip)
print("test known:", known.sum(), "new:", (~known).sum())

shift = drift_table(jun, jul[known])
jul_dd = dedrift(jul, shift)
ref = jul_dd[known].copy()
ref[T] = ref.Zip.map(jun.set_index("Zip")[T]).values
query = jul_dd[~known].copy()


def tuned_models(geo, seed):
    base = get_models(seed)
    path = f"{OUT}/best_lgb_mae{'_geo' if geo else ''}.json"
    if os.path.exists(path):
        p = json.load(open(path))["params"].copy()
        if "huber_alpha" in p:
            p["alpha"] = p.pop("huber_alpha")
        p.update(subsample_freq=1, verbose=-1, n_jobs=4, random_state=seed)
        base["lgb_tuned"] = lambda: lgb.LGBMRegressor(**p)
        print("using tuned lgb", p)
    else:
        base["lgb_tuned"] = base["lgb"]
    return base


def new_zip_preds(geo):
    J, R, Q = jun, ref, query
    if geo:
        J, R, Q = add_geo(J), add_geo(R), add_geo(Q)
    names = ["lgb_tuned", "lgb_huber", "cat"]
    allp = []
    for s in SEEDS:
        preds, cols = fit_predict(J, R, Q, names, seed=s, models=tuned_models(geo, s))
        allp.append(np.mean([preds[n] for n in names], 0))
        print(f"geo={geo} seed {s} done; mean pred {allp[-1].mean():.2f}", flush=True)
    return np.mean(allp, 0)


def known_preds(momentum=False):
    y_jun = jul.loc[known, "Zip"].map(jun.set_index("Zip")[T]).values.astype(float)
    if not momentum:
        return y_jun
    # Variant: May->June delta model (y_prev + State), applied to June->July and rounded (untestable offline)
    X = pd.DataFrame({"y_prev": may.set_index("Zip").loc[jun.Zip, T].values,
                      "state_code": jun.State.astype("category").cat.codes.values})
    d = jun[T].values - X.y_prev.values
    m = lgb.LGBMRegressor(objective="l1", n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4).fit(X, d)
    cats = dict(zip(jun.State.astype("category").cat.categories, range(100)))
    Xq = pd.DataFrame({"y_prev": y_jun, "state_code": jul.loc[known, "State"].map(cats).values})
    return np.clip(np.round(y_jun + m.predict(Xq)), 0, 100)


def write(name, p_known, p_new):
    sub = ss[["Zip", "State"]].copy()
    sub["Zip"] = sub["Zip"].astype(int).astype(str).str.zfill(5)  # keep 5-digit ZIPs
    sub[T] = np.nan
    sub.loc[known.values, T] = p_known
    sub.loc[~known.values, T] = p_new
    assert sub[T].notna().all() and (sub.Zip.values == te.Zip.astype(str).str.zfill(5).values).all()
    sub[T] = sub[T].clip(0, 100)
    sub.to_csv(f"{SUB}/{name}.csv", index=False)
    print("wrote", name, sub[T].describe().round(2).to_dict())


if __name__ == "__main__":
    os.makedirs(SUB, exist_ok=True)
    kp = known_preds(False)
    # baselines for the report
    write("sub00_baseline_known_last_new_statemean", kp,
          query.State.map(jun.groupby("State")[T].mean()).fillna(jun[T].mean()).values)
    p_nogeo = new_zip_preds(False)
    write("sub01_known_last__new_ens_nogeo", kp, p_nogeo)
    p_geo = new_zip_preds(True)
    write("sub02_known_last__new_ens_geo", kp, p_geo)
    write("sub03_known_momentum__new_ens_geo", known_preds(True), p_geo)
    write("sub04_known_last__new_ens_geo_rounded", kp, np.round(p_geo))
    pd.DataFrame({"Zip": query.Zip.values, "nogeo": p_nogeo, "geo": p_geo}).to_csv(f"{OUT}/new_zip_test_preds.csv", index=False)
    print("corr nogeo vs geo new-zip preds", np.corrcoef(p_nogeo, p_geo)[0, 1])
