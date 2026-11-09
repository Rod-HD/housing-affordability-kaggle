"""v9 submission: v8 blend machinery with the Realtor.com feature set (v5) available to TabPFN and the GBMs,
weights from an NNLS json fitted on the July-proxy OOF predictions (nested-CV checked in ens_rmse.py).
Usage: python make_submission_v9.py NAME [WEIGHTS_JSON]
Seen ZIPs: unchanged from v4/v5 (sub09)."""
import os, re, sys, json, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(ROOT, ".env"), encoding="utf8"):
    if line.startswith("TABPFN_TOKEN="):
        os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()
os.environ.setdefault("HF_HOME", "D:/Programs/huggingface_cache")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "D:/Programs/huggingface_cache/tabpfn")
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["EXTRA_GEO"] = "1"
import numpy as np, pandas as pd, torch, lightgbm as lgb
from common import *
from pipeline import fit_predict, build_matrices, drift_table, dedrift
from nn_model import prep, mlp_fit_predict
from extra_sources import add_realtor

warnings.filterwarnings("ignore")
NAME = sys.argv[1] if len(sys.argv) > 1 else "sub13_realtor_v9"
WFILE = sys.argv[2] if len(sys.argv) > 2 else f"{OUT}/ens_weights_v9.json"
W = {k: v for k, v in json.load(open(WFILE)).items() if v > 0.005}
s = sum(W.values()); W = {k: v / s for k, v in W.items()}
print("weights:", {k: round(v, 3) for k, v in W.items()}, flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EXT = {"v3": lambda d: add_irs(add_zillow(add_geo(d))), "v4": lambda d: add_hist(add_irs(add_zillow(add_geo(d)))),
       "v5": lambda d: add_realtor(add_hist(add_irs(add_zillow(add_geo(d)))))}

tr, te, ss = load()
jun = tr[tr.Month == JUN].reset_index(drop=True)
known = te.Zip.isin(jun.Zip)
shift = drift_table(jun, te[known])
jul_dd = dedrift(te, shift)
ref = jul_dd[known].copy(); ref[T] = ref.Zip.map(jun.set_index("Zip")[T]).values
q = jul_dd[~known].copy()
_mats = {}


def mats(fs):
    if fs not in _mats:
        _mats[fs] = build_matrices(EXT[fs](jun), EXT[fs](ref), EXT[fs](q))
    return _mats[fs]


def tabpfn_pred(var, n_est, seed, arg):
    from tabpfn import TabPFNRegressor
    Xt, yt, Xq = mats(var if var in ("v4", "v5") else "v3")
    if var == "sel":
        g = lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, num_leaves=31, min_child_samples=30, subsample=0.7,
                              subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=8,
                              importance_type="gain").fit(Xt, yt)
        keep = Xt.columns[np.argsort(-g.feature_importances_)[:int(arg)]]; Xt, Xq = Xt[keep], Xq[keep]
    m = TabPFNRegressor(device=DEV, n_estimators=n_est, ignore_pretraining_limits=True, random_state=seed, memory_saving_mode=True)
    m.fit(Xt.values.astype("float32"), yt)
    return np.clip(m.predict(Xq.values.astype("float32")), 0, 100)


def tabm_pred(fs):
    from pytabkit import TabM_D_Regressor
    Xt, yt, Xq = mats(fs)
    med = Xt.median(); miss = [c for c in Xt if Xt[c].isna().any() or Xq[c].isna().any()]
    Ft, Fq = Xt.fillna(med).astype("float32"), Xq.fillna(med).astype("float32")
    for c in miss:
        Ft[c + "_na"] = Xt[c].isna().astype("float32"); Fq[c + "_na"] = Xq[c].isna().astype("float32")
    m = TabM_D_Regressor(device=DEV + (":0" if DEV == "cuda" else ""), random_state=0, n_cv=1, verbosity=0)
    m.fit(Ft, yt.astype("float32"))
    return np.clip(np.asarray(m.predict(Fq)).ravel(), 0, 100)


p_new = np.zeros((~known).sum())
for suffix, fs in (("@FINAL", "v4"), ("@GBM_v5", "v5")):
    gbm = {k: v for k, v in W.items() if k.endswith(suffix)}
    if gbm:
        names = [k.split("@")[0] for k in gbm]; acc = []
        for sd in (0, 1, 2):
            p, cols = fit_predict(EXT[fs](jun), EXT[fs](ref), EXT[fs](q), names, seed=sd)
            acc.append(sum(gbm[k] * p[k.split("@")[0]] for k in gbm)); print(f"gbm {fs} seed {sd}: {len(cols)} features", flush=True)
        p_new += np.mean(acc, 0)
for k, w in W.items():
    if k.endswith(("@FINAL", "@GBM_v5")):
        continue
    name = k.split("@")[0]
    if name == "mlp":
        Xt, yt, Xq = mats("v3"); wt = np.sqrt(Xt["SampleSize"].clip(lower=100).values); Zt, Zq = prep(Xt, Xq)
        p = np.clip(np.mean([mlp_fit_predict(Zt, yt, wt, Zq, s) for s in range(5)], 0), 0, 100)
    elif name.startswith("tabm"):
        p = tabm_pred("v4" if name.endswith("_v4") else "v3")
    else:
        var, arg, n_est, seed = re.fullmatch(r"(v5|v4|base|sel)(\d*)_n(\d+)_r(\d+)", name).groups()
        p = tabpfn_pred(var, int(n_est), int(seed), arg)
    p_new += w * p; print(f"{name} done (w={w:.3f})", flush=True)
    if DEV == "cuda":
        torch.cuda.empty_cache()
p_new = np.clip(p_new, 0, 100)

prev = pd.read_csv(f"{SUB}/sub09_gbm_mlp_v5.csv")
ref12 = pd.read_csv(f"{SUB}/sub12_tabpfn32_v3v4_v8.csv")
sub = ss[["Zip", "State"]].copy(); sub["Zip"] = sub.Zip.astype(int).astype(str).str.zfill(5)
sub[T] = prev[T].values
sub.loc[~known.values, T] = p_new
assert sub[T].notna().all() and sub[T].between(0, 100).all()
sub.to_csv(f"{SUB}/{NAME}.csv", index=False)
print("wrote", NAME, "| RMS change vs sub12 on new ZIPs: %.3f" % np.sqrt(((sub[T] - ref12[T])[~known.values] ** 2).mean()))
