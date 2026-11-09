"""v11 submission builder (any feature set from featuresets.py, any model family).
Usage: python make_submission_v11.py NAME WEIGHTS_JSON
Weight keys (same names as the OOF files used to fit the weights):
  <fs>_n<N>_r<S>@TPF     TabPFN v3.5 with N estimators, seed S, on feature set fs ("base" = v3)
  <col>@GBM_<fs>         LightGBM from pipeline.get_models (lgb / lgb_slow_h_w), averaged over 3 seeds
  <model><arg>_<fs>@NEW  TabICL / TabDPT / xRFM (exp_newmodels.run)
Seen ZIPs: unchanged from sub09 (June label + Zillow price change). Per-component predictions are cached in
outputs/sub_parts so several blends can be written without refitting."""
import os, re, sys, json, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(ROOT, ".env"), encoding="utf8"):
    if line.startswith("TABPFN_TOKEN="):
        os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()
os.environ.setdefault("HF_HOME", "D:/Programs/huggingface_cache")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "D:/Programs/huggingface_cache/tabpfn")
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["EXTRA_GEO"] = "1"
import numpy as np, pandas as pd, torch
from common import *
from pipeline import fit_predict, build_matrices, drift_table, dedrift
from featuresets import FEATURE_SETS

warnings.filterwarnings("ignore")
NAME, WFILE = sys.argv[1], sys.argv[2]
W = {k: v for k, v in json.load(open(WFILE)).items() if v > 0.005}
s = sum(W.values()); W = {k: v / s for k, v in W.items()}
print("weights:", {k: round(v, 3) for k, v in W.items()}, flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PARTS = f"{OUT}/sub_parts"; os.makedirs(PARTS, exist_ok=True)

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
        E = FEATURE_SETS[fs]; _mats[fs] = build_matrices(E(jun), E(ref), E(q))
    return _mats[fs]


def component(key):
    path = f"{PARTS}/{key.replace('@', '__')}.npy"
    if os.path.exists(path):
        return np.load(path)
    name, kind = key.split("@")
    if kind == "TPF":
        fs, n_est, seed = re.fullmatch(r"(\w+?)_n(\d+)_r(\d+)", name).groups()
        from tabpfn import TabPFNRegressor
        Xt, yt, Xq = mats("v3" if fs == "base" else fs)
        m = TabPFNRegressor(device=DEV, n_estimators=int(n_est), ignore_pretraining_limits=True, random_state=int(seed),
                            memory_saving_mode=True)
        m.fit(Xt.values.astype("float32"), yt); p = m.predict(Xq.values.astype("float32"))
    elif kind.startswith("GBM_"):
        fs = kind[4:]; E = FEATURE_SETS[fs]
        p = np.mean([fit_predict(E(jun), E(ref), E(q), [name], seed=sd)[0][name] for sd in (0, 1, 2)], 0)
    elif kind == "NEW":
        from exp_newmodels import run
        model, arg, fs = re.fullmatch(r"([a-z]+)(\d*)_(\w+)", name).groups()
        Xt, yt, Xq = mats(fs); p = run(model, arg, Xt, np.asarray(yt, float), Xq)
    else:
        raise SystemExit(f"unknown key {key}")
    p = np.clip(np.asarray(p, float).ravel(), 0, 100); np.save(path, p)
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return p


p_new = np.zeros((~known).sum())
for k, w in W.items():
    p_new += w * component(k); print(f"{k} done (w={w:.3f})", flush=True)
p_new = np.clip(p_new, 0, 100)

prev = pd.read_csv(f"{SUB}/sub09_gbm_mlp_v5.csv")
ref12 = pd.read_csv(f"{SUB}/sub12_tabpfn32_v3v4_v8.csv")
sub = ss[["Zip", "State"]].copy(); sub["Zip"] = sub.Zip.astype(int).astype(str).str.zfill(5)
sub[T] = prev[T].values
sub.loc[~known.values, T] = p_new
assert len(sub) == 8534 and sub[T].notna().all() and sub[T].between(0, 100).all()
assert (sub.Zip.str.len() == 5).all() and not sub.duplicated(["Zip", "State"]).any()
sub.to_csv(f"{SUB}/{NAME}.csv", index=False)
d = (sub[T] - ref12[T])[~known.values]
print("wrote", NAME, "| vs sub12 on new ZIPs: RMS %.3f, max %.2f" % (np.sqrt((d ** 2).mean()), d.abs().max()))
