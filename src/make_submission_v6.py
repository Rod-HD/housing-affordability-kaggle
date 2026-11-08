"""v6 submission = v5 ensemble + TabPFN (tabular foundation model), weights from outputs/ens_weights_v6.json (NNLS on OOF).
TabPFN and the MLP use the v3 feature set (as validated on the cached folds); GBMs use the v4 feature set.
Seen ZIPs: unchanged from v4/v5."""
import os, sys, json, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(ROOT, ".env"), encoding="utf8"):
    if line.startswith("TABPFN_TOKEN="):
        os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()
os.environ.setdefault("HF_HOME", "D:/Programs/huggingface_cache")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "D:/Programs/huggingface_cache/tabpfn")
os.environ["TABPFN_NO_BROWSER"] = "1"
os.environ["EXTRA_GEO"] = "1"
import numpy as np, pandas as pd
from common import *
from pipeline import fit_predict, build_matrices, drift_table, dedrift
from nn_model import prep, mlp_fit_predict

warnings.filterwarnings("ignore")
NAME = sys.argv[1] if len(sys.argv) > 1 else "sub10_gbm_mlp_tabpfn_v6"
W = {k.split("@")[0]: v for k, v in json.load(open(os.environ.get("ENS_WEIGHTS", f"{OUT}/ens_weights_v6.json"))).items() if v > 0.005}
s = sum(W.values()); W = {k: v / s for k, v in W.items()}
print("weights:", {k: round(v, 3) for k, v in W.items()})
EXT_GBM = lambda d: add_hist(add_irs(add_zillow(add_geo(d))))
EXT_V3 = lambda d: add_irs(add_zillow(add_geo(d)))

tr, te, ss = load()
jun = tr[tr.Month == JUN].reset_index(drop=True)
known = te.Zip.isin(jun.Zip)
shift = drift_table(jun, te[known])
jul_dd = dedrift(te, shift)
ref = jul_dd[known].copy(); ref[T] = ref.Zip.map(jun.set_index("Zip")[T]).values
q = jul_dd[~known].copy()

p_new = np.zeros((~known).sum())
gbm = [m for m in W if m not in ("mlp", "tabpfn")]
if gbm:
    acc = []
    for sd in (0, 1, 2):
        p, cols = fit_predict(EXT_GBM(jun), EXT_GBM(ref), EXT_GBM(q), gbm, seed=sd)
        acc.append(sum(W[m] * p[m] for m in gbm)); print(f"gbm seed {sd}: {len(cols)} features", flush=True)
    p_new += np.mean(acc, 0)
if "mlp" in W or "tabpfn" in W:
    Xt, yt, Xq = build_matrices(EXT_V3(jun), EXT_V3(ref), EXT_V3(q))
    if "mlp" in W:
        wt = np.sqrt(Xt["SampleSize"].clip(lower=100).values); Zt, Zq = prep(Xt, Xq)
        p_mlp = np.clip(np.mean([mlp_fit_predict(Zt, yt, wt, Zq, s) for s in range(5)], 0), 0, 100)
        p_new += W["mlp"] * p_mlp; print("mlp done", flush=True)
    if "tabpfn" in W:
        from tabpfn import TabPFNRegressor
        m = TabPFNRegressor(device="cpu", n_estimators=int(os.environ.get("TABPFN_NEST", "2")), ignore_pretraining_limits=True, random_state=0)
        m.fit(Xt.values.astype("float32"), yt)
        p_tab = np.clip(m.predict(Xq.values.astype("float32")), 0, 100)
        p_new += W["tabpfn"] * p_tab; print("tabpfn done", flush=True)
p_new = np.clip(p_new, 0, 100)

prev = pd.read_csv(f"{SUB}/sub09_gbm_mlp_v5.csv")
sub = ss[["Zip", "State"]].copy(); sub["Zip"] = sub.Zip.astype(int).astype(str).str.zfill(5)
sub[T] = prev[T].values
sub.loc[~known.values, T] = p_new
assert sub[T].notna().all() and sub[T].between(0, 100).all()
sub.to_csv(f"{SUB}/{NAME}.csv", index=False)
print("wrote", NAME, "| RMS change vs sub09 on new ZIPs: %.3f" % np.sqrt(((sub[T] - prev[T])[~known.values] ** 2).mean()))
