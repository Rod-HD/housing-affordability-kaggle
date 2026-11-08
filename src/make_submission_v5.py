"""v5 submission = v4 GBMs + MLP (different model family, added for ensemble diversity).
Weights (NNLS, nested-CV RMSE 4.956 on new ZIPs): lgb_slow_h_w 0.506, lgb 0.191, cat_w 0.064, mlp 0.239.
GBMs use the v4 feature set (with purchase-year cost features); the MLP uses the v3 feature set, exactly as validated.
Seen ZIPs: unchanged (June label + Zillow-change adjustment)."""
import os, sys, json, warnings
os.environ["EXTRA_GEO"] = "1"
import numpy as np, pandas as pd
from common import *
from pipeline import fit_predict, build_matrices, drift_table, dedrift
from nn_model import prep, mlp_fit_predict

warnings.filterwarnings("ignore")
NAME = sys.argv[1] if len(sys.argv) > 1 else "sub09_gbm_mlp_v5"
W = {k.split("@")[0]: v for k, v in json.load(open(f"{OUT}/ens_weights_final_nn.json")).items() if v > 0.005}
s = sum(W.values()); W = {k: v / s for k, v in W.items()}
print("weights:", {k: round(v, 3) for k, v in W.items()})
EXT_GBM = lambda d: add_hist(add_irs(add_zillow(add_geo(d))))
EXT_MLP = lambda d: add_irs(add_zillow(add_geo(d)))

tr, te, ss = load()
jun = tr[tr.Month == JUN].reset_index(drop=True)
known = te.Zip.isin(jun.Zip)
shift = drift_table(jun, te[known])
jul_dd = dedrift(te, shift)
ref = jul_dd[known].copy(); ref[T] = ref.Zip.map(jun.set_index("Zip")[T]).values
q = jul_dd[~known].copy()

gbm = [m for m in W if m != "mlp"]
p_gbm = []
for sd in (0, 1, 2):
    p, cols = fit_predict(EXT_GBM(jun), EXT_GBM(ref), EXT_GBM(q), gbm, seed=sd)
    p_gbm.append(sum(W[m] * p[m] for m in gbm)); print(f"gbm seed {sd}: {len(cols)} features", flush=True)
p_new = np.mean(p_gbm, 0)
if "mlp" in W:
    Xt, yt, Xq = build_matrices(EXT_MLP(jun), EXT_MLP(ref), EXT_MLP(q))
    wt = np.sqrt(Xt["SampleSize"].clip(lower=100).values)
    Zt, Zq = prep(Xt, Xq)
    p_mlp = np.clip(np.mean([mlp_fit_predict(Zt, yt, wt, Zq, s) for s in range(5)], 0), 0, 100)
    print(f"mlp: {Xt.shape[1]} features, mean {p_mlp.mean():.2f}; corr with gbm {np.corrcoef(p_mlp, p_new)[0,1]:.3f}")
    p_new = p_new + W["mlp"] * p_mlp
p_new = np.clip(p_new, 0, 100)

prev = pd.read_csv(f"{SUB}/sub08_huber_weighted_hist_v4.csv")       # seen-ZIP part identical to v4
sub = ss[["Zip", "State"]].copy(); sub["Zip"] = sub.Zip.astype(int).astype(str).str.zfill(5)
sub[T] = prev[T].values
sub.loc[~known.values, T] = p_new
assert sub[T].notna().all() and sub[T].between(0, 100).all()
sub.to_csv(f"{SUB}/{NAME}.csv", index=False)
print("wrote", NAME, "| RMS change vs sub08 on new ZIPs: %.3f" % np.sqrt(((sub[T] - prev[T])[~known.values] ** 2).mean()))
