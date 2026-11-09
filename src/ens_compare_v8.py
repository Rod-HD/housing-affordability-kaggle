"""Compare a small set of pre-declared blends on the July-proxy OOF predictions.
Weights: NNLS (sum to 1), always evaluated nested (fitted on 4/5 of rows, scored on 1/5, 5 repeats).
Paired bootstrap against the sub11 recipe guards against picking a blend that only won by chance."""
import json, warnings
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from common import OUT, load, JUN

warnings.filterwarnings("ignore")
F = {"lgb_slow_h_w": ("FINAL", "lgb_slow_h_w"), "lgb": ("FINAL", "lgb"), "mlp": ("NN", "mlp"), "tabpfn8": ("TABPFN8", "tabpfn"),
     "base_n32_r0": ("TPF_base_n32_r0", "base_n32_r0"), "v4_n32_r0": ("TPF_v4_n32_r0", "v4_n32_r0"),
     "sel100_n8_r0": ("TPF_sel100_n8_r0", "sel100_n8_r0"), "v4_n8_r1": ("TPF_v4_n8_r1", "v4_n8_r1"),
     "base_n8_r1": ("TPF_base_n8_r1", "base_n8_r1"), "tabm_cv1": ("PTK_tabm_cv1", "tabm_cv1"), "tabm_cv1_v4": ("PTK_tabm_cv1_v4", "tabm_cv1_v4"),
     "realmlp_cv1": ("PTK_realmlp_cv1", "realmlp_cv1")}
P, y = {}, None
for n, (tag, col) in F.items():
    o = pd.read_csv(f"{OUT}/rmse_oof_{tag}_s0.csv").sort_values("Zip"); y = o.y.values; P[n] = o[col].values
GBM = ["lgb_slow_h_w", "lgb"]
SETS = {"sub11 recipe": GBM + ["mlp", "tabpfn8"],
        "A: GBM + TPF32 v3": GBM + ["base_n32_r0"],
        "B: GBM + TPF32 v3+v4": GBM + ["base_n32_r0", "v4_n32_r0"],
        "C: B + sel100": GBM + ["base_n32_r0", "v4_n32_r0", "sel100_n8_r0"],
        "D: C + extra seeds": GBM + ["base_n32_r0", "v4_n32_r0", "sel100_n8_r0", "v4_n8_r1", "base_n8_r1"],
        "E: C + TabM": GBM + ["base_n32_r0", "v4_n32_r0", "sel100_n8_r0", "tabm_cv1", "tabm_cv1_v4"],
        "F: C + MLP + RealMLP": GBM + ["base_n32_r0", "v4_n32_r0", "sel100_n8_r0", "mlp", "realmlp_cv1"]}
rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))
fit = lambda A, b: (lambda w: w / w.sum())(nnls(A, b)[0])
def nested(X):
    out = np.zeros((5, len(y)))
    for s in range(5):
        for a, b in KFold(5, shuffle=True, random_state=s).split(X):
            out[s, b] = X[b] @ fit(X[a], y[a])
    return out
print("single:", {n: round(rmse(p), 4) for n, p in P.items()})
base = None; res = {}
rng = np.random.default_rng(0)
idx = rng.integers(0, len(y), (2000, len(y)))
for name, cols in SETS.items():
    X = np.c_[[P[c] for c in cols]].T; N = nested(X); w = fit(X, y); pm = N.mean(0)
    scores = [rmse(n) for n in N]
    if base is None: base = N
    d = ((base.mean(0) - y) ** 2 - (pm - y) ** 2)
    gain = np.array([np.sqrt(((base.mean(0) - y)[i] ** 2).mean()) - np.sqrt(((pm - y)[i] ** 2).mean()) for i in idx])
    print(f"{name:24s} in-sample {rmse(X @ w):.4f} nested {np.mean(scores):.4f} | gain vs sub11 {gain.mean():+.4f} [{np.percentile(gain, 2.5):+.4f}, {np.percentile(gain, 97.5):+.4f}] | P(gain>0) {(gain > 0).mean():.3f}")
    print("    w:", {c: round(float(v), 3) for c, v in zip(cols, w) if v > 0.005})
    res[name] = dict(zip(cols, map(float, w)))
json.dump(res, open(f"{OUT}/ens_compare_v8.json", "w"), indent=2)
