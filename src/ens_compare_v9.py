"""Realtor (v5) blends vs the sub12 recipe. Compare a small set of pre-declared blends on the July-proxy OOF predictions.
Weights: NNLS (sum to 1), always evaluated nested (fitted on 4/5 of rows, scored on 1/5, 5 repeats).
Paired bootstrap against the sub12 recipe guards against picking a blend that only won by chance."""
import json, warnings
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from common import OUT, ZILLOW_PATH, load, JUN

warnings.filterwarnings("ignore")
F = {"lgb_slow_h_w": ("FINAL", "lgb_slow_h_w"), "lgb": ("FINAL", "lgb"), "base_n32_r0": ("TPF_base_n32_r0", "base_n32_r0"), "v4_n32_r0": ("TPF_v4_n32_r0", "v4_n32_r0"),
     "v5_n32_r0": ("TPF_v5_n32_r0", "v5_n32_r0"), "lgb_slow_h_w_v5": ("GBM_v5", "lgb_slow_h_w"), "lgb_v5": ("GBM_v5", "lgb")}
P, y = {}, None
for n, (tag, col) in F.items():
    o = pd.read_csv(f"{OUT}/rmse_oof_{tag}_s0.csv").sort_values("Zip"); y = o.y.values; P[n] = o[col].values; ZIPS = o.Zip.values
GBM = ["lgb_slow_h_w", "lgb"]; GBM5 = ["lgb_slow_h_w_v5", "lgb_v5"]
T32 = ["base_n32_r0", "v4_n32_r0"]
SETS = {"sub12 recipe (B)": GBM + T32, "G: GBM5 + TPF32 v5": GBM5 + ["v5_n32_r0"], "H: G + TPF32 v3 + v4": GBM5 + ["v5_n32_r0"] + T32, "I: B + TPF32 v5": GBM + T32 + ["v5_n32_r0"]}
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
    z = pd.read_csv(ZILLOW_PATH); z = z[z.Month == JUN].set_index("Zip"); ok = z.zhvi.reindex(ZIPS).notna().values
    k = np.where(ok)[0]; ik = rng.integers(0, len(k), (2000, len(k)))
    g2 = np.array([np.sqrt(((base.mean(0) - y)[k[i]] ** 2).mean()) - np.sqrt(((pm - y)[k[i]] ** 2).mean()) for i in ik])
    print(f"    excl ZHVI-missing: {np.sqrt(((pm - y)[ok] ** 2).mean()):.4f} | gain {g2.mean():+.4f} [{np.percentile(g2, 2.5):+.4f}, {np.percentile(g2, 97.5):+.4f}]")
    print(f"{name:24s} in-sample {rmse(X @ w):.4f} nested {np.mean(scores):.4f} | gain vs sub12 {gain.mean():+.4f} [{np.percentile(gain, 2.5):+.4f}, {np.percentile(gain, 97.5):+.4f}] | P(gain>0) {(gain > 0).mean():.3f}")
    print("    w:", {c: round(float(v), 3) for c, v in zip(cols, w) if v > 0.005})
    res[name] = dict(zip(cols, map(float, w)))
json.dump(res, open(f"{OUT}/ens_compare_v9.json", "w"), indent=2)
