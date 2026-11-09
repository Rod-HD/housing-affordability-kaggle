"""sub15 (no Realtor.com) blends vs the sub12 recipe, on the July-proxy OOF predictions.
Blend sets were declared before the 32-estimator TabPFN results were known. Weights: NNLS (sum 1) or Caruana greedy
selection, always scored nested (fitted on 4/5 of ZIPs, scored on 1/5, 5 repeats). Paired bootstrap vs sub12.
Rule: among sets with P(gain > 0) > 0.9, take the fewest components within 0.005 of the best nested score."""
import json, os, warnings
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from common import OUT, ZILLOW_PATH, JUN

warnings.filterwarnings("ignore")
SRC = {  # weight key -> (oof file tag, column)
    "lgb_slow_h_w@FINAL": ("FINAL", "lgb_slow_h_w"), "lgb@FINAL": ("FINAL", "lgb"),
    "base_n32_r0@TPF": ("TPF_base_n32_r0", "base_n32_r0"), "v4_n32_r0@TPF": ("TPF_v4_n32_r0", "v4_n32_r0"),
    "lgb@GBM_w3": ("GBM_w3", "lgb"), "lgb_slow_h_w@GBM_w3": ("GBM_w3", "lgb_slow_h_w"),
    "lgb_slow_h_w@GBM_w6": ("GBM_w6", "lgb_slow_h_w"),
    "w3_n32_r0@TPF": ("TPF_w3_n32_r0", "w3_n32_r0"), "w5_n32_r0@TPF": ("TPF_w5_n32_r0", "w5_n32_r0"),
    "tabicl8_w3@NEW": ("NEW_tabicl8_w3", "tabicl8_w3"), "tabicl8_v4@NEW": ("NEW_tabicl8_v4", "tabicl8_v4"),
    "xrfm_v4@NEW": ("NEW_xrfm_v4", "xrfm_v4"),
}
P, y = {}, None
for k, (tag, col) in SRC.items():
    f = f"{OUT}/rmse_oof_{tag}_s0.csv"
    if os.path.exists(f):
        o = pd.read_csv(f).sort_values("Zip"); y = o.y.values; P[k] = o[col].values; ZIPS = o.Zip.values
    else:
        print("missing", k)
BASE = ["lgb_slow_h_w@FINAL", "lgb@FINAL", "base_n32_r0@TPF", "v4_n32_r0@TPF"]
S1 = ["lgb@GBM_w3", "lgb_slow_h_w@GBM_w3", "w3_n32_r0@TPF"]
S2 = S1 + ["w5_n32_r0@TPF"]
SETS = {"sub12 recipe": BASE, "S1: GBM w3 + TPF32 w3": S1, "S2: S1 + TPF32 w5": S2,
        "S3: S2 + TabICL + xRFM": S2 + ["tabicl8_w3@NEW", "tabicl8_v4@NEW", "xrfm_v4@NEW"],
        "S4: S2 + sub12 TabPFNs": S2 + ["base_n32_r0@TPF", "v4_n32_r0@TPF"],
        "S5: Caruana over all": "caruana"}
rmse = lambda p, m=slice(None): float(np.sqrt(((p - y)[m] ** 2).mean()))
nnls_w = lambda A, b: (lambda w: w / w.sum())(nnls(A, b)[0])


def caruana_w(A, b, steps=40):
    """Greedy forward selection with replacement (Caruana et al. 2004), starting from the best single model."""
    cnt = np.zeros(A.shape[1]); cur = np.zeros(len(b))
    for t in range(steps):
        sc = [np.sqrt((((cur * t + A[:, j]) / (t + 1) - b) ** 2).mean()) for j in range(A.shape[1])]
        j = int(np.argmin(sc)); cnt[j] += 1; cur = (cur * t + A[:, j]) / (t + 1)
    return cnt / cnt.sum()


def nested(X, fit):
    out = np.zeros((5, len(y)))
    for s in range(5):
        for a, b in KFold(5, shuffle=True, random_state=s).split(X):
            out[s, b] = X[b] @ fit(X[a], y[a])
    return out


print("single:", {k: round(rmse(p), 4) for k, p in P.items()})
z = pd.read_csv(ZILLOW_PATH); z = z[z.Month == JUN].set_index("Zip"); ok = z.zhvi.reindex(ZIPS).notna().values
rng = np.random.default_rng(0); I = rng.integers(0, len(y), (3000, len(y)))
k_ok = np.where(ok)[0]; I2 = rng.integers(0, len(k_ok), (3000, len(k_ok)))
base_pred = None; res = {}
for name, cols in SETS.items():
    if cols == "caruana":
        cols = list(P); fit = caruana_w
    else:
        cols = [c for c in cols if c in P]; fit = nnls_w
    X = np.c_[[P[c] for c in cols]].T; N = nested(X, fit); pm = N.mean(0); w = fit(X, y)
    if base_pred is None:
        base_pred = pm
    e0, e1 = (base_pred - y) ** 2, (pm - y) ** 2
    g = np.sqrt(e0[I].mean(1)) - np.sqrt(e1[I].mean(1))
    g2 = np.sqrt(e0[k_ok][I2].mean(1)) - np.sqrt(e1[k_ok][I2].mean(1))
    print(f"{name:26s} nested {np.mean([rmse(n) for n in N]):.4f} | gain vs sub12 {g.mean():+.4f} "
          f"[{np.percentile(g, 2.5):+.4f}, {np.percentile(g, 97.5):+.4f}] P {(g > 0).mean():.3f} | "
          f"excl ZHVI-missing {rmse(pm, ok):.4f} gain {g2.mean():+.4f} [{np.percentile(g2, 2.5):+.4f}, {np.percentile(g2, 97.5):+.4f}]")
    print("    w:", {c: round(float(v), 3) for c, v in zip(cols, w) if v > 0.005})
    res[name] = {"nested": float(np.mean([rmse(n) for n in N])), "P": float((g > 0).mean()),
                 "weights": {c: float(v) for c, v in zip(cols, w) if v > 0.005}}
json.dump(res, open(f"{OUT}/ens_compare_v11.json", "w"), indent=2)
