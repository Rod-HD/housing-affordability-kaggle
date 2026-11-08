"""RMSE-optimal ensemble weights (non-negative least squares, sum-to-1) with nested CV to avoid optimism."""
import sys, glob, json
import numpy as np, pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from common import OUT, metrics

files = sys.argv[1:] or sorted(glob.glob(f"{OUT}/rmse_oof_*_s0.csv"))
P, y = {}, None
for f in files:
    tag = f.split("rmse_oof_")[1].split("_s")[0]
    o = pd.read_csv(f).sort_values("Zip")
    y = o.y.values
    for c in o.columns[2:]:
        P[f"{c}@{tag}"] = o[c].values
names = list(P); X = np.c_[[P[n] for n in names]].T
print("single models (RMSE):")
for n in sorted(names, key=lambda n: metrics(y, P[n])["RMSE"]):
    print(f"  {n:22s} {metrics(y, P[n])['RMSE']:.3f}")


def fit_w(Xa, ya):
    w, _ = nnls(Xa, ya); return w / w.sum()


# nested: weights fitted on 4/5 of OOF rows, evaluated on held-out 1/5, repeated
sc = []
for s in range(5):
    pred = np.zeros_like(y, dtype=float)
    for a, b in KFold(5, shuffle=True, random_state=s).split(X):
        pred[b] = X[b] @ fit_w(X[a], y[a])
    sc.append(metrics(y, pred)["RMSE"])
w = fit_w(X, y)
print("\nNNLS weights (full):", {n: round(v, 3) for n, v in zip(names, w) if v > 0.005})
print("NNLS ensemble RMSE in-sample %.3f | nested-CV %.3f ± %.3f" % (metrics(y, X @ w)["RMSE"], np.mean(sc), np.std(sc)))
print("equal-average of all: %.3f" % metrics(y, X.mean(1))["RMSE"])
json.dump({n: float(v) for n, v in zip(names, w)}, open(f"{OUT}/ens_rmse_weights.json", "w"), indent=2)
