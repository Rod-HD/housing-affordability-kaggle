"""v2 submission, optimised for the official metric: 0.2*RMSE(seen ZIPs) + 0.8*RMSE(new ZIPs).
Seen ZIPs: June label (persistence). New ZIPs: NNLS-weighted ensemble (weights from ens_rmse.py, nested-CV checked)
on geo + similarity-neighbour features (EXTRA_GEO=1), July features de-drifted."""
import os, sys, json, warnings
os.environ["EXTRA_GEO"] = "1"
import numpy as np, pandas as pd
from common import *
from pipeline import fit_predict, drift_table, dedrift

warnings.filterwarnings("ignore")
SEEDS = [0, 1, 2]
NAME = sys.argv[1] if len(sys.argv) > 1 else "sub06_rmse_nnls_ens_geo_v2"
W = {k.split("@")[0]: v for k, v in json.load(open(f"{OUT}/ens_rmse_weights.json")).items() if v > 0.005}
s = sum(W.values()); W = {k: v / s for k, v in W.items()}
print("weights:", {k: round(v, 3) for k, v in W.items()})

tr, te, ss = load()
jun = tr[tr.Month == JUN].reset_index(drop=True)
known = te.Zip.isin(jun.Zip)
shift = drift_table(jun, te[known])
jul_dd = dedrift(te, shift)
ref = jul_dd[known].copy(); ref[T] = ref.Zip.map(jun.set_index("Zip")[T]).values
J, R, Q = add_geo(jun), add_geo(ref), add_geo(jul_dd[~known].copy())
preds = []
for sd in SEEDS:
    p, _ = fit_predict(J, R, Q, list(W), seed=sd)
    preds.append(sum(W[m] * p[m] for m in W))
    print(f"seed {sd} done, mean {preds[-1].mean():.2f}", flush=True)
p_new = np.clip(np.mean(preds, 0), 0, 100)
sub = ss[["Zip", "State"]].copy(); sub["Zip"] = sub.Zip.astype(int).astype(str).str.zfill(5)
sub[T] = np.nan
sub.loc[known.values, T] = te.loc[known, "Zip"].map(jun.set_index("Zip")[T]).values
sub.loc[~known.values, T] = p_new
assert sub[T].notna().all() and sub[T].between(0, 100).all()
sub.to_csv(f"{SUB}/{NAME}.csv", index=False)
old = pd.read_csv(f"{SUB}/sub02_known_last__new_ens_geo.csv")[T].values
print("wrote", NAME, "| RMS diff vs sub02 on new ZIPs: %.3f" % np.sqrt(((sub[T].values - old)[~known.values] ** 2).mean()))
