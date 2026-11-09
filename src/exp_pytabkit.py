"""Modern deep tabular models (pytabkit: RealMLP-TD, TabM-D) on the cached July-proxy folds.
Usage: python exp_pytabkit.py MODEL [N_CV] [CACHE]
  MODEL  realmlp | tabm | realmlp_hpo-free variants are the pytabkit "tuned default" settings
  N_CV   inner bagging folds (default 1 = single 80/20 split for early stopping)
Output: outputs/rmse_oof_PTK_<model>_cv<N_CV>_s0.csv"""
import os, pickle, sys, time, warnings
import numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold
from common import OUT, load, JUN

warnings.filterwarnings("ignore")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "realmlp"
N_CV = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CACHE = sys.argv[3] if len(sys.argv) > 3 else "fold_cache_full.pkl"
DEV = os.environ.get("PTK_DEV") or ("cuda:0" if torch.cuda.is_available() else "cpu")
TAG = f"{MODEL}_cv{N_CV}" + ("_v4" if "v4" in CACHE else "")
from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor

folds = pickle.load(open(f"{OUT}/{CACHE}", "rb"))
tr, _, _ = load(); zips = np.sort(tr[tr.Month == JUN].Zip.values)
splits = list(KFold(5, shuffle=True, random_state=200).split(zips))


def make():
    kw = dict(device=DEV, random_state=0, n_cv=N_CV, verbosity=0)
    return {"realmlp": lambda: RealMLP_TD_Regressor(**kw), "tabm": lambda: TabM_D_Regressor(**kw)}[MODEL]()


def fill(Xt, Xv):  # median fill + missing flags (both models expect finite numerics)
    med = Xt.median(); miss = [c for c in Xt if Xt[c].isna().any() or Xv[c].isna().any()]
    ft = Xt.fillna(med).astype("float32"); fv = Xv.fillna(med).astype("float32")
    for c in miss:
        ft[c + "_na"] = Xt[c].isna().astype("float32"); fv[c + "_na"] = Xv[c].isna().astype("float32")
    return ft, fv


oof = np.zeros(len(zips)); y_all = np.zeros(len(zips)); t0 = time.time()
for f, ((Xt, yt, Xv, yv), (a, b)) in enumerate(zip(folds, splits)):
    Ft, Fv = fill(Xt, Xv)
    m = make(); m.fit(Ft, yt.astype("float32"))
    p = np.clip(np.asarray(m.predict(Fv)).ravel(), 0, 100)
    oof[b] = p; y_all[b] = yv
    print(f"[{TAG}] fold {f} {time.time()-t0:.0f}s RMSE {np.sqrt(((p - yv) ** 2).mean()):.3f}", flush=True)
print(f"[{TAG}] OOF RMSE", round(float(np.sqrt(((oof - y_all) ** 2).mean())), 4), flush=True)
pd.DataFrame({"Zip": zips, "y": y_all, TAG: oof}).to_csv(f"{OUT}/rmse_oof_PTK_{TAG}_s0.csv", index=False)
