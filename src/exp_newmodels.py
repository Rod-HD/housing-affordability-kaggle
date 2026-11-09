"""Recent tabular models on the cached July-proxy folds (GPU).
Usage: python exp_newmodels.py MODEL CACHE [ARG]
  tabicl  TabICLv2 regressor (tabicl 2.2.0), ARG = n_estimators (default 8)
  tabdpt  TabDPT (tabdpt 1.3.2) with retrieval context, ARG = context size (default 1024)
  xrfm    xRFM kernel machine (xrfm 0.4.5), inner 85/15 split for its validation set
Output: outputs/rmse_oof_NEW_<model><arg>_<cache>_s0.csv; per-fold parts in outputs/new_parts (resumable)."""
import os, pickle, sys, time, warnings
os.environ.setdefault("HF_HOME", "D:/Programs/huggingface_cache")
import numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold, train_test_split
from common import OUT, load, JUN

warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fill(Xt, Xv):
    """Median fill + missing flags, standardised on the training part (for models that need finite inputs)."""
    med = Xt.median(); miss = [c for c in Xt if Xt[c].isna().any() or Xv[c].isna().any()]
    ft, fv = Xt.fillna(med), Xv.fillna(med)
    for c in miss:
        ft[c + "_na"] = Xt[c].isna().astype(float); fv[c + "_na"] = Xv[c].isna().astype(float)
    mu, sd = ft.mean(), ft.std().replace(0, 1)
    return ((ft - mu) / sd).values.astype("float32"), ((fv - mu) / sd).values.astype("float32")


def run(MODEL, ARG, Xt, yt, Xv):
    if MODEL == "tabicl":
        from tabicl import TabICLRegressor
        m = TabICLRegressor(n_estimators=int(ARG or 8), device=DEV, random_state=0, batch_size=1)
        m.fit(Xt.values.astype("float32"), yt)
        return m.predict(Xv.values.astype("float32"))
    if MODEL == "tabdpt":
        from tabdpt import TabDPTRegressor
        Ft, Fv = fill(Xt, Xv)
        m = TabDPTRegressor(device=DEV, compile=False, use_flash=False, context_reduction="retrieval", verbose=False)
        m.fit(Ft, yt)
        return m.predict(Fv, n_ensembles=8, context_size=int(ARG or 1024), batch_size=64, seed=0)
    if MODEL == "xrfm":
        from xrfm import xRFM
        Ft, Fv = fill(Xt, Xv)
        a, b = train_test_split(np.arange(len(yt)), test_size=0.15, random_state=0)
        m = xRFM(device=DEV, tuning_metric="mse", random_state=0, verbose=False)
        m.fit(torch.tensor(Ft[a]), torch.tensor(yt[a], dtype=torch.float32)[:, None],
              torch.tensor(Ft[b]), torch.tensor(yt[b], dtype=torch.float32)[:, None])
        return np.asarray(m.predict(torch.tensor(Fv))).ravel()
    raise SystemExit(f"unknown model {MODEL}")


if __name__ == "__main__":
    MODEL = sys.argv[1]; CACHE = sys.argv[2] if len(sys.argv) > 2 else "v4"
    ARG = sys.argv[3] if len(sys.argv) > 3 else ""
    TAG = f"{MODEL}{ARG}_{CACHE}"
    folds = pickle.load(open(f"{OUT}/fold_cache_{CACHE}.pkl", "rb"))
    tr, _, _ = load(); zips = np.sort(tr[tr.Month == JUN].Zip.values)
    splits = list(KFold(5, shuffle=True, random_state=200).split(zips))
    oof = np.zeros(len(zips)); y_all = np.zeros(len(zips)); t0 = time.time()
    for f, ((Xt, yt, Xv, yv), (a, b)) in enumerate(zip(folds, splits)):
        part = f"{OUT}/new_parts/{TAG}_f{f}.npy"
        if os.path.exists(part):
            p = np.load(part)
        else:
            p = np.clip(np.asarray(run(MODEL, ARG, Xt, np.asarray(yt, float), Xv), float).ravel(), 0, 100)
            os.makedirs(os.path.dirname(part), exist_ok=True); np.save(part, p)
            if DEV == "cuda":
                torch.cuda.empty_cache()
        oof[b] = p; y_all[b] = yv
        print(f"[{TAG}] fold {f} {time.time()-t0:.0f}s RMSE {np.sqrt(((p - yv) ** 2).mean()):.3f}", flush=True)
    print(f"[{TAG}] OOF RMSE", round(float(np.sqrt(((oof - y_all) ** 2).mean())), 4), flush=True)
    pd.DataFrame({"Zip": zips, "y": y_all, TAG: oof}).to_csv(f"{OUT}/rmse_oof_NEW_{TAG}_s0.csv", index=False)
