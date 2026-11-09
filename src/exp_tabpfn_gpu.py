"""TabPFN variants on the cached July-proxy folds (GPU if available).
Usage: python exp_tabpfn_gpu.py VARIANT [N_EST] [SEED] [ARG]
  base   plain TabPFN, N_EST estimators, random_state SEED
  sel    top-ARG features by in-fold LightGBM gain (fit on the training part only)
  stack  + inner out-of-fold LightGBM prediction as an extra feature
  ft     fine-tune TabPFN weights on the training part (ARG = epochs)
  v4     v4 feature set (fold_cache_v4.pkl, adds purchase-year costs)
  v5     v4 + Realtor.com features (fold_cache_v5.pkl)
Output: outputs/rmse_oof_TPF_<tag>_s0.csv (columns Zip, y, <tag>)"""
import os, pickle, sys, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(ROOT, ".env"), encoding="utf8"):
    if line.startswith("TABPFN_TOKEN="):
        os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()
os.environ.setdefault("HF_HOME", "D:/Programs/huggingface_cache")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "D:/Programs/huggingface_cache/tabpfn")
os.environ["TABPFN_NO_BROWSER"] = "1"
import numpy as np, pandas as pd, torch, lightgbm as lgb
from sklearn.model_selection import KFold
from common import OUT, load, JUN

warnings.filterwarnings("ignore")
VAR = sys.argv[1] if len(sys.argv) > 1 else "base"
N_EST = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ARG = sys.argv[4] if len(sys.argv) > 4 else ""
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TAG = f"{VAR}{ARG}_n{N_EST}_r{SEED}"
from tabpfn import TabPFNRegressor

cache = f"fold_cache_{VAR}.pkl" if VAR in ("v4", "v5") else "fold_cache_full.pkl"
folds = pickle.load(open(f"{OUT}/{cache}", "rb"))
tr, _, _ = load(); zips = np.sort(tr[tr.Month == JUN].Zip.values)
splits = list(KFold(5, shuffle=True, random_state=200).split(zips))
LGB = dict(n_estimators=1200, learning_rate=0.02, num_leaves=31, min_child_samples=30, subsample=0.7, subsample_freq=1,
           colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=8)


def tabpfn(Xt, yt, Xv):
    if VAR == "ft":
        from tabpfn.finetuning import FinetunedTabPFNRegressor
        m = FinetunedTabPFNRegressor(device=DEV, epochs=int(ARG or 30), random_state=SEED, n_estimators_final_inference=N_EST,
                                     n_finetune_ctx_plus_query_samples=int(os.environ.get("TPF_FT_CTX", "2048")),
                                     extra_regressor_kwargs=dict(ignore_pretraining_limits=True))
    else:
        m = TabPFNRegressor(device=DEV, n_estimators=N_EST, ignore_pretraining_limits=True, random_state=SEED,
                            memory_saving_mode=True if os.environ.get("TPF_MEMSAVE") else "auto")
    m.fit(Xt.values.astype("float32"), yt)
    return m.predict(Xv.values.astype("float32"))


oof = np.zeros(len(zips)); y_all = np.zeros(len(zips)); t0 = time.time()
ONLY = [int(x) for x in os.environ.get("TPF_FOLDS", "").split(",") if x]
for f, ((Xt, yt, Xv, yv), (a, b)) in enumerate(zip(folds, splits)):
    if ONLY and f not in ONLY:
        continue
    if VAR == "sel":
        g = lgb.LGBMRegressor(**LGB, importance_type="gain").fit(Xt, yt)
        keep = Xt.columns[np.argsort(-g.feature_importances_)[:int(ARG)]]
        Xt, Xv = Xt[keep], Xv[keep]
    if VAR == "stack":
        inner = np.zeros(len(yt))
        for ia, ib in KFold(5, shuffle=True, random_state=7).split(Xt):
            inner[ib] = lgb.LGBMRegressor(**LGB).fit(Xt.iloc[ia], yt[ia]).predict(Xt.iloc[ib])
        Xt = Xt.assign(lgb_pred=inner)
        Xv = Xv.assign(lgb_pred=lgb.LGBMRegressor(**LGB).fit(Xt.drop(columns="lgb_pred"), yt).predict(Xv))
    p = np.clip(tabpfn(Xt, yt, Xv), 0, 100)
    oof[b] = p; y_all[b] = yv
    print(f"[{TAG}] fold {f} {time.time()-t0:.0f}s RMSE {np.sqrt(((p - yv) ** 2).mean()):.3f}", flush=True)
if ONLY:
    sys.exit()
print(f"[{TAG}] OOF RMSE", round(float(np.sqrt(((oof - y_all) ** 2).mean())), 4), flush=True)
pd.DataFrame({"Zip": zips, "y": y_all, TAG: oof}).to_csv(f"{OUT}/rmse_oof_TPF_{TAG}_s0.csv", index=False)
