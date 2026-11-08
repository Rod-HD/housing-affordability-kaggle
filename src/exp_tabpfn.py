"""TabPFN (tabular foundation model, Prior Labs) on the cached July-proxy folds, for comparison and ensembling.
Needs TABPFN_TOKEN in .env and the license accepted at https://ux.priorlabs.ai (Licenses tab).
Weights are cached under D:/Programs/huggingface_cache. Output: outputs/rmse_oof_TABPFN_s0.csv"""
import os, pickle, sys, time, warnings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(ROOT, ".env"), encoding="utf8"):
    if line.startswith("TABPFN_TOKEN="):
        os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip()
os.environ.setdefault("HF_HOME", "D:/Programs/huggingface_cache")
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", "D:/Programs/huggingface_cache/tabpfn")
os.environ["TABPFN_NO_BROWSER"] = "1"
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import OUT, load, JUN

warnings.filterwarnings("ignore")
N_EST = int(sys.argv[1]) if len(sys.argv) > 1 else 4
from tabpfn import TabPFNRegressor

folds = pickle.load(open(f"{OUT}/fold_cache_full.pkl", "rb"))
tr, _, _ = load(); zips = np.sort(tr[tr.Month == JUN].Zip.values)
splits = list(KFold(5, shuffle=True, random_state=200).split(zips))
oof = np.zeros(len(zips)); y_all = np.zeros(len(zips)); t0 = time.time()
for f, ((Xt, yt, Xv, yv), (a, b)) in enumerate(zip(folds, splits)):
    m = TabPFNRegressor(device="cpu", n_estimators=N_EST, ignore_pretraining_limits=True, random_state=0)
    m.fit(Xt.values.astype("float32"), yt)
    p = np.clip(m.predict(Xv.values.astype("float32")), 0, 100)
    oof[b] = p; y_all[b] = yv
    print(f"fold {f} {time.time()-t0:.0f}s RMSE {np.sqrt(((p - yv) ** 2).mean()):.3f}", flush=True)
print("TabPFN OOF RMSE", round(float(np.sqrt(((oof - y_all) ** 2).mean())), 4))
pd.DataFrame({"Zip": zips, "y": y_all, "tabpfn": oof}).to_csv(f"{OUT}/rmse_oof_TABPFN_s0.csv", index=False)
