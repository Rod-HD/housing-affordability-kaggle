"""Figures for the report (PNG, outputs/figures)."""
import os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

warnings.filterwarnings("ignore")
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#d6d5d0",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
                     "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"})


def hbar(labels, vals, title, xlabel, fname, highlight=None, errs=None, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(labels) + 1.4))
    cols = [BLUE if (highlight and highlight(l)) else GRAY for l in labels]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols, height=0.62, xerr=errs, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(v + (errs[i] if errs is not None else 0) + max(vals) * 0.012, i, fmt.format(v), va="center", fontsize=8.5, color=INK2)
    ax.set_title(title, loc="left"); ax.set_xlabel(xlabel)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


tr, te, ss = load()
jun = tr[tr.Month == JUN].set_index("Zip").sort_index(); may = tr[tr.Month == MAY].set_index("Zip").sort_index()

# 1. month-to-month change of the target
d = (jun[T] - may[T]).clip(-6, 6)
fig, ax = plt.subplots(figsize=(7, 3.4))
vc = d.value_counts().sort_index()
ax.bar(vc.index, vc.values, color=BLUE, width=0.8)
ax.set_title("Thay đổi target tháng 5→6 theo ZIP (cắt ±6)", loc="left"); ax.set_xlabel("điểm %"); ax.set_ylabel("số ZIP")
ax.grid(axis="x", visible=False); fig.tight_layout(); fig.savefig(f"{FIG}/01_target_delta_may_jun.png", dpi=150); plt.close(fig)

# 2. feature drift June -> July (standardised by June std), measured on the same ZIPs
jul = te.set_index("Zip").loc[jun.index.intersection(te.Zip)]
cols = [c for c in RAW_NUM if c not in ("AnnualMedianHouseholdIncome", "RenterPercent")]
smd = pd.Series({c: (jul[c] - jun.loc[jul.index, c]).median() / jun[c].std() for c in cols}).sort_values()
smd2 = pd.Series({c: (jun[c] - may[c]).median() / jun[c].std() for c in cols}).loc[smd.index]
fig, ax = plt.subplots(figsize=(8, 4.2)); yy = np.arange(len(smd))
ax.barh(yy - 0.2, smd2.values, height=0.38, color=GRAY, label="tháng 5→6")
ax.barh(yy + 0.2, smd.values, height=0.38, color=ORANGE, label="tháng 6→7 (test)")
ax.set_yticks(yy, smd.index); ax.axvline(0, color=INK2, lw=0.8); ax.legend(frameon=False, loc="lower right")
ax.set_title("Dịch chuyển đặc trưng trên cùng ZIP (trung vị, đơn vị std)", loc="left"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{FIG}/02_feature_drift.png", dpi=150); plt.close(fig)

# 3. known-ZIP strategies
k = pd.read_csv(f"{OUT}/known_zip_results.csv")
hbar(k.strategy.tolist(), k.MAE.tolist(), "ZIP đã biết: dự đoán tháng 6 từ tháng 5 (MAE)", "MAE",
     "03_known_zip_strategies.png", highlight=lambda l: l.startswith("K1"), fmt="{:.3f}")

# 4. new-ZIP model ladder
s = pd.read_csv(f"{OUT}/newzip_cv_summary.csv", header=[0, 1], index_col=0)
lab = list(s.index); mae = s[("MAE", "mean")].values; sd = s[("MAE", "std")].values
if os.path.exists(f"{OUT}/geo_cv_summary.csv"):
    g = pd.read_csv(f"{OUT}/geo_cv_summary.csv", header=[0, 1], index_col=0)
    lab += [i + " (+geo ngoài)" for i in g.index if i.endswith("_geo")]
    mae = np.r_[mae, g.loc[[i for i in g.index if i.endswith("_geo")], ("MAE", "mean")].values]
    sd = np.r_[sd, g.loc[[i for i in g.index if i.endswith("_geo")], ("MAE", "std")].values]
o = np.argsort(mae)[::-1]
hbar([lab[i] for i in o], mae[o], "ZIP mới: so sánh mô hình (GroupKFold 5-fold, MAE ± std)", "MAE (điểm %)",
     "04_new_zip_models.png", highlight=lambda l: "geo" in l or l in ("lgb_te", "lgb_full_huber", "cat_full"), errs=sd[o])

# 5. July-proxy scenarios
j = pd.read_csv(f"{OUT}/julyproxy_folds.csv")
jj = j[j.model.str.startswith("mean")].groupby("scenario").MAE.agg(["mean", "std"])
hbar(jj.index.tolist(), jj["mean"].tolist(), "Xử lý dịch chuyển đặc trưng tháng 7 (July-proxy, MAE)", "MAE",
     "05_july_proxy.png", highlight=lambda l: l.startswith("S3"), errs=jj["std"].values, fmt="{:.3f}")

# 6. error by target band (best OOF) + by state size
oo = pd.read_csv(f"{OUT}/newzip_oof.csv"); y = oo.y.values
p = oo[[c for c in oo.columns if c.split("|")[0] in ("lgb_te", "lgb_full_huber", "cat_full")]].mean(1).values
band = pd.cut(y, [-1, 0, 10, 30, 50, 70, 90, 100], labels=["0", "1-10", "11-30", "31-50", "51-70", "71-90", "91-100"])
e = pd.DataFrame({"band": band, "ae": np.abs(y - p), "bias": p - y}).groupby("band").agg(ae=("ae", "mean"), bias=("bias", "mean"), n=("ae", "size"))
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.bar(e.index.astype(str), e.ae, color=BLUE, width=0.6, label="MAE")
ax.plot(e.index.astype(str), e.bias, color=ORANGE, lw=2, marker="o", ms=6, label="sai lệch TB (dự đoán − thực)")
ax.axhline(0, color=INK2, lw=0.8); ax.legend(frameon=False); ax.set_xlabel("khoảng target thực"); ax.grid(axis="x", visible=False)
ax.set_title("ZIP mới: sai số theo khoảng target (co về trung bình ở hai đầu)", loc="left")
fig.tight_layout(); fig.savefig(f"{FIG}/06_error_by_band.png", dpi=150); plt.close(fig)
e.to_csv(f"{OUT}/error_by_band.csv")

# 7. feature importance of final LGBM (geo) on all June rows
import lightgbm as lgb
from pipeline import feature_frame
J = add_basic_features(add_geo(tr[tr.Month == JUN].reset_index(drop=True)))
ft = oof_target_features(J); sm = {s_: i for i, s_ in enumerate(sorted(J.State.unique()))}
X = feature_frame(J, ft, sm); cols = RAW_NUM + ENG_NUM + ["state_code"] + ft.columns.tolist()
m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, verbose=-1, n_jobs=4).fit(X[cols], J[T])
imp = pd.Series(m.booster_.feature_importance("gain"), cols); imp = (imp / imp.sum()).sort_values(ascending=False)
imp.to_csv(f"{OUT}/final_feature_importance.csv")
top = imp.head(20)[::-1]
hbar(top.index.tolist()[::-1], (top.values * 100)[::-1].tolist(), "Top 20 đặc trưng (LightGBM, % gain)", "% tổng gain",
     "07_feature_importance.png", highlight=lambda l: l.startswith(("te_", "geo", "knn", "cnt_")), fmt="{:.1f}")
print("figures written to", FIG)
