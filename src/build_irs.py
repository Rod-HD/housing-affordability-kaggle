"""EXTERNAL DATA: IRS SOI individual income tax statistics by ZIP (2022, fallback 2021), public at irs.gov/pub/irs-soi.
Per ZIP: number of returns by AGI bracket (the within-ZIP income distribution the competition data lacks),
mortgage-interest and real-estate-tax prevalence/amounts (owner housing-cost proxies)."""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGES = np.array([1, 25e3, 50e3, 75e3, 100e3, 200e3])  # lower edges of agi_stub 1..6
COLS = ["zipcode", "agi_stub", "N1", "N2", "A00100", "N19300", "A19300", "N18500", "A18500", "ELDERLY", "N00200", "A00200",
        "N02650", "A02650"]


def load(yr):
    fn = os.path.join(ROOT, "external", "irs", f"{yr}zpallagi.csv")
    hdr = pd.read_csv(fn, nrows=0).columns
    use = [c for c in COLS if c in hdr]
    d = pd.read_csv(fn, usecols=use)
    d = d[(d.zipcode > 0) & (d.zipcode < 99999)]
    return d


out = []
for yr in ("22", "21"):
    d = load(yr)
    piv = d.pivot_table(index="zipcode", columns="agi_stub", values="N1", aggfunc="sum").reindex(columns=range(1, 7)).fillna(0)
    agi = d.pivot_table(index="zipcode", columns="agi_stub", values="A00100", aggfunc="sum").reindex(columns=range(1, 7)).fillna(0)
    tot = d.groupby("zipcode")[[c for c in COLS[2:] if c in d.columns]].sum()
    f = pd.DataFrame(index=piv.index)
    n = piv.sum(1)
    for k in range(1, 7):
        f[f"irs_share_b{k}"] = piv[k] / n
    f["irs_n_returns"] = n
    f["irs_mean_agi"] = tot["A00100"] * 1000 / tot["N1"]
    f["irs_mean_agi_top"] = agi[6] * 1000 / piv[6].replace(0, np.nan)     # mean AGI of $200k+ returns (Pareto tail)
    f["irs_mortgage_share"] = tot["N19300"] / tot["N1"]
    f["irs_mortgage_int"] = tot["A19300"] * 1000 / tot["N19300"].replace(0, np.nan)
    f["irs_retax_share"] = tot["N18500"] / tot["N1"]
    f["irs_retax_amt"] = tot["A18500"] * 1000 / tot["N18500"].replace(0, np.nan)
    f["irs_elderly_share"] = tot["ELDERLY"] / tot["N1"]
    f["irs_persons_per_return"] = tot["N2"] / tot["N1"]
    f["irs_wage_share"] = tot["N00200"] / tot["N1"]
    f["irs_year"] = int(yr)
    out.append(f)
F = out[0].combine_first(out[1])
F.index.name = "Zip"
F.reset_index().to_csv(os.path.join(ROOT, "external", "irs_features.csv"), index=False)
print(F.shape); print(F.describe().T[["mean", "50%"]].round(3))
