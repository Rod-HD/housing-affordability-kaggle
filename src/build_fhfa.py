"""FHFA annual house price index for 5-digit ZIP codes (all transactions, public; last updated 2026-03-31).
Output: external/fhfa_features.csv (Zip, fhfa_g1/g3/g5/g10/g20 log growth to the latest year, fhfa_y<year> log index)."""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
x = pd.read_excel(os.path.join(ROOT, "external", "fhfa", "hpi_at_zip5.xlsx"), skiprows=6, header=None,
                  names=["Zip", "Year", "chg", "hpi", "hpi1990", "hpi2000"], dtype={"Zip": str})
x = x[x.Zip.str.fullmatch(r"\d{5}", na=False)]; x["Zip"] = x.Zip.astype(int)
x["hpi"] = pd.to_numeric(x.hpi, errors="coerce"); x["Year"] = pd.to_numeric(x.Year, errors="coerce")
w = np.log(x.pivot_table(index="Zip", columns="Year", values="hpi"))
last = int(w.columns.max()); print("years", int(w.columns.min()), "-", last, "| zips", len(w))
f = pd.DataFrame(index=w.index)
for k in (1, 3, 5, 10, 20):
    f[f"fhfa_g{k}"] = w[last] - w[last - k]
for y in range(2000, last + 1):
    f[f"fhfa_y{y}"] = w[y]
f.reset_index().to_csv(os.path.join(ROOT, "external", "fhfa_features.csv"), index=False)
tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv")); j = tr[tr.Month == "6/1/2026"].merge(f.reset_index(), on="Zip", how="left")
for c in ["fhfa_g1", "fhfa_g5", "fhfa_g10", "fhfa_g20"]:
    print(f"{c:9s} cov {j[c].notna().mean():.3f} spearman {j[c].corr(j.AffordabilityPercentageTrue, method='spearman'):+.3f}")
