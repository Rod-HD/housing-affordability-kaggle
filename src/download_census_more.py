"""ACS 5-year (ZCTA) cost-to-income DISTRIBUTIONS and property taxes, which complement B25106 near the 30% threshold:
B25070 gross rent as % of household income (renters, 10 brackets + not computed),
B25091 owner costs as % of income by mortgage status, B25103 median real estate taxes by mortgage status.
These are historical survey tables (2020-2024), not the competition's Melissa labels.
Needs CENSUS_API_KEY in .env. Output: external/acs_more_zcta.csv"""
import os
import numpy as np, pandas as pd
from download_census import read_key, get_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARS = ([f"B25070_{i:03d}E" for i in range(1, 12)] + [f"B25091_{i:03d}E" for i in range(1, 24)]
        + ["B25103_001E", "B25103_002E", "B25103_003E"])
EDGES = [10, 15, 20, 25, 30, 35, 40, 50]          # upper edges of the first 8 brackets (%), last bracket = 50+


def shares(counts, name):
    """counts: columns = 10 brackets (<10 ... 50+), returns cumulative shares below 20/25/30/35/40% of computed households."""
    tot = counts.sum(1).replace(0, np.nan); cum = counts.cumsum(1).div(tot, axis=0)
    out = {f"{name}_lt{e}": cum.iloc[:, k] for k, e in enumerate(EDGES) if e in (20, 25, 30, 35, 40)}
    out[f"{name}_ge50"] = counts.iloc[:, -1] / tot
    return pd.DataFrame(out), tot


if __name__ == "__main__":
    key = read_key(); d = None
    for year in (2024, 2023):
        try:
            rows = get_json(f"https://api.census.gov/data/{year}/acs/acs5?get={','.join(VARS)}"
                            f"&for=zip%20code%20tabulation%20area:*&key={key}")
            d = pd.DataFrame(rows[1:], columns=rows[0]).set_index("zip code tabulation area"); break
        except Exception as e:
            print(year, "failed:", str(e)[:120])
    d = d.apply(pd.to_numeric, errors="coerce"); d = d.where(d > -1e8)
    d.index = d.index.astype(int); d.index.name = "Zip"; print("ACS year", year, d.shape)
    rent, rtot = shares(d[[f"B25070_{i:03d}E" for i in range(2, 11)]], "acsm_rent")       # 9 brackets, 011 = not computed
    om, omt = shares(d[[f"B25091_{i:03d}E" for i in range(3, 12)]], "acsm_ownm")         # with mortgage, 012 = not computed
    on, ont = shares(d[[f"B25091_{i:03d}E" for i in range(14, 23)]], "acsm_ownnm")
    f = pd.concat([rent, om, on], axis=1)
    f["acsm_rent_notcomp"] = d["B25070_011E"] / d["B25070_001E"].replace(0, np.nan)
    f["acsm_own_notcomp"] = (d["B25091_012E"] + d["B25091_023E"]) / d["B25091_001E"].replace(0, np.nan)
    f["acsm_own_lt30"] = (f["acsm_ownm_lt30"] * omt + f["acsm_ownnm_lt30"] * ont) / (omt + ont)
    f["acsm_all_lt30"] = (f["acsm_rent_lt30"] * rtot + f["acsm_own_lt30"] * (omt + ont)) / (rtot + omt + ont)
    f["acsm_all_lt25"] = (f["acsm_rent_lt25"] * rtot + f["acsm_ownm_lt25"] * omt + f["acsm_ownnm_lt25"] * ont) / (rtot + omt + ont)
    f["acsm_all_lt35"] = (f["acsm_rent_lt35"] * rtot + f["acsm_ownm_lt35"] * omt + f["acsm_ownnm_lt35"] * ont) / (rtot + omt + ont)
    f["acsm_tax"] = d["B25103_001E"]; f["acsm_tax_mort"] = d["B25103_002E"]; f["acsm_tax_nomort"] = d["B25103_003E"]
    f["acsm_year"] = year
    f.reset_index().to_csv(os.path.join(ROOT, "external", "acs_more_zcta.csv"), index=False)
    print("features", f.shape)
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    j = tr[tr.Month == "6/1/2026"].merge(f.reset_index(), on="Zip", how="left")
    for c in ["acsm_rent_lt30", "acsm_own_lt30", "acsm_all_lt30", "acsm_all_lt25", "acsm_all_lt35", "acsm_tax"]:
        print(f"{c:18s} cov {j[c].notna().mean():.3f} spearman {j[c].corr(j.AffordabilityPercentageTrue, method='spearman'):+.3f}")
