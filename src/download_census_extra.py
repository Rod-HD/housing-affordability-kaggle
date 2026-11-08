"""ACS 5-year (ZCTA) tables that describe tenure-specific income and costs, which the competition data lacks:
B25119 median household income by tenure, B19001 household income distribution (16 brackets),
B25071 median gross rent % of income, B25092 median owner costs % of income (with / without mortgage),
B25088 median owner costs, B25081 mortgage status, B25077 median value, B25064 median gross rent, B25003 tenure.
Needs CENSUS_API_KEY in .env. Output: external/acs_extra_zcta.csv"""
import os, sys
import numpy as np, pandas as pd
from download_census import read_key, get_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARS = (["B25119_001E", "B25119_002E", "B25119_003E", "B25071_001E", "B25092_001E", "B25092_002E", "B25092_003E",
         "B25088_002E", "B25088_003E", "B25077_001E", "B25064_001E", "B25003_001E", "B25003_002E", "B25003_003E"]
        + [f"B25081_{i:03d}E" for i in range(1, 10)]
        + [f"B19001_{i:03d}E" for i in range(1, 18)])

if __name__ == "__main__":
    key = read_key()
    frames = []
    for chunk in (VARS[:24], VARS[24:]):              # API limit: 50 variables per call
        url = (f"https://api.census.gov/data/2023/acs/acs5?get={','.join(chunk)}"
               f"&for=zip%20code%20tabulation%20area:*&key={key}")
        rows = get_json(url)
        d = pd.DataFrame(rows[1:], columns=rows[0]).set_index("zip code tabulation area")
        frames.append(d.apply(pd.to_numeric, errors="coerce"))
    d = pd.concat(frames, axis=1)
    d = d.where(d > -1e8)                             # Census uses -666666666 etc. for missing
    d.index = d.index.astype(int); d.index.name = "Zip"
    d.to_csv(os.path.join(ROOT, "external", "acs_extra_raw.csv"))
    print("raw", d.shape)
    # B25081: find the 'without a mortgage' line as the column that completes the total
    tot = d["B25081_001E"]; withm = d["B25081_002E"]
    cand = {c: ((d[c] + withm - tot).abs() / tot.clip(lower=1)).median() for c in [f"B25081_{i:03d}E" for i in range(3, 10)]}
    nom = min(cand, key=cand.get); print("B25081 no-mortgage column:", nom, round(cand[nom], 4))
    f = pd.DataFrame(index=d.index)
    f["acs_inc_all"] = d["B25119_001E"]; f["acs_inc_owner"] = d["B25119_002E"]; f["acs_inc_renter"] = d["B25119_003E"]
    f["acs_rent_pct_inc"] = d["B25071_001E"]; f["acs_own_pct_inc"] = d["B25092_001E"]
    f["acs_own_pct_inc_mort"] = d["B25092_002E"]; f["acs_own_pct_inc_nomort"] = d["B25092_003E"]
    f["acs_owncost_mort"] = d["B25088_002E"]; f["acs_owncost_nomort"] = d["B25088_003E"]
    f["acs_value"] = d["B25077_001E"]; f["acs_rent"] = d["B25064_001E"]
    f["acs_owner_share2"] = d["B25003_002E"] / d["B25003_001E"]
    f["acs_mortgage_share"] = withm / tot.replace(0, np.nan)
    br = d[[f"B19001_{i:03d}E" for i in range(2, 18)]].div(d["B19001_001E"].replace(0, np.nan), axis=0)
    br.columns = [f"acs_incb{i:02d}" for i in range(2, 18)]
    f = pd.concat([f, br], axis=1)
    f.reset_index().to_csv(os.path.join(ROOT, "external", "acs_extra_zcta.csv"), index=False)
    print("features", f.shape)
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    j = tr[tr.Month == "6/1/2026"].merge(f.reset_index(), on="Zip", how="left")
    for c in ["acs_inc_renter", "acs_inc_owner", "acs_rent_pct_inc", "acs_own_pct_inc", "acs_own_pct_inc_mort", "acs_mortgage_share", "acs_value"]:
        print(f"{c:22s} cov {j[c].notna().mean():.3f} spearman {j[c].corr(j.AffordabilityPercentageTrue, method='spearman'):+.3f}")
