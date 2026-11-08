"""Download ACS 5-year table B25106 (tenure by housing costs as % of household income) for every ZCTA
and derive the share of NOT cost-burdened households (owners, renters, all).
Needs a free Census API key in the project-root .env file:  CENSUS_API_KEY=xxxxxxxx
api.census.gov rejects some regions; the r.jina.ai reader is used as a fallback relay.
Output: external/acs_b25106_zcta.csv"""
import io, json, os, sys, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YEARS = ("2023", "2022")


def read_key():
    env = os.path.join(ROOT, ".env")
    key = os.environ.get("CENSUS_API_KEY", "")
    if not key and os.path.exists(env):
        for line in open(env, encoding="utf8"):
            if line.strip().startswith("CENSUS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("CENSUS_API_KEY is empty: put your key in .env (see .env.example)")
    return key


def get_json(url):
    for u in (url, "https://r.jina.ai/" + url):
        try:
            txt = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=600).read().decode()
            i = txt.find("[[")
            if i >= 0:
                return json.loads(txt[i:txt.rfind("]]") + 2])
        except Exception as e:
            print("  failed via", u[:40], "->", type(e).__name__)
    raise RuntimeError("could not download " + url.split("&key=")[0])


OWN_BURDEN = [6, 10, 14, 18, 22]    # owner 30%+ by income band
RENT_BURDEN = [28, 32, 36, 40, 44]  # renter 30%+ by income band
VARS = sorted({1, 2, 23, 24, 45, 46, *OWN_BURDEN, *RENT_BURDEN})

if __name__ == "__main__":
    key = read_key()
    for yr in YEARS:
        get = ",".join(f"B25106_{v:03d}E" for v in VARS)
        url = f"https://api.census.gov/data/{yr}/acs/acs5?get={get}&for=zip%20code%20tabulation%20area:*&key={key}"
        try:
            rows = get_json(url); break
        except RuntimeError as e:
            print(yr, e)
    else:
        sys.exit("download failed for all years")
    d = pd.DataFrame(rows[1:], columns=rows[0])
    d["Zip"] = d["zip code tabulation area"].astype(int)
    v = {c: pd.to_numeric(d[f"B25106_{c:03d}E"], errors="coerce") for c in VARS}
    own_b = sum(v[c] for c in OWN_BURDEN); rent_b = sum(v[c] for c in RENT_BURDEN)
    own_den = v[2] - v[23]; rent_den = v[24] - v[45] - v[46]
    out = pd.DataFrame({"Zip": d.Zip,
                        "acs_owner_ok": 100 * (1 - own_b / own_den.replace(0, np.nan)),
                        "acs_renter_ok": 100 * (1 - rent_b / rent_den.replace(0, np.nan)),
                        "acs_all_ok": 100 * (1 - (own_b + rent_b) / (own_den + rent_den).replace(0, np.nan)),
                        "acs_owner_share": 100 * v[2] / v[1].replace(0, np.nan),
                        "acs_households": v[1], "acs_year": int(yr)})
    path = os.path.join(ROOT, "external", "acs_b25106_zcta.csv")
    out.to_csv(path, index=False)
    print("saved", path, out.shape)
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    j = tr[tr.Month == "6/1/2026"].merge(out, on="Zip", how="left")
    for c in ("acs_all_ok", "acs_owner_ok", "acs_renter_ok"):
        print(f"{c:15s} coverage {j[c].notna().mean():.3f}  spearman with target {j[c].corr(j.AffordabilityPercentageTrue, method='spearman'):+.3f}")
