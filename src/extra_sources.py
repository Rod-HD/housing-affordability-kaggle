"""Additional public data sources kept apart from common.py so they can be switched off without touching the validated pipeline."""
import os
import numpy as np, pandas as pd
from common import ROOT, ZFEATS

RTR_PATH = os.path.join(ROOT, "external", "realtor_features.csv")


def add_realtor(d):
    """Realtor.com listing prices, size and market activity. Requires add_zillow first (zhvi, zhvi_f)."""
    F = pd.read_csv(RTR_PATH)
    d = d.merge(F, on=["Zip", "Month"], how="left")
    inc = d["AnnualMedianHouseholdIncome"].clip(lower=1000)
    p = d["rtr_price_12m"]
    z = d["zhvi"]
    scale = float((z / p).replace([np.inf, -np.inf], np.nan).median())    # listing price -> ZHVI scale, no target involved
    best = z.fillna(p * scale)
    r = 0.068 / 12; fac = r / (1 - (1 + r) ** -360)
    d["rtr_log_pti"] = np.log(p / inc)
    d["rtr_log_ppsf_inc"] = np.log(d["rtr_ppsf_12m"] / inc)
    d["rtr_vs_zhvi"] = np.log(p / d["zhvi_f"])
    d["rtr_rent_yield"] = 12 * d["RentIndex"] / p
    d["rtr_own_burden"] = 12 * p * scale * 0.8 * fac / inc
    d["rtr_best_log_pti"] = np.log(best / inc)
    d["rtr_zhvi_missing"] = z.isna().astype(float)
    new = [c for c in d.columns if c.startswith("rtr_")]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


HUD_PATH = os.path.join(ROOT, "external", "hud_safmr_features.csv")
HOT_PATH = os.path.join(ROOT, "external", "realtor_hot_features.csv")


def add_hud(d):
    """HUD Small Area Fair Market Rents (FY2026): ZIP-level rent levels by bedroom count. Requires add_irs first."""
    from common import _irs_frac_above
    d = d.merge(pd.read_csv(HUD_PATH), on="Zip", how="left")
    inc = d["AnnualMedianHouseholdIncome"].clip(lower=1000)
    for k in (1, 2, 3):
        d[f"hud_log_rti_{k}br"] = np.log(12 * d[f"hud_safmr_{k}br"] / inc)
    d["hud_vs_rentindex"] = np.log(d["RentIndex"] / d["hud_safmr_2br"])
    ok = (d["irs_share_b1"].notna() & d["hud_safmr_2br"].notna()).values
    for k in (1, 2):
        a = np.full(len(d), np.nan); a[ok] = _irs_frac_above(d[ok], 12 * d.loc[ok, f"hud_safmr_{k}br"].values / 0.3)
        d[f"hud_irs_rent_ok_{k}br"] = a
    new = [c for c in d.columns if c.startswith("hud_")]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


def add_hot(d):
    """Realtor.com market hotness (supply/demand scores, listing views relative to the US)."""
    d = d.merge(pd.read_csv(HOT_PATH), on=["Zip", "Month"], how="left")
    new = [c for c in d.columns if c.startswith("hot_")]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


ACSX_PATH = os.path.join(ROOT, "external", "acs_extra_zcta.csv")


def add_fill(d):
    """Best available home value without listing data: Zillow ZHVI, else ACS median value, else HUD rent x price/rent.
    Scales are medians over ZIPs that have both values (no target involved). Requires add_zillow and add_hud."""
    a = pd.read_csv(ACSX_PATH, usecols=["Zip", "acs_value"])
    d = d.merge(a.rename(columns={"acs_value": "fill_acs_value"}), on="Zip", how="left")
    inc = d["AnnualMedianHouseholdIncome"].clip(lower=1000)
    z = d["zhvi"]; av = d["fill_acs_value"]; hr = 12 * d["hud_safmr_2br"]
    s_acs = float((z / av).replace([np.inf, -np.inf], np.nan).median())
    s_hud = float((z / hr).replace([np.inf, -np.inf], np.nan).median())
    best = z.fillna(av * s_acs).fillna(hr * s_hud)
    r = 0.068 / 12; fac = r / (1 - (1 + r) ** -360)
    d["fill_best_log_pti"] = np.log(best / inc)
    d["fill_best_own_burden"] = 12 * best * (0.8 * fac + 0.015 / 12) / inc
    d["fill_src"] = np.select([z.notna(), av.notna()], [0.0, 1.0], 2.0)
    d["fill_acs_vs_zhvi"] = np.log(av * s_acs / d["zhvi_f"])
    d = d.drop(columns="fill_acs_value")
    new = ["fill_best_log_pti", "fill_best_own_burden", "fill_src", "fill_acs_vs_zhvi"]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


def qmap_table(prev_rows, cur_rows, cols):
    """Quantile mapping current month -> previous month for each drifting feature, fitted on the same (seen) ZIPs."""
    qs = np.linspace(0, 1, 201); tab = {}
    for c in cols:
        a, b = prev_rows[c].dropna().values, cur_rows[c].dropna().values
        if len(a) > 50 and len(b) > 50:
            tab[c] = (np.quantile(b, qs), np.quantile(a, qs))
    return tab


def qmap(rows, tab):
    rows = rows.copy()
    for c, (src, dst) in tab.items():
        v = rows[c].values.astype(float); ok = ~np.isnan(v)
        src_u, idx = np.unique(src, return_index=True)
        v[ok] = np.interp(v[ok], src_u, dst[idx]); rows[c] = v
    return rows


ACSM_PATH = os.path.join(ROOT, "external", "acs_more_zcta.csv")
FHFA_PATH = os.path.join(ROOT, "external", "fhfa_features.csv")


def add_acsm(d):
    """ACS 2020-24 cost-to-income distributions (B25070, B25091) and property taxes (B25103) by ZCTA."""
    a = pd.read_csv(ACSM_PATH).drop(columns="acsm_year")
    d = d.merge(a, on="Zip", how="left")
    inc = d["AnnualMedianHouseholdIncome"].clip(lower=1000)
    d["acsm_tax_to_inc"] = d["acsm_tax"] / inc
    if "acs_value" in d:
        d["acsm_tax_to_value"] = d["acsm_tax"] / d["acs_value"]
    o = d["OwnerPercent"] / 100            # re-weight ACS tenure shares with the competition's owner share
    d["acsm_mix_lt30"] = o * d["acsm_own_lt30"] + (1 - o) * d["acsm_rent_lt30"]
    new = [c for c in d.columns if c.startswith("acsm_")]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


def add_fhfa(d):
    """FHFA ZIP5 annual HPI: log growth over 1-20 years and since the median move-in year."""
    F = pd.read_csv(FHFA_PATH).set_index("Zip")
    yrs = sorted(int(c[6:]) for c in F.columns if c.startswith("fhfa_y")); last = yrs[-1]
    G = F[[f"fhfa_g{k}" for k in (1, 3, 5, 10, 20)]]
    d = d.merge(G.reset_index(), on="Zip", how="left")
    L = d["MedianHomeLengthofResidence"].fillna(9).values
    yr = np.clip(np.round(2026 - L), yrs[0], last).astype(int)
    M = F.reindex(d["Zip"].values)
    then = np.array([M[f"fhfa_y{y}"].values[i] for i, y in enumerate(yr)])
    d["fhfa_since_movein"] = M[f"fhfa_y{last}"].values - then
    new = list(G.columns) + ["fhfa_since_movein"]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d
