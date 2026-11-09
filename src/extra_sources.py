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
