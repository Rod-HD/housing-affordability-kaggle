"""EXTERNAL DATA: Realtor.com inventory metrics by ZIP (public research CSV). For each competition month, keep that month's
values plus a trailing 12-month median (months after the target month are never used).
Output: external/realtor_features.csv (Zip, Month, rtr_*)."""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONTHS = {"5/1/2026": 202605, "6/1/2026": 202606, "7/1/2026": 202607}
COLS = {"median_listing_price": "price", "median_listing_price_per_square_foot": "ppsf", "median_square_feet": "sqft",
        "active_listing_count": "active", "new_listing_count": "new", "median_days_on_market": "dom",
        "price_reduced_share": "reduced", "pending_ratio": "pending", "median_listing_price_yy": "price_yy"}
d = pd.read_csv(os.path.join(ROOT, "external", "realtor", "zip_history.csv"), usecols=["month_date_yyyymm", "postal_code", *COLS],
                dtype={"postal_code": str})
d = d[d.month_date_yyyymm.between(202506, 202607) & d.postal_code.fillna("").str.fullmatch(r"\d{1,5}")].rename(columns=COLS)
d["Zip"] = d.postal_code.astype(int)
d = d.drop_duplicates(["Zip", "month_date_yyyymm"])
frames = []
for mon, ym in MONTHS.items():
    start = ym - 100 + 1 if ym % 100 < 12 else ym - 11
    w = d[d.month_date_yyyymm.between(start, ym)]
    a = w.groupby("Zip")[["price", "ppsf", "sqft", "active"]].median().add_suffix("_12m")
    a["n_12m"] = w.groupby("Zip").price.count()
    cur = d[d.month_date_yyyymm == ym].set_index("Zip")[list(COLS.values())]
    f = a.join(cur, how="left").add_prefix("rtr_").reset_index(); f["Month"] = mon
    frames.append(f)
R = pd.concat(frames, ignore_index=True)
R.to_csv(os.path.join(ROOT, "external", "realtor_features.csv"), index=False)
print(R.shape, R.groupby("Month").rtr_price_12m.count().to_dict())
