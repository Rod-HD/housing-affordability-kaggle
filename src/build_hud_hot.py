"""EXTERNAL DATA: (1) HUD Small Area Fair Market Rents FY2026 (revised) by ZIP; (2) Realtor.com market hotness by ZIP.
Hotness: value in the target month plus trailing 12-month median (nothing after the target month).
Outputs: external/hud_safmr_features.csv (Zip, hud_*), external/realtor_hot_features.csv (Zip, Month, hot_*)."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
h = pd.read_excel(os.path.join(ROOT, "external", "hud", "fy2026_safmrs_revised.xlsx"))
h.columns = [" ".join(str(c).split()) for c in h.columns]
keep = {f"SAFMR {k}BR": f"hud_safmr_{k}br" for k in range(5)}
h = h[["ZIP Code", *keep]].rename(columns={"ZIP Code": "Zip", **keep})
h = h.groupby("Zip", as_index=False).mean()          # ZIPs split across HUD areas: average
h.to_csv(os.path.join(ROOT, "external", "hud_safmr_features.csv"), index=False)
print("hud", h.shape)

MONTHS = {"5/1/2026": 202605, "6/1/2026": 202606, "7/1/2026": 202607}
COLS = {"hotness_score": "score", "supply_score": "supply", "demand_score": "demand",
        "page_view_count_per_property_vs_us": "views_vs_us", "median_dom_vs_us": "dom_vs_us", "median_listing_price_vs_us": "price_vs_us"}
d = pd.read_csv(os.path.join(ROOT, "external", "realtor_hot", "zip_hotness_history.csv"), usecols=["month_date_yyyymm", "postal_code", *COLS],
                dtype={"postal_code": str})
d = d[d.month_date_yyyymm.between(202506, 202607) & d.postal_code.fillna("").str.fullmatch(r"\d{1,5}")].rename(columns=COLS)
d["Zip"] = d.postal_code.astype(int); d = d.drop_duplicates(["Zip", "month_date_yyyymm"])
frames = []
for mon, ym in MONTHS.items():
    w = d[d.month_date_yyyymm.between(ym - 100 + 1, ym)]
    a = w.groupby("Zip")[["score", "supply", "demand", "views_vs_us"]].median().add_suffix("_12m")
    cur = d[d.month_date_yyyymm == ym].set_index("Zip")[list(COLS.values())]
    f = a.join(cur, how="left").add_prefix("hot_").reset_index(); f["Month"] = mon; frames.append(f)
R = pd.concat(frames, ignore_index=True)
R.to_csv(os.path.join(ROOT, "external", "realtor_hot_features.csv"), index=False)
print("hot", R.shape, R.groupby("Month").hot_score_12m.count().to_dict())
