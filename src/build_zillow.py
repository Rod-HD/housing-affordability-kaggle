"""EXTERNAL DATA (Zillow Research public CSVs, cited by the competition itself): build a compact ZIP x month table.
For each competition month m (May/Jun/Jul 2026) we take the Zillow value at the end of that month, plus lags
(1, 3, 12 months) for the mid-tier ZHVI. Output: external/zillow_features.csv (Zip, Month, features...)."""
import os, glob
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZD = os.path.join(ROOT, "external", "zillow")
SERIES = {
    "zhvi": "zhvi_zip.csv",
    "zhvi_raw": "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv",
    "zhvi_low": "Zip_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv",
    "zhvi_high": "Zip_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv",
    "zhvi_sfr": "Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv",
    "zhvi_condo": "Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv",
    **{f"zhvi_b{k}": f"Zip_zhvi_bdrmcnt_{k}_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv" for k in range(1, 6)},
    "zori_z": "Zip_zori_uc_sfrcondomfr_sm_month.csv",
}
MONTHS = {"5/1/2026": "2026-05", "6/1/2026": "2026-06", "7/1/2026": "2026-07"}


def month_col(cols, ym):
    c = [x for x in cols if x.startswith(ym)]
    return c[0] if c else None


def shift_ym(ym, k):
    y, m = map(int, ym.split("-")); t = y * 12 + (m - 1) - k
    return f"{t // 12}-{t % 12 + 1:02d}"


frames = []
for name, fn in SERIES.items():
    hdr = pd.read_csv(os.path.join(ZD, fn), nrows=0).columns
    need = {"RegionName"}
    lags = (0, 1, 3, 12) if name in ("zhvi", "zhvi_raw") else (0,)
    for ym in MONTHS.values():
        for k in lags:
            c = month_col(hdr, shift_ym(ym, k))
            if c: need.add(c)
    d = pd.read_csv(os.path.join(ZD, fn), usecols=list(need))
    d["Zip"] = d.RegionName.astype(int)
    per_month = []
    for mon, ym in MONTHS.items():
        out = pd.DataFrame({"Zip": d.Zip, "Month": mon})
        for k in lags:
            c = month_col(hdr, shift_ym(ym, k))
            out[name if k == 0 else f"{name}_lag{k}"] = d[c].values if c else np.nan
        per_month.append(out)
    frames.append(pd.concat(per_month).drop_duplicates(["Zip", "Month"]).set_index(["Zip", "Month"]))
    print(name, "rows", len(d))
Z = pd.concat(frames, axis=1)
Z.reset_index().to_csv(os.path.join(ROOT, "external", "zillow_features.csv"), index=False)
print(Z.shape, Z.notna().mean().round(3).to_dict())
