"""Shared utilities: data loading, feature engineering, leak-free target/neighbour features,
and the validation scheme that mimics the test set (known ZIPs + unseen ZIPs)."""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
SUB = os.path.join(ROOT, "submissions")
T = "AffordabilityPercentageTrue"
MAY, JUN, JUL = "5/1/2026", "6/1/2026", "7/1/2026"

RAW_NUM = ["SampleSize", "RentIndex", "AnnualMedianHouseholdIncome", "OwnerPercent", "RenterPercent",
           "MedianHomeLengthofResidence", "AvgNumberofChildren", "AvgNumberofAdults", "AvgHouseholdSize",
           "AvgGenerationsinHousehold", "YoungAdultinHouseholdPercent", "SeniorAdultinHouseholdPercent"]
CAT = ["State", "Metro", "City"]


def load():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    for d in (tr, te):
        d["Metro"] = d["Metro"].fillna("NA_METRO")
        d["City"] = d["City"].fillna("NA_CITY")
        d["CityKey"] = d["State"] + "|" + d["City"]
        d["zip3"] = d["Zip"] // 100
        d["zip2"] = d["Zip"] // 1000
    return tr, te, ss


def add_basic_features(d):
    """Deterministic, label-free engineered features."""
    d = d.copy()
    inc = d["AnnualMedianHouseholdIncome"]
    rent = d["RentIndex"]
    d["rent_missing"] = rent.isna().astype(int)
    d["log_income"] = np.log(inc)
    d["log_rent"] = np.log(rent.clip(lower=300, upper=20000))
    d["rent_burden"] = (rent.clip(upper=20000) * 12 / inc)          # annual rent / income
    d["log_rent_burden"] = np.log(d["rent_burden"])
    d["log_sample"] = np.log1p(d["SampleSize"])
    d["child_share"] = d["AvgNumberofChildren"] / d["AvgHouseholdSize"]
    d["adult_minus_child"] = d["AvgNumberofAdults"] - d["AvgNumberofChildren"]
    d["hh_per_gen"] = d["AvgHouseholdSize"] / d["AvgGenerationsinHousehold"]
    d["owner_x_child"] = d["OwnerPercent"] * d["AvgNumberofChildren"]
    d["senior_minus_young"] = d["SeniorAdultinHouseholdPercent"] - d["YoungAdultinHouseholdPercent"]
    d["owner_per_tenure"] = d["OwnerPercent"] / (d["MedianHomeLengthofResidence"] + 1)
    d["income_per_person"] = inc / d["AvgHouseholdSize"]
    return d


ENG_NUM = ["rent_missing", "log_income", "log_rent", "rent_burden", "log_rent_burden", "log_sample",
           "child_share", "adult_minus_child", "hh_per_gen", "owner_x_child", "senior_minus_young",
           "owner_per_tenure", "income_per_person"]


# ---------------------------------------------------------------------------------------------
# Target-derived features (computed only from a "labeled" reference set -> no leakage)
# ---------------------------------------------------------------------------------------------
def _smoothed_group_mean(lab, qry, key, y, prior, alpha):
    g = lab.groupby(key)[y].agg(["sum", "count"])
    s = qry[key].map(g["sum"]).fillna(0.0).values
    n = qry[key].map(g["count"]).fillna(0.0).values
    return (s + alpha * prior) / (n + alpha), n


def _zip_knn(lab_zip, lab_y, lab_feat, q_zip, ks=(1, 3, 5, 10, 20), exclude_same=True):
    """Nearest labeled ZIPs by numeric ZIP-code distance (a proxy for geographic proximity)."""
    order = np.argsort(lab_zip)
    lz, ly = lab_zip[order], lab_y[order]
    lf = lab_feat[order] if lab_feat is not None else None
    K = max(ks)
    W = K + 1
    pos = np.searchsorted(lz, q_zip)
    cand = pos[:, None] + np.arange(-W, W + 1)[None, :]
    valid = (cand >= 0) & (cand < len(lz))
    cand_c = np.clip(cand, 0, len(lz) - 1)
    dist = np.abs(lz[cand_c] - q_zip[:, None]).astype(float)
    if exclude_same:
        valid &= dist > 0
    dist[~valid] = np.inf
    srt = np.argsort(dist, axis=1)[:, :K]
    nd = np.take_along_axis(dist, srt, axis=1)
    ni = np.take_along_axis(cand_c, srt, axis=1)
    ny = ly[ni]
    out = {}
    for k in ks:
        out[f"knn{k}_mean"] = ny[:, :k].mean(1)
        if k >= 3:
            out[f"knn{k}_std"] = ny[:, :k].std(1)
            out[f"knn{k}_median"] = np.median(ny[:, :k], 1)
    w = 1.0 / (1.0 + nd[:, :10])
    out["knn10_wmean"] = (ny[:, :10] * w).sum(1) / w.sum(1)
    out["knn1_dist"] = np.log1p(nd[:, 0])
    out["knn5_dist"] = np.log1p(nd[:, 4])
    if lf is not None:  # neighbour averages of raw features -> "relative to neighbourhood" features
        nf = lf[ni[:, :10]]  # (n, 10, f)
        out["_nbr_feat_mean"] = np.nanmean(nf, axis=1)
    return out


NBR_FEATS = ["log_rent", "log_income", "AvgNumberofChildren", "OwnerPercent", "AvgHouseholdSize",
             "MedianHomeLengthofResidence"]


def target_features(lab, qry, y=T, exclude_same=True, use_knn=True, use_te=True):
    """lab: labeled reference rows (one row per ZIP). qry: rows to featurize."""
    prior = lab[y].mean()
    f = pd.DataFrame(index=qry.index)
    if use_te:
        for key, a in [("State", 5), ("Metro", 5), ("CityKey", 3), ("zip3", 3), ("zip2", 5)]:
            m, n = _smoothed_group_mean(lab, qry, key, y, prior, a)
            f[f"te_{key}"] = m
            f[f"cnt_{key}"] = n
    if use_knn:
        lf = lab[NBR_FEATS].values.astype(float)
        o = _zip_knn(lab["Zip"].values, lab[y].values.astype(float), lf, qry["Zip"].values,
                     exclude_same=exclude_same)
        nfm = o.pop("_nbr_feat_mean")
        for k, v in o.items():
            f[k] = v
        for i, c in enumerate(NBR_FEATS):
            f[f"rel_{c}"] = qry[c].values - nfm[:, i]
    if "latitude" in lab.columns and "latitude" in qry.columns:
        f = pd.concat([f, geo_target_features(lab, qry, y=y, exclude_same=exclude_same)], axis=1)
        f["lat"] = qry["latitude"].values; f["lon"] = qry["longitude"].values
    return f


def oof_target_features(lab_rows, n_splits=5, seed=0, y=T, **kw):
    """Target features for training rows computed out-of-fold (grouped by ZIP) so the model
    never sees a row's own label (or same-ZIP label) inside its features."""
    zips = lab_rows["Zip"].unique()
    kf = KFold(n_splits, shuffle=True, random_state=seed)
    parts = []
    for tr_i, va_i in kf.split(zips):
        trz, vaz = set(zips[tr_i]), set(zips[va_i])
        lab = lab_rows[lab_rows.Zip.isin(trz)]
        qry = lab_rows[lab_rows.Zip.isin(vaz)]
        parts.append(target_features(lab, qry, y=y, **kw))
    return pd.concat(parts).loc[lab_rows.index]


GEO_PATH = os.path.join(ROOT, "external", "us_zip_latlon.csv")
ZILLOW_PATH = os.path.join(ROOT, "external", "zillow_features.csv")
ZFEATS = []  # filled by add_zillow
SDOH_PATH = os.path.join(ROOT, "external", "cdc_sdoh_zcta.csv")  # EXTERNAL: CDC PLACES SDOH by ZCTA (ACS 2017-21)


def add_zillow(d):
    """EXTERNAL DATA (Zillow Research ZHVI/ZORI by ZIP, public; cited by the competition).
    Month-matched home values + structural housing-cost features."""
    from scipy.stats import norm
    z = pd.read_csv(ZILLOW_PATH)
    d = d.merge(z, on=["Zip", "Month"], how="left")
    inc = d["AnnualMedianHouseholdIncome"]
    V = d["zhvi"].fillna(d["zhvi_sfr"]).fillna(d["zhvi_low"]).fillna(d["zhvi_high"])
    d["zhvi_imputed"] = V.isna().astype(int)
    if V.isna().any() and "latitude" in d.columns:
        # geo-impute missing home values from the 8 nearest Zillow ZIPs (all 26k Zillow ZIPs, same month)
        from sklearn.neighbors import BallTree
        g = pd.read_csv(GEO_PATH, dtype={"postal_code": str})
        g = g[g.postal_code.str.isdigit()].drop_duplicates("postal_code"); g["Zip"] = g.postal_code.astype(int)
        for mon in d.loc[V.isna(), "Month"].unique():
            zm = z[(z.Month == mon) & z.zhvi.notna()].merge(g[["Zip", "latitude", "longitude"]], on="Zip")
            tree = BallTree(np.radians(zm[["latitude", "longitude"]].values), metric="haversine")
            idx = d.index[V.isna() & (d.Month == mon)]
            dist, ind = tree.query(np.radians(d.loc[idx, ["latitude", "longitude"]].values), k=8)
            w = 1 / (dist * 6371 + 1.0)
            V.loc[idx] = np.exp((np.log(zm.zhvi.values[ind]) * w).sum(1) / w.sum(1))
    d["zhvi_f"] = V
    d["log_pti"] = np.log(V / inc)
    for t in ("zhvi_low", "zhvi_high", "zhvi_sfr", "zhvi_condo", "zhvi_b2", "zhvi_b3", "zhvi_b4"):
        d[f"log_pti_{t[5:]}"] = np.log(d[t] / inc)
    d["log_pti_raw"] = np.log(d["zhvi_raw"].fillna(V) / inc)
    d["zhvi_raw_g1"] = np.log(d["zhvi_raw"] / d["zhvi_raw_lag1"])
    d["zhvi_raw_g12"] = np.log(d["zhvi_raw"] / d["zhvi_raw_lag12"])
    d["zhvi_tier_spread"] = np.log(d["zhvi_high"] / d["zhvi_low"])
    d["zhvi_g12"] = np.log(d["zhvi"] / d["zhvi_lag12"])
    d["zhvi_g3"] = np.log(d["zhvi"] / d["zhvi_lag3"])
    d["zhvi_g1"] = np.log(d["zhvi"] / d["zhvi_lag1"])
    r = 0.068 / 12
    pi_fac = 0.8 * r / (1 - (1 + r) ** -360)           # P&I per $ of value, 80% LTV, 30y @6.8%
    own_cost = V * (pi_fac + 0.015 / 12)                 # + ~1.5%/yr tax & insurance
    rent = d["RentIndex"].fillna(d["zori_z"])
    d["own_burden"] = own_cost * 12 / inc
    d["rent_burden2"] = rent * 12 / inc
    o = d["OwnerPercent"] / 100
    d["mix_burden"] = o * d["own_burden"] + (1 - o) * d["rent_burden2"].fillna(d["own_burden"] * 0.7)
    d["price_to_rent"] = np.log(V / (rent * 12))
    for s in (0.6, 0.9):  # lognormal-income share of households paying <=30%
        p_own = norm.cdf((np.log(inc) - np.log(own_cost * 12 / 0.3)) / s)
        p_rent = norm.cdf((np.log(inc) - np.log(rent * 12 / 0.3)) / s)
        d[f"struct_s{s}"] = 100 * (o * p_own + (1 - o) * np.where(np.isnan(p_rent), p_own, p_rent))
    sd = pd.read_csv(SDOH_PATH)
    d = d.merge(sd, on="Zip", how="left")
    new = list(sd.columns[1:]) + ["log_pti_raw", "zhvi_raw_g1", "zhvi_raw_g12", "zhvi_f", "zhvi_imputed", "log_pti"] + [f"log_pti_{t[5:]}" for t in ("zhvi_low", "zhvi_high", "zhvi_sfr", "zhvi_condo", "zhvi_b2", "zhvi_b3", "zhvi_b4")] + \
          ["zhvi_tier_spread", "zhvi_g12", "zhvi_g3", "zhvi_g1", "own_burden", "rent_burden2", "mix_burden", "price_to_rent",
           "struct_s0.6", "struct_s0.9"]
    ZFEATS[:] = new
    return d


def add_geo(d):
    """EXTERNAL DATA (GeoNames via pgeocode, CC-BY): ZIP centroid lat/lon + county."""
    g = pd.read_csv(GEO_PATH, dtype={"postal_code": str})
    g = g[g.postal_code.str.isdigit()].drop_duplicates("postal_code")
    g["Zip"] = g.postal_code.astype(int)
    g["CountyKey"] = g.state_code.fillna("") + "|" + g.county_name.fillna("")
    d = d.merge(g[["Zip", "latitude", "longitude", "CountyKey"]], on="Zip", how="left")
    # fallback for unmatched ZIPs: nearest ZIP code's coordinates
    miss = d.latitude.isna()
    if miss.any():
        gz = g.sort_values("Zip"); zz = gz.Zip.values
        pos = np.clip(np.searchsorted(zz, d.loc[miss, "Zip"].values), 0, len(zz) - 1)
        d.loc[miss, "latitude"] = gz.latitude.values[pos]
        d.loc[miss, "longitude"] = gz.longitude.values[pos]
        d.loc[miss, "CountyKey"] = gz.CountyKey.values[pos]
    return d


def geo_target_features(lab, qry, y=T, ks=(3, 5, 10, 20), exclude_same=True):
    from sklearn.neighbors import BallTree
    R = 6371.0
    L = np.radians(lab[["latitude", "longitude"]].values)
    Q = np.radians(qry[["latitude", "longitude"]].values)
    tree = BallTree(L, metric="haversine")
    K = max(ks) + 1
    dist, ind = tree.query(Q, k=K)
    dist *= R
    ly = lab[y].values.astype(float); lz = lab["Zip"].values
    same = lz[ind] == qry["Zip"].values[:, None]
    if exclude_same:
        dist = np.where(same, np.inf, dist)
    o = np.argsort(dist, 1)[:, :max(ks)]
    dist = np.take_along_axis(dist, o, 1); ind = np.take_along_axis(ind, o, 1)
    ny = ly[ind]
    f = pd.DataFrame(index=qry.index)
    for k in ks:
        f[f"geo{k}_mean"] = ny[:, :k].mean(1)
        f[f"geo{k}_std"] = ny[:, :k].std(1)
    w = 1 / (dist[:, :10] + 1.0)
    f["geo10_idw"] = (ny[:, :10] * w).sum(1) / w.sum(1)
    f["geo1_dist"] = np.log1p(dist[:, 0]); f["geo5_dist"] = np.log1p(dist[:, 4])
    for r in (5, 15, 40):
        msk = dist <= r
        n = msk.sum(1)
        f[f"geo_r{r}_mean"] = np.where(n > 0, (ny * msk).sum(1) / np.maximum(n, 1), np.nan)
        f[f"geo_r{r}_n"] = n
    prior = lab[y].mean()
    m, n = _smoothed_group_mean(lab, qry, "CountyKey", y, prior, 3)
    f["te_County"] = m; f["cnt_County"] = n
    if os.environ.get("EXTRA_GEO") == "1":
        # neighbours that are geographically close AND demographically similar
        sim_cols = ["log_rent", "log_income", "AvgNumberofChildren", "OwnerPercent", "AvgHouseholdSize", "rent_burden"]
        LF = lab[sim_cols].astype(float); mu, sd = LF.mean(), LF.std()
        lz_ = ((LF - mu) / sd).fillna(0).values
        qz_ = ((qry[sim_cols].astype(float) - mu) / sd).fillna(0).values
        dz2 = ((lz_[ind] - qz_[:, None, :]) ** 2).sum(2)          # (n, K)
        for bw in (0.5, 1.0, 2.0):
            w = np.exp(-dist / 15.0) * np.exp(-dz2 / (2 * bw ** 2))
            w = np.where(np.isfinite(dist), w, 0)
            f[f"geo_sim{bw}_mean"] = (ny * w).sum(1) / np.maximum(w.sum(1), 1e-9)
            f[f"geo_sim{bw}_wsum"] = w.sum(1)
        w = 1 / (dist + 1.0); w = np.where(np.isfinite(dist), w, 0)
        f["geo20_idw"] = (ny * w).sum(1) / w.sum(1)
        f["geo5_idw"] = (ny[:, :5] * w[:, :5]).sum(1) / w[:, :5].sum(1)
        # neighbours' rent burden vs own: affordability gradient proxy
        nb_rb = lab["rent_burden"].values[ind]
        f["rel_rent_burden_geo"] = qry["rent_burden"].values - np.nanmean(nb_rb[:, :10], 1)
    if "log_pti" in lab.columns and "log_pti" in qry.columns:
        f = pd.concat([f, gwr_features(lab, qry, y, exclude_same)], axis=1)
    return f


def gwr_features(lab, qry, y=T, exclude_same=True, K=40, bw_km=(10.0, 30.0)):
    """Geographically weighted (local) ridge regression of y on [1, log_pti, log rent burden, owner share],
    fitted on the K nearest labelled ZIPs of each query and evaluated at the query's own covariates."""
    from sklearn.neighbors import BallTree
    cols = ["log_pti", "log_rent_burden", "OwnerPercent"]
    LX = lab[cols].astype(float); mu = LX.mean(); sd = LX.std()
    LX = ((LX - mu) / sd).fillna(0).values; QX = ((qry[cols].astype(float) - mu) / sd).fillna(0).values
    tree = BallTree(np.radians(lab[["latitude", "longitude"]].values), metric="haversine")
    dist, ind = tree.query(np.radians(qry[["latitude", "longitude"]].values), k=K + 1)
    dist = dist * 6371.0
    if exclude_same:
        same = lab["Zip"].values[ind] == qry["Zip"].values[:, None]
        dist = np.where(same, np.inf, dist)
    ly = lab[y].values.astype(float)
    Xn = np.concatenate([np.ones((*ind.shape, 1)), LX[ind]], axis=2)        # (n, K+1, p)
    xq = np.concatenate([np.ones((len(QX), 1)), QX], axis=1)                 # (n, p)
    yn = ly[ind]
    out = pd.DataFrame(index=qry.index)
    lam = np.diag([1e-3, 1.0, 1.0, 1.0])  # ridge on slopes (shrinks towards local mean)
    for bw in bw_km:
        w = np.where(np.isfinite(dist), np.exp(-(dist / bw) ** 2), 0.0) + 1e-6 * np.isfinite(dist)
        XtW = Xn.transpose(0, 2, 1) * w[:, None, :]
        A = XtW @ Xn + lam[None]
        b = (XtW @ yn[:, :, None])[:, :, 0]
        beta = np.linalg.solve(A, b[..., None])[..., 0]
        out[f"gwr{int(bw)}_pred"] = (beta * xq).sum(1)
        out[f"gwr{int(bw)}_slope_pti"] = beta[:, 1]
        out[f"gwr{int(bw)}_neff"] = w.sum(1) ** 2 / (w ** 2).sum(1)
    return out



IRS_PATH = os.path.join(ROOT, "external", "irs_features.csv")


def _irs_frac_above(F, t):
    """Share of tax returns with AGI >= t, from the 6-bracket IRS distribution (log-linear within brackets,
    Pareto above $200k with alpha from the bracket mean)."""
    edges = np.array([1.0, 25e3, 50e3, 75e3, 100e3, 200e3])
    S = F[[f"irs_share_b{k}" for k in range(1, 7)]].values
    t = np.asarray(t, float)
    above = np.zeros(len(t))
    for k in range(6):
        lo = edges[k]
        if k < 5:
            hi = edges[k + 1]
            frac_in = np.clip((np.log(hi) - np.log(np.clip(t, lo, hi))) / (np.log(hi) - np.log(lo)), 0, 1)
        else:
            m = F["irs_mean_agi_top"].values
            alpha = np.clip(m / np.maximum(m - lo, 1.0), 1.05, 6.0)
            frac_in = np.where(t <= lo, 1.0, (lo / np.maximum(t, lo)) ** alpha)
        above += S[:, k] * frac_in
    return above


def add_irs(d):
    """EXTERNAL DATA: IRS SOI ZIP income distribution + structural share-above-threshold features.
    Requires add_zillow first (uses zhvi_f, RentIndex/zori)."""
    F = pd.read_csv(IRS_PATH)
    d = d.merge(F, on="Zip", how="left")
    r = 0.068 / 12; pi_fac = r / (1 - (1 + r) ** -360)
    V = d["zhvi_raw"].fillna(d["zhvi_f"]) if "zhvi_raw" in d else d["zhvi_f"]
    rent = d["RentIndex"].fillna(d["zori_z"])
    o = d["OwnerPercent"].values / 100
    new = [c for c in F.columns if c != "Zip"]
    ok = d["irs_share_b1"].notna().values
    for ltv in (0.8, 0.95):
        own_annual = 12 * V * (ltv * pi_fac + 0.015 / 12)
        for mult in (1.0, 1.3):
            a_own = np.full(len(d), np.nan); a_own[ok] = _irs_frac_above(d[ok], (own_annual[ok] * mult / 0.3))
            d[f"irs_own_ok_l{ltv}_m{mult}"] = a_own; new.append(f"irs_own_ok_l{ltv}_m{mult}")
    for mult in (1.0, 1.3):
        a_rent = np.full(len(d), np.nan); a_rent[ok] = _irs_frac_above(d[ok], (12 * rent[ok] * mult / 0.3).fillna(1e9))
        a_rent[rent.isna().values] = np.nan
        d[f"irs_rent_ok_m{mult}"] = a_rent; new.append(f"irs_rent_ok_m{mult}")
    d["irs_mix_ok"] = o * d["irs_own_ok_l0.8_m1.0"] + (1 - o) * d["irs_rent_ok_m1.0"].fillna(d["irs_own_ok_l0.8_m1.0"])
    d["irs_median_ratio"] = d["AnnualMedianHouseholdIncome"] / d["irs_mean_agi"]
    new += ["irs_mix_ok", "irs_median_ratio"]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


ACS_PATH = os.path.join(ROOT, "external", "acs_b25106_zcta.csv")


def add_acs(d):
    """EXTERNAL DATA: ACS 5-year B25106 by ZCTA (src/download_census.py): share of NOT cost-burdened households."""
    a = pd.read_csv(ACS_PATH).drop(columns=["acs_year"])
    d = d.merge(a, on="Zip", how="left")
    o = d["OwnerPercent"] / 100
    d["acs_mix_ok"] = o * d["acs_owner_ok"] + (1 - o) * d["acs_renter_ok"]
    new = [c for c in a.columns if c != "Zip"] + ["acs_mix_ok"]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


ACSX_PATH = os.path.join(ROOT, "external", "acs_extra_zcta.csv")
_ACS_EDGES = np.array([1, 10e3, 15e3, 20e3, 25e3, 30e3, 35e3, 40e3, 45e3, 50e3, 60e3, 75e3, 100e3, 125e3, 150e3, 200e3])


def _acs_frac_above(S, t):
    """Share of households with income >= t from the 16 ACS B19001 brackets (log-linear within, Pareto a=2 on top)."""
    t = np.asarray(t, float); above = np.zeros(len(t))
    for k in range(16):
        lo = _ACS_EDGES[k]
        if k < 15:
            hi = _ACS_EDGES[k + 1]
            fr = np.clip((np.log(hi) - np.log(np.clip(t, lo, hi))) / (np.log(hi) - np.log(lo)), 0, 1)
        else:
            fr = np.where(t <= lo, 1.0, (lo / np.maximum(t, lo)) ** 2.0)
        above += S[:, k] * fr
    return above


def add_acs_extra(d):
    """EXTERNAL DATA: ACS 5-yr tenure-specific incomes/costs + income distribution (src/download_census_extra.py).
    Requires add_zillow first."""
    a = pd.read_csv(ACSX_PATH)
    d = d.merge(a, on="Zip", how="left")
    r = 0.068 / 12; pi_fac = r / (1 - (1 + r) ** -360)
    V = d["zhvi_raw"].fillna(d["zhvi_f"]) if "zhvi_raw" in d else d["zhvi_f"]
    rent = d["RentIndex"].fillna(d["zori_z"]).fillna(d["acs_rent"])
    own_annual = 12 * V * (0.8 * pi_fac + 0.015 / 12)
    o = d["OwnerPercent"] / 100
    d["acsx_rent_burden_renterinc"] = 12 * rent / d["acs_inc_renter"]
    d["acsx_own_burden_ownerinc"] = own_annual / d["acs_inc_owner"]
    d["acsx_log_pti_ownerinc"] = np.log(V / d["acs_inc_owner"])
    d["acsx_renter_inc_ratio"] = d["acs_inc_renter"] / d["AnnualMedianHouseholdIncome"]
    d["acsx_owner_inc_ratio"] = d["acs_inc_owner"] / d["AnnualMedianHouseholdIncome"]
    d["acsx_zhvi_vs_acsvalue"] = np.log(V / d["acs_value"])
    # mortgage-free owners only pay tax+insurance; owners with a mortgage pay P&I too
    m = d["acs_mortgage_share"].fillna(0.6)
    own_cost_mix = 12 * V * (m * 0.8 * pi_fac + 0.015 / 12)
    d["acsx_own_burden_mortmix"] = own_cost_mix / d["acs_inc_owner"]
    S = d[[f"acs_incb{i:02d}" for i in range(2, 18)]].values
    ok = ~np.isnan(S).any(1)
    new = [c for c in a.columns if c != "Zip"] + ["acsx_rent_burden_renterinc", "acsx_own_burden_ownerinc", "acsx_log_pti_ownerinc",
                                                  "acsx_renter_inc_ratio", "acsx_owner_inc_ratio", "acsx_zhvi_vs_acsvalue",
                                                  "acsx_own_burden_mortmix"]
    for name, cost in [("own", own_annual), ("own_mortmix", own_cost_mix), ("rent", 12 * rent)]:
        for mult in (1.0, 1.4):
            v = np.full(len(d), np.nan)
            okk = ok & cost.notna().values
            v[okk] = _acs_frac_above(S[okk], cost.values[okk] * mult / 0.3)
            d[f"acsx_ok_{name}_m{mult}"] = 100 * v; new.append(f"acsx_ok_{name}_m{mult}")
    d["acsx_ok_mix"] = o * d["acsx_ok_own_mortmix_m1.0"] + (1 - o) * d["acsx_ok_rent_m1.0"].fillna(d["acsx_ok_own_mortmix_m1.0"])
    new.append("acsx_ok_mix")
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


# Freddie Mac PMMS 30-year fixed, annual average (%), public; 2025-2026 approximate.
PMMS = {2000: 8.05, 2001: 6.97, 2002: 6.54, 2003: 5.83, 2004: 5.84, 2005: 5.87, 2006: 6.41, 2007: 6.34, 2008: 6.03,
        2009: 5.04, 2010: 4.69, 2011: 4.45, 2012: 3.66, 2013: 3.98, 2014: 4.17, 2015: 3.85, 2016: 3.65, 2017: 3.99,
        2018: 4.54, 2019: 3.94, 2020: 3.11, 2021: 2.96, 2022: 5.34, 2023: 6.81, 2024: 6.72, 2025: 6.6, 2026: 6.3}
HIST_PATH = os.path.join(ROOT, "external", "zhvi_annual.csv")


def _pmt(rate_pct, years=30):
    r = np.asarray(rate_pct, float) / 1200
    return r / (1 - (1 + r) ** (-12 * years))


def add_hist(d):
    """Owner cost at PURCHASE time: ZHVI of the ZIP in the year households moved in (2026 - median length of
    residence) and the mortgage rate of that year. Uses external/zhvi_annual.csv (June value per year)."""
    h = pd.read_csv(HIST_PATH).set_index("Zip")
    inc = d["AnnualMedianHouseholdIncome"].values
    L = d["MedianHomeLengthofResidence"].fillna(9).values
    V_now = (d["zhvi_raw"].fillna(d["zhvi_f"]) if "zhvi_raw" in d else d["zhvi_f"]).values
    new = []
    for name, mult in (("L", 1.0), ("halfL", 0.5), ("1p5L", 1.5)):
        yr = np.clip(np.round(2026 - L * mult), 2000, 2026).astype(int)
        Vb = np.array([h.at[z, str(y)] if (z in h.index and str(y) in h.columns) else np.nan for z, y in zip(d["Zip"].values, yr)])
        Vb = np.where(np.isnan(Vb), V_now, Vb)
        rate = np.array([PMMS[y] for y in yr])
        pay = 12 * 0.8 * Vb * _pmt(rate)
        tax = 0.015 * V_now                                   # taxes/insurance follow today's value
        paid_off = (L * mult) >= 30
        d[f"hist_own_burden_{name}"] = (np.where(paid_off, 0, pay) + tax) / inc
        d[f"hist_log_price_then_{name}"] = np.log(Vb / inc)
        d[f"hist_rate_{name}"] = rate
        new += [f"hist_own_burden_{name}", f"hist_log_price_then_{name}", f"hist_rate_{name}"]
    d["hist_appreciation_L"] = np.log(V_now / np.exp(d["hist_log_price_then_L"].values) / inc)
    d["hist_burden_now_vs_then"] = d["own_burden"] / d["hist_own_burden_L"] if "own_burden" in d else np.nan
    new += ["hist_appreciation_L", "hist_burden_now_vs_then"]
    ZFEATS.extend([c for c in new if c not in ZFEATS])
    return d


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    mae = np.abs(y - p).mean()
    rmse = np.sqrt(((y - p) ** 2).mean())
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def log_result(rows, path=os.path.join(OUT, "experiments.csv")):
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
