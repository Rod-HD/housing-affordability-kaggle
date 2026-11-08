"""Download every public external dataset used by the pipeline into ./external, then build the compact tables.
Sources: GeoNames (via pgeocode), Zillow Research public CSVs, IRS SOI ZIP data, CDC PLACES SDOH by ZCTA.
Run:  .venv\\Scripts\\python.exe src\\download_external.py"""
import io, os, subprocess, sys, urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "external")
ZURL = "https://files.zillowstatic.com/research/public_csvs"
ZILLOW = {
    "zhvi_zip.csv": "zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv": "zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv",
    "Zip_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv": "zhvi/Zip_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv",
    "Zip_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv": "zhvi/Zip_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv",
    "Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv": "zhvi/Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv",
    "Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv": "zhvi/Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv",
    **{f"Zip_zhvi_bdrmcnt_{k}_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv":
       f"zhvi/Zip_zhvi_bdrmcnt_{k}_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv" for k in range(1, 6)},
    "Zip_zori_uc_sfrcondomfr_sm_month.csv": "zori/Zip_zori_uc_sfrcondomfr_sm_month.csv",
}
IRS = {f"{y}zpallagi.csv": f"https://www.irs.gov/pub/irs-soi/{y}zpallagi.csv" for y in ("22", "21")}
# data.cdc.gov may refuse some regions; the r.jina.ai reader is used as a fallback relay.
CDC = "https://data.cdc.gov/resource/bumh-rgsq.csv?$select=locationid,measureid,data_value,moe&$limit=50000&$order=locationid,measureid&$offset={}"


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return print("skip (exists)", os.path.basename(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
        f.write(r.read())
    print("ok", os.path.basename(path), os.path.getsize(path))


def geonames():
    os.environ["PGEOCODE_DATA_DIR"] = os.path.join(EXT, "pgeocode")
    import pgeocode
    d = pgeocode.Nominatim("us")._data
    d[["postal_code", "latitude", "longitude", "state_code", "county_name", "place_name"]].to_csv(
        os.path.join(EXT, "us_zip_latlon.csv"), index=False)
    print("ok us_zip_latlon.csv")


def cdc():
    out = os.path.join(EXT, "cdc_sdoh_zcta.csv")
    if os.path.exists(out):
        return print("skip (exists) cdc_sdoh_zcta.csv")
    parts = []
    for off in range(0, 350000, 50000):
        url = CDC.format(off)
        try:
            txt = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read().decode()
        except Exception:
            txt = urllib.request.urlopen("https://r.jina.ai/" + url, timeout=300).read().decode()
        i = txt.find('"locationid"')
        if i < 0:
            break
        df = pd.read_csv(io.StringIO(txt[i:]), dtype={"locationid": str})
        if df.empty:
            break
        parts.append(df)
    d = pd.concat(parts).drop_duplicates(["locationid", "measureid"])
    w = d.pivot(index="locationid", columns="measureid", values="data_value"); w.columns = ["sdoh_" + c for c in w.columns]
    w["sdoh_HCOST_moe"] = d.pivot(index="locationid", columns="measureid", values="moe")["HCOST"]
    w.index = w.index.astype(int); w.index.name = "Zip"; w.to_csv(out)
    print("ok cdc_sdoh_zcta.csv", w.shape)


if __name__ == "__main__":
    geonames()
    for fn, rel in ZILLOW.items():
        fetch(f"{ZURL}/{rel}", os.path.join(EXT, "zillow", fn))
    for fn, url in IRS.items():
        fetch(url, os.path.join(EXT, "irs", fn))
    cdc()
    py = sys.executable
    subprocess.run([py, os.path.join(ROOT, "src", "build_zillow.py")], check=True)
    subprocess.run([py, os.path.join(ROOT, "src", "build_irs.py")], check=True)
