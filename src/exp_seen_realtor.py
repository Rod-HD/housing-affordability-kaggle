"""Seen ZIPs: does the ZIP's own Realtor.com listing-price change add to the ZHVI-change adjustment used since sub07?
Honest protocol as exp_seen_z.py: fit on the May->June transition with ZIP-grouped CV, own-ZIP features only."""
import warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from common import *

warnings.filterwarnings("ignore")
tr, te, ss = load()
z = pd.read_csv(ZILLOW_PATH); r = pd.read_csv(f"{ROOT}/external/realtor_features.csv")
d = tr.merge(z, on=["Zip", "Month"], how="left").merge(r, on=["Zip", "Month"], how="left")
m = d[d.Month == MAY].set_index("Zip").sort_index(); j = d[d.Month == JUN].set_index("Zip").loc[m.index]
y0, y = m[T].values.astype(float), j[T].values.astype(float); s = y0 * (100 - y0) / 100
X = pd.DataFrame({"zhvi": np.log(j.zhvi_raw / m.zhvi_raw).values * s,
                  "rtr_price": np.log(j.rtr_price / m.rtr_price).values * s,
                  "rtr_ppsf": np.log(j.rtr_ppsf / m.rtr_ppsf).values * s,
                  "rtr_12m": np.log(j.rtr_price_12m / m.rtr_price_12m).values * s}, index=m.index)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-50, 50)
print("persistence RMSE %.4f" % np.sqrt(((y - y0) ** 2).mean()))
for cols in (["zhvi"], ["zhvi", "rtr_12m"], ["zhvi", "rtr_price"], ["zhvi", "rtr_ppsf"], ["zhvi", "rtr_price", "rtr_ppsf", "rtr_12m"]):
    sc = []
    for rs in range(5):
        p = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=rs).split(X):
            mdl = LinearRegression(fit_intercept=False).fit(X.iloc[a][cols], (y - y0)[a]); p[b] = y0[b] + mdl.predict(X.iloc[b][cols])
        sc.append(np.sqrt(((y - p) ** 2).mean()))
    coef = LinearRegression(fit_intercept=False).fit(X[cols], y - y0).coef_
    print(f"{'+'.join(cols):35s} CV RMSE {np.mean(sc):.4f}  coef {np.round(coef, 3)}")
