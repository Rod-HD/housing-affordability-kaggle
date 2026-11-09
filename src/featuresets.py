"""Named feature sets (each is a function rows -> rows with external features added). Shared by the fold-cache
builder and the submission builder so CV and submission always use identical preprocessing.
v* sets with 5/7/8 use Realtor.com data; w* sets use only government / Zillow / GeoNames sources."""
from common import add_geo, add_zillow, add_irs, add_hist, add_acs, add_acs_extra
from extra_sources import add_realtor, add_hud, add_hot, add_fill, add_acsm, add_fhfa

V3 = lambda d: add_irs(add_zillow(add_geo(d)))
V4 = lambda d: add_hist(V3(d))
W3 = lambda d: add_fill(add_acs_extra(add_acs(add_hud(V4(d)))))
FEATURE_SETS = {
    "v3": V3, "v4": V4,
    "v5": lambda d: add_realtor(V4(d)),
    "v6": lambda d: add_hud(add_realtor(V4(d))),
    "v7": lambda d: add_hot(add_realtor(V4(d))),
    "v8": lambda d: add_hot(add_hud(add_realtor(V4(d)))),
    "w1": lambda d: add_hud(V4(d)),                       # + HUD SAFMR
    "w2": lambda d: add_acs_extra(add_acs(V4(d))),        # + ACS cost burden, tenure incomes, income brackets
    "w3": W3,                                             # + HUD + ACS + best-available home value
    "w4": lambda d: add_acsm(W3(d)),                      # + ACS cost-to-income distributions, property tax
    "w5": lambda d: add_fhfa(W3(d)),                      # + FHFA ZIP5 house price index
    "w6": lambda d: add_fhfa(add_acsm(W3(d))),
}
