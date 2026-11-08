import numpy as np, pandas as pd, itertools
from scipy.optimize import minimize
from common import *
o=pd.read_csv(f"{OUT}/newzip_oof.csv"); y=o.y.values
cols=[c for c in o.columns if '|' in c]
P={c.split('|')[0]:o[c].values for c in cols}
good=['lgb_te','lgb_full','lgb_full_huber','cat_full','xgb_full','lgb_full_l1','cat_full_mae','hgb_full','ridge_full','et_full']
for k in [2,3,4]:
    best=min(itertools.combinations(good,k), key=lambda c: np.abs(y-np.mean([P[m] for m in c],0)).mean())
    p=np.mean([P[m] for m in best],0); print('best equal-avg',k,best,metrics(y,p))
# optimised weights (MAE) -- note: in-sample on OOF, small optimism
X=np.c_[[P[m] for m in good]].T
f=lambda w: np.abs(y-X@(np.abs(w)/np.abs(w).sum())).mean()
r=minimize(f,np.ones(len(good))/len(good),method='Nelder-Mead',options={'maxiter':4000})
w=np.abs(r.x)/np.abs(r.x).sum(); print(dict(zip(good,w.round(3))), 'MAE',r.fun)
# spatial autocorrelation of residuals (by zip order)
p=np.mean([P[m] for m in ['lgb_te','cat_full','lgb_full_huber']],0); res=y-p
s=np.argsort(o.Zip.values); rs=res[s]
for k in [1,2,3]:
    print('residual corr lag',k, np.corrcoef(rs[k:],rs[:-k])[0,1])
# median of preds vs rounding
print('rounded', metrics(y,np.round(p)))
print('error by target band'); print(pd.DataFrame({'y':pd.cut(y,[-1,0,10,30,50,70,90,100]),'ae':np.abs(res)}).groupby('y').ae.agg(['mean','count']))
