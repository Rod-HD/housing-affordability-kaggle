import pandas as pd, numpy as np
pd.set_option('display.width',220); pd.set_option('display.max_columns',40)
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
T='AffordabilityPercentageTrue'
m=tr[tr.Month=='5/1/2026'].set_index('Zip').sort_index(); j=tr[tr.Month=='6/1/2026'].set_index('Zip').sort_index()
print('same index', (m.index==j.index).all())
print('state/city/metro same', (m.State==j.State).mean(), (m.City.fillna('')==j.City.fillna('')).mean(), (m.Metro.fillna('')==j.Metro.fillna('')).mean())
cols=[c for c in tr.columns if c not in ['Zip','Month','State','City','Metro',T]]
d=(j[cols+[T]]-m[cols+[T]])
print('corr of dTarget with dFeatures:'); print(d.corr()[T].sort_values())
print('corr dT with levels (June):'); print(pd.concat([d[T],j[cols]],axis=1).corr()[T].sort_values())
# RentIndex relative change
r=np.log(j.RentIndex/m.RentIndex); print('corr dT vs dlogRent', np.corrcoef(r.dropna(), d[T][r.notna()])[0,1])
# Does target delta relate to target level? (bounded)
print(pd.crosstab(pd.cut(j[T],[-1,0,10,30,50,70,90,99,100]), d[T].clip(-3,3)))
# missing pattern of rent
print('rent missing May,Jun,Jul known', m.RentIndex.isna().sum(), j.RentIndex.isna().sum(), te.RentIndex.isna().sum())
print('target where rent missing', j.loc[j.RentIndex.isna(),T].describe())
# target 0 / 100 analysis
print(j[j[T]==0][cols].median()); print(j[j[T]>0][cols].median())
