import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv')
T='AffordabilityPercentageTrue'
te['new']=(~te.Zip.isin(tr.Zip)).astype(int)
num=[c for c in te.columns if c not in ['Zip','Month','State','City','Metro','new']]
print(te.groupby('new')[num].median().T)
print('rent missing new vs known', te.groupby('new').RentIndex.apply(lambda s:s.isna().mean()))
X=te[num+['Zip']].copy()
for feats,name in [(num,'features'),(num+['Zip'],'features+zip')]:
    p=cross_val_predict(lgb.LGBMClassifier(n_estimators=300,learning_rate=.05,verbose=-1,n_jobs=8),te[feats],te.new,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
    print('adversarial AUC new-vs-known (July)',name, roc_auc_score(te.new,p))
# zip spatial smoothness: corr of target with nearest zip's target in June
j=tr[tr.Month=='6/1/2026'].sort_values('Zip').reset_index(drop=True)
y=j[T].values; z=j.Zip.values
for k in [1,2,5]:
    nb=(np.roll(y,k)+np.roll(y,-k))/2
    print('lag',k,'corr with neighbour-by-zip', np.corrcoef(y[k:-k],nb[k:-k])[0,1], 'MAE', np.abs(y[k:-k]-nb[k:-k]).mean())
j['zip3']=j.Zip//100
g=j.groupby('zip3')[T]
print('zip3 groups',g.ngroups,'eta2 zip3', 1-((j[T]-g.transform('mean'))**2).sum()/((j[T]-j[T].mean())**2).sum())
for c in ['State','Metro','City']:
    g=j.groupby(c)[T]; print('eta2',c,g.ngroups, 1-((j[T]-g.transform('mean'))**2).sum()/((j[T]-j[T].mean())**2).sum())
# new zips: distance to nearest known zip
kn=np.sort(tr.Zip.unique()); nz=te.loc[te.new==1,'Zip'].values
idx=np.searchsorted(kn,nz); d=np.minimum(np.abs(kn[np.clip(idx,0,len(kn)-1)]-nz),np.abs(kn[np.clip(idx-1,0,len(kn)-1)]-nz))
print('new zip nearest known zip distance', pd.Series(d).describe())
kd=np.diff(kn); print('known zip spacing', pd.Series(kd).describe())
