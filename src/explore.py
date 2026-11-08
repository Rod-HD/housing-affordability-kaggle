import pandas as pd, numpy as np
pd.set_option('display.width',220); pd.set_option('display.max_columns',40)
tr=pd.read_csv('data/train.csv'); te=pd.read_csv('data/test.csv'); ss=pd.read_csv('data/sample_submission.csv')
T='AffordabilityPercentageTrue'
print(tr.Month.value_counts(), te.Month.value_counts())
print('target integer?', (tr[T]==tr[T].round()).mean())
print('ss order == test order?', (ss.Zip.values==te.Zip.values).all(), ss[T].nunique())
known=set(tr.Zip); te['known']=te.Zip.isin(known)
print('test known', te.known.sum(), 'new', (~te.known).sum())
print('dup zip in test', te.Zip.duplicated().sum())
# new zips: do their Metro/City appear in train?
new=te[~te.known]
print('new zip metro seen', new.Metro.isin(tr.Metro).mean(), 'city seen', (new.City+new.State).isin(tr.City+tr.State).mean(), 'state seen', new.State.isin(tr.State).mean())
print(new.State.value_counts().head(15))
# zip numeric sort neighbours
m=tr[tr.Month=='5/1/2026'].set_index('Zip'); j=tr[tr.Month=='6/1/2026'].set_index('Zip')
cols=[c for c in tr.columns if c not in ['Zip','Month','State','City','Metro',T]]
d=(j[cols+[T]]-m[cols+[T]])
print('feature change May->Jun (frac changed):'); print((d.abs()>1e-9).mean())
print(d.describe().T)
t7=te[te.known].set_index('Zip')
d2=(t7[cols]-j.loc[t7.index,cols])
print('feature change Jun->Jul (frac changed):'); print((d2.abs()>1e-9).mean()); print(d2.describe().T)
print('target delta dist', d[T].value_counts().sort_index().head(40))
# categorical consistency
print('state/city/metro same across months', (m.State==j.State).mean(), (m.City.fillna('')==j.City.fillna('')).mean(), (m.Metro.fillna('')==j.Metro.fillna('')).mean())
