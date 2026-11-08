"""June-only, fold-local structural affordability models. Run from any directory."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, ndtr
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'outputs' / 'structural'; WORK.mkdir(parents=True, exist_ok=True)
TARGET = 'AffordabilityPercentageTrue'
FEATURES = ['MedianHomeLengthofResidence', 'SeniorAdultinHouseholdPercent',
            'AvgNumberofAdults', 'AvgNumberofChildren', 'YoungAdultinHouseholdPercent',
            'AvgGenerationsinHousehold']


class Preparation:
    """Two small log regressions, fit on training covariates only."""
    def fit(self, df, tier):
        self.tier = tier
        income = np.log(df.AnnualMedianHouseholdIncome.to_numpy())
        rent = df.RentIndex.fillna(df.zori_z).to_numpy(float)
        valid = np.isfinite(rent) & (rent > 0)
        self.rent_coef = np.linalg.lstsq(np.c_[np.ones(valid.sum()), income[valid]],
                                         np.log(rent[valid]), rcond=None)[0]
        rent = np.where(valid, rent, np.exp(self.rent_coef[0] + self.rent_coef[1]*income))
        home = self.home(df)
        valid = np.isfinite(home) & (home > 0)
        self.home_coef = np.linalg.lstsq(np.c_[np.ones(valid.sum()), np.log(rent[valid])],
                                         np.log(home[valid]), rcond=None)[0]
        self.mean = df[FEATURES].mean().to_numpy()
        self.std = df[FEATURES].std().to_numpy()
        return self

    def home(self, df):
        order = list(dict.fromkeys([self.tier, 'zhvi', 'zhvi_sfr', 'zhvi_low', 'zhvi_high']))
        return df[order].where(df[order] > 0).bfill(axis=1).iloc[:, 0].to_numpy(float)

    def transform(self, df):
        income = df.AnnualMedianHouseholdIncome.to_numpy(float)
        rent = df.RentIndex.fillna(df.zori_z).to_numpy(float)
        rent = np.where(np.isfinite(rent) & (rent > 0), rent,
                        np.exp(self.rent_coef[0] + self.rent_coef[1]*np.log(income)))
        home = self.home(df)
        home = np.where(np.isfinite(home) & (home > 0), home,
                        np.exp(self.home_coef[0] + self.home_coef[1]*np.log(rent)))
        x = np.clip((df[FEATURES].to_numpy(float)-self.mean)/self.std, -5, 5)
        # Include affordability context in flexible sigma and cost functions.
        x = np.c_[x, np.log(income/90000), df.OwnerPercent.to_numpy()/100-.65]
        return income, rent, home, df.OwnerPercent.to_numpy()/100, x


def predict(p, data, kind):
    income, rent, home, owner, x = data
    # Fixed 6.5% 30-year mortgage; LTV absorbs effective historical borrowing cost.
    rate = .065/12
    payment = rate/(1-(1+rate)**-360)
    sigma = np.exp(p[0])
    ltv = expit(p[1])
    tax = np.exp(p[2])
    rent_scale = np.exp(p[3])
    free = expit(p[4] + p[5]*x[:, 0] + p[6]*x[:, 1])
    cost_scale = 1.
    renter_shift = 0.
    if kind >= 1:
        sigma = np.exp(np.clip(p[0]+x[:, :4]@p[7:11], -4, 2))
    if kind >= 2:
        cost_scale = np.exp(np.clip(x@p[11:19], -4, 4))
    if kind >= 3:
        renter_shift = x[:, :6]@p[19:25]
    mortgage_cost = home*(ltv*payment + tax/12)*cost_scale
    free_cost = home*tax/12*cost_scale
    prob_mortgage = ndtr(np.log(income*.3/(12*mortgage_cost))/sigma)
    prob_free = ndtr(np.log(income*.3/(12*free_cost))/sigma)
    prob_rent = ndtr((np.log(income*.3/(12*rent*rent_scale))+renter_shift)/sigma)
    return np.clip(100*(owner*((1-free)*prob_mortgage+free*prob_free)+(1-owner)*prob_rent), 0, 100)


def fit(data, y, kind):
    size = [7, 11, 19, 25][kind]
    initial = np.zeros(size)
    initial[:7] = [np.log(.5), 1., np.log(.02), 0., -2., .2, .2]
    lower = np.full(size, -2.)
    upper = np.full(size, 2.)
    lower[:7] = [-4, -7, np.log(.001), -1.5, -9, -3, -3]
    upper[:7] = [1, 7, np.log(.10), 1.5, 5, 3, 3]
    best = None
    # Two deterministic starts reduce sensitivity to local minima.
    for sigma in (.35, .9):
        initial[0] = np.log(sigma)
        result = least_squares(lambda p: predict(p, data, kind)-y, initial,
                               bounds=(lower, upper), max_nfev=220, ftol=2e-6,
                               xtol=2e-6, gtol=2e-5)
        if best is None or result.cost < best.cost:
            best = result
    return best


def main():
    started = time.time()
    zillow = pd.read_csv(ROOT/'external/zillow_features.csv')
    assert not zillow.duplicated(['Zip', 'Month']).any()
    train = pd.read_csv(ROOT/'data/train.csv')
    train = train.loc[train.Month.eq('6/1/2026')].sort_values('Zip').reset_index(drop=True)
    test = pd.read_csv(ROOT/'data/test.csv')
    assert train.Zip.is_unique and test.Zip.is_unique
    train = train.merge(zillow, on=['Zip', 'Month'], how='left', validate='one_to_one')
    test = test.merge(zillow, on=['Zip', 'Month'], how='left', validate='one_to_one')
    y = train[TARGET].to_numpy(float)
    folds = list(KFold(5, shuffle=True, random_state=200).split(train.Zip.to_numpy()))
    variants = [('constant_mid', 'zhvi', 0), ('sigma_mid', 'zhvi', 1),
                ('cost_mid', 'zhvi', 2), ('flex_mid', 'zhvi', 3),
                ('flex_low', 'zhvi_low', 3), ('flex_sfr', 'zhvi_sfr', 3)]
    records, predictions = [], {}
    for name, tier, kind in variants:
        oof = np.full(len(train), np.nan)
        fold_scores, diagnostics = [], []
        for fold, (tr, va) in enumerate(folds):
            prep = Preparation().fit(train.iloc[tr], tier)
            result = fit(prep.transform(train.iloc[tr]), y[tr], kind)
            oof[va] = predict(result.x, prep.transform(train.iloc[va]), kind)
            rmse = float(np.sqrt(np.mean((oof[va]-y[va])**2)))
            fold_scores.append(rmse)
            diagnostics.append({'fold': fold, 'nfev': result.nfev, 'status': int(result.status)})
            print(f'{name} fold {fold}: RMSE={rmse:.6f}, nfev={result.nfev}', flush=True)
        predictions[name] = oof
        records.append(dict(name=name, tier=tier, kind=kind, parameters=[7,11,19,25][kind],
                            rmse=float(np.sqrt(np.mean((oof-y)**2))),
                            mae=float(np.mean(np.abs(oof-y))), folds=fold_scores,
                            diagnostics=diagnostics))
    records.sort(key=lambda row: row['rmse'])
    best, second = records[:2]
    difference = float(np.sqrt(np.mean((predictions[best['name']]-predictions[second['name']])**2)))
    selected = [best] + ([second] if difference >= .25 else [])
    out_oof, out_test = pd.DataFrame({'Zip': train.Zip}), pd.DataFrame({'Zip': test.Zip})
    fitted = {}
    for i, row in enumerate(selected):
        col = 'struct_pred' if i == 0 else 'struct_pred2'
        prep = Preparation().fit(train, row['tier'])
        result = fit(prep.transform(train), y, row['kind'])
        out_oof[col] = predictions[row['name']]
        out_test[col] = predict(result.x, prep.transform(test), row['kind'])
        fitted[col] = {'variant': row['name'], 'parameters': result.x.tolist(),
                       'rent_regression': prep.rent_coef.tolist(), 'home_regression': prep.home_coef.tolist(),
                       'feature_mean': prep.mean.tolist(), 'feature_std': prep.std.tolist(),
                       'optimizer_status': int(result.status), 'nfev': result.nfev}
    for frame in (out_oof, out_test):
        assert frame.notna().all().all()
        assert np.isfinite(frame.to_numpy()).all()
        assert frame.iloc[:, 1:].ge(0).all().all() and frame.iloc[:, 1:].le(100).all().all()
    (ROOT/'outputs').mkdir(exist_ok=True)
    out_oof.to_csv(ROOT/'outputs/structural_oof.csv', index=False)
    out_test.to_csv(ROOT/'outputs/structural_test.csv', index=False)
    (WORK/'fit_details.json').write_text(json.dumps({'metrics': records, 'fitted': fitted}, indent=2))
    table = '| Variant | Structural parameters | OOF RMSE | OOF MAE | Fold RMSEs |\n|---|---:|---:|---:|---|\n'
    for r in records:
        table += f"| {r['name']} | {r['parameters']} | {r['rmse']:.6f} | {r['mae']:.6f} | " + ', '.join(f'{v:.4f}' for v in r['folds']) + ' |\n'
    report = '# Structural affordability results\n\n'+table
    report += f"\nBest: **{best['name']}**, OOF RMSE **{best['rmse']:.6f}**, MAE **{best['mae']:.6f}**.\n"
    report += f"Runner-up OOF prediction RMS difference: {difference:.6f} percentage points. "
    report += ('Included as `struct_pred2` (threshold 0.25).\n' if len(selected)>1 else 'Not included because difference is below 0.25.\n')
    report += '\n## Method and validation\n\n'
    report += '- Only June training rows are used, sorted by Zip; KFold(5, shuffle=True, random_state=200). Each held-out ZIP is predicted exactly once. May rows are unused.\n'
    report += '- Unweighted least squares and unweighted pooled OOF metrics; no target-derived preprocessing. All imputation regressions and feature scaling are refitted within each fold.\n'
    report += '- Lognormal household income; affordability is the normal CDF of log(0.30 × annual median income / annual housing cost), divided by sigma. Owner and renter probabilities are mixed by OwnerPercent/100.\n'
    report += '- Mortgage cost uses a fixed 6.5% rate, 30-year amortization, fitted LTV and annual tax plus insurance. Mortgage-free owners pay tax plus insurance only; their share is logistic in residence length and senior percentage. Rent has a fitted multiplier.\n'
    report += '- constant: seven structural parameters; sigma: four additional log-sigma slopes; cost: eight additional log-owner-cost slopes; flex: six additional renter log-income-threshold slopes. Slopes are bounded and features standardized; these are effective predictive parameters, not causal estimates.\n'
    report += '- Feature order: '+', '.join(FEATURES)+', log(income/90000), OwnerPercent/100 − 0.65. Sigma uses first four; renter adjustment uses first six.\n'
    report += '- Tier first, then zhvi/sfr/low/high fallbacks. RentIndex falls back to zori_z. Remaining rent uses log-income regression; remaining home value uses log-rent regression. Each has two coefficients: including these, the largest model has 29 fitted predictive coefficients. Scaling statistics are preprocessing, not optimized model coefficients.\n'
    report += '- July test features are joined to July Zillow data. Final selected models are refitted on all June rows. Predictions are clipped to [0,100]. Model selection uses these same OOF scores; they are not a nested estimate of selection performance.\n'
    report += f'\nOutputs: {len(out_oof)} June OOF rows and {len(out_test)} July test rows, validated finite, unique ZIPs, and bounded predictions. Runtime: {(time.time()-started)/60:.2f} minutes.\n'
    report += '\nReproduce: `.venv\\Scripts\\python.exe src/structural_model.py`. Optimizer diagnostics and final coefficients are in `fit_details.json`.\n'
    (WORK/'results.md').write_text(report, encoding='utf-8')
    print('\n'+table, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
