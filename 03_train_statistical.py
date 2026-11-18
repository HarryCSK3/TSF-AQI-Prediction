"""
Statistical Models: ARIMA, SARIMA, SARIMAX, Holt-Winters, ETS
Walk-forward validation: N_SPLITS folds (>=20), expanding training window,
each fold validated on the next day (4 steps at 6-hour resolution).
Run this yourself: `python 03_train_statistical.py`
"""
import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from walk_forward import get_fold_boundaries, compute_metrics, summarize_folds, N_SPLITS, VAL_SIZE

df = pd.read_csv('data/anand_vihar_raw.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
series = df['aqi']
exog = df[['pm25', 'pm10', 'no2', 'temperature', 'humidity']]
datetimes = df['datetime']

SEASONAL_PERIOD = 4  # daily cycle at 6h resolution
folds = get_fold_boundaries(len(series), n_splits=N_SPLITS, val_size=VAL_SIZE)

model_names = ['ARIMA', 'SARIMA', 'SARIMAX', 'Holt-Winters', 'ETS']
fold_metrics = {name: [] for name in model_names}
fold_times = {name: {'train': [], 'pred': []} for name in model_names}
all_predictions = {name: [] for name in model_names}
all_actuals = []
all_pred_datetimes = []

for fold_i, (train_end, val_start, val_end) in enumerate(folds, start=1):
    print(f"Fold {fold_i}/{N_SPLITS}  (train size={train_end}, validating steps {val_start}:{val_end})")

    train_y = series.iloc[:train_end]
    val_y = series.iloc[val_start:val_end]
    train_exog = exog.iloc[:train_end]
    val_exog = exog.iloc[val_start:val_end]

    all_actuals.extend(val_y.values.tolist())
    all_pred_datetimes.extend(datetimes.iloc[val_start:val_end].tolist())

    # ---- ARIMA ----
    t0 = time.time()
    fit = ARIMA(train_y, order=(3, 1, 2)).fit()
    t_train = time.time() - t0
    t0 = time.time()
    pred = fit.forecast(steps=VAL_SIZE)
    t_pred = time.time() - t0
    fold_metrics['ARIMA'].append(compute_metrics(val_y, pred))
    fold_times['ARIMA']['train'].append(t_train)
    fold_times['ARIMA']['pred'].append(t_pred)
    all_predictions['ARIMA'].extend(pred.values.tolist())

    # ---- SARIMA ----
    t0 = time.time()
    fit = SARIMAX(train_y, order=(2, 1, 2), seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    t_train = time.time() - t0
    t0 = time.time()
    pred = fit.forecast(steps=VAL_SIZE)
    t_pred = time.time() - t0
    fold_metrics['SARIMA'].append(compute_metrics(val_y, pred))
    fold_times['SARIMA']['train'].append(t_train)
    fold_times['SARIMA']['pred'].append(t_pred)
    all_predictions['SARIMA'].extend(pred.values.tolist())

    # ---- SARIMAX ----
    t0 = time.time()
    fit = SARIMAX(train_y, exog=train_exog, order=(2, 1, 2), seasonal_order=(1, 1, 1, SEASONAL_PERIOD),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    t_train = time.time() - t0
    t0 = time.time()
    pred = fit.forecast(steps=VAL_SIZE, exog=val_exog)
    t_pred = time.time() - t0
    fold_metrics['SARIMAX'].append(compute_metrics(val_y, pred))
    fold_times['SARIMAX']['train'].append(t_train)
    fold_times['SARIMAX']['pred'].append(t_pred)
    all_predictions['SARIMAX'].extend(pred.values.tolist())

    # ---- Holt-Winters ----
    t0 = time.time()
    fit = ExponentialSmoothing(train_y, trend='add', seasonal='add',
                                seasonal_periods=SEASONAL_PERIOD).fit()
    t_train = time.time() - t0
    t0 = time.time()
    pred = fit.forecast(steps=VAL_SIZE)
    t_pred = time.time() - t0
    fold_metrics['Holt-Winters'].append(compute_metrics(val_y, pred))
    fold_times['Holt-Winters']['train'].append(t_train)
    fold_times['Holt-Winters']['pred'].append(t_pred)
    all_predictions['Holt-Winters'].extend(pred.values.tolist())

    # ---- ETS ----
    t0 = time.time()
    fit = ETSModel(train_y, error='add', trend='add', seasonal='add',
                   seasonal_periods=SEASONAL_PERIOD).fit(disp=False)
    t_train = time.time() - t0
    t0 = time.time()
    pred = fit.forecast(steps=VAL_SIZE)
    t_pred = time.time() - t0
    fold_metrics['ETS'].append(compute_metrics(val_y, pred))
    fold_times['ETS']['train'].append(t_train)
    fold_times['ETS']['pred'].append(t_pred)
    all_predictions['ETS'].extend(pred.values.tolist())

# ---- Aggregate across folds ----
summary_rows = {}
for name in model_names:
    summary = summarize_folds(fold_metrics[name])
    summary['AvgTrainTime'] = float(np.mean(fold_times[name]['train']))
    summary['AvgPredTime'] = float(np.mean(fold_times[name]['pred']))
    summary_rows[name] = summary

summary_df = pd.DataFrame(summary_rows).T
summary_df.to_csv('models/statistical_walkforward_summary.csv')

# per-fold detail (for full transparency / your own analysis)
detail_rows = []
for name in model_names:
    for i, m in enumerate(fold_metrics[name], start=1):
        detail_rows.append({'model': name, 'fold': i, **m,
                             'train_time': fold_times[name]['train'][i-1],
                             'pred_time': fold_times[name]['pred'][i-1]})
pd.DataFrame(detail_rows).to_csv('models/statistical_walkforward_folds.csv', index=False)

with open('models/statistical_results.pkl', 'wb') as f:
    pickle.dump({
        'predictions': {k: np.array(v) for k, v in all_predictions.items()},
        'actuals': np.array(all_actuals),
        'pred_datetimes': pd.to_datetime(all_pred_datetimes),
        'summary': summary_rows,
    }, f)

print("\nDone. Per-fold results: models/statistical_walkforward_folds.csv")
print("Aggregated (mean/std over folds): models/statistical_walkforward_summary.csv")
