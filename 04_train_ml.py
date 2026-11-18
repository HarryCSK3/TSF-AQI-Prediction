"""
Machine Learning Models: XGBoost, LightGBM, Random Forest
Walk-forward validation: N_SPLITS folds (>=20), expanding training window,
each fold validated on the next day (4 steps at 6-hour resolution).
Uses lag + rolling + datetime engineered features (data/features_ml.csv).
Run this yourself: `python 04_train_ml.py`
"""
import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from walk_forward import get_fold_boundaries, compute_metrics, summarize_folds, N_SPLITS, VAL_SIZE

feat = pd.read_csv('data/features_ml.csv', parse_dates=['datetime'])
TARGET = 'aqi'
drop_cols = ['datetime', TARGET, 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3',
             'temperature', 'humidity', 'wind_speed', 'visibility']
feature_cols = [c for c in feat.columns if c not in drop_cols]

X_all = feat[feature_cols]
y_all = feat[TARGET]
datetimes = feat['datetime']

# features_ml.csv has fewer rows than the raw series (lag features drop the first
# few rows with NaN). To keep fold datetimes aligned with the statistical/DL
# scripts (which use the full raw series), compute fold boundaries on the RAW
# timeline length, then shift indices to match this shorter, offset dataframe.
raw_len = len(pd.read_csv('data/anand_vihar_raw.csv'))
offset = raw_len - len(feat)
folds_global = get_fold_boundaries(raw_len, n_splits=N_SPLITS, val_size=VAL_SIZE)
folds = [(te - offset, vs - offset, ve - offset) for (te, vs, ve) in folds_global]

model_builders = {
    'XGBoost': lambda: XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42),
    'LightGBM': lambda: LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.03,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
    'Random Forest': lambda: RandomForestRegressor(n_estimators=300, max_depth=10,
                                                     random_state=42, n_jobs=-1),
}

fold_metrics = {name: [] for name in model_builders}
fold_times = {name: {'train': [], 'pred': []} for name in model_builders}
all_predictions = {name: [] for name in model_builders}
all_actuals = []
all_pred_datetimes = []

for fold_i, (train_end, val_start, val_end) in enumerate(folds, start=1):
    print(f"Fold {fold_i}/{N_SPLITS}  (train size={train_end}, validating steps {val_start}:{val_end})")

    X_train, y_train = X_all.iloc[:train_end], y_all.iloc[:train_end]
    X_val, y_val = X_all.iloc[val_start:val_end], y_all.iloc[val_start:val_end]

    all_actuals.extend(y_val.values.tolist())
    all_pred_datetimes.extend(datetimes.iloc[val_start:val_end].tolist())

    for name, builder in model_builders.items():
        model = builder()
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0
        t0 = time.time()
        pred = model.predict(X_val)
        t_pred = time.time() - t0
        fold_metrics[name].append(compute_metrics(y_val, pred))
        fold_times[name]['train'].append(t_train)
        fold_times[name]['pred'].append(t_pred)
        all_predictions[name].extend(pred.tolist())

# ---- Aggregate across folds ----
summary_rows = {}
for name in model_builders:
    summary = summarize_folds(fold_metrics[name])
    summary['AvgTrainTime'] = float(np.mean(fold_times[name]['train']))
    summary['AvgPredTime'] = float(np.mean(fold_times[name]['pred']))
    summary_rows[name] = summary

pd.DataFrame(summary_rows).T.to_csv('models/ml_walkforward_summary.csv')

detail_rows = []
for name in model_builders:
    for i, m in enumerate(fold_metrics[name], start=1):
        detail_rows.append({'model': name, 'fold': i, **m,
                             'train_time': fold_times[name]['train'][i-1],
                             'pred_time': fold_times[name]['pred'][i-1]})
pd.DataFrame(detail_rows).to_csv('models/ml_walkforward_folds.csv', index=False)

with open('models/ml_results.pkl', 'wb') as f:
    pickle.dump({
        'predictions': {k: np.array(v) for k, v in all_predictions.items()},
        'actuals': np.array(all_actuals),
        'pred_datetimes': pd.to_datetime(all_pred_datetimes),
        'summary': summary_rows,
        'feature_cols': feature_cols,
    }, f)

# Also fit each model once on ALL data and save it (useful for inspection / feature importance)
for name, builder in model_builders.items():
    final_model = builder()
    final_model.fit(X_all, y_all)
    with open(f'models/{name.replace(" ", "_").lower()}_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

print("\nDone. Per-fold results: models/ml_walkforward_folds.csv")
print("Aggregated (mean/std over folds): models/ml_walkforward_summary.csv")
