"""
Train FINAL production models (on full dataset) for deployment.
Best model selected from comparison = XGBoost family (best MAE/MAPE, fast, and
supports multi-target lag-based forecasting needed for the web app).

We train one XGBoost regressor per target: aqi, pm25, pm10, no2, so2, co, o3
using the same lag/rolling/datetime feature set, so the Flask app can do
recursive (iterative) multi-step forecasting for any future date.
"""
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor

feat = pd.read_csv('./data/features_ml.csv', parse_dates=['datetime'])

TARGETS = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
drop_cols = ['datetime'] + TARGETS + ['temperature', 'humidity', 'wind_speed', 'visibility']
feature_cols = [c for c in feat.columns if c not in drop_cols]

models = {}
for target in TARGETS:
    print(f"Training deployment model for {target}...")
    X = feat[feature_cols]
    y = feat[target]
    model = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X, y)
    models[target] = model

with open('./models/deployment_models.pkl', 'wb') as f:
    pickle.dump({'models': models, 'feature_cols': feature_cols, 'targets': TARGETS}, f)

# Also save the last LAGS window of raw history needed to bootstrap recursive forecasting
raw = pd.read_csv('./data/anand_vihar_raw.csv', parse_dates=['datetime'])
raw = raw.sort_values('datetime').reset_index(drop=True)
history_cols = ['datetime', 'aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3',
                 'temperature', 'humidity', 'wind_speed', 'visibility']
last_history = raw[history_cols].tail(40).reset_index(drop=True)  # enough for lag_28
last_history.to_csv('./models/last_history.csv', index=False)

print("\nDeployment models trained and saved.")
print("Feature columns:", len(feature_cols))
print("Last known datetime in dataset:", raw['datetime'].max())
