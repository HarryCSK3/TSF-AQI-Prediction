"""
Feature Engineering for ML and DL models
- Lag features, rolling stats, datetime features for ML
- Scaled sequences for DL (done at model-train time)
"""
import pandas as pd
import numpy as np

df = pd.read_csv('./data/anand_vihar_raw.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

TARGET = 'aqi'
POLLUTANTS = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']

feat = df[['datetime', TARGET] + POLLUTANTS + ['temperature','humidity','wind_speed','visibility']].copy()

# ---- Datetime features ----
feat['hour'] = feat['datetime'].dt.hour
feat['day'] = feat['datetime'].dt.day
feat['month'] = feat['datetime'].dt.month
feat['weekday'] = feat['datetime'].dt.weekday
feat['is_weekend'] = (feat['weekday'] >= 5).astype(int)
feat['quarter'] = feat['datetime'].dt.quarter
feat['day_of_year'] = feat['datetime'].dt.dayofyear
# cyclical encodings
feat['hour_sin'] = np.sin(2*np.pi*feat['hour']/24)
feat['hour_cos'] = np.cos(2*np.pi*feat['hour']/24)
feat['month_sin'] = np.sin(2*np.pi*feat['month']/12)
feat['month_cos'] = np.cos(2*np.pi*feat['month']/12)

# ---- Lag features (in steps of 6-hour reading; 4 steps = 1 day) ----
LAGS = [1, 2, 3, 4, 8, 28]  # 6h, 12h, 18h, 1 day, 2 days, 1 week
for lag in LAGS:
    feat[f'aqi_lag_{lag}'] = feat[TARGET].shift(lag)
    for p in POLLUTANTS:
        feat[f'{p}_lag_{lag}'] = feat[p].shift(lag)

# ---- Rolling features (window in steps) ----
for window in [4, 8, 28]:  # 1 day, 2 days, 1 week
    feat[f'aqi_roll_mean_{window}'] = feat[TARGET].shift(1).rolling(window).mean()
    feat[f'aqi_roll_std_{window}'] = feat[TARGET].shift(1).rolling(window).std()
    feat[f'aqi_roll_max_{window}'] = feat[TARGET].shift(1).rolling(window).max()
    feat[f'aqi_roll_min_{window}'] = feat[TARGET].shift(1).rolling(window).min()

feat = feat.dropna().reset_index(drop=True)
feat.to_csv('./data/features_ml.csv', index=False)

print("Feature engineered dataset shape:", feat.shape)
print("Columns:", feat.columns.tolist())
print("\nSaved to data/features_ml.csv")
