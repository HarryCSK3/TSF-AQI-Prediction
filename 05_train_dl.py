"""
Deep Learning Models: LSTM, GRU
Walk-forward validation: N_SPLITS folds (>=20), expanding training window,
each fold validated on the next day (4 steps at 6-hour resolution).
Sequence generation via sliding window over scaled AQI + pollutant features.

NOTE: retraining a neural net 20+ times is inherently slow. If this takes too
long on your machine, lower N_SPLITS in walk_forward.py (e.g. to 20 is already
the minimum requested; you can raise EPOCHS/lower LOOKBACK to trade accuracy
for speed).

Run this yourself: `python 05_train_dl.py`
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from walk_forward import get_fold_boundaries, compute_metrics, summarize_folds, N_SPLITS, VAL_SIZE

tf.random.set_seed(42)
np.random.seed(42)

df = pd.read_csv('data/anand_vihar_raw.csv', parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

FEATURES = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'temperature', 'humidity']
data = df[FEATURES].values
datetimes = df['datetime']
LOOKBACK = 28   # 7 days of 6-hour readings
EPOCHS = 25
PATIENCE = 5

folds = get_fold_boundaries(len(data), n_splits=N_SPLITS, val_size=VAL_SIZE)


def make_sequences(arr, lookback, target_idx=0):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i])
        y.append(arr[i, target_idx])
    return np.array(X), np.array(y)


def build_lstm(n_features):
    m = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m


def build_gru(n_features):
    m = Sequential([
        GRU(64, activation='tanh', return_sequences=True, input_shape=(LOOKBACK, n_features)),
        Dropout(0.2),
        GRU(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m


model_builders = {'LSTM': build_lstm, 'GRU': build_gru}
n_features = data.shape[1]

fold_metrics = {name: [] for name in model_builders}
fold_times = {name: {'train': [], 'pred': []} for name in model_builders}
all_predictions = {name: [] for name in model_builders}
all_actuals = []
all_pred_datetimes = []

for fold_i, (train_end, val_start, val_end) in enumerate(folds, start=1):
    print(f"Fold {fold_i}/{N_SPLITS}  (train size={train_end}, validating steps {val_start}:{val_end})")

    # scaler fit ONLY on training data of this fold (avoid leakage)
    scaler = MinMaxScaler()
    train_raw = data[:train_end]
    scaler.fit(train_raw)

    # need LOOKBACK points of history before val_start to build validation sequences
    context_start = val_start - LOOKBACK
    val_raw = data[context_start:val_end]

    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)

    X_train, y_train = make_sequences(train_scaled, LOOKBACK)
    X_val, y_val = make_sequences(val_scaled, LOOKBACK)

    aqi_min, aqi_max = scaler.data_min_[0], scaler.data_max_[0]

    def inverse_aqi(v):
        return v * (aqi_max - aqi_min) + aqi_min

    true_aqi = inverse_aqi(y_val)
    all_actuals.extend(true_aqi.tolist())
    all_pred_datetimes.extend(datetimes.iloc[val_start:val_end].tolist())

    for name, builder in model_builders.items():
        model = builder(n_features)
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        t0 = time.time()
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_split=0.1,
                  callbacks=[es], verbose=0)
        t_train = time.time() - t0
        t0 = time.time()
        pred_scaled = model.predict(X_val, verbose=0).flatten()
        t_pred = time.time() - t0
        pred_aqi = inverse_aqi(pred_scaled)

        fold_metrics[name].append(compute_metrics(true_aqi, pred_aqi))
        fold_times[name]['train'].append(t_train)
        fold_times[name]['pred'].append(t_pred)
        all_predictions[name].extend(pred_aqi.tolist())

# ---- Aggregate across folds ----
summary_rows = {}
for name in model_builders:
    summary = summarize_folds(fold_metrics[name])
    summary['AvgTrainTime'] = float(np.mean(fold_times[name]['train']))
    summary['AvgPredTime'] = float(np.mean(fold_times[name]['pred']))
    summary_rows[name] = summary

pd.DataFrame(summary_rows).T.to_csv('models/dl_walkforward_summary.csv')

detail_rows = []
for name in model_builders:
    for i, m in enumerate(fold_metrics[name], start=1):
        detail_rows.append({'model': name, 'fold': i, **m,
                             'train_time': fold_times[name]['train'][i-1],
                             'pred_time': fold_times[name]['pred'][i-1]})
pd.DataFrame(detail_rows).to_csv('models/dl_walkforward_folds.csv', index=False)

with open('models/dl_results.pkl', 'wb') as f:
    pickle.dump({
        'predictions': {k: np.array(v) for k, v in all_predictions.items()},
        'actuals': np.array(all_actuals),
        'pred_datetimes': pd.to_datetime(all_pred_datetimes),
        'summary': summary_rows,
    }, f)

# ---- Fit a final model on ALL data for later inspection ----
final_scaler = MinMaxScaler()
final_scaler.fit(data)
data_scaled_full = final_scaler.transform(data)
X_full, y_full = make_sequences(data_scaled_full, LOOKBACK)
for name, builder in model_builders.items():
    final_model = builder(n_features)
    es = EarlyStopping(monitor='loss', patience=PATIENCE, restore_best_weights=True)
    final_model.fit(X_full, y_full, epochs=EPOCHS, batch_size=32, callbacks=[es], verbose=0)
    final_model.save(f'models/{name.lower()}_model.keras')

with open('models/dl_scaler.pkl', 'wb') as f:
    pickle.dump({'scaler': final_scaler, 'features': FEATURES, 'lookback': LOOKBACK}, f)

print("\nDone. Per-fold results: models/dl_walkforward_folds.csv")
print("Aggregated (mean/std over folds): models/dl_walkforward_summary.csv")
