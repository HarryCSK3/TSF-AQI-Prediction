"""
Forecast engine used by the Flask app.
Rebuilds the exact same lag/rolling/datetime features used in training,
for a single future timestamp, given a rolling history buffer.
Supports recursive (iterative) multi-step forecasting into the future,
and direct lookup for dates that already exist in the historical dataset.
"""
import pandas as pd
import numpy as np
import pickle
import os

BASE = os.path.dirname(os.path.abspath(__file__))

TARGETS = ['aqi', 'pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
LAGS = [1, 2, 3, 4, 8, 28]
ROLL_WINDOWS = [4, 8, 28]
STEP_HOURS = [6, 12, 18, 23]  # the 4 daily reading slots present in the dataset

_model_bundle = None
_history_df = None
_raw_full = None


def _load():
    global _model_bundle, _history_df, _raw_full
    if _model_bundle is None:
        with open(os.path.join(BASE, 'models', 'deployment_models.pkl'), 'rb') as f:
            _model_bundle = pickle.load(f)
        _history_df = pd.read_csv(os.path.join(BASE, 'models', 'last_history.csv'), parse_dates=['datetime'])
        raw_path = os.path.join(BASE, 'data', 'anand_vihar_raw.csv')
        if os.path.exists(raw_path):
            _raw_full = pd.read_csv(raw_path, parse_dates=['datetime'])
    return _model_bundle, _history_df, _raw_full


def _build_feature_row(buffer, next_dt, feature_cols):
    """buffer: list of dicts (ascending order) with keys aqi + pollutants.
    Builds one feature row (numpy array) matching training-time feature engineering."""
    row = {}
    row['hour'] = next_dt.hour
    row['day'] = next_dt.day
    row['month'] = next_dt.month
    row['weekday'] = next_dt.weekday()
    row['is_weekend'] = int(row['weekday'] >= 5)
    row['quarter'] = (next_dt.month - 1) // 3 + 1
    row['day_of_year'] = next_dt.timetuple().tm_yday
    row['hour_sin'] = np.sin(2 * np.pi * row['hour'] / 24)
    row['hour_cos'] = np.cos(2 * np.pi * row['hour'] / 24)
    row['month_sin'] = np.sin(2 * np.pi * row['month'] / 12)
    row['month_cos'] = np.cos(2 * np.pi * row['month'] / 12)

    n = len(buffer)
    for lag in LAGS:
        idx = n - lag
        idx = idx if idx >= 0 else 0
        src = buffer[idx]
        row[f'aqi_lag_{lag}'] = src['aqi']
        for p in ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']:
            row[f'{p}_lag_{lag}'] = src[p]

    aqi_vals_all = [b['aqi'] for b in buffer]
    for window in ROLL_WINDOWS:
        window_vals = np.array(aqi_vals_all[max(0, n - window):n], dtype=float)
        row[f'aqi_roll_mean_{window}'] = window_vals.mean()
        row[f'aqi_roll_std_{window}'] = window_vals.std(ddof=1) if len(window_vals) > 1 else 0.0
        row[f'aqi_roll_max_{window}'] = window_vals.max()
        row[f'aqi_roll_min_{window}'] = window_vals.min()

    return np.array([[row[c] for c in feature_cols]], dtype=float)


def _predict_next(buffer, next_dt, bundle):
    feature_cols = bundle['feature_cols']
    X = _build_feature_row(buffer, next_dt, feature_cols)
    preds = {}
    for target in TARGETS:
        model = bundle['models'][target]
        val = float(model.predict(X)[0])
        preds[target] = max(val, 0.0)
    preds['aqi'] = min(preds['aqi'], 500.0)
    return preds


def aqi_category(aqi):
    if aqi <= 50:
        return 'Good', '#2ecc71'
    elif aqi <= 100:
        return 'Satisfactory', '#a3d900'
    elif aqi <= 200:
        return 'Moderate', '#f1c40f'
    elif aqi <= 300:
        return 'Poor', '#e67e22'
    elif aqi <= 400:
        return 'Very Poor', '#e74c3c'
    else:
        return 'Severe', '#7d0d0d'


def forecast_for_date(target_date_str):
    """
    target_date_str: 'YYYY-MM-DD'
    Returns list of 4 dicts (one per 6-hour slot) with aqi + pollutants.
    If date exists in historical data -> returns actual recorded values (mode='historical').
    Otherwise -> recursively forecasts forward from the last known data point (mode='forecast').
    """
    bundle, history, raw_full = _load()
    target_date = pd.Timestamp(target_date_str)

    # ---- Case 1: historical date already in dataset ----
    if raw_full is not None:
        day_rows = raw_full[raw_full['datetime'].dt.date == target_date.date()]
        if len(day_rows) > 0:
            out = []
            for _, r in day_rows.sort_values('datetime').iterrows():
                cat, color = aqi_category(r['aqi'])
                out.append({
                    'time': r['datetime'].strftime('%H:%M'),
                    'datetime': r['datetime'].strftime('%Y-%m-%d %H:%M'),
                    'aqi': round(float(r['aqi']), 1),
                    'pm25': round(float(r['pm25']), 1),
                    'pm10': round(float(r['pm10']), 1),
                    'no2': round(float(r['no2']), 1),
                    'so2': round(float(r['so2']), 1),
                    'co': round(float(r['co']), 2),
                    'o3': round(float(r['o3']), 1),
                    'temperature': round(float(r['temperature']), 1),
                    'humidity': round(float(r['humidity']), 1),
                    'category': cat,
                    'color': color,
                })
            return out, 'historical'

    # ---- Case 2: future date -> recursive forecast ----
    last_known_dt = history['datetime'].max()
    if target_date.date() < last_known_dt.date():
        # date before dataset start, or a gap: not supported, fallback to nearest historical
        return [], 'unavailable'

    MAX_FORECAST_DAYS = 90
    days_ahead = (target_date.date() - last_known_dt.date()).days
    if days_ahead > MAX_FORECAST_DAYS:
        return [], 'too_far'

    # lightweight rolling buffer (list of dicts) - avoids expensive DataFrame concat per step
    buffer = history.to_dict('records')
    last_temp = buffer[-1]['temperature']
    last_hum = buffer[-1]['humidity']
    results = []

    # walk forward hour-slot by hour-slot until we've generated the full target day
    target_slots = [pd.Timestamp(f"{target_date.date()} {h:02d}:00:00") for h in STEP_HOURS]

    # Determine all slots we need to step through from last_known_dt+1 slot to last target slot
    all_slots = []
    cursor = last_known_dt
    while True:
        cur_hour = cursor.hour
        cur_date = cursor.date()
        remaining = [h for h in STEP_HOURS if h > cur_hour]
        if remaining:
            next_slot = pd.Timestamp(f"{cur_date} {remaining[0]:02d}:00:00")
        else:
            next_day = cursor + pd.Timedelta(days=1)
            next_slot = pd.Timestamp(f"{next_day.date()} {STEP_HOURS[0]:02d}:00:00")
        all_slots.append(next_slot)
        cursor = next_slot
        if next_slot >= target_slots[-1]:
            break
        if len(all_slots) > 1500:  # hard safety cap (~1 year of 6-hourly slots)
            break

    for slot_dt in all_slots:
        preds = _predict_next(buffer, slot_dt, bundle)
        new_row = {'aqi': preds['aqi'], 'pm25': preds['pm25'], 'pm10': preds['pm10'],
                   'no2': preds['no2'], 'so2': preds['so2'], 'co': preds['co'], 'o3': preds['o3'],
                   'temperature': last_temp, 'humidity': last_hum}
        buffer.append(new_row)
        if slot_dt in target_slots:
            cat, color = aqi_category(preds['aqi'])
            results.append({
                'time': slot_dt.strftime('%H:%M'),
                'datetime': slot_dt.strftime('%Y-%m-%d %H:%M'),
                'aqi': round(preds['aqi'], 1),
                'pm25': round(preds['pm25'], 1),
                'pm10': round(preds['pm10'], 1),
                'no2': round(preds['no2'], 1),
                'so2': round(preds['so2'], 1),
                'co': round(preds['co'], 2),
                'o3': round(preds['o3'], 1),
                'temperature': round(float(new_row['temperature']), 1),
                'humidity': round(float(new_row['humidity']), 1),
                'category': cat,
                'color': color,
            })

    return results, 'forecast'
