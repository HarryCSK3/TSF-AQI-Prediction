"""
Shared walk-forward (expanding-window) validation utility.
Every model is validated the same way: N_SPLITS folds, each with an expanding
training window and a fixed-size validation window, walking forward through
time so validation never sees the future relative to its training data.
"""
import numpy as np

N_SPLITS = 20         # minimum 20 iterations per model, as required
VAL_SIZE = 4            # 4 steps = 1 day at 6-hour resolution (matches the forecast horizon)


def get_fold_boundaries(n_total, n_splits=N_SPLITS, val_size=VAL_SIZE):
    """
    Returns a list of (train_end, val_start, val_end) index tuples.
    Expanding window: train always starts at 0 and grows; validation is the
    next val_size points immediately after train, walking forward to the end
    of the series so the final fold's validation ends at the most recent data.
    """
    n_val_total = n_splits * val_size
    min_train_size = n_total - n_val_total
    if min_train_size < val_size * 4:
        raise ValueError("Not enough data for the requested number of splits/val_size.")

    folds = []
    for i in range(n_splits):
        train_end = min_train_size + i * val_size
        val_start = train_end
        val_end = train_end + val_size
        folds.append((train_end, val_start, val_end))
    return folds


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}


def summarize_folds(fold_metrics_list):
    """fold_metrics_list: list of dicts (one per fold) -> returns mean/std summary dict."""
    keys = fold_metrics_list[0].keys()
    summary = {}
    for k in keys:
        vals = [f[k] for f in fold_metrics_list]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std'] = float(np.std(vals))
    return summary
