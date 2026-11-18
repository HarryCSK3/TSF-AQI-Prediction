"""
Combine walk-forward validation results from all 10 models into ONE comparison
plot + metrics comparison charts. Aggregated over >=20 folds per model.
Run this yourself: `python 06_compare_models.py` (after running scripts 03-05)
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOTS = 'plots'

with open('models/statistical_results.pkl', 'rb') as f:
    stat = pickle.load(f)
with open('models/ml_results.pkl', 'rb') as f:
    ml = pickle.load(f)
with open('models/dl_results.pkl', 'rb') as f:
    dl = pickle.load(f)

# Statistical, ML and DL scripts all use the SAME fold boundaries (walk_forward.py),
# so their concatenated fold-validation predictions line up on the same datetimes.
pred_datetimes = pd.to_datetime(stat['pred_datetimes'])
actuals = stat['actuals']

all_predictions = {}
all_predictions.update(stat['predictions'])
all_predictions.update(ml['predictions'])
all_predictions.update(dl['predictions'])

all_summary = {}
all_summary.update(stat['summary'])
all_summary.update(ml['summary'])
all_summary.update(dl['summary'])

colors = {
    'ARIMA': '#e74c3c', 'SARIMA': '#e67e22', 'SARIMAX': '#f1c40f',
    'Holt-Winters': '#2ecc71', 'ETS': '#1abc9c',
    'XGBoost': '#3498db', 'LightGBM': '#2980b9', 'Random Forest': '#9b59b6',
    'LSTM': '#34495e', 'GRU': '#7f8c8d'
}

# ---------------- Combined single plot: Actual vs ALL models (all folds concatenated) ----------------
plt.figure(figsize=(18, 9))
plt.plot(pred_datetimes, actuals, label='Actual AQI', color='black', linewidth=2.2, zorder=10)
for name, pred in all_predictions.items():
    plt.plot(pred_datetimes, pred, label=name, color=colors.get(name), linewidth=1.1, alpha=0.8)
plt.title('Anand Vihar - Actual vs Predicted AQI: All 10 Models (concatenated walk-forward validation folds)')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend(loc='upper left', ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig(f'{PLOTS}/08_all_models_combined.png', dpi=120)
plt.close()

# ---------------- Metrics comparison (mean across folds, with std as error bars) ----------------
summary_df = pd.DataFrame(all_summary).T
summary_df = summary_df.sort_values('RMSE_mean')
summary_df.to_csv('models/all_models_comparison.csv')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for ax, metric in zip(axes.flat, ['MAE', 'RMSE', 'MAPE', 'MSE']):
    sorted_df = summary_df.sort_values(f'{metric}_mean')
    bar_colors = [colors.get(idx, '#95a5a6') for idx in sorted_df.index]
    ax.barh(sorted_df.index, sorted_df[f'{metric}_mean'],
            xerr=sorted_df[f'{metric}_std'], color=bar_colors, capsize=3)
    ax.set_title(f'{metric} (mean \u00b1 std over folds, lower = better)')
    ax.invert_yaxis()
plt.suptitle('Model Comparison Across Error Metrics (Walk-Forward Validation)', fontsize=15)
plt.tight_layout()
plt.savefig(f'{PLOTS}/09_metrics_comparison.png', dpi=120)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
time_df = summary_df.sort_values('AvgTrainTime')
axes[0].barh(time_df.index, time_df['AvgTrainTime'], color=[colors.get(i, '#95a5a6') for i in time_df.index])
axes[0].set_title('Average Training Time per Fold (seconds)')
axes[0].invert_yaxis()
time_df2 = summary_df.sort_values('AvgPredTime')
axes[1].barh(time_df2.index, time_df2['AvgPredTime'] * 1000, color=[colors.get(i, '#95a5a6') for i in time_df2.index])
axes[1].set_title('Average Prediction Time per Fold (milliseconds)')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(f'{PLOTS}/10_time_comparison.png', dpi=120)
plt.close()

best_model = summary_df.index[0]
with open('models/best_model_name.txt', 'w') as f:
    f.write(best_model)

print("Comparison complete.")
print("Combined plot: plots/08_all_models_combined.png")
print("Metrics comparison: plots/09_metrics_comparison.png")
print("Full table: models/all_models_comparison.csv")
