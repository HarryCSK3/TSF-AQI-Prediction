"""
Flask web application - Anand Vihar AQI Forecasting System
Pure Flask + Jinja templates + CSS. No JavaScript.
The user picks a date, submits a normal HTML form, and the page reloads
with the forecast rendered server-side (including a server-generated chart image).
"""
import os
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from flask import Flask, render_template, request

from forecast_engine import forecast_for_date

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
raw = pd.read_csv(os.path.join(BASE, 'data', 'anand_vihar_raw.csv'), parse_dates=['datetime'])
DATA_MIN_DATE = raw['datetime'].min().strftime('%Y-%m-%d')
DATA_MAX_DATE = raw['datetime'].max().strftime('%Y-%m-%d')
FORECAST_MAX_DATE = (raw['datetime'].max() + pd.Timedelta(days=90)).strftime('%Y-%m-%d')

try:
    model_comparison = pd.read_csv(os.path.join(BASE, 'models', 'all_models_comparison.csv'), index_col=0)
    display_df = pd.DataFrame({
        'MAE': model_comparison['MAE_mean'],
        'RMSE': model_comparison['RMSE_mean'],
        'MAPE': model_comparison['MAPE_mean'],
        'TrainTime': model_comparison['AvgTrainTime'],
    }).round(3)
    model_comparison_records = display_df.reset_index().rename(columns={'index': 'model'}).to_dict('records')
except Exception:
    model_comparison_records = []

# XGBoost is the model actually deployed: it ranks in the top tier on MAE/MAPE/RMSE among
# the 10 models compared, trains fast, and — unlike ETS/ARIMA/SARIMA which are univariate —
# it can be trained per-pollutant to power the multi-pollutant recursive forecast the app needs.
BEST_MODEL_NAME = 'XGBoost'


def make_chart_base64(readings):
    """Render the AQI-across-the-day line chart server-side and return it as a base64 PNG,
    so it can be embedded directly in the HTML with no JavaScript charting library."""
    times = [r['time'] for r in readings]
    values = [r['aqi'] for r in readings]
    colors = [r['color'] for r in readings]

    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=140)
    fig.patch.set_alpha(0)
    ax.set_facecolor('#161d24')
    ax.plot(times, values, color='#6fd0a8', linewidth=2.4, zorder=2)
    ax.scatter(times, values, color=colors, s=90, zorder=3, edgecolor='#0e1318', linewidth=1.2)
    for t, v in zip(times, values):
        ax.annotate(f'{v:.0f}', (t, v), textcoords="offset points", xytext=(0, 10),
                    ha='center', color='#e7ecef', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 520)
    ax.set_ylabel('AQI', color='#94a3ac')
    ax.tick_params(colors='#94a3ac')
    for spine in ax.spines.values():
        spine.set_color('#262f38')
    ax.grid(axis='y', color='#262f38', linewidth=0.6)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@app.route('/', methods=['GET'])
def index():
    date_str = request.args.get('date', DATA_MAX_DATE)

    context = {
        'data_min': DATA_MIN_DATE,
        'data_max': DATA_MAX_DATE,
        'forecast_max': FORECAST_MAX_DATE,
        'models': model_comparison_records,
        'best_model': BEST_MODEL_NAME,
        'station': 'Anand Vihar, Delhi',
        'selected_date': date_str,
        'error': None,
        'readings': None,
        'mode': None,
        'summary': None,
        'chart_image': None,
    }

    try:
        results, mode = forecast_for_date(date_str)
    except Exception as e:
        context['error'] = f'Something went wrong: {e}'
        return render_template('index.html', **context)

    if mode == 'unavailable':
        context['error'] = f'No data available before {DATA_MIN_DATE}.'
    elif mode == 'too_far':
        context['error'] = (f'Forecasts are supported up to {FORECAST_MAX_DATE} '
                             f'(90 days beyond the last recorded date {DATA_MAX_DATE}).')
    elif not results:
        context['error'] = 'No prediction could be generated for this date.'
    else:
        avg_aqi = round(sum(r['aqi'] for r in results) / len(results), 1)
        max_aqi = max(r['aqi'] for r in results)
        min_aqi = min(r['aqi'] for r in results)
        mid_category = results[len(results) // 2]['category']
        mid_color = results[len(results) // 2]['color']

        context.update({
            'readings': results,
            'mode': mode,
            'summary': {'avg_aqi': avg_aqi, 'max_aqi': max_aqi, 'min_aqi': min_aqi,
                        'category': mid_category, 'color': mid_color,
                        'gauge_pct': round(min(avg_aqi / 500, 1) * 100, 1)},
            'chart_image': make_chart_base64(results),
        })

    return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
