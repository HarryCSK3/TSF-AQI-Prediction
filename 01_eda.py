"""
EDA for Anand Vihar AQI dataset
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams['figure.figsize'] = (14, 6)
sns.set_style('whitegrid')

df = pd.read_csv('./data/anand_vihar_raw.csv', parse_dates=['datetime'])
df = df.set_index('datetime')

PLOTS = './plots'

# 1. AQI trend over time
plt.figure()
plt.plot(df.index, df['aqi'], color='#c0392b', linewidth=0.6)
plt.title('Anand Vihar - AQI Trend (2020-2025)')
plt.xlabel('Date'); plt.ylabel('AQI')
plt.tight_layout()
plt.savefig(f'{PLOTS}/01_aqi_trend.png', dpi=110)
plt.close()

# 2. AQI distribution
plt.figure()
sns.histplot(df['aqi'], bins=50, kde=True, color='#2980b9')
plt.title('AQI Distribution')
plt.tight_layout()
plt.savefig(f'{PLOTS}/02_aqi_distribution.png', dpi=110)
plt.close()

# 3. Monthly / seasonal variation
plt.figure()
monthly = df.groupby('month')['aqi'].mean()
sns.barplot(x=monthly.index, y=monthly.values, color='#e67e22')
plt.title('Average AQI by Month')
plt.xlabel('Month'); plt.ylabel('Average AQI')
plt.tight_layout()
plt.savefig(f'{PLOTS}/03_monthly_aqi.png', dpi=110)
plt.close()

plt.figure()
season_order = ['winter','summer','monsoon','post_monsoon']
season_avg = df.groupby('season')['aqi'].mean()
season_avg = season_avg.reindex([s for s in season_order if s in season_avg.index])
sns.barplot(x=season_avg.index, y=season_avg.values, color='#8e44ad')
plt.title('Average AQI by Season')
plt.tight_layout()
plt.savefig(f'{PLOTS}/04_seasonal_aqi.png', dpi=110)
plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(10, 8))
cols = ['aqi','pm25','pm10','no2','so2','co','o3','temperature','humidity','wind_speed','visibility']
corr = df[cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(f'{PLOTS}/05_correlation_heatmap.png', dpi=110)
plt.close()

# 5. Outlier detection (boxplot)
plt.figure()
sns.boxplot(x=df['aqi'], color='#16a085')
plt.title('AQI Outlier Detection (Boxplot)')
plt.tight_layout()
plt.savefig(f'{PLOTS}/06_aqi_boxplot.png', dpi=110)
plt.close()

# 6. Seasonal decomposition (using daily resample of AQI)
daily = df['aqi'].resample('D').mean().interpolate()
decomp = seasonal_decompose(daily, model='additive', period=365)
fig = decomp.plot()
fig.set_size_inches(14, 9)
plt.tight_layout()
plt.savefig(f'{PLOTS}/07_seasonal_decomposition.png', dpi=110)
plt.close()

# 7. Stationarity tests
adf_result = adfuller(df['aqi'].dropna())
kpss_result = kpss(df['aqi'].dropna(), regression='c', nlags='auto')

report = f"""
STATIONARITY ANALYSIS - Anand Vihar AQI (6-hour interval series)
==================================================================
ADF Test:
  ADF Statistic : {adf_result[0]:.4f}
  p-value       : {adf_result[1]:.6f}
  Critical Values: {adf_result[4]}
  Conclusion    : {"Stationary (reject H0)" if adf_result[1] < 0.05 else "Non-Stationary (fail to reject H0)"}

KPSS Test:
  KPSS Statistic : {kpss_result[0]:.4f}
  p-value        : {kpss_result[1]:.6f}
  Critical Values: {kpss_result[3]}
  Conclusion     : {"Stationary" if kpss_result[1] > 0.05 else "Non-Stationary"}

Basic Statistics:
{df['aqi'].describe().to_string()}

Outlier count (IQR method): {(( (df['aqi'] < (df['aqi'].quantile(.25) - 1.5*(df['aqi'].quantile(.75)-df['aqi'].quantile(.25)))) | (df['aqi'] > (df['aqi'].quantile(.75) + 1.5*(df['aqi'].quantile(.75)-df['aqi'].quantile(.25)))) )).sum()}
"""
print(report)
with open('./plots/stationarity_report.txt', 'w') as f:
    f.write(report)

print("EDA complete. Plots saved to:", PLOTS)
