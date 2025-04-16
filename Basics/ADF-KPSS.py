import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# Define symbols and date range
symbols = ['^HSI', '^KS11', '^GSPC', '^N225']
start_date = '1991-01-01'
end_date = '2024-12-31'

data_dict = {}
results = []
for sym in symbols:
    df = yf.download(sym, start=start_date, end=end_date)
    series = df['Close'].dropna() # Adj_Close is no longer use by yfinancne
    data_dict[sym] = series

    # Augmented Dickey-Fuller (ADF) test
    adf_result = adfuller(series)
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]
    
    # KPSS test (using regression='c' for constant)
    kpss_result = kpss(series, regression='c', nlags="auto") 
    kpss_stat = kpss_result[0]
    kpss_pvalue = kpss_result[1]

    results.append({
        'Symbol': sym,
        'ADF Statistic': adf_stat,
        'ADF p-value': adf_pvalue,
        'KPSS Statistic': kpss_stat,
        'KPSS p-value': kpss_pvalue
    })

result_df = pd.DataFrame(results)
print("Stationarity Test Results:")
print(result_df)

adf_pvalues = [r['ADF p-value'] for r in results]
kpss_pvalues = [r['KPSS p-value'] for r in results]

# Cumulative Returns (for demonstration of data)
fig, ax = plt.subplots(figsize=(15, 10))
colors = plt.cm.tab20.colors

for i, sym in enumerate(symbols):
    series = data_dict[sym]
    cumulative_returns = (series / series.iloc[0]) - 1
    ax.plot(series.index, cumulative_returns, label=sym, color=colors[i])

ax.set_title("Cumulative Returns of Indices")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend()
plt.tight_layout()
plt.savefig('Basics/Visualizations/ADF-KPSS-cumulative_returns_combined.png')
plt.show()

# P values
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(symbols))
width = 0.35

ax.bar([i - width/2 for i in x], adf_pvalues, width, label='ADF p-value')
ax.bar([i + width/2 for i in x], kpss_pvalues, width, label='KPSS p-value')
ax.set_xticks(x)
ax.set_xticklabels(symbols)
ax.set_ylabel('p-value')
ax.set_title('Stationarity Test p-values Comparison')
ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
ax.axhline(y=0.1, color='blue', linestyle='--', label='p=0.1')
ax.legend()
plt.savefig('Basics/Visualizations/ADF-KPSS-p_values_comparison.png')
plt.show()

# Test Statistics
fig, ax = plt.subplots(figsize=(10, 6))
adf_stats = [r['ADF Statistic'] for r in results]
kpss_stats = [r['KPSS Statistic'] for r in results]

ax.bar([i - width/2 for i in x], adf_stats, width, label='ADF Statistic')
ax.bar([i + width/2 for i in x], kpss_stats, width, label='KPSS Statistic')
ax.set_xticks(x)
ax.set_xticklabels(symbols)
ax.set_ylabel('Test Statistic')
ax.set_title('Stationarity Test Statistics Comparison')
ax.axhline(y=0.0, color='black', linestyle='--', label='Zero Line')
ax.legend()
plt.savefig('Basics/Visualizations/ADF-KPSS-test_statistics_comparison.png')
plt.show()

# Log Difference and Stationarity Test
log_diff_results = []
for sym in symbols:
    series = data_dict[sym]
    log_diff_series = series.apply(lambda x: np.log(x)).diff().dropna()
    
    adf_result = adfuller(log_diff_series)
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]

    kpss_result = kpss(log_diff_series, regression='c', nlags="auto")
    kpss_stat = kpss_result[0]
    kpss_pvalue = kpss_result[1]

    log_diff_results.append({
        'Symbol': sym,
        'ADF Statistic': adf_stat,
        'ADF p-value': adf_pvalue,
        'KPSS Statistic': kpss_stat,
        'KPSS p-value': kpss_pvalue
    })

log_diff_result_df = pd.DataFrame(log_diff_results)
print("Log Difference Stationarity Test Results:")
print(log_diff_result_df)

# Log Difference
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs = axs.flatten()
for i, sym in enumerate(symbols):
    log_diff_series = data_dict[sym].apply(lambda x: np.log(x)).diff().dropna()
    axs[i].plot(log_diff_series.index, log_diff_series, label=sym, color=colors[i])
    axs[i].set_title(f"Log Difference of {sym}")
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("Log Difference")
    axs[i].legend()

plt.tight_layout()
plt.savefig('Basics/Visualizations/ADF-KPSS-log_difference.png')
plt.show()
