import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import periodogram

start_date = "1991-01-01"
end_date = "2024-12-31"

symbols = ['^HSI', '^KS11', '^GSPC', '^N225']
monthly_returns = {}

for sym in symbols:
    df_daily = yf.download(sym, start=start_date, end=end_date)
    
    if isinstance(df_daily.columns, pd.MultiIndex):
        close_col = ('Close', sym)
        df_daily = df_daily[close_col].to_frame()
        df_daily.columns = ['Close']
    
    df_daily.dropna(subset=["Close"], inplace=True)
    df_monthly = df_daily["Close"].resample("ME").last().dropna()
    monthly_returns[sym] = df_monthly.pct_change().dropna()

# Periodogram
def get_periodogram(series):
    freqs, power = periodogram(series, scaling='density')
    return freqs, power

periodograms = {}
eps = 1e-12

for sym in symbols:
    freqs, power = get_periodogram(monthly_returns[sym])
    valid_idx = (freqs > eps)
    freqs_val = freqs[valid_idx]
    power_val = power[valid_idx]
    period = (1.0 / freqs_val) / 12.0
    periodograms[sym] = (period, power_val)

# Plot
fig, axs = plt.subplots(nrows=len(symbols), ncols=1, figsize=(16, 16), sharex=True)
fig.suptitle("Periodogram of Monthly Returns", fontsize=16)

colors = ["blue", "green", "orange", "purple"]

for i, sym in enumerate(symbols):
    period, power_val = periodograms[sym]
    axs[i].plot(period, power_val, label=sym, color=colors[i])
    axs[i].set_xlim(0, 10)
    axs[i].set_ylabel("Spectral Power")
    axs[i].set_title(f"{sym}: 0–10 Years")
    axs[i].grid(True, linestyle='--', alpha=0.5)
    axs[i].legend(loc="best")
    axs[i].axvline(x=3.5, color="red", linestyle="--", alpha=0.7, label="3.5-year ref") # Reference line
    axs[i].fill_betweenx(y=[0, max(power_val)], x1=0, x2=1, color='lightgrey', alpha=0.5) # Shaded area

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Basics/Visualizations/Periodogram-monthly_returns.png')
plt.close()
