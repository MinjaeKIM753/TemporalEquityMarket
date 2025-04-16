import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

start_date = "1991-01-01"
end_date = "2024-12-31"

symbols = ['^HSI', '^KS11', '^GSPC', '^N225']

max_lags = 120  # 10years

fig, axs = plt.subplots(nrows=len(symbols), ncols=2, figsize=(12, 16), sharex=False)

for i, sym in enumerate(symbols):
    df_d = yf.download(sym, start=start_date, end=end_date)
    
    # MultiIndex - Cannot dropna with MultiIndex
    if isinstance(df_d.columns, pd.MultiIndex):
        close_col = ('Close', sym)
        df_d = df_d[close_col].to_frame()
        df_d.columns = ['Close']
    
    df_d.dropna(subset=["Close"], inplace=True)

    df_m = df_d.resample('M').last()
    df_m.dropna(subset=["Close"], inplace=True)

    df_m["LogClose"] = np.log(df_m["Close"])

    n_obs = len(df_m["LogClose"])
    max_allowed_lags = min(max_lags, n_obs // 2)

    # Plot ACF
    plot_acf(df_m["LogClose"], ax=axs[i, 0], lags=max_allowed_lags, 
            title=f"{sym} - ACF (Log Monthly Close)")
    axs[i, 0].set_xlabel("Lag (months)")
    axs[i, 0].set_ylabel("Autocorrelation")

    # Plot PACF
    plot_pacf(df_m["LogClose"], ax=axs[i, 1], lags=max_allowed_lags, 
            title=f"{sym} - PACF (Log Monthly Close)")
    axs[i, 1].set_xlabel("Lag (months)")
    axs[i, 1].set_ylabel("Partial Autocorrelation")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Basics/Visualizations/ACF-PACF-LogMonthlyClose.png')
plt.close()
