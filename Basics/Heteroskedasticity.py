import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model

tickers = ["^HSI", "^KS11", "^GSPC", "^N225"]
start_date = "1991-01-01"
end_date = "2024-12-31"

data = yf.download(tickers, start=start_date, end=end_date)["Close"]
data.dropna(inplace=True)

returns = np.log(data / data.shift(1)).dropna()

# ARCH Testing
arch_pvalues = {}
for ticker in tickers:
    test_stat, p_value, _, _ = het_arch(returns[ticker])
    arch_pvalues[ticker] = p_value
    print(f"{ticker}: ARCH test p-value = {p_value:.4f}")

# Rolling Volatility
window = 252  # 1yr
plt.figure(figsize=(10, 6))
for ticker in tickers:
    rolling_vol = returns[ticker].rolling(window=window).std() * np.sqrt(window)
    plt.plot(rolling_vol.index, rolling_vol, label=ticker)
plt.title("252-Day Rolling Annualized Volatility (Rolling Window = 252)")
plt.xlabel("Date")
plt.ylabel("Annualized Volatility")
plt.legend()
plt.savefig('Basics/Visualizations/Rolling-volatility.png')
plt.close()

# GARCH(1,1) Conditional Volatility
fig, axs = plt.subplots(nrows=len(tickers), ncols=1, figsize=(12, 24), sharex=True)
fig.suptitle("GARCH(1,1) Conditional Volatility", fontsize=16)

for i, ticker in enumerate(tickers):
    ret = returns[ticker] * 100
    model = arch_model(ret, vol='Garch', p=1, q=1, dist='Normal')
    res = model.fit(disp='off')
    axs[i].plot(res.conditional_volatility.index, res.conditional_volatility, label=f'{ticker} Conditional Volatility')
    axs[i].set_title(f"{ticker} GARCH(1,1) Conditional Volatility")
    axs[i].set_ylabel("Volatility")
    axs[i].legend()

plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Basics/Visualizations/GARCH-conditional-volatility-stacked.png')
plt.close()
