# Financial Time Series Analysis Basics

This folder contains Python scripts and generated plots for performing basic financial time series analysis tasks on major stock market indices.

**Common Settings:**
- **Markets Analyzed:** Hang Seng (^HSI), KOSPI (^KS11), S&P 500 (^GSPC), Nikkei 225 (^N225)
- **Data Period:** January 1, 1991 - December 31, 2024 (data fetched using `yfinance`)

## Python Scripts

- **`ACF.py`**:
    - Computes and plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for the log of monthly closing prices.
    - **Parameters:** Maximum lag (`max_lags`) set to 120 months (10 years).
    - Helps identify potential ARMA model orders.
- **`ADF-KPSS.py`**:
    - Performs Augmented Dickey-Fuller (ADF) and Kwiatkowski–Phillips–Schmidt–Shin (KPSS) tests to check for stationarity.
    - Analyzes both raw closing prices and their log differences.
    - **Parameters:** KPSS test uses automatic lag selection (`nlags="auto"`) and tests for stationarity around a constant (`regression='c'`).
    - Includes visualizations comparing test statistics and p-values across indices, and plots of cumulative returns and log differences.
- **`Heteroskedasticity.py`**:
    - Analyzes time-varying volatility (heteroskedasticity) using log returns.
    - Performs ARCH tests to detect autoregressive conditional heteroskedasticity.
    - Calculates and plots rolling annualized volatility.
    - Fits GARCH(1,1) models and plots the conditional volatility for each index.
    - **Parameters:** Rolling volatility window (`window`) set to 252 days (approx. 1 year). GARCH model uses p=1, q=1, and assumes a Normal distribution.
- **`Periodogoram.py`**:
    - Calculates and plots the periodogram of monthly percentage returns to identify dominant cyclical frequencies.
    - **Parameters:** Plots focus on periods between 0 and 10 years. Uses 'density' scaling for the periodogram. Includes a reference line at 3.5 years.

## Generated Plots (`.png`)

This folder also contains various plots generated by the scripts above, visualizing the results of the analyses for the specified financial indices and data transformations (e.g., monthly returns, log differences, log monthly close). **All plots are saved in the `Basics/Visualizations/` subfolder.** These include:

- `Visualizations/ACF-PACF-LogMonthlyClose.png`: ACF/PACF plots for log monthly closing prices.
- `Visualizations/ADF-KPSS-*.png`: Plots related to ADF and KPSS stationarity tests (cumulative returns, p-value comparison, statistic comparison, log difference series).
- `Visualizations/GARCH-conditional-volatility-*.png`: Plots showing GARCH(1,1) conditional volatility for individual indices and stacked together.
- `Visualizations/Rolling-volatility.png`: Plot of 252-day rolling annualized volatility.
- `Visualizations/Periodogram-monthly_returns.png`: Periodograms for monthly returns. 