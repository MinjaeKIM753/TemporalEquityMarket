# Economic Policy Uncertainty Analysis

This directory contains scripts and data for analyzing the impact of various economic policy uncertainty indices on stock market returns and volatility.

## Overview

The main script (`uncertainty.py`) performs the following steps:
1.  Loads various economic policy uncertainty indices (EPU, TPU, CPU, UCT) for different regions (Global, HK, KR, US, JP, CN).
2.  Downloads corresponding stock market index data (HSI, KOSPI, S&P 500 via SPY, Nikkei 225) using `yfinance`.
3.  Calculates monthly log returns and rolling volatility for the market indices.
4.  Merges the uncertainty indices and market data based on monthly frequency.
5.  Performs several analyses for each market:
    - **Rolling Window Analysis**: Visualizes the rolling mean of uncertainty indices and market volatility.
    - **Regression Analysis (OLS)**: Investigates the linear relationship between market returns (dependent variable) and various uncertainty indices (independent variables).
    - **Granger Causality Test**: Tests whether uncertainty indices can predict future market returns.
6.  Generates and saves visualizations for each analysis step.

## Data (`Policy/datasets/`)

The analysis utilizes several economic policy uncertainty indices. The raw data files are stored in the `Policy/datasets/` subdirectory.

**Primary Data Source:** All indices are obtained from the **Economic Policy Uncertainty** website: [https://www.policyuncertainty.com/index.html](https://www.policyuncertainty.com/index.html)

Specific indices and their citations:

- **`HK EPU.xlsx`**: Hong Kong Economic Policy Uncertainty Index
    - *Source*: [https://www.policyuncertainty.com/hk_monthly.html](https://www.policyuncertainty.com/hk_monthly.html)
    - *Citation*: Luk, P., Cheng, M., Ng, P., & Wong, K. (n.d.). *Hong Kong Economic Policy Uncertainty Index*. Economic Policy Uncertainty.

- **`KR EPU.xls`**: South Korea Economic Policy Uncertainty Index
    - *Source*: [https://www.policyuncertainty.com/korea_monthly.html](https://www.policyuncertainty.com/korea_monthly.html)
    - *Citation*: Baker, S. R., Bloom, N., & Davis, S. J. (n.d.). *South Korea Economic Policy Uncertainty Index*. Economic Policy Uncertainty.

- **`US EPU.xlsx`**: United States Economic Policy Uncertainty Index
    - *Source*: [https://www.policyuncertainty.com/us_monthly.html](https://www.policyuncertainty.com/us_monthly.html)
    - *Citation*: Baker, S. R., Bloom, N., & Davis, S. J. (n.d.). *U.S. Economic Policy Uncertainty Index*. Economic Policy Uncertainty.

- **`JP EPU.xlsx`**: Japan Economic Policy Uncertainty Index
    - *Source*: [https://www.policyuncertainty.com/japan_monthly.html](https://www.policyuncertainty.com/japan_monthly.html)
    - *Citation*: Arbatli, E. C., Davis, S. J., Ito, A., Miake, N., & Saito, I. (n.d.). *Japan Economic Policy Uncertainty Index*. Economic Policy Uncertainty.

- **`Global EPU.xlsx`**: Global Economic Policy Uncertainty Index
    - *Source*: [https://www.policyuncertainty.com/global_monthly.html](https://www.policyuncertainty.com/global_monthly.html)
    - *Citation*: Davis, S. J. (n.d.). *Global Economic Policy Uncertainty Index*. Economic Policy Uncertainty. (Based on Davis, S. J. (2016). *An Index of Global Economic Policy Uncertainty*. NBER Working Paper No. 22740).

- **`CN TPU.xlsx`**: China Trade Policy Uncertainty Index
    - *Source*: [https://www.policyuncertainty.com/china_monthly.html](https://www.policyuncertainty.com/china_monthly.html)
    - *Citation*: Davis, S. J., Liu, D., & Sheng, X. S. (n.d.). *China Trade Policy Uncertainty Index*. Economic Policy Uncertainty. (Based on Davis, S. J., Liu, D., & Sheng, X. S. (2019). *Economic Policy Uncertainty in China Since 1949: The View from Mainland Newspapers*).

- **`US TPU.xlsx`**: United States Trade Policy Uncertainty Index
    - *Source*: [https://www.policyuncertainty.com/trade_uncertainty.html](https://www.policyuncertainty.com/trade_uncertainty.html)
    - *Citation*: Caldara, D., & Iacoviello, M. (n.d.). *U.S. Trade Policy Uncertainty Index*. Economic Policy Uncertainty. (Methodology may vary, check source for specific index details).

- **`CPU index.csv`**: Climate Policy Uncertainty Index
    - *Source*: [https://www.policyuncertainty.com/climate_uncertainty.html](https://www.policyuncertainty.com/climate_uncertainty.html)
    - *Citation*: Gavriilidis, K. (n.d.). *Climate Policy Uncertainty Index*. Economic Policy Uncertainty. (Based on Gavriilidis, K. (2021). *Measuring Climate Policy Uncertainty*. Working Paper).

- **`UCT.csv`**: US-China Geopolitical Tension Index
    - *Source*: [https://www.policyuncertainty.com/US_China_Tension.html](https://www.policyuncertainty.com/US_China_Tension.html)
    - *Citation*: Davis, S. J., Liu, D., & Sheng, X. S. (n.d.). *U.S.-China Geopolitical Tension Index*. Economic Policy Uncertainty. (Based on Davis, S. J., Liu, D., & Sheng, X. S. (2021). *An Index of U.S.-China Policy Tension*. Working Paper).

## Script (`Policy/uncertainty.py`)

This script orchestrates the data loading, processing, analysis, and visualization.

- **Key Libraries**: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `yfinance`.
- **Input**: Reads data files from `Policy/datasets/`.
- **Output**: Saves generated plots to `Policy/Visualizations/`.

## Visualizations (`Policy/Visualizations/`)

The script generates the following plots, saved in the `Policy/Visualizations/` subdirectory:

- `[Country]_Rolling_Window.png`: Shows rolling means of uncertainty indices and market volatility for each country (HK, KR, US, JP).
- `[Country]_Regression_Summary.png`: Displays OLS regression coefficients and model fit statistics for each country.
- `[Country]_Granger_Test_Results.png`: Shows p-values from the Granger Causality test for each uncertainty index predicting market returns, across different lags.
- `Combined_Regression_Results.png`: Compares regression coefficients across all analyzed countries.
- `Combined_Granger_Results.png`: Compares Granger causality p-values across all analyzed countries.

## How to Run

Ensure the required libraries are installed:

```bash
pip install pandas numpy matplotlib statsmodels yfinance openpyxl
```

Then, execute the main script from the root directory of the project:

```bash
python Policy/uncertainty.py
```

The script will load the data, perform the analyses, print summaries (like regression results) to the console, and save the plots in `Policy/Visualizations/`. 