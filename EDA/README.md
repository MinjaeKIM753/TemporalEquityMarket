# Exploratory Data Analysis (EDA) for Financial Time Series Strategies

This folder contains Python scripts designed to perform exploratory data analysis on various technical indicator-based trading strategies applied to major stock market indices.

## Overview

The primary goal of this EDA is to evaluate the historical performance of common trading strategies derived from technical indicators. The analysis involves fetching historical market data, calculating indicators, generating trading signals, backtesting the strategies, evaluating their performance, and visualizing the results.

## Key Scripts

- **`FinancialAnalyzer.py`**: A core class responsible for:
    - Fetching historical financial data using the `yfinance` library.
    - Calculating a wide range of technical indicators (Moving Averages, RSI, MACD, Bollinger Bands, Stochastic Oscillator, Ichimoku Cloud, OBV, CMF, Force Index, Parabolic SAR, etc.).
    - Original code can be fetched from MINJAEKIM753/FinancialAnalyzer.
- **`EDA.py`**: The main script that orchestrates the EDA process:
    - Defines global settings (tickers, date range).
    - Utilizes `FinancialAnalyzer` to get data and indicators.
    - Implements functions to generate trading signals based on indicator values.
    - Defines helper functions for performance calculation (annualized return, volatility, Sharpe ratio, max drawdown) and plotting.
    - Backtests each individual strategy.
    - Identifies the best-performing subset of strategies using a combination approach (majority vote).
    - Generates and saves various comparative plots to the `EDA/Visualizations/` subfolder.

## Workflow

1.  **Data Fetching**: Downloads daily historical closing prices and volume for specified indices (`^HSI`, `^KS11`, `^GSPC`, `^N225`) for the period 1991-01-01 to 2024-12-31.
2.  **Indicator Calculation**: Computes numerous technical indicators using `FinancialAnalyzer.py`.
3.  **Signal Generation**: Translates indicator values into simple binary (long/flat) trading signals using predefined rules (e.g., MA crossover, RSI threshold, MACD line vs. signal line).
4.  **Backtesting**: Simulates the application of each strategy, calculating daily returns based on the generated signals (assuming trades occur based on the previous day's signal).
5.  **Performance Evaluation**: Calculates key performance metrics for each individual strategy.
6.  **Strategy Combination**: Explores combinations of strategies to find a subset that yields superior performance compared to individual strategies (using a majority vote signal).
7.  **Visualization**: Creates plots comparing the performance of different strategies, including cumulative returns, rolling performance metrics (returns, Sharpe ratio), and signal persistence.

## Strategies Evaluated

The analysis evaluates strategies based on the following indicators:

- Moving Average (MA) Crossover (50-day vs 200-day)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Trend Following (Close vs 200-day SMA)
- Stochastic Oscillator
- Ichimoku Cloud
- On-Balance Volume (OBV)
- Chaikin Money Flow (CMF)
- Force Index
- Parabolic SAR
- Disparity Index
- TRIX
- Three Line Break

## Generated Plots (`EDA/Visualizations/`)

All generated plots are saved in the `EDA/Visualizations/` subfolder. They include:

- `*-StrategyCumulativeReturns.png`: Comparison of cumulative returns for all individual strategies and a Buy & Hold baseline.
- `*-CombinedStrategyWithBest.png`: Comparison including the best-performing combined strategy.
- `*-RollingCumulativeReturns.png`: Rolling window (252-day) cumulative returns for strategies.
- `*-RollingSharpeRatios.png`: Rolling window (252-day) Sharpe ratios for strategies.
- `*-CombinedSignalPersistence.png`: Analysis of average returns following a buy signal over different holding periods (horizons).

## How to Run

Execute the main script to perform the analysis and generate the plots:

```bash
python EDA/EDA.py
``` 