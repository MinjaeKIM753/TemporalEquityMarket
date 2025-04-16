# Temporal Equity Market Analysis

This project explores various aspects of financial time series analysis, focusing on major equity markets (primarily Hong Kong (Hang Seng Index), South Korea (KOSPI Composite), with comparison to USA (S&P500), Japan (Nikkei Daily)) and investigating the influence of technical indicators, macroeconomic events, policy uncertainty, and other external factors.

## Project Goal

The primary goal is to apply diverse quantitative methods to understand market behavior, evaluate trading strategies, and assess the impact of external information (news, policy) on market dynamics.

## Directory Structure & Contents

The project is organized into the following main directories:

```
TemporalEquityMarket/
├── Basics/
├── EDA/
├── Macro/
│   └── Events/
├── Others/
├── Policy/
└── README.md  (This file)
```

### `Basics/`

-   **Aim**: To understand the fundamental statistical properties of major financial market indices.
-   **Approach**: Implements foundational time series analyses including:
    -   Plotting **Autocorrelation and Partial Autocorrelation Functions (ACF/PACF)** to identify potential model orders (`ACF.py`).
    -   Performing **Augmented Dickey-Fuller (ADF)** and **Kwiatkowski–Phillips–Schmidt–Shin (KPSS)** tests for stationarity (`ADF-KPSS.py`).
    -   Analyzing heteroskedasticity using **GARCH models** and rolling volatility (`Heteroskedasticity.py`).
    -   Calculating and plotting the **Periodogram** to identify dominant cyclical frequencies (`Periodogoram.py`).
-   **Contents**: Python scripts for each analysis, generated visualizations (`Visualizations/`), and a detailed `README.md`.

### `EDA/`

-   **Aim**: To perform Exploratory Data Analysis (EDA) on technical indicator-based trading strategies and evaluate their historical performance.
-   **Approach**: Uses a custom `FinancialAnalyzer` class (`FinancialAnalyzer.py`) to calculate various **technical indicators (MA Crossover, RSI, MACD, Bollinger Bands, Stochastic, Ichimoku, OBV, CMF, Force Index, Parabolic SAR, etc.)**. The main `EDA.py` script generates trading signals, performs **backtesting**, calculates performance metrics (**Sharpe Ratio, Max Drawdown**), identifies optimal combinations using **majority voting**, and conducts **signal persistence analysis**.
-   **Contents**: Core analysis scripts, helper utilities (`utils/`), evaluation code (`evaluate/`), potentially data handling (`data/`), generated plots (`Visualizations/`), and a detailed `README.md`.

### `Macro/Events/`

-   **Aim**: To investigate the impact of macroeconomic events on financial markets by fetching events, validating them, and analyzing their sentiment.
-   **Approach**: Leverages the **OpenAI API (e.g., `gpt-4o-mini`)** to fetch (`events.py`) and optionally validate (`validator.py`) historical macroeconomic events based on detailed prompts. Performs sentiment analysis on event descriptions using Transformer models (**`ProsusAI/finbert`**, **`cardiffnlp/twitter-roberta-base-sentiment`**) via `sentiment.py`. This script also correlates sentiment scores with market returns (**Pearson Correlation**, **Cross-Correlation Function - CCF**, **Granger Causality**), analyzes keyword impact (using **NLTK** for noun extraction and **TF-IDF**), and explores optimal sentiment weighting (using **Linear Regression**).
-   **Contents**: Python scripts for event fetching, sentiment analysis, validation, and post-processing. Contains directories for cached data (`Saves/`), processed API outputs (`Processed_files/`), generated plots (`Visualizations/`), and data files (`.jsonl`, `.pkl`) including prompts and API responses (see disclaimer in `Macro/Events/README.md`). Requires an `api_key.txt` file.

### `Others/`

-   **Aim**: To explore potential relationships between market movements (specifically Hong Kong and US markets) and a diverse set of alternative/external factors.
-   **Approach**: Loads various datasets (themed characteristics, weather, aging population, crypto, etc.) using `data/data.py` and `data/preprocessing.py`. Applies advanced evaluation metrics (`evaluate/metrics.py`, `evaluate/evaluate.py`) such as **distance correlation**, **mutual information**, **transfer entropy**, and **cointegration tests (e.g., Engle-Granger)**, alongside standard **correlation** and **causality tests**. Interprets these metrics using custom logic in `evaluate/analysis.py`.
-   **Contents**: Data handling scripts (`data/`), evaluation/analysis scripts (`evaluate/`), utility/settings files (`utils/`), a main script (`main.py`), and a `README.md`. Note: Focus is on HK/US, and some data sources/analyses might be experimental or not fully utilized in the current `main.py`.

### `Policy/`

-   **Aim**: To analyze the relationship between various economic policy uncertainty indices and stock market returns/volatility.
-   **Approach**: Loads EPU, TPU, CPU, and UCT indices (sourced from [policyuncertainty.com](https://www.policyuncertainty.com/)) and market index data (`uncertainty.py`). Performs **rolling window analysis**, **Ordinary Least Squares (OLS) regression**, and **Granger causality tests** to assess the impact of policy uncertainty on different markets (HK, KR, US, JP).
-   **Contents**: The core analysis script (`uncertainty.py`), datasets downloaded from the source (`datasets/`), generated plots (`Visualizations/`), and a detailed `README.md` including data citations.

## Data Sources

Key data sources used across the project include:

-   **Market Data**: Yahoo Finance (`yfinance` library).
-   **Economic Policy Uncertainty Indices**: Economic Policy Uncertainty website ([https://www.policyuncertainty.com/](https://www.policyuncertainty.com/)).
-   **Macroeconomic Events**: Generated via OpenAI API calls.
-   **Alternative Data (`Others/`)**: Sourced from various providers (details within `Others/data/`).

## Running the Analyses

Each subdirectory (`Basics`, `EDA`, `Macro/Events`, `Others`, `Policy`) contains its own `README.md` file with specific instructions on dependencies and how to run the respective analyses. Generally, you will need Python and libraries listed in the individual READMEs (`pip install ...`). For `Macro/Events`, an OpenAI API key stored in `Macro/Events/api_key.txt` is required.

## Disclaimer

-   This project is intended for educational and research purposes. 
-   The analyses and strategies presented do not constitute financial advice.
-   **All data used (except API keys) are disclosed in the directory, with outputs. (Timespan : 1990~2024)**
-   The `Macro/Events` directory contains prompts and data generated via API calls as part of a capstone project; this information is disclosed within the code and data files.
-   Ensure you have the necessary API keys and permissions before running scripts that interact with external services. 
-   Paper with analyses will be disclosed later. 