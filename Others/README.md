# Capstone

# 1. Data Collection
### Interaction Effect
- Themed Characteristics (Working Days) (KMeans + TF-IDF on LongBusinessSummary -> Mean Change in %p)

### Analyst Effect
The opinion of exploratory analysis on stock price will affect traders on decision making.
- ~~EDA opinion~~ (Working Days) (Not finished)
- Institutional outlook (Working Days) (Bloomberg opinon)

### Exposure Effect
- Google Trends (Daily) (US + Local)
- ~~Reddit Mentions (Daily)~~ (US + Local(?)) 

### Indirect Effect
- Weather Data (Daily) (Longitudinal, Latitudinal coordinate)
- Aging Population (Annual) (percentage over 65)
- Cryptocurrency Price & Volume (Daily)

### Direct Effect
- US Interest Rate Changes + Announcement date (Working Days) (Sparse)
- Connected Market (Working Days)

### Holiday Effect and 
- Local Holidays (Dates) (Sparse)
- Non Local Holidays (Dates) (Sparse)
- Lunar Calendar (Dates) (Sparse)

# 2. Data Processing
a) Identify the target comparison data (ticker/market)

b) Reorganize the ticker/market price by same timestamp (Dates/Working Days/Daily/Annual)

c) Evaluate with the following metrics
### Causality
- Granger Causality (p-value)
### Correlation
- Cross-Correlation Function (lagged)
- Canonical Coherence Analysis (time-frequency)
### Similarity
- Dynamic Time Wrapping (for asynchronized)
### Nonlinearity
- Mutual Information
- Distance Correlation
### Information transmission
- Transfer Entropy
### Cointegration
- Engle-Granger (Residual)
- Johansen Test (Covariance)
### Decomposition
- Empirical Mode Decomposition (EMD) + Correlation
- Feature Importance

d) display the results in dataframe. 


# Brief Explanations

## Themed Characteristics
Purpose : Identify the relationship between theme/sector trend and individual ticker
Steps : 
1. Identify n cluster of the sample tickers (from sample.py)


2. Calculate mean(ticker['adj_closed'].pct_change()). 

3. Preprocess in the way that there is Date in index, avgclosedchanges of a designated cluster (will have n separate dataframes)

```
processed['themed_characteristics_time_series'] = themed_list
```

4. Prepare comparison data:
```
comparison['themed_characteristics_time_series'] = list[n][len(processed['themed_characteristics_time_series'][n])]
```

5. Analysis:
    for n in range(len(processed['themed_characteristics_time_series'])):
        for m in range (comparison[n])
            for evaluation metrics:
                mean(evaluationmetric(processed['themed_characteristics_time_series'][n], comparison['themed_characteristics_time_series'][n][m]))

# Indirect Effect (weather)
Purpose : Examine indirect and seemingly unrelated items to set baseline of our investigation.

# Holiday Effect (Powell's announcement, Nonlocal holidays, local holidays, lunar calendar)
Purpose : Examine the holiday effect related impacts onto market indicies. (for powell's announcement, it also has the change in interest rate)



# Metrics

Causation : 
- Standard Granger Causality (for Standard Causality) / Toda-Yamamoto Granger Causality (for nonstationary Causality) / Diks-Panchenko Nonlinear Causality (for nonlinear Causality)
    *Toda Yamamoto with regularization to handle colinearity, used t-statistics.

Correlation : 
- Cross Correlation Function and Lagged Correlation
    1. Calculate CCF and normalize. 
    3. Find maximum correlation w.r.t window size defined by lag. 
    4. Get CCF correlation and lagged correlation respectively
    *p values were calculated in t-statistics. 
    *min_points = 30, returns np.nan if below.
    *max_lag_ratio = 0.2, uses 20% of the sample legth.
    returns ccf_correlation, ccf_lag, ccf_p_value, lagged_correlation, lagged_p_value
