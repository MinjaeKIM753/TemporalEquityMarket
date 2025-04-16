# evaluate/analysis.py
import pandas as pd


def analyze_correlation_max(value):
    value = float(value.iloc[0]) if hasattr(value, 'iloc') else float(value)
    if value > 0.7: return "strong linear relationship"
    elif value > 0.3: return "moderate linear relationship"
    else: return "weak linear relationship"

def analyze_correlation_lag(value):
    value = float(value.iloc[0]) if hasattr(value, 'iloc') else float(value)
    if value == 0: return "contemporaneous relationship"
    elif value > 0: return f"target leads comparison by {value} periods"
    else: return f"target lags comparison by {abs(value)} periods"

def analyze_correlation_distance(value):
    value = float(value.iloc[0]) if hasattr(value, 'iloc') else float(value)
    if value > 0.7: return "strong nonlinear relationship"
    elif value > 0.3: return "moderate nonlinear relationship"
    else: return "weak nonlinear relationship"

def analyze_causality_p_values(max_p, min_p):
    max_p = float(max_p.iloc[0]) if hasattr(max_p, 'iloc') else float(max_p)
    min_p = float(min_p.iloc[0]) if hasattr(min_p, 'iloc') else float(min_p)
    if min_p < 0.01: return "strong causal relationship"
    elif min_p < 0.05: return "moderate causal relationship"
    elif min_p < 0.1: return "weak causal relationship"
    else: return "no significant causal relationship"

def analyze_mutual_info(value):
    value = float(value.iloc[0]) if hasattr(value, 'iloc') else float(value)
    if value > 5: return "strong information dependency"
    elif value > 2: return "moderate information dependency"
    else: return "weak information dependency"

def analyze_transfer_entropy(value):
    value = float(value.iloc[0]) if hasattr(value, 'iloc') else float(value)
    if value > 0.1: return "significant information flow"
    elif value > 0.05: return "moderate information flow"
    else: return "minimal information flow"

def analyze_cointegration(p_value):
    p_value = float(p_value.iloc[0]) if hasattr(p_value, 'iloc') else float(p_value)
    if p_value < 0.01: return "strong long-term equilibrium"
    elif p_value < 0.05: return "moderate long-term equilibrium"
    else: return "no significant long-term relationship"

def analyze_rolling_correlation(mean, std):
    mean = float(mean.iloc[0]) if hasattr(mean, 'iloc') else float(mean)
    std = float(std.iloc[0]) if hasattr(std, 'iloc') else float(std)
    stability = "stable" if std < 0.2 else "volatile"
    if mean > 0.7: return f"strong and {stability} relationship over time"
    elif mean > 0.3: return f"moderate and {stability} relationship over time"
    else: return f"weak and {stability} relationship over time"

def interpret_metrics(df):
    interpreted = pd.DataFrame(index=df.index, columns=df.columns, dtype='object')
    for col in df.columns:
        for idx in df.index:
            if idx == 'correlation_max':
                interpreted.at[idx, col] = analyze_correlation_max(df.at[idx, col])
            elif idx == 'correlation_lag':
                interpreted.at[idx, col] = analyze_correlation_lag(df.at[idx, col])
            elif idx == 'correlation_distance':
                interpreted.at[idx, col] = analyze_correlation_distance(df.at[idx, col])
            elif idx == 'causality_p_value_max':
                interpreted.at[idx, col] = analyze_causality_p_values(df.at[idx, col], df.at['causality_p_value_min', col])
            elif idx == 'mutual_info':
                interpreted.at[idx, col] = analyze_mutual_info(df.at[idx, col])
            elif idx == 'transfer_entropy':
                interpreted.at[idx, col] = analyze_transfer_entropy(df.at[idx, col])
            elif idx == 'cointegration_p_value':
                interpreted.at[idx, col] = analyze_cointegration(df.at[idx, col])
            elif idx == 'mean_rolling_correlation':
                interpreted.at[idx, col] = analyze_rolling_correlation(df.at[idx, col], df.at['std_rolling_correlation', col])
    
    return interpreted