import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import cwt, morlet2
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.stattools import grangercausalitytests, coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import dcor

def evaluate_causality(target_df, comparison_df, threshold_lag=5):
    merged = pd.merge(target_df, comparison_df, left_index=True, right_index=True)
    if merged.empty or len(merged) < threshold_lag + 2:
        raise ValueError(f"Insufficient data points: {len(merged)}")
        
    x = (merged.iloc[:, 0] - merged.iloc[:, 0].mean()) / merged.iloc[:, 0].std()
    y = (merged.iloc[:, 1] - merged.iloc[:, 1].mean()) / merged.iloc[:, 1].std()
    
    maxlag = min(threshold_lag, len(merged) // 10)
    
    try:
        granger_result = grangercausalitytests(merged, maxlag=maxlag, verbose=False)
        granger_p_values = [granger_result[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
    except:
        granger_p_values = [np.nan] * maxlag
        
    def toda_yamamoto_test(x, y, lag):
        data = pd.DataFrame({'y': y, 'x': x})
        total_lag = lag + 1
        
        for i in range(total_lag):
            data[f'x_lag_{i+1}'] = data['x'].shift(i+1)
            data[f'y_lag_{i+1}'] = data['y'].shift(i+1)
        
        data = data.dropna()
        Y = data['y']
        X = sm.add_constant(data[[col for col in data.columns if col != 'y']])
        
        try:
            model = OLS(Y, X).fit()
            r_matrix = np.zeros((lag, len(model.params)))
            for i in range(lag):
                r_matrix[i, i+1] = 1
            
            p_value = float(model.f_test(r_matrix).pvalue)
            return max(min(p_value, 0.9999), 1e-16)
        except:
            return np.nan

    ty_p_values = [toda_yamamoto_test(x, y, i+1) for i in range(maxlag)]
    
    def diks_panchenko_test(x, y, lag, epsilon=0.5):
        if len(x) - lag < 30:
            return np.nan
            
        x_lag = x[:-lag]
        y_lag = y[:-lag]
        y_future = y[lag:]
        
        def normalize_windows(data, window_size=50):
            normalized = np.zeros_like(data)
            for i in range(0, len(data), window_size):
                window = data[i:i+window_size]
                if len(window) >= 3:
                    normalized[i:i+window_size] = (window - np.mean(window)) / (np.std(window) + 1e-10)
            return normalized
        
        x_lag, y_lag, y_future = map(normalize_windows, [x_lag, y_lag, y_future])
        
        def compute_distance_matrix(X, Y):
            return np.abs(X.reshape(-1, 1) - Y.reshape(-1, 1).T) < epsilon
        
        Dx = compute_distance_matrix(x_lag, x_lag)
        Dy = compute_distance_matrix(y_lag, y_lag)
        Dz = compute_distance_matrix(y_future, y_future)
        
        np.fill_diagonal(Dx, 0)
        np.fill_diagonal(Dy, 0)
        np.fill_diagonal(Dz, 0)
        
        joint_prob = np.mean(Dx & Dy & Dz)
        marginal_prob = np.mean(Dx) * np.mean(Dy & Dz)
        var_est = np.var(Dx & Dy & Dz) / (len(x) - lag)
        
        if var_est < 1e-15:
            return 0.5
        
        t_stat = (joint_prob - marginal_prob) / np.sqrt(var_est)
        return max(min(2 * (1 - stats.norm.cdf(abs(t_stat))), 1.0), 1e-16)
        
    dp_p_values = [diks_panchenko_test(x, y, i+1) for i in range(maxlag)]
    
    return {'g_p': granger_p_values, 'ty_p': ty_p_values, 'dp_p': dp_p_values}

def evaluate_correlation(target_df, comparison_df, normalize=True, min_points=30, max_lag_ratio=0.2):
    x = target_df.iloc[:, 0].dropna().values
    y = comparison_df.iloc[:, 0].dropna().values
    
    if len(x) < min_points or len(y) < min_points:
        raise ValueError(f"Both series must have at least {min_points} points")
    
    n = min(len(x), len(y))
    x, y = (x[:n] - np.mean(x[:n])) / np.std(x[:n]), (y[:n] - np.mean(y[:n])) / np.std(y[:n])
    
    ccf = np.correlate(x, y, mode='full')
    if normalize:
        denominators = np.array([(n - abs(k)) for k in range(-n + 1, n)])
        ccf = np.divide(ccf, denominators, where=denominators != 0)
    
    max_possible_lag = int(n * max_lag_ratio)
    center_idx = len(ccf) // 2
    search_start = center_idx - max_possible_lag
    search_end = center_idx + max_possible_lag + 1
    
    max_idx = search_start + np.argmax(np.abs(ccf[search_start:search_end]))
    max_lag = max_idx - center_idx
    ccf_corr = np.clip(ccf[max_idx], -0.9999, 0.9999)
    
    if max_lag >= 0:
        x_lag, y_lag = x[max_lag:], y[:-max_lag] if max_lag > 0 else y
    else:
        x_lag, y_lag = x[:max_lag], y[-max_lag:]
        
    if len(x_lag) < min_points:
        raise ValueError("Insufficient data points after lag")
        
    lagged_corr = np.clip(np.corrcoef(x_lag, y_lag)[0, 1], -0.9999, 0.9999)
    
    def get_pvalue(r, n):
        if n < 3: return np.nan
        if n > 100 and abs(r) > 0.8: return 1e-16
        df = n - 2
        t = r * np.sqrt(df / (1 - r**2))
        return max(2 * (1 - stats.t.cdf(abs(t), df)), 1e-16)

    def compute_cca(x, y, num_scales=8):
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        
        scales = np.logspace(1, np.log10(n/2), num_scales)
        
        cwt_x = cwt(x, morlet2, scales)
        cwt_y = cwt(y, morlet2, scales)
        
        csd = np.abs(cwt_x * np.conj(cwt_y))
        asd_x = np.abs(cwt_x * np.conj(cwt_x))
        asd_y = np.abs(cwt_y * np.conj(cwt_y))
        
        coherence = np.real(csd**2 / (asd_x * asd_y + 1e-10))
        
        mean_coherence = np.clip(np.mean(coherence), 0, 1)
        
        z_stat = np.sqrt(n-3) * 0.5 * np.log1p(2 * mean_coherence / (1 - mean_coherence + 1e-10))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        p_value = max(p_value, 1e-16)
        
        return {
            'coherence': float(mean_coherence),
            'p_value': float(p_value)
        }

    cca_results = compute_cca(x, y)
    
    return {
        'ccf_correlation': float(ccf_corr),
        'ccf_lag': int(max_lag),
        'ccf_p_value': float(get_pvalue(ccf_corr, n)),
        'lagged_correlation': float(lagged_corr),
        'lagged_p_value': float(get_pvalue(lagged_corr, len(x_lag))),
        'cca_coherence': float(cca_results['coherence']),
        'cca_p_value': float(cca_results['p_value'])
    }

def evaluate_nonlinearity(target_df, comparison_df):
    merged = pd.merge(target_df, comparison_df, left_index=True, right_index=True)
    if merged.empty:
        return {'mutual_information': np.nan, 'distance_correlation': np.nan}
    try:
        x = pd.qcut(merged.iloc[:,0], 20, labels=False, duplicates='drop')
        y = pd.qcut(merged.iloc[:,1], 20, labels=False, duplicates='drop')
        return {
            'mutual_information': mutual_info_score(x, y),
            'distance_correlation': dcor.distance_correlation(merged.iloc[:,0], merged.iloc[:,1])
        }
    except:
        return {'mutual_information': np.nan, 'distance_correlation': np.nan}

def evaluate_information_transmission(target_df, comparison_df):
    merged = pd.merge(target_df, comparison_df, left_index=True, right_index=True)
    x, y = merged.iloc[:,0], merged.iloc[:,1]
    bins = np.linspace(start=min(x.min(), y.min()), stop=max(x.max(), y.max()), num=10)
    return {'transfer_entropy': mutual_info_score(np.digitize(y, bins)[1:], np.digitize(x, bins)[:-1])}

def evaluate_cointegration(target_df, comparison_df):
    merged = pd.merge(target_df, comparison_df, left_index=True, right_index=True)
    score, p_value, _ = coint(merged.iloc[:,0], merged.iloc[:,1])
    return {'score': score, 'p_value': max(p_value, 1e-16)}

def evaluate_decomposition(target_df, comparison_df):
    merged = pd.merge(target_df, comparison_df, left_index=True, right_index=True)
    rolling_corr = merged.iloc[:,0].rolling(window=30).corr(merged.iloc[:,1])
    return {'mean_rolling_correlation': rolling_corr.mean(), 
            'std_rolling_correlation': rolling_corr.std()}