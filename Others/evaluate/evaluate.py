# evaluate/evaluate.py
from evaluate.metrics import *
from utils.utils import dbg_print


def add_to_dict(analysis:dict, counts, target_df, comparison_df):
    def det(target_key, value):
        if value is np.nan:
            analysis[target_key] += analysis[target_key]/counts
        else:
            analysis[target_key] += value
    causality, correlation, nonlinearity, information_transmission, cointegration, decomposition = evaluation(target_df, comparison_df)

    dbg_print(f'Current Count: {counts}, values = {causality},{correlation},{nonlinearity},{information_transmission}, {cointegration}, {decomposition}')
    # Causality
    det('causality_granger_p_value_max', max(causality['g_p']))
    det('causality_granger_p_value_min', min(causality['g_p']))
    det('causality_toda_yamamoto_p_value_max', max(causality['ty_p']))
    det('causality_toda_yamamoto_p_value_min', min(causality['g_p']))
    det('causality_diks_panchenko_p_value_max', max(causality['dp_p']))
    det('causality_diks_panchenko_p_value_min', min(causality['dp_p']))
    # Correlation
    det('correlation_ccf_max', correlation['ccf_correlation'])
    det('correlation_ccf_maxlag', correlation['ccf_lag'])
    det('correlation_ccf_p_value', correlation['ccf_p_value'])
    det('correlation_lagged', correlation['lagged_correlation'])
    det('correlation_lagged_p_value', correlation['lagged_p_value'])
    det('correlation_cca_coherence', correlation['cca_coherence'])
    det('correlation_cca_p_value', correlation['cca_p_value'])

    
    analysis['mutual_info'] += analysis['mutual_info'] if nonlinearity['mutual_information'] is np.nan else nonlinearity['mutual_information']
    analysis['correlation_distance'] += analysis['correlation_distance'] if nonlinearity['distance_correlation'] is np.nan else nonlinearity['distance_correlation']
    analysis['transfer_entropy'] += analysis['transfer_entropy'] if information_transmission['transfer_entropy'] is np.nan else information_transmission['transfer_entropy']
    analysis['cointegration_p_value'] += analysis['cointegration_p_value'] if cointegration['p_value'] is np.nan else cointegration['p_value']
    analysis['cointegration_score'] += analysis['cointegration_score']/counts if cointegration['score'] is np.nan else cointegration['score']
    analysis['mean_rolling_correlation'] += analysis['mean_rolling_correlation']/counts if decomposition['mean_rolling_correlation'] is np.nan else decomposition['mean_rolling_correlation']
    analysis['std_rolling_correlation'] += analysis['std_rolling_correlation']/counts if decomposition['std_rolling_correlation'] is np.nan else decomposition['std_rolling_correlation']
    return analysis


def evaluation(target_df, comparison_df):
    return (
        evaluate_causality(target_df, comparison_df),
        evaluate_correlation(target_df, comparison_df),
        evaluate_nonlinearity(target_df, comparison_df),
        evaluate_information_transmission(target_df, comparison_df),
        evaluate_cointegration(target_df, comparison_df),
        evaluate_decomposition(target_df, comparison_df)
    )

def evaluate_data(processed, compared):
    result = {}
    for data in processed.keys():
        analysis = {
            # Causation
            'causality_granger_p_value_max': 0,
            'causality_granger_p_value_min': 0,
            'causality_toda_yamamoto_p_value_max': 0,
            'causality_toda_yamamoto_p_value_min': 0,
            'causality_diks_panchenko_p_value_max': 0,
            'causality_diks_panchenko_p_value_min': 0,
            # Correlation
            'correlation_ccf_max': 0,
            'correlation_ccf_maxlag': 0,
            'correlation_ccf_p_value': 0,
            'correlation_lagged': 0,
            'correlation_lagged_p_value': 0,
            'correlation_cca_coherence': 0,
            'correlation_cca_p_value': 0,
            'correlation_distance': 0,
            'mutual_info': 0,
            'transfer_entropy': 0,
            'cointegration_p_value': 0,
            'cointegration_score': 0,
            'mean_rolling_correlation': 0,
            'std_rolling_correlation': 0
        }
        if data == 'themed_characteristics_time_series':
            try: 
                counts = 0
                n_clusters = len(processed['themed_characteristics_time_series'])
                for n in range(n_clusters):
                    for m in range(len(compared['themed_characteristics_time_series'][n])):
                        counts += 1
                        analysis = add_to_dict(analysis, counts, processed['themed_characteristics_time_series'][n], compared['themed_characteristics_time_series'][n][m])
                if counts > 0:
                    for key in analysis.keys():
                        analysis[key] /= counts
                    result[data] = analysis
            except Exception as e:
                dbg_print(f"[ERROR] Failed to evaluate {data}: {e}")
        if data == 'aging_population':
            try:
                counts = 0
                for market_df in compared[data]:
                    counts += 1
                    analysis = add_to_dict(analysis, counts, processed[data], market_df)
                if counts > 0:
                    for key in analysis.keys():
                        analysis[key] /= counts
                    result[data] = analysis
            except Exception as e:
                dbg_print(f"[ERROR] Failed to evaluate {data}: {e}")
    return pd.DataFrame.from_dict(result)
