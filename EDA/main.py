from data.data import GlobalDataHandler
from data.preprocessing import preprocess_data
from evaluate.metrics import *
from utils.utils import dbg_print

if __name__ == "__main__":
    debug = True
    
    data_loader = GlobalDataHandler(country='hong kong', start='2022-01-01', end='2023-12-31', debug=debug)
    data_loader.load_all_data()

    print(data_loader.themed_characteristics_clusters)
    
    data_dict = {
        'themed_characteristics_time_series': data_loader.themed_characteristics_time_series,
        'weather_data': data_loader.weather_data,
        'powell_speeches': data_loader.powell_speeches,
        'non_local_holidays': data_loader.non_local_holidays,
        'hk_holidays': data_loader.hk_holidays,
        'lunar_calendar': data_loader.lunar_calendar,
        'aging_population': data_loader.aging_population,
        'google_trends_local_df': data_loader.google_trends_local_df,
        'google_trends_us_df': data_loader.google_trends_us_df,
        'bitcoin_prices': data_loader.bitcoin_prices,
        'crypto_volume': data_loader.crypto_volume,
        'institutional_outlook': data_loader.institutional_outlook,
        'connected_markets': data_loader.connected_markets  # list of dfs
    }
    processed_data = preprocess_data(data_dict, debug=debug)

    causality_res = evaluate_causality(processed_data)
    correlation_res = evaluate_correlation(processed_data)
    nonlinearity_res = evaluate_nonlinearity(processed_data)
    info_transmission_res = evaluate_information_transmission(processed_data)
    cointegration_res = evaluate_cointegration(processed_data)
    decomposition_res = evaluate_decomposition(processed_data)

    # 2d) Combine results
    final_df = combine_metrics_results(causality_res, correlation_res, nonlinearity_res,
                                       info_transmission_res, cointegration_res, decomposition_res)

    if final_df is not None:
        print("\nFinal Evaluation Results DataFrame:")
        print(final_df)
    else:
        print("\nNo final results to display (metrics not implemented).")
