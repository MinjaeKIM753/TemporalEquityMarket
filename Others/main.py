from data.data import GlobalDataHandler
from data.preprocessing import preprocess_data
from evaluate.metrics import *
from evaluate.evaluate import evaluate_data
from evaluate.analysis import interpret_metrics
from utils.utils import dbg_print
from utils.settings import setting

if __name__ == "__main__":
    
    data_loader = GlobalDataHandler()
    data_loader.load_all_data()

    #[Debug] Cluster checking
    if setting.debug: 
        data_loader.themed_characteristics_clusters
    
    data_dict = {
        'themed_characteristics_clusters': data_loader.themed_characteristics_clusters,
        'themed_characteristics_time_series': data_loader.themed_characteristics_time_series,
        'weather_data': data_loader.weather_data,
        #'powell_speeches': data_loader.powell_speeches,
        #'non_local_holidays': data_loader.non_local_holidays,
        #'local_holidays': data_loader.local_holidays,
        #'lunar_calendar': data_loader.lunar_calendar,
        'aging_population': data_loader.aging_population,
        #'google_trends_local_df': data_loader.google_trends_local_df,
        #'google_trends_us_df': data_loader.google_trends_us_df,
        'bitcoin_prices': data_loader.bitcoin_prices,
        'crypto_volume': data_loader.crypto_volume,
        #'institutional_outlook': data_loader.institutional_outlook,
        #'connected_markets': data_loader.connected_markets  # list of dfs
    }
    processed_data, comparison_data = preprocess_data(data_dict)
    
    results = evaluate_data(processed_data, comparison_data)
    dbg_print(results)
    interpreted_results = interpret_metrics(results)
    dbg_print(interpreted_results)
    