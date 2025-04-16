# data/preprocessing.py
import pandas as pd
import yfinance as yf
from utils.utils import dbg_print
from utils.settings import setting

def get_market_indices():
    if setting.country == 'hong kong': return ['^HSI', '^HSCE']  
    elif setting.country == 'south korea': return ['^KS11', '^KQ11']  
    return ['^GSPC', '^IXIC']  

def fetch_market_data(indices):
    market_dfs = []
    for index in indices:
        try:
            df = yf.download(index, start=setting.start, end=setting.end, progress=False)
            if not df.empty:
                df_close = pd.DataFrame(df['Close'])
                df_close.columns = [f'{index}_close']
                market_dfs.append(df_close)
        except Exception as e:
            dbg_print(f"Failed to fetch data for {index}: {e}")
    return market_dfs

def preprocess_data(data_dict):
    if data_dict is None:
        if setting.debug: dbg_print("[DEBUG] No data_dict provided to preprocess. Returning empty dictionary.")
        return {}, {}
    processed = {}
    comparison = {}
    market_indices = get_market_indices()
    market_data = fetch_market_data(market_indices)

    dbg_print("Starting preprocessing for themed_characteristics_time_series")
    tcts = data_dict.get('themed_characteristics_time_series')
    if tcts is not None:
        try:
            if 'date' in tcts.columns:
                tcts.set_index('date', inplace=True)
            themed_list = []
            for col in tcts.columns:
                df_col = tcts[[col]].copy()
                df_col.dropna(inplace=True)
                themed_list.append(df_col)
            processed['themed_characteristics_time_series'] = themed_list
            clusters = data_dict.get('themed_characteristics_clusters')
            if clusters is not None:
                cluster_data = []
                for _, row in clusters.iterrows():
                    tickers = row['ticker']
                    cluster_dfs = []
                    for ticker in tickers:
                        try:
                            df = yf.download(ticker, 
                                           start=setting.start, 
                                           end=setting.end, 
                                           progress=False)
                            if not df.empty:
                                pct_change = pd.DataFrame(df['Close'].pct_change())
                                if not pct_change.empty:
                                    pct_change.columns = [f'{ticker}_closed_pct_change']
                                    pct_change.dropna(inplace=True)
                                    cluster_dfs.append(pct_change)
                        except Exception as e:
                            dbg_print(f"Failed to fetch data for {ticker}: {e}")
                    if cluster_dfs:
                        cluster_data.append(cluster_dfs)
                comparison['themed_characteristics_time_series'] = cluster_data
            dbg_print("Processed themed_characteristics_time_series successfully.")
        except Exception as e:
            dbg_print(f"[ERROR] Failed to process themed_characteristics_time_series: {e}")

    dbg_print("Starting preprocessing for weather_data")
    wd = data_dict.get('weather_data')
    if wd is not None:
        try:
            if 'date' in wd.columns:
                wd['date'] = pd.to_datetime(wd['date'], errors='coerce')
                wd.dropna(subset=['date'], inplace=True)
                wd.set_index('date', inplace=True)
                wd.index.names = ['Date']
            if 'mean_temp' in wd.columns:
                wd = wd[['mean_temp']]
            processed['weather_data'] = wd
            comparison['weather_data'] = market_data
            dbg_print("Processed weather_data successfully.")
        except Exception as e: dbg_print(f"[ERROR] Failed to process weather_data: {e}")

    dbg_print("Starting preprocessing for powell_speeches")
    ps = data_dict.get('powell_speeches')
    if ps is not None:
        try:
            if 'date' in ps.columns:
                ps['date'] = pd.to_datetime(ps['date'], errors='coerce')
                ps.dropna(subset=['date'], inplace=True)
                ps.set_index('date', inplace=True)
                ps.index.names = ['Date']
            if 'rate_change' in ps.columns:
                ps = ps[['rate_change']]
            processed['powell_speeches'] = ps
            comparison['powell_speeches'] = market_data
            dbg_print("Processed powell_speeches successfully.")
        except Exception as e: dbg_print(f"[ERROR] Failed to process powell_speeches: {e}")

    dbg_print("Starting preprocessing for non_local_holidays")
    nh = data_dict.get('non_local_holidays')
    if nh is not None and not nh.empty:
        try:
            if 'date' in nh.columns:
                nh['date'] = pd.to_datetime(nh['date'], errors='coerce')
                nh.dropna(subset=['date'], inplace=True)
            non_local_list = []
            for ctry in nh['country'].dropna().unique():
                sub_df = nh[nh['country'] == ctry].copy()
                sub_df = pd.DataFrame(1, index=sub_df['date'], columns=['is_holiday'])
                sub_df.index.names = ['Date']
                non_local_list.append(sub_df)
            processed['non_local_holidays'] = non_local_list
            comparison['non_local_holidays'] = market_data
            dbg_print("Processed non_local_holidays successfully.")
        except Exception as e: dbg_print(f"[ERROR] Failed to process non_local_holidays: {e}")

    dbg_print("Starting preprocessing for local_holidays")
    lhol = data_dict.get('local_holidays')
    if lhol is not None:
        try:
            if 'date' in lhol.columns:
                lhol['date'] = pd.to_datetime(lhol['date'], errors='coerce')
                lhol.dropna(subset=['date'], inplace=True)
            lhol = pd.DataFrame(1, index=lhol['date'], columns=['is_holiday'])
            lhol.index.names = ['Date']
            processed['local_holidays'] = lhol
            comparison['local_holidays'] = market_data
            dbg_print("Processed local_holidays successfully.")
        except Exception as e: dbg_print(f"[ERROR] Failed to process local_holidays: {e}")

    dbg_print("Starting preprocessing for lunar_calendar")
    lc = data_dict.get('lunar_calendar')
    if lc is not None:
        try:
            if 'gregorian_date' in lc.columns:
                lc.rename(columns={'gregorian_date': 'date'}, inplace=True)
            if 'date' in lc.columns:
                lc['date'] = pd.to_datetime(lc['date'], errors='coerce')
                lc.dropna(subset=['date'], inplace=True)
            lc = pd.DataFrame(1, index=lc['date'], columns=['is_lunar_holiday'])
            lc.index.names = ['Date']
            processed['lunar_calendar'] = lc
            comparison['lunar_calendar'] = market_data
            dbg_print("Processed lunar_calendar successfully.")
        except Exception as e: dbg_print(f"[ERROR] Failed to process lunar_calendar: {e}")

    dbg_print("Starting preprocessing for aging_population")
    ap = data_dict.get('aging_population')
    if ap is not None:
        try:
            ap_copy = ap[['year', 'population_15-64_percent']].copy()
            ap_copy.loc[:, 'year'] = pd.to_datetime(ap_copy['year'].astype(str))
            ap_copy.set_index('year', inplace=True)
            ap_copy.index.names = ['Date']
            
            daily_idx = pd.date_range(start=ap_copy.index.min(), end=ap_copy.index.max(), freq='D')
            daily_data = ap_copy.reindex(daily_idx)
            daily_data = daily_data.interpolate(method='linear')
            
            unrestricted_market_data = []
            market_indices = get_market_indices()
            for index in market_indices:
                try:
                    df = yf.download(index, start=daily_data.index.min(), end=daily_data.index.max(), progress=False)
                    if not df.empty:
                        df_close = pd.DataFrame(df['Close'])
                        df_close.columns = [f'{index}_close']
                        unrestricted_market_data.append(df_close)
                except Exception as e:
                    dbg_print(f"Failed to fetch data for {index}: {e}")
            
            processed['aging_population'] = daily_data
            comparison['aging_population'] = unrestricted_market_data
            dbg_print("Processed aging_population successfully.")
        except Exception as e: dbg_print(f"[ERROR] Failed to process aging_population: {e}")

    dbg_print("Starting preprocessing for connected_markets")
    cm = data_dict.get('connected_markets')
    if cm is not None:
        try:
            connected_list = []
            for df_cm in cm:
                if 'Index' not in df_cm.columns:
                    continue
                for idx_val in df_cm['Index'].dropna().unique():
                    sub = df_cm[df_cm['Index'] == idx_val]
                    adj_close_df = sub[['Close']].copy()
                    volume_df = sub[['Volume']].copy()
                    connected_list.append(adj_close_df)
                    connected_list.append(volume_df)
            processed['connected_markets'] = connected_list
            comparison['connected_markets'] = market_data
            dbg_print("Processed connected_markets successfully.")
        except Exception as e: dbg_print(f"[ERROR] Failed to process connected_markets: {e}")
    dbg_print("Preprocessing completed")
    return processed, comparison