import pandas as pd
from utils.utils import dbg_print

def preprocess_data(data_dict, debug=False):
    print(data_dict)
    print(data_dict is None)
    if data_dict is None:
        if debug:
            print("[DEBUG] No data_dict provided to preprocess. Returning empty dictionary.")
        return {}

    processed = {}

    # 1. Themed Characteristics Time Series
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
            dbg_print("Processed themed_characteristics_time_series successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process themed_characteristics_time_series: {e}")

    # 2. Weather Data
    dbg_print("Starting preprocessing for weather_data")
    wd = data_dict.get('weather_data')
    if wd is not None:
        try:
            if 'date' in wd.columns:
                wd['date'] = pd.to_datetime(wd['date'], errors='coerce')
                wd.dropna(subset=['date'], inplace=True)
                wd.set_index('date', inplace=True)
            if 'mean_temp' in wd.columns:
                wd = wd[['mean_temp']]
            processed['weather_data'] = wd
            dbg_print("Processed weather_data successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process weather_data: {e}")

    # 3. Powell Announcements
    dbg_print("Starting preprocessing for powell_speeches")
    ps = data_dict.get('powell_speeches')
    if ps is not None:
        try:
            if 'date' in ps.columns:
                ps['date'] = pd.to_datetime(ps['date'], errors='coerce')
                ps.dropna(subset=['date'], inplace=True)
                ps.set_index('date', inplace=True)
            if 'rate_change' in ps.columns:
                ps = ps[['rate_change']]
            processed['powell_speeches'] = ps
            dbg_print("Processed powell_speeches successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process powell_speeches: {e}")

    # 4. Non-local Holidays
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
                sub_df = sub_df[['date', 'localName']].copy()
                sub_df.set_index('date', inplace=True)
                non_local_list.append(sub_df)
            processed['non_local_holidays'] = non_local_list
            dbg_print("Processed non_local_holidays successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process non_local_holidays: {e}")

    # 5. Local holidays (hk_holidays)
    dbg_print("Starting preprocessing for hk_holidays")
    lhol = data_dict.get('hk_holidays')
    if lhol is not None:
        try:
            if 'date' in lhol.columns:
                lhol['date'] = pd.to_datetime(lhol['date'], errors='coerce')
                lhol.dropna(subset=['date'], inplace=True)
                lhol.set_index('date', inplace=True)
            lhol = lhol[['localName']]
            processed['hk_holidays'] = lhol
            dbg_print("Processed hk_holidays successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process hk_holidays: {e}")

    # 6. Lunar Calendar
    dbg_print("Starting preprocessing for lunar_calendar")
    lc = data_dict.get('lunar_calendar')
    if lc is not None:
        try:
            if 'gregorian_date' in lc.columns:
                lc.rename(columns={'gregorian_date': 'date'}, inplace=True)
            if 'date' in lc.columns:
                lc['date'] = pd.to_datetime(lc['date'], errors='coerce')
                lc.dropna(subset=['date'], inplace=True)
                lc.set_index('date', inplace=True)
            lc = lc[['lunar_event']]
            processed['lunar_calendar'] = lc
            dbg_print("Processed lunar_calendar successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process lunar_calendar: {e}")

    # 7. Aging Population
    dbg_print("Starting preprocessing for aging_population")
    ap = data_dict.get('aging_population')
    if ap is not None:
        try:
            if 'year' in ap.columns:
                # year is assumed numeric or string representing a year
                ap['year'] = pd.to_numeric(ap['year'], errors='coerce')
                ap.dropna(subset=['year'], inplace=True)
                ap.set_index('year', inplace=True)
            ap = ap[['population_over_65_percent']]
            processed['aging_population'] = ap
            dbg_print("Processed aging_population successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process aging_population: {e}")

    # 8. Connected Markets
    dbg_print("Starting preprocessing for connected_markets")
    cm = data_dict.get('connected_markets')
    if cm is not None:
        try:
            connected_list = []
            for df_cm in cm:
                if 'date' in df_cm.columns:
                    df_cm['date'] = pd.to_datetime(df_cm['date'], errors='coerce')
                    df_cm.dropna(subset=['date'], inplace=True)
                    df_cm.set_index('date', inplace=True)
                if 'Index' not in df_cm.columns:
                    print("[WARN] connected_markets df missing 'Index' column.")
                    continue
                for idx_val in df_cm['Index'].dropna().unique():
                    sub = df_cm[df_cm['Index'] == idx_val]
                    adj_close_df = sub[['Adj Close']].copy()
                    volume_df = sub[['Volume']].copy()

                    connected_list.append(adj_close_df)
                    connected_list.append(volume_df)
            processed['connected_markets'] = connected_list
            dbg_print("Processed connected_markets successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process connected_markets: {e}")

    dbg_print("Preprocessing completed.")
    return processed
