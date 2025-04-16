import pandas as pd
import numpy as np
import requests
import time
import feedparser
import yfinance as yf
from pytrends.request import TrendReq
from pandas_datareader import wb, data as pdr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from .sample import SampleTickers
from utils.utils import dbg_print
import re

class GlobalDataHandler:
    def __init__(self, country='hong kong', start='2022-01-01', end='2023-12-31', debug=False):
        self.country = country.lower()
        self.start = start
        self.end = end
        self.debug = debug

        self.themed_characteristics_clusters = None
        self.themed_characteristics_time_series = None
        self.weather_data = None
        self.powell_speeches = None
        self.non_local_holidays = None
        self.hk_holidays = None
        self.lunar_calendar = None
        self.aging_population = None
        self.google_trends_local_df = None
        self.google_trends_us_df = None
        self.bitcoin_prices = None
        self.crypto_volume = None
        self.analyst_opinions = None
        self.institutional_outlook = None
        self.connected_markets = []
        self.governance_scores = None

        self.sample_tickers = SampleTickers()

    def verify_data(self, df, name, allow_all_zeros=False):
        if df is None or df.empty:
            raise ValueError(f"{name} data is empty or could not be fetched.")
        if not allow_all_zeros:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.size > 0:
                non_nan_count = numeric_df.notna().sum().sum()
                all_zero = (numeric_df.fillna(0) == 0).all().all()
                if non_nan_count == 0 or all_zero:
                    raise ValueError(f"{name} data is all NaNs or zeros.")

    def get_country_tickers(self):
        if self.country == 'hong kong':
            return self.sample_tickers.hk_samples()
        elif self.country == 'south korea':
            return self.sample_tickers.sk_samples()
        else:
            return self.sample_tickers.us_samples()

    def get_themed_characteristics(self, tickers=None, n_clusters=5):
        try:
            if tickers is None:
                tickers = self.get_country_tickers()

            if not tickers:
                print("No tickers available for this country in themed characteristics.")
                return

            descriptions = []
            valid_tickers = []
            for t in tickers:
                info = None
                for attempt in range(3):
                    try:
                        stock = yf.Ticker(t)
                        info = stock.info
                        break
                    except:
                        time.sleep(1)
                if info and 'longBusinessSummary' in info and info['longBusinessSummary']:
                    descriptions.append(info['longBusinessSummary'])
                    valid_tickers.append(t)

            if not descriptions:
                print("No descriptions fetched for any ticker in Themed Characteristics.")
                return

            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            X = vectorizer.fit_transform(descriptions)

            n_clusters = n_clusters if len(valid_tickers) >= n_clusters else len(valid_tickers)
            if n_clusters < 1:
                print("Not enough tickers to form clusters in Themed Characteristics.")
                return

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X)

            cluster_map = pd.DataFrame({'ticker': valid_tickers, 'cluster': labels})

            feature_names = vectorizer.get_feature_names_out()
            cluster_centers = kmeans.cluster_centers_
            top_keywords_per_cluster = []
            top_n = 5
            for c in range(n_clusters):
                sorted_indices = cluster_centers[c].argsort()[::-1]
                top_features = [feature_names[idx] for idx in sorted_indices[:top_n]]
                top_keywords_per_cluster.append(top_features)

            cluster_map['theme_name'] = cluster_map['cluster'].apply(lambda x: ", ".join(top_keywords_per_cluster[x]))

            cluster_names = cluster_map[['cluster', 'theme_name']].drop_duplicates().set_index('cluster')['theme_name']
            cluster_returns_dict = {}
            for c in range(n_clusters):
                cluster_tickers = cluster_map[cluster_map['cluster'] == c]['ticker'].tolist()
                daily_returns_list = []
                for ct in cluster_tickers:
                    dfp = yf.download(ct, start=self.start, end=self.end, progress=False)
                    if dfp.empty:
                        continue
                    dfp['pct_change'] = dfp['Close'].pct_change()
                    daily_returns_list.append(dfp[['pct_change']])
                if daily_returns_list:
                    merged = pd.concat(daily_returns_list, axis=1)
                    daily_mean = merged.mean(axis=1)
                    cluster_name = cluster_names[c]
                    cluster_returns_dict[cluster_name] = daily_mean
                else:
                    # No data for this cluster?
                    cluster_name = cluster_names[c]
                    cluster_returns_dict[cluster_name] = pd.Series(dtype=float)
            if cluster_returns_dict:
                self.themed_characteristics_time_series = pd.DataFrame(cluster_returns_dict)
            self.themed_characteristics_clusters = cluster_map.groupby(['cluster', 'theme_name'])['ticker'].apply(list).reset_index()

            dbg_print("Finished get_themed_characteristics.")
        except Exception as e:
            print(f"Failed to get Themed Characteristics: {e}")

    def get_powell_announcements(self):

        try:
            start_dt = pd.to_datetime(self.start)
            end_dt = pd.to_datetime(self.end)
            dff = pdr.DataReader('DFF', 'fred', start_dt, end_dt)
            dff['rate_change'] = dff['DFF'].diff().fillna(0.0)
            df = dff.copy()
            df.reset_index(inplace=True)
            df.rename(columns={'DATE': 'date', 'DFF': 'effr'}, inplace=True)
            df['date'] = df['date'].dt.strftime("%Y-%m-%d")

            self.verify_data(df[['rate_change']], "Powell’s Announcements (Rate Changes)", allow_all_zeros=True)
            self.powell_speeches = df[['date', 'rate_change']]
            dbg_print("Finished get_powell_announcements.")
        except Exception as e:
            print(f"Failed to fetch Powell's announcements: {e}")


    def get_weather_data(self):
        try:
            if self.country == 'hong kong':
                lat, lon = 22.3193, 114.1694
            elif self.country == 'south korea':
                lat, lon = 37.5665, 126.9780
            else:
                lat, lon = 37.7749, -122.4194

            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": self.start,
                "end_date": self.end,
                "daily": "temperature_2m_mean",
                "timezone": "Asia/Hong_Kong"
            }
            url = "https://archive-api.open-meteo.com/v1/archive"
            r = requests.get(url, params=params)
            data = r.json()
            if "daily" in data and "temperature_2m_mean" in data["daily"]:
                df = pd.DataFrame({
                    "date": data["daily"]["time"],
                    "mean_temp": data["daily"]["temperature_2m_mean"]
                })
                self.verify_data(df, "Weather - Average Temperature")
                self.weather_data = df
            else:
                print("No temperature data returned for weather.")
            dbg_print("Finished get_weather_data.")
        except Exception as e:
            print(f"Failed to fetch Weather data: {e}")

    def get_hk_holidays(self):
        try:
            country_code = 'HK'
            if self.country == 'south korea':
                country_code = 'KR'
            elif self.country == 'us':
                country_code = 'US'

            year_start = pd.to_datetime(self.start).year
            year_end = pd.to_datetime(self.end).year

            # Data Accessing
            all_hols = []
            for year in range(year_start, year_end+1):
                url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
                response = requests.get(url).json()
                # Parsing
                for h in response:
                    hol_date = pd.to_datetime(h['date'])
                    if pd.to_datetime(self.start) <= hol_date <= pd.to_datetime(self.end):
                        # Keep only date and localName
                        all_hols.append({'date': h['date'], 'localName': h['localName']})
            df = pd.DataFrame(all_hols).astype(str)
            df.drop_duplicates(inplace=True)
            self.verify_data(df, f"Holidays ({self.country})")
            self.hk_holidays = df
            dbg_print("Finished get_hk_holidays.")
        except Exception as e:
            print(f"Failed to fetch Holidays: {e}")

    def get_non_local_holidays(self):
        try:
            year_start = pd.to_datetime(self.start).year
            year_end = pd.to_datetime(self.end).year
            countries = ['US', 'GB', 'JP']
            holidays = []
            for year in range(year_start, year_end+1):
                for country in countries:
                    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"
                    resp = requests.get(url).json()
                    for h in resp:
                        hol_date = pd.to_datetime(h['date'])
                        if pd.to_datetime(self.start) <= hol_date <= pd.to_datetime(self.end):
                            holidays.append({'country': country, 'date': h['date'], 'localName': h['localName']})
            df = pd.DataFrame(holidays).astype(str)
            df.drop_duplicates(inplace=True)
            self.verify_data(df, "Non-local Holidays")
            self.non_local_holidays = df
            dbg_print("Finished get_non_local_holidays.")
        except Exception as e:
            print(f"Failed to fetch Non-local Holidays: {e}")

    def get_lunar_calendar(self):
        chinese_new_year_dates = {
            2022: '2022-02-01',
            2023: '2023-01-22',
            2024: '2024-02-10',
            2025: '2025-01-29'
        }
        try:
            start_dt = pd.to_datetime(self.start)
            end_dt = pd.to_datetime(self.end)
            events = []
            for y, d_str in chinese_new_year_dates.items():
                cny_date = pd.to_datetime(d_str)
                if start_dt <= cny_date <= end_dt:
                    events.append({'gregorian_date': cny_date.date(), 'lunar_event': f"Chinese New Year {y}"})
            df = pd.DataFrame(events)
            if not df.empty:
                self.verify_data(df, "Lunar Calendar Effect")
                self.lunar_calendar = df
            else:
                print("No Lunar Calendar events found in the given range.")
            dbg_print("Finished get_lunar_calendar.")
        except Exception as e:
            print(f"Failed to fetch Lunar Calendar data: {e}")

    def get_aging_population_data(self):
        try:
            if self.country == 'hong kong':
                country_code = 'HKG'
            elif self.country == 'south korea':
                country_code = 'KOR'
            else:
                country_code = 'USA'

            indicator = 'SP.POP.65UP.TO.ZS'
            data = wb.download(indicator=indicator, country=country_code, start=2000, end=2021)
            data = data.apply(pd.to_numeric, errors='coerce')
            data.reset_index(inplace=True)
            data.rename(columns={indicator: 'population_over_65_percent'}, inplace=True)

            if data.empty:
                print("Aging Population data is empty after conversion.")
            else:
                self.verify_data(data, f"Aging Population ({self.country})", allow_all_zeros=False)
                self.aging_population = data
            dbg_print("Finished get_aging_population_data.")
        except Exception as e:
            print(f"Failed to fetch Aging Population data: {e}")

    def get_google_trends(self, tickers=None):
        def fetch_trend_data(pytrends, kw_list, timeframe, geo, retries=3, wait=60):
            for i in range(retries):
                try:
                    pytrends.build_payload(kw_list, timeframe=timeframe, geo=geo)
                    df = pytrends.interest_over_time()
                    return df
                except Exception as e:
                    if "429" in str(e) or "Read timed out" in str(e):
                        time.sleep(wait * (i+1))
                    else:
                        raise e
            print(f"Failed to fetch Google Trends data for {kw_list} after {retries} retries.")
            return pd.DataFrame()

        try:
            if self.country == 'hong kong':
                local_geo = 'HK'
            elif self.country == 'south korea':
                local_geo = 'KR'
            else:
                local_geo = 'US'
            timeframe = f"{self.start} {self.end}"

            if tickers is None:
                tickers = self.get_country_tickers()
                if not tickers:
                    print("No country tickers available for Google Trends.")
                    return

            pytrends = TrendReq(hl='en-US', tz=360)
            local_dfs = []
            us_dfs = []
            for t in tickers:
                time.sleep(2)
                df_local = fetch_trend_data(pytrends, [t], timeframe, local_geo)
                time.sleep(2)
                df_us = fetch_trend_data(pytrends, [t], timeframe, 'US')

                if df_local is not None and not df_local.empty:
                    df_local = df_local[df_local['isPartial'] == True]
                    if not df_local.empty:
                        df_local['ticker'] = t
                        local_dfs.append(df_local)

                if df_us is not None and not df_us.empty:
                    df_us = df_us[df_us['isPartial'] == True]
                    if not df_us.empty:
                        df_us['ticker'] = t
                        us_dfs.append(df_us)

            if local_dfs:
                self.google_trends_local_df = pd.concat(local_dfs)
            if us_dfs:
                self.google_trends_us_df = pd.concat(us_dfs)

            dbg_print("Finished get_google_trends.")
        except Exception as e:
            print(f"Failed to fetch Google Trends data: {e}")

    def get_bitcoin_prices(self):
        try:
            start_dt = pd.to_datetime(self.start)
            end_dt = pd.to_datetime(self.end)
            days = (end_dt - start_dt).days
            if days <= 0:
                print("Invalid date range for Bitcoin Prices.")
                return
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
            # Data Accessing
            response = requests.get(url)
            if response.status_code != 200:
                print("Failed to fetch Bitcoin data: Non-200 response.")
                return
            data = response.json()
            prices = data.get('prices', [])
            if not prices:
                print("No Bitcoin price data returned by CoinGecko.")
                return
            # Parsing
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            if df.empty:
                print("Bitcoin price data is empty after filtering by date.")
                return
            self.verify_data(df[['price']], "Bitcoin - Daily Closed Price")
            self.bitcoin_prices = df[['date', 'price']]
            dbg_print("Finished get_bitcoin_prices.")
        except Exception as e:
            print(f"Failed to fetch Bitcoin Prices: {e}")

    def get_crypto_volume(self):
        try:
            start_dt = pd.to_datetime(self.start)
            end_dt = pd.to_datetime(self.end)
            days = (end_dt - start_dt).days
            if days <= 0:
                print("Invalid date range for Crypto Volume.")
                return
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}"
            # Data Accessing
            response = requests.get(url)
            if response.status_code != 200:
                print("Failed to fetch Crypto volume data: Non-200 response.")
                return
            data = response.json()
            volumes = data.get('total_volumes', [])
            if not volumes:
                print("No Crypto volume data returned.")
                return
            # Parsing
            df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]
            if df.empty:
                print("Crypto volume data is empty after filtering by date.")
                return
            self.verify_data(df[['volume']], "Cryptocurrency Trading Volume")
            self.crypto_volume = df[['date', 'volume']]
            dbg_print("Finished get_crypto_volume.")
        except Exception as e:
            print(f"Failed to fetch Crypto Volume: {e}")

    def get_analyst_opinions(self):
        dbg_print("Finished get_analyst_opinions (empty).")

    def get_institutional_outlook(self):
        try:
            url = "https://www.bloomberg.com/markets/rss"
            feed = feedparser.parse(url)
            if not feed.entries:
                print("No data returned for Institutional Outlook.")
                dbg_print("Finished get_institutional_outlook with no data.")
                return
            items = []
            for entry in feed.entries[:20]:
                title = entry.title
                if 'expect' in title.lower() or 'forecast' in title.lower():
                    outlook = 'outlook provided'
                else:
                    outlook = 'no specific outlook'
                items.append({'date': entry.published, 'institution_name': 'Various (Bloomberg)', 'market_outlook': outlook, 'title': title})
            df = pd.DataFrame(items)
            if df.empty:
                print("Institutional Outlook data is empty.")
                dbg_print("Finished get_institutional_outlook with empty data.")
                return
            self.verify_data(df, "Bank/Hedgefunds Opinion")
            self.institutional_outlook = df
            dbg_print("Finished get_institutional_outlook.")
        except Exception as e:
            print(f"Failed to fetch Institutional Outlook: {e}")

    def get_connected_markets(self):
        try:
            # Return list of dataframes with Adj Close and Volume
            if self.country == 'hong kong':
                indexes = ['^HSI', '^HSCE']
            elif self.country == 'south korea':
                indexes = ['^KS11', '^KQ11']
            else:
                indexes = ['^GSPC', '^IXIC']

            dfs = []
            for idx in indexes:
                df = yf.download(idx, start=self.start, end=self.end, progress=False)
                if not df.empty:
                    # Extract only Adj Close and Volume
                    df = df[['Adj Close', 'Volume']]
                    df['Index'] = idx
                    dfs.append(df)
                else:
                    print(f"No data returned for connected market index {idx}.")
            self.connected_markets = dfs
            dbg_print("Finished get_connected_markets.")
        except Exception as e:
            print(f"Failed to fetch Connected Markets data: {e}")

    def get_governance_scores(self):
        dbg_print("Finished get_governance_scores (empty).")

    def load_all_data(self):
        self.get_themed_characteristics()
        self.get_weather_data()
        self.get_powell_announcements()
        self.get_hk_holidays()
        self.get_non_local_holidays()
        self.get_lunar_calendar()
        self.get_aging_population_data()
        #self.get_google_trends()
        self.get_bitcoin_prices()
        self.get_crypto_volume()
        self.get_analyst_opinions()
        self.get_institutional_outlook()
        self.get_connected_markets()
        self.get_governance_scores()
        dbg_print("Finished load_all_data.")
