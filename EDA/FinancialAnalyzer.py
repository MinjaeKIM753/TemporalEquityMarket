import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class FinancialAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = None

    def get_data(self, start_date, end_date):
        self.data = yf.Ticker(self.ticker).history(start=start_date, end=end_date)
        return self.data

    def calculate_sma(self, window=50):
        return self.data['Close'].rolling(window=window).mean()

    def calculate_ema(self, span):
        return self.data['Close'].ewm(span=span, adjust=False).mean()

    def calculate_dema(self, period=20):
        ema1 = self.calculate_ema(period)
        ema2 = self.calculate_ema(period)
        return 2 * ema1 - ema2

    def calculate_tema(self, period=20):
        ema1 = self.calculate_ema(period)
        ema2 = self.calculate_ema(period)
        ema3 = self.calculate_ema(period)
        return 3 * ema1 - 3 * ema2 + ema3

    def calculate_macd(self, fast=12, slow=26, signal=9):
        ema_fast = self.data['Close'].ewm(span=fast, adjust = False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust = False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span = signal, adjust = False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def calculate_ppo(self, fast_period=12, slow_period=26, signal_period=9):
        fast_ema = self.data['Close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.data['Close'].ewm(span=slow_period, adjust=False).mean()
        ppo = ((fast_ema - slow_ema) / slow_ema) * 100
        signal_line = ppo.ewm(span=signal_period, adjust=False).mean()
        histogram = ppo - signal_line
        return ppo, signal_line, histogram

    def calculate_rsi(self, window=14):
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_stochastic_rsi(self, period=14, smooth_k=3, smooth_d=3):
        rsi = self.calculate_rsi(period)
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        k = stoch_rsi.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        return k, d

    def calculate_stochastic(self, k_period=14, d_period=3, slow_period=3):
        low_min = self.data['Low'].rolling(window=k_period).min()
        high_max = self.data['High'].rolling(window=k_period).max()
        k_fast = 100 * (self.data['Close'] - low_min) / (high_max - low_min)
        d_fast = k_fast.rolling(window=d_period).mean()
        k_slow = d_fast
        d_slow = k_slow.rolling(window=slow_period).mean()
        return k_fast, d_fast, k_slow, d_slow

    def calculate_price_roc(self, period=12):
        return ((self.data['Close'] - self.data['Close'].shift(period)) / self.data['Close'].shift(period)) * 100

    def calculate_bollinger_bands(self, window=20, num_std=2):
        sma = self.calculate_sma(window)
        std = self.data['Close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return sma, upper_band, lower_band

    def calculate_keltner_channels(self, ema_window=20, atr_window=10, multiplier=2):
        ema = self.calculate_ema(ema_window)
        atr = self.calculate_atr(atr_window)
        upper = ema + multiplier * atr
        lower = ema - multiplier * atr
        return ema, upper, lower

    def calculate_atr(self, window=14):
        tr = np.maximum(
            self.data['High'] - self.data['Low'],
            np.abs(self.data['High'] - self.data['Close'].shift(1)),
            np.abs(self.data['Low'] - self.data['Close'].shift(1))
        )
        return tr.rolling(window=window).mean()

    def calculate_adx(self, period=14):
        plus_dm = self.data['High'].diff()
        minus_dm = self.data['Low'].shift(1) - self.data['Low']
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = np.maximum(
            self.data['High'] - self.data['Low'],
            np.abs(self.data['High'] - self.data['Close'].shift(1)),
            np.abs(self.data['Low'] - self.data['Close'].shift(1))
        )
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * plus_dm.rolling(window=period).sum() / atr
        minus_di = 100 * minus_dm.rolling(window=period).sum() / atr

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return plus_di, minus_di, adx

    def calculate_obv(self):
        return (np.sign(self.data['Close'].diff()) * self.data['Volume']).cumsum()

    def calculate_cmf(self, period=20):
        mfm = ((self.data['Close'] - self.data['Low']) - (self.data['High'] - self.data['Close'])) / (self.data['High'] - self.data['Low'])
        mfv = mfm * self.data['Volume']
        cmf = mfv.rolling(window=period).sum() / self.data['Volume'].rolling(window=period).sum()
        return cmf

    def calculate_chaikin_osc(self, fast_period=3, slow_period=10):
        mfm = ((self.data['Close'] - self.data['Low']) - (self.data['High'] - self.data['Close'])) / (self.data['High'] - self.data['Low'])
        mfm = mfm.fillna(0)
        mfv = mfm * self.data['Volume']

        chaikin_osc = self.calculate_ema(fast_period) - self.calculate_ema(slow_period)
        return chaikin_osc

    def calculate_force_index(self, period=13):
        fi = (self.data['Close'] - self.data['Close'].shift(1)) * self.data['Volume']
        return fi.ewm(span=period, adjust=False).mean()

    def calculate_ar_br(self, ar_period=26, br_period=26):
        ar_high = self.data['High'] - self.data['Open']
        ar_low = self.data['Open'] - self.data['Low']
        ar = (ar_high.rolling(window=ar_period).sum() / ar_low.rolling(window=ar_period).sum()) * 100

        prev_close = self.data['Close'].shift(1)
        br_high = self.data['High'] - prev_close
        br_low = prev_close - self.data['Low']
        br = (br_high.rolling(window=br_period).sum() / br_low.rolling(window=br_period).sum()) * 100

        return ar, br

    def calculate_trix(self, window=15):
        ema1 = self.calculate_ema(window)
        ema2 = self.calculate_ema(window)
        ema3 = self.calculate_ema(window)
        return (ema3 - ema3.shift(1)) / ema3.shift(1) * 100

    def calculate_mass_index(self, window=9, window2=25):
        range_ema1 = self.calculate_ema(window)
        range_ema2 = self.calculate_ema(window)
        mass = range_ema1 / range_ema2
        return mass.rolling(window=window2).sum()

    def calculate_parabolic_sar(self, step=0.02, max_step=0.2):
        sar = self.data['Low'].copy()
        af = pd.Series(step, index=sar.index)
        ep = self.data['High'].copy()
        long = True

        for i in range(1, len(sar)):
            if long:
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            else:
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

            if long and sar[i] > self.data['Low'].iloc[i]:
                long = False
                sar[i] = ep.iloc[:i+1].max()
                ep[i] = self.data['Low'].iloc[i]
            elif not long and sar[i] < self.data['High'].iloc[i]:
                long = True
                sar[i] = ep.iloc[:i+1].min()
                ep[i] = self.data['High'].iloc[i]

            if long:
                ep[i] = max(ep[i], self.data['High'].iloc[i])
            else:
                ep[i] = min(ep[i], self.data['Low'].iloc[i])

            af[i] = min(af[i-1] + step, max_step) if ep[i] != ep[i-1] else af[i-1]

        return sar

    def calculate_three_line_break(self, window=3):
        tlb = pd.Series(index=self.data.index, dtype=float)
        last_high = self.data['Close'].iloc[0]
        last_low = self.data['Close'].iloc[0]

        for i in range(window, len(self.data['Close'])):
            if self.data['Close'].iloc[i] > last_high:
                tlb.iloc[i] = 1
                last_high = self.data['Close'].iloc[i]
                last_low = min(self.data['Close'].iloc[i-window+1:i+1])
            elif self.data['Close'].iloc[i] < last_low:
                tlb.iloc[i] = -1
                last_low = self.data['Close'].iloc[i]
                last_high = max(self.data['Close'].iloc[i-window+1:i+1])
            else:
                tlb.iloc[i] = 0

        return tlb

    def calculate_ichimoku(self, conversion_period=9, base_period=26, leading_span_b_period=52):
        conversion_line = (self.data['High'].rolling(window=conversion_period).max() + self.data['Low'].rolling(window=conversion_period).min()) / 2
        base_line = (self.data['High'].rolling(window=base_period).max() + self.data['Low'].rolling(window=base_period).min()) / 2
        leading_span_a = ((conversion_line + base_line) / 2).shift(base_period)
        leading_span_b = ((self.data['High'].rolling(window=leading_span_b_period).max() + self.data['Low'].rolling(window=leading_span_b_period).min()) / 2).shift(base_period)
        lagging_span = self.data['Close'].shift(-base_period)

        return pd.DataFrame({
            'Conversion Line': conversion_line,
            'Base Line': base_line,
            'Leading Span A': leading_span_a,
            'Leading Span B': leading_span_b,
            'Lagging Span': lagging_span
        })

    def calculate_disparity(self, windows=[5, 10, 20, 60, 120]):
        disparities = pd.DataFrame(index=self.data.index)
        for window in windows:
            ma = self.calculate_sma(window)
            disparity = (self.data['Close'] / ma) * 100
            disparities[f'Disparity_{window}'] = disparity
        return disparities

    def calculate_adxr(self, period=14):
        pdi, mdi, adx = self.calculate_adx(period)
        adxr = (adx + adx.shift(period)) / 2
        return pdi, mdi, adx, adxr


    def analyze_indicators(self):
        analysis = pd.DataFrame({
            'Trend': ['Bullish' if self.data['Close'].iloc[-1] > self.data['SMA_200'].iloc[-1] else 'Bearish'],
            'MACD': ['Bullish' if self.data['MACD'].iloc[-1] > self.data['MACD_Signal'].iloc[-1] else 'Bearish'],
            'RSI': [self.data['RSI'].iloc[-1]],
            'Stochastic': ['Overbought' if self.data['K_Fast'].iloc[-1] > 80 else 'Oversold' if self.data['K_Fast'].iloc[-1] < 20 else 'Neutral'],
            'BB_Width': [(self.data['BB_Upper'].iloc[-1] - self.data['BB_Lower'].iloc[-1]) / self.data['Close'].iloc[-1]],
            'ATR': [self.data['ATR'].iloc[-1]],
            'OBV_Trend': ['Positive' if self.data['OBV'].diff().iloc[-1] > 0 else 'Negative'],
            'CMF': [self.data['CMF'].iloc[-1]],
            'Force_Index': [self.data['Force_Index'].iloc[-1]],
            'ADX': [self.data['ADX'].iloc[-1]],
            'ADXR': [self.data['ADXR'].iloc[-1]],
            'Parabolic_SAR': ['Bullish' if self.data['Parabolic_SAR'].iloc[-1] < self.data['Close'].iloc[-1] else 'Bearish'],
            'Ichimoku_Trend': ['Bullish' if (self.data['Close'].iloc[-1] > self.data['Leading Span A'].iloc[-1] and
                                             self.data['Close'].iloc[-1] > self.data['Leading Span B'].iloc[-1])
                               else 'Bearish' if (self.data['Close'].iloc[-1] < self.data['Leading Span A'].iloc[-1] and
                                                  self.data['Close'].iloc[-1] < self.data['Leading Span B'].iloc[-1])
                               else 'Neutral'],
            'Disparity': ['Overbought' if self.data['Disparity_10'].iloc[-1] > 105 else 'Oversold' if self.data['Disparity_10'].iloc[-1] < 95 else 'Neutral'],
            'Price_ROC': [self.data['Price_ROC'].iloc[-1]],
            'AR': [self.data['AR'].iloc[-1]],
            'BR': [self.data['BR'].iloc[-1]],
            'TRIX': [self.data['TRIX'].iloc[-1]],
            'Mass_Index': [self.data['Mass_Index'].iloc[-1]],
            'Three_Line_Break': ['Bullish' if self.data['Three_Line_Break'].iloc[-1] > 0 else 'Bearish' if self.data['Three_Line_Break'].iloc[-1] < 0 else 'Neutral'],
        })
        return analysis

    def visualize_moving_averages(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['Close'], label='Close', color='black')
        ax.plot(self.data.index, self.data['SMA_50'], label='50-day SMA', linewidth=0.75)
        ax.plot(self.data.index, self.data['SMA_200'], label='200-day SMA', linewidth=0.75)
        ax.plot(self.data.index, self.data['EMA_50'], label='50-day EMA', linewidth=0.75)
        ax.plot(self.data.index, self.data['EMA_200'], label='200-day EMA', linewidth=0.75)
        ax.plot(self.data.index, self.data['DEMA_20'], label='20-day DEMA', linewidth=0.75)
        ax.plot(self.data.index, self.data['TEMA_20'], label='20-day TEMA', linewidth=0.75)
        ax.set_title('Moving Averages (Trend Indicator) - Price Trend Analysis\n'
                     f'SMA Periods: 50, 200; EMA Periods: 50, 200; DEMA Period: 20; TEMA Period: 20')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        ax.legend()

        if show_details:
            equations = (r'Equations:' '\n'
                         r'$SMA_n = \frac{1}{n}\sum_{i=1}^n P_i$' '\n'
                         r'$EMA_t = \alpha \cdot P_t + (1-\alpha) \cdot EMA_{t-1}, \alpha = \frac{2}{n+1}$' '\n'
                         r'$DEMA = 2 \cdot EMA_n - EMA(EMA_n)$' '\n'
                         r'$TEMA = 3 \cdot EMA_n - 3 \cdot EMA(EMA_n) + EMA(EMA(EMA_n))$')
            interpretation = ('Interpretation:' '\n'
                              ' - Price > Long-Term MA : Bullish' '\n'
                              ' - Price < Long-Term MA : Bearish' '\n'
                              ' - Shorter-Term MAs > Longer-Term MAs : Bullish' '\n'
                              ' - Shorter-Term MAs < Longer-Term MAs : Bearish')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_ichimoku_cloud(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['Close'], label='Close', color='black')
        ax.plot(self.data.index, self.data['Conversion Line'], label='Conversion Line', color='red', linewidth=0.75)
        ax.plot(self.data.index, self.data['Base Line'], label='Base Line', color='blue', linewidth=0.75)
        ax.plot(self.data.index, self.data['Leading Span A'], label='Leading Span A', color='green', linewidth=0.75)
        ax.plot(self.data.index, self.data['Leading Span B'], label='Leading Span B', color='brown', linewidth=0.75)
        ax.fill_between(self.data.index, self.data['Leading Span A'], self.data['Leading Span B'],
                        where=self.data['Leading Span A'] >= self.data['Leading Span B'], facecolor='limegreen', alpha=0.5)
        ax.fill_between(self.data.index, self.data['Leading Span A'], self.data['Leading Span B'],
                        where=self.data['Leading Span A'] < self.data['Leading Span B'], facecolor='lightcoral', alpha=0.5)
        ax.plot(self.data.index, self.data['Lagging Span'], label='Lagging Span', color='orange', linewidth=0.75)
        ax.set_title('Ichimoku Cloud (Trend Indicator) - Trend and Support/Resistance Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        ax.legend()

        if show_details:
            equations = (r'Equations:' '\n'
                         r'$Conversion Line = \frac{9d High + 9d Low}{2}$' '\n'
                         r'$Base Line = \frac{26d High + 26d Low}{2}$' '\n'
                         r'$Leading Span A = \frac{Conversion Line + Base Line}{2}$' '\n'
                         r'$Leading Span B = \frac{52d High + 52d Low}{2}$' '\n'
                         r'$Lagging Span = Close_{t-26}$')
            interpretation = ('Interpretation:' '\n'
                              ' - Price above cloud : Bullish' '\n'
                              ' - Price below cloud : Bearish' '\n'
                              ' - Conversion Line > Base Line : Bullish' '\n'
                              ' - Conversion Line < Base Line : Bearish' '\n'
                              ' - Green cloud : Bullish' '\n'
                              ' - Red cloud : Bearish')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_dmi_adx_adxr(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['ADX'], label='ADX', color='purple')
        ax.plot(self.data.index, self.data['ADXR'], label='ADXR', color='magenta')
        ax.fill_between(self.data.index, self.data['PDI'], self.data['MDI'],
                        where=self.data['PDI'] >= self.data['MDI'], facecolor='green', alpha=0.3)
        ax.fill_between(self.data.index, self.data['PDI'], self.data['MDI'],
                        where=self.data['PDI'] < self.data['MDI'], facecolor='red', alpha=0.3)
        ax.plot(self.data.index, self.data['PDI'], label='PDI', color='green', linewidth=0.75)
        ax.plot(self.data.index, self.data['MDI'], label='MDI', color='red', linewidth=0.75)
        ax.set_title('DMI, ADX, and ADXR (Trend Strength Indicators) - Directional Movement Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend()

        if show_details:
            equations = (r'Equations:' '\n'
                         r'$DM^+ = H_t - H_{t-1}, DM^- = L_{t-1} - L_t$' '\n'
                         r'$TR = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$' '\n'
                         r'$DI^+ = 100 \cdot \frac{SMA(DM^+)}{SMA(TR)}, DI^- = 100 \cdot \frac{SMA(DM^-)}{SMA(TR)}$' '\n'
                         r'$ADX = 100 \cdot \frac{SMA(|DI^+ - DI^-|)}{DI^+ + DI^-}$' '\n'
                         r'$ADXR = \frac{ADX_t + ADX_{t-n}}{2}$')
            interpretation = ('Interpretation:' '\n'
                              ' - ADX and ADXR > 25 : Strong Trend' '\n'
                              ' - PDI > MDI : Bullish' '\n'
                              ' - MDI > PDI : Bearish')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_disparity_index(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        disparity_windows = [5, 10, 20, 60, 120]
        for window in disparity_windows:
            if len(self.data) >= window:
                ax.plot(self.data.index, self.data[f'Disparity_{window}'], label=f'{window}-day')
        ax.axhline(y=100, color='grey', linestyle='--', label='Baseline')
        ax.set_title('Disparity Index (Trend Indicator) - Price Divergence from Moving Averages')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Disparity', fontsize=14)
        ax.legend()

        if show_details:
            equations = r'Equation: $Disparity_n = \frac{Close}{SMA_n} \times 100$'
            interpretation = ('Interpretation:' '\n'
                              ' - Disparity > 100 : Price > MA (and potentially Overbought)' '\n'
                              ' - Disparity < 100 : Price < MA (and potentially Oversold)' '\n'
                              ' - Price = Lower Low and Disparity = Higher Low : Bullish (Bullish Reversal)' '\n'
                              ' - Price = Higher High and Disparity = Lower High : Bearish (Bearish Reversal)')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_macd_ppo(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax2 = ax.twinx()

        # MACD
        ax.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        ax.plot(self.data.index, self.data['MACD_Signal'], label='MACD Signal', color='red', linestyle='dashdot')
        ax.bar(self.data.index, self.data['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3, width=1)

        # PPO
        ax2.plot(self.data.index, self.data['PPO'], label='PPO', color='navy')
        ax2.plot(self.data.index, self.data['PPO_Signal'], label='PPO Signal', color='orange', linestyle='dashdot')
        ax2.bar(self.data.index, self.data['PPO_Histogram'], label='PPO Histogram', color='lightgray', alpha=0.3, width=1)

        # Shared zero line
        ax.axhline(y=0, color='grey', linestyle='--', label='Zero Line')

        ax.set_title('MACD and PPO (Momentum Indicators) - Trend Momentum Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('MACD', fontsize=14)
        ax2.set_ylabel('PPO', fontsize=14)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        if show_details:
            equations = (r'Equations:' '\n'
                         r'MACD = EMA_{12}(Close) - EMA_{26}(Close)' '\n'
                         r'Signal = EMA_9(MACD)' '\n'
                         r'Histogram = MACD - Signal' '\n'
                         r'PPO = \frac{EMA_{12}(Close) - EMA_{26}(Close)}{EMA_{26}(Close)} \times 100')
            interpretation = ('Interpretation:' '\n'
                              ' - MACD/PPO Line > Signal : Bullish' '\n'
                              ' - MACD/PPO Line < Signal : Bearish' '\n'
                              ' - Histogram > 0 : Bullish' '\n'
                              ' - Histogram < 0 : Bearish' '\n'
                              ' - Price = Lower Low and MACD/PPO = Higher Low : Bullish' '\n'
                              ' - Price = Higher High and MACD/PPO = Lower High : Bearish')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_rsi_stochastic_rsi(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax2 = ax.twinx()

        # RSI
        ax.plot(self.data.index, self.data['RSI'], label='RSI')
        ax.axhline(y=70, color='r', linestyle='--', label='Overbought (70, 0.7)')
        ax.axhline(y=30, color='g', linestyle='--', label='Oversold (30, 0.3)')
        ax.fill_between(self.data.index, 70, 100, color='r', alpha=0.1)
        ax.fill_between(self.data.index, 0, 30, color='g', alpha=0.1)

        # Stochastic RSI
        ax2.plot(self.data.index, self.data['Stoch_RSI_K'], label='Stoch RSI %K', color='orange')
        ax2.plot(self.data.index, self.data['Stoch_RSI_D'], label='Stoch RSI %D', color='purple')

        ax.set_title('RSI and Stochastic RSI (Momentum Indicators) - Overbought/Oversold Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('RSI', fontsize=14)
        ax2.set_ylabel('Stochastic RSI', fontsize=14)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, 1)

        if show_details:
            equations = (r'Equations:' '\n'
                         r'RSI = 100 - \frac{100}{1 + \frac{EMA(U, n)}{EMA(D, n)}}' '\n'
                         r'StochRSI = \frac{RSI - RSI_{min}}{RSI_{max} - RSI_{min}}')
            interpretation = ('Interpretation:' '\n'
                              'RSI:' '\n'
                              ' - RSI > 70 : Overbought' '\n'
                              ' - RSI < 30 : Oversold' '\n'
                              'Stochastic RSI:' '\n'
                              ' - Values > 0.8 : Overbought' '\n'
                              ' - Values < 0.2 : Oversold' '\n'
                              ' - %K crosses above %D : Bullish' '\n'
                              ' - %K crosses below %D : Bearish')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_stochastic_oscillator(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['K_Fast'], label='%K Fast', color='purple')
        ax.plot(self.data.index, self.data['D_Fast'], label='%D Fast', color='navy')
        ax.plot(self.data.index, self.data['K_Slow'], label='%K Slow', color='violet')
        ax.plot(self.data.index, self.data['D_Slow'], label='%D Slow', color='blue')
        ax.axhline(y=80, color='r', linestyle='--', label='Overbought (80)')
        ax.axhline(y=20, color='g', linestyle='--', label='Oversold (20)')
        ax.fill_between(self.data.index, 80, 100, color='r', alpha=0.1)
        ax.fill_between(self.data.index, 0, 20, color='g', alpha=0.1)
        ax.set_ylim(0, 100)
        ax.set_title('Stochastic Oscillator (Momentum Indicator) - Overbought/Oversold Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend()

        if show_details:
            equations = (r'Equations:' '\n'
                         r'$\%K = \frac{C_t - L_{14}}{H_{14} - L_{14}} \times 100$' '\n'
                         r'$\%D = SMA_3(\%K)$')
            interpretation = ('Interpretation:' '\n'
                              ' - Values > 80 : Overbought' '\n'
                              ' - Values < 20 : Oversold' '\n'
                              ' - %K > %D : Bullish' '\n'
                              ' - %K < %D : Bearish' '\n'
                              ' - Divergences can signal potential reversals')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_price_roc(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['Price_ROC'], label='Price ROC')
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.fill_between(self.data.index, 0, self.data['Price_ROC'], where=self.data['Price_ROC'] >= 0, facecolor='green', alpha=0.3, label='+ve Momentum')
        ax.fill_between(self.data.index, 0, self.data['Price_ROC'], where=self.data['Price_ROC'] < 0, facecolor='red', alpha=0.3, label='-ve Momentum')
        ax.set_title('Price Rate of Change (Momentum Indicator) - Price Momentum Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('ROC', fontsize=14)
        ax.legend()

        if show_details:
            equations = r'Equation: $ROC = \frac{Close_t - Close_{t-n}}{Close_{t-n}} \times 100$'
            interpretation = ('Interpretation:' '\n'
                              ' - Values > 0 : +ve Momentum' '\n'
                              ' - Values < 0 : -ve Momentum' '\n'
                              ' - Value crossing 0 : Potential Trend Changes' '\n'
                              ' - Divergences can signal potential reversals')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_bollinger_keltner(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['Close'], label='Close', color='black')
        ax.plot(self.data.index, self.data['BB_Upper'], label='BB Upper', color='red', linestyle=':')
        ax.plot(self.data.index, self.data['BB_Lower'], label='BB Lower', color='green', linestyle=':')
        ax.plot(self.data.index, self.data['KC_Upper'], label='KC Upper', color='red', linestyle='--')
        ax.plot(self.data.index, self.data['KC_Lower'], label='KC Lower', color='green', linestyle='--')
        ax.set_title('Bollinger Bands and Keltner Channels (Volatility Indicators) - Price Volatility Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        ax.legend()

        if show_details:
            equations = (r'Equations:' '\n'
                         r'Bollinger Bands:' '\n'
                         r'$BB_{Upper} = SMA_n + k \times \sigma_n$' '\n'
                         r'$BB_{Lower} = SMA_n - k \times \sigma_n$' '\n'
                         r'Keltner Channels:' '\n'
                         r'$KC_{Upper} = EMA_n + a \times ATR_n$' '\n'
                         r'$KC_{Lower} = EMA_n - a \times ATR_n$')
            interpretation = ('Interpretation:' '\n'
                              ' - Price > Upper : Overbought' '\n'
                              ' - Price < Lower : Oversold' '\n'
                              ' - Narrowing Band : Low Volatility' '\n'
                              ' - Widening Band : High Volatility')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_atr(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['ATR'], label='ATR')
        ax.set_title('Average True Range (Volatility Indicator) - Price Volatility Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('ATR', fontsize=14)
        ax.legend()

        if show_details:
            equations = r'Equation: $ATR = EMA_n(TR), TR = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$'
            interpretation = ('Interpretation:' '\n'
                              ' - High ATR : High Volatility' '\n'
                              ' - Low ATR : Low Volatility')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_obv(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['OBV'], label='OBV', color='blue')
        ax.set_title('On-Balance Volume (Volume Indicator) - Volume Trend Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('OBV', fontsize=14)
        ax.legend()

        if show_details:
            equations = (r'Equation:' '\n'
                         r'$OBV_t = OBV_{t-1} + Volume_t$ if $Close_t > Close_{t-1}$' '\n'
                         r'$OBV_t = OBV_{t-1}$ if $Close_t = Close_{t-1}$' '\n'
                         r'$OBV_t = OBV_{t-1} - Volume_t$ if $Close_t < Close_{t-1}$')
            interpretation = ('Interpretation:' '\n'
                              ' - Rising OBV : Uptrend' '\n'
                              ' - Falling OBV : Downtrend' '\n'
                              ' - Divergences can signal potential reversals')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_cmf_chaikin_osc(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax2 = ax.twinx()

        # Chaikin Money Flow
        ax.plot(self.data.index, self.data['CMF'], label='CMF', color='green')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.fill_between(self.data.index, 0, self.data['CMF'], where=self.data['CMF'] >= 0, facecolor='green', alpha=0.1, label='CMF +ve')
        ax.fill_between(self.data.index, 0, self.data['CMF'], where=self.data['CMF'] < 0, facecolor='red', alpha=0.1, label='CMF -ve')

        # Chaikin Oscillator
        ax2.plot(self.data.index, self.data['Chaikin_OSC'], label='Chaikin OSC', color='gold')
        ax2.fill_between(self.data.index, 0, self.data['Chaikin_OSC'], where=self.data['Chaikin_OSC'] >= 0, facecolor='green', alpha=0.1, label='Chaikin OSC +ve')
        ax2.fill_between(self.data.index, 0, self.data['Chaikin_OSC'], where=self.data['Chaikin_OSC'] < 0, facecolor='red', alpha=0.1, label='Chaikin OSC -ve')

        ax.set_title('Chaikin Money Flow and Chaikin Oscillator (Volume Indicators) - Volume-Price Relationship Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('CMF', fontsize=14)
        ax2.set_ylabel('Chaikin Oscillator', fontsize=14)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        if show_details:
            equations = (r'Equation:' '\n'
                         r'MFM = \frac{Close - Low}{High - Low} - \frac{High - Close}{High - Low}' '\n'
                         r'CMF = \frac{\sum_{i=1}^n MFM_i \times Volume_i}{\sum_{i=1}^n Volume_i}' '\n'
                         r'ADL = \sum (MFM \times Volume)' '\n'
                         r'Chaikin OSC = EMA_3(ADL) - EMA_{10}(ADL)')
            interpretation = ('Interpretation:' '\n'
                              ' - CMF > 0 : Buying Pressure' '\n'
                              ' - CMF < 0 : Selling Pressure' '\n'
                              ' - Chaikin OSC > 0 : Bullish' '\n'
                              ' - Chaikin OSC < 0 : Bearish')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_force_index(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['Force_Index'], label='Force Index', color='black')
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.fill_between(self.data.index, 0, self.data['Force_Index'], where=self.data['Force_Index'] >= 0, facecolor='green', alpha=0.3, label='+ve FI')
        ax.fill_between(self.data.index, 0, self.data['Force_Index'], where=self.data['Force_Index'] < 0, facecolor='red', alpha=0.3, label='-ve FI')
        ax.set_title('Force Index (Volume Indicator) - Price and Volume Trend Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Force Index', fontsize=14)
        ax.legend()

        if show_details:
            equations = r'Equation: $FI = (Close_t - Close_{t-1}) \times Volume_t$'
            interpretation = ('Interpretation:' '\n'
                              ' - Positive FI : Bullish Pressure' '\n'
                              ' - Negative FI : Bearish Pressure' '\n'
                              ' - FI crossing 0 : Potential Trend Changes')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_ar_br(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['AR'], label='AR', color='blue')
        ax.plot(self.data.index, self.data['BR'], label='BR', color='navy')
        ax.axhline(y=100, color='grey', linestyle='--', label='Neutral Line')
        ax.fill_between(self.data.index, 100, self.data['AR'], where=self.data['AR'] >= 100, facecolor='green', alpha=0.2)
        ax.fill_between(self.data.index, self.data['AR'], 100, where=self.data['AR'] < 100, facecolor='red', alpha=0.2)
        ax.fill_between(self.data.index, 100, self.data['BR'], where=self.data['BR'] >= 100, facecolor='green', alpha=0.2)
        ax.fill_between(self.data.index, self.data['BR'], 100, where=self.data['BR'] < 100, facecolor='red', alpha=0.2)
        ax.set_title('AR (Accumulation Ratio) and BR (Buying Ratio) (Volume Indicators) - Buying Pressure Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.legend()

        if show_details:
            equations = (r'Equations:' '\n'
                         r'$AR = \frac{\sum_{i=1}^n (H_i - O_i)}{\sum_{i=1}^n (O_i - L_i)} \times 100$' '\n'
                         r'$BR = \frac{\sum_{i=1}^n (H_i - C_{i-1})}{\sum_{i=1}^n (C_{i-1} - L_i)} \times 100$')
            interpretation = ('Interpretation:' '\n'
                              ' - AR/BR > 100 : Strong Buying Pressure' '\n'
                              ' - AR/BR < 100 : Strong Selling Pressure' '\n'
                              ' - AR Rises and BR Falls : Buying Pressure' '\n'
                              ' - AR Falls and BR Rises : Selling Pressure')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_trix_mass_index(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax2 = ax.twinx()

        # TRIX
        ax.plot(self.data.index, self.data['TRIX'], label='TRIX', color='b')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.fill_between(self.data.index, 0, self.data['TRIX'], where=self.data['TRIX'] >= 0, facecolor='green', alpha=0.2)
        ax.fill_between(self.data.index, 0, self.data['TRIX'], where=self.data['TRIX'] < 0, facecolor='red', alpha=0.2)

        # Mass Index
        ax2.plot(self.data.index, self.data['Mass_Index'], label='Mass Index', color='r')
        ax2.axhline(y=27, color='r', linestyle='--', label='MI = 27')
        ax2.axhline(y=26.5, color='r', linestyle='--', label='MI = 26.5')

        ax.set_title('TRIX and Mass Index (Trend Reversal Indicators) - Trend and Reversal Analysis')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('TRIX', fontsize=14)
        ax2.set_ylabel('Mass Index', fontsize=14)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        if show_details:
            equations = (r'Equations:' '\n'
                         r'TRIX = \frac{EMA_1 - EMA_{t-1}}{EMA_{t-1}} \times 100' '\n'
                         r'EMA_1 = EMA_n(EMA_n(EMA_n(Close)))' '\n'
                         r'Mass Index = \sum_{i=1}^{25} \frac{EMA_9(H-L)}{EMA_9(EMA_9(H-L))}')
            interpretation = ('Interpretation:' '\n'
                              ' - TRIX > 0 : Bullish' '\n'
                              ' - TRIX < 0 : Bearish' '\n'
                              ' - TRIX crossing zero : Potential trend change' '\n'
                              ' - MI > 27 then < 26.5 : Potential reversal')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_parabolic_sar(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['Close'], label='Close', color='black')
        ax.scatter(self.data.index, self.data['Parabolic_SAR'], label='Parabolic SAR', color='blue', marker='.')
        ax.fill_between(self.data.index, self.data['Close'], self.data['Parabolic_SAR'],
                        where=(self.data['Parabolic_SAR'] < self.data['Close']),
                        facecolor='green', alpha=0.3, label='Bullish')
        ax.fill_between(self.data.index, self.data['Close'], self.data['Parabolic_SAR'],
                        where=(self.data['Parabolic_SAR'] >= self.data['Close']),
                        facecolor='red', alpha=0.3, label='Bearish')
        ax.set_title('Parabolic SAR (Trend and Reversal Indicator) - Trend Direction and Potential Reversals')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        ax.legend()

        if show_details:
            equations = (r'Equations:' '\n'
                         r'$SAR_{t+1} = SAR_t + \alpha(EP_t - SAR_t)$' '\n'
                         r'$\alpha_{t+1} = \min(\alpha_t + 0.02, 0.2)$ if $EP_{t+1} = EP_t$, else $0.02$' '\n'
                         r'$EP_t = $ Extreme Point (highest high for long, lowest low for short)')
            interpretation = ('Interpretation:' '\n'
                              ' - SAR below price : Uptrend' '\n'
                              ' - SAR above price : Downtrend' '\n'
                              ' - SAR crossing price : Potential trend reversal')
            fig.text(0.1, -0.05, equations, fontsize=10, verticalalignment='top')
            fig.text(0.6, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def visualize_three_line_break(self, show_details=False):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.data.index, self.data['Close'], label='Close', color='gray', alpha=0.5)
        for i in range(1, len(self.data)):
            if self.data['Three_Line_Break'].iloc[i] == 1:
                ax.bar(self.data.index[i], self.data['Close'].iloc[i] - self.data['Close'].iloc[i-1],
                       bottom=self.data['Close'].iloc[i-1], color='green', width=1)
            elif self.data['Three_Line_Break'].iloc[i] == -1:
                ax.bar(self.data.index[i], self.data['Close'].iloc[i] - self.data['Close'].iloc[i-1],
                       bottom=self.data['Close'].iloc[i], color='red', width=1)
        ax.set_title('Three Line Break Chart (Trend & Reversal Chart Technique) - Trend Direction and Potential Reversals')
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Price', fontsize=14)
        ax.legend(['Close', 'Bullish Break', 'Bearish Break'])

        if show_details:
            interpretation = ('Interpretation:' '\n'
                              ' - Green bars : Bullish trend' '\n'
                              ' - Red bars : Bearish trend' '\n'
                              ' - Alternating colors : Potential consolidation or reversal')
            fig.text(0.1, -0.05, interpretation, fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.show()

    def comprehensive_analysis(self, start_date, end_date, plot=True):
        self.get_data(start_date, end_date)
        self.calculate_all_indicators()

        if plot:
            self.visualize_moving_averages()
            self.visualize_ichimoku_cloud()
            self.visualize_dmi_adx_adxr()
            self.visualize_disparity_index()
            self.visualize_macd_ppo()
            self.visualize_rsi_stochastic_rsi()
            self.visualize_stochastic_oscillator()
            self.visualize_price_roc()
            self.visualize_bollinger_keltner()
            self.visualize_atr()
            self.visualize_obv()
            self.visualize_cmf_chaikin_osc()
            self.visualize_force_index()
            self.visualize_ar_br()
            self.visualize_trix_mass_index()
            self.visualize_parabolic_sar()
            self.visualize_three_line_break()

        performance_metrics = self.calculate_returns_and_metrics()
        indicator_analysis = self.analyze_indicators()
        interpretations = self.interpret_indicators(indicator_analysis)

        all_metrics = pd.concat([performance_metrics, indicator_analysis], axis=1)

    def calculate_all_indicators(self):
        self.data['SMA_50'] = self.calculate_sma(50)
        self.data['SMA_200'] = self.calculate_sma(200)
        self.data['EMA_50'] = self.calculate_ema(50)
        self.data['EMA_200'] = self.calculate_ema(200)
        self.data['DEMA_20'] = self.calculate_dema()
        self.data['TEMA_20'] = self.calculate_tema()
        self.data['MACD'], self.data['MACD_Signal'], self.data['MACD_Histogram'] = self.calculate_macd()
        self.data['PPO'], self.data['PPO_Signal'], self.data['PPO_Histogram'] = self.calculate_ppo()
        self.data['RSI'] = self.calculate_rsi()
        self.data['Stoch_RSI_K'], self.data['Stoch_RSI_D'] = self.calculate_stochastic_rsi()
        self.data['K_Fast'], self.data['D_Fast'], self.data['K_Slow'], self.data['D_Slow'] = self.calculate_stochastic()
        self.data['Price_ROC'] = self.calculate_price_roc()
        self.data['BB_MA'], self.data['BB_Upper'], self.data['BB_Lower'] = self.calculate_bollinger_bands()
        self.data['KC_MA'], self.data['KC_Upper'], self.data['KC_Lower'] = self.calculate_keltner_channels()
        self.data['ATR'] = self.calculate_atr()
        self.data['OBV'] = self.calculate_obv()
        self.data['CMF'] = self.calculate_cmf()
        self.data['Chaikin_OSC'] = self.calculate_chaikin_osc()
        self.data['Force_Index'] = self.calculate_force_index()
        self.data['AR'], self.data['BR'] = self.calculate_ar_br()
        self.data['TRIX'] = self.calculate_trix()
        self.data['Mass_Index'] = self.calculate_mass_index()
        self.data['Parabolic_SAR'] = self.calculate_parabolic_sar()
        self.data['Three_Line_Break'] = self.calculate_three_line_break()
        
        ichimoku = self.calculate_ichimoku()
        self.data = pd.concat([self.data, ichimoku], axis=1)
        
        disparities = self.calculate_disparity()
        self.data = pd.concat([self.data, disparities], axis=1)
        
        pdi, mdi, adx, adxr = self.calculate_adxr()
        self.data['PDI'] = pdi
        self.data['MDI'] = mdi
        self.data['ADX'] = adx
        self.data['ADXR'] = adxr

        return self.data

    def comprehensive_analysis(self, start_date, end_date, plot=True):
        self.get_data(start_date, end_date)
        self.calculate_all_indicators()

        if plot:
            self.visualize_moving_averages()
            self.visualize_ichimoku_cloud()
            self.visualize_dmi_adx_adxr()
            self.visualize_disparity_index()
            self.visualize_macd_ppo()
            self.visualize_rsi_stochastic_rsi()
            self.visualize_stochastic_oscillator()
            self.visualize_price_roc()
            self.visualize_bollinger_keltner()
            self.visualize_atr()
            self.visualize_obv()
            self.visualize_cmf_chaikin_osc()
            self.visualize_force_index()
            self.visualize_ar_br()
            self.visualize_trix_mass_index()
            self.visualize_parabolic_sar()
            self.visualize_three_line_break()

        performance_metrics = self.calculate_returns_and_metrics()
        indicator_analysis = self.analyze_indicators()
        interpretations = self.interpret_indicators(indicator_analysis)

        all_metrics = pd.concat([performance_metrics, indicator_analysis], axis=1)

        results = {
            'Performance Metrics': performance_metrics,
            'Indicator Analysis': indicator_analysis,
            'Interpretations': interpretations,
            'All Metrics': all_metrics,
            'Data': self.data
        }

        return results  # Add this line to return the results

    def calculate_returns_and_metrics(self):
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Cumulative_Returns'] = (1 + self.data['Returns']).cumprod()

        total_days = len(self.data)
        annualized_return = (self.data['Cumulative_Returns'].iloc[-1] ** (252 / total_days)) - 1
        annualized_volatility = self.data['Returns'].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / annualized_volatility  # Assuming 2% risk-free rate

        max_drawdown = (self.data['Cumulative_Returns'] / self.data['Cumulative_Returns'].cummax() - 1).min()

        metrics = pd.DataFrame({
            'Annualized Return': [annualized_return],
            'Annualized Volatility': [annualized_volatility],
            'Sharpe Ratio': [sharpe_ratio],
            'Max Drawdown': [max_drawdown]
        })

    def interpret_indicators(self, analysis):
        interpretations = []

        interpretations.append(f"Overall trend is {analysis['Trend'].values[0]}.")
        interpretations.append(f"MACD indicates a {analysis['MACD'].values[0]} momentum.")

        rsi = analysis['RSI'].values[0]
        if rsi > 70:
            interpretations.append(f"RSI at {rsi:.2f} suggests overbought conditions.")
        elif rsi < 30:
            interpretations.append(f"RSI at {rsi:.2f} suggests oversold conditions.")
        else:
            interpretations.append(f"RSI at {rsi:.2f} is in neutral territory.")

        interpretations.append(f"Stochastic Oscillator indicates {analysis['Stochastic'].values[0]} conditions.")

        bb_width = analysis['BB_Width'].values[0]
        interpretations.append(f"Bollinger Band width is {bb_width:.2f}, indicating {'high' if bb_width > 0.1 else 'low'} volatility.")

        atr = analysis['ATR'].values[0]
        interpretations.append(f"ATR is at {atr:.2f}, indicating {'high' if atr > self.data['ATR'].mean() else 'low'} volatility.")

        interpretations.append(f"On-Balance Volume trend is {analysis['OBV_Trend'].values[0]}.")

        cmf = analysis['CMF'].values[0]
        interpretations.append(f"Chaikin Money Flow at {cmf:.2f} suggests {'buying' if cmf > 0 else 'selling'} pressure.")

        force_index = analysis['Force_Index'].values[0]
        interpretations.append(f"Force Index at {force_index:.2f} indicates {'bullish' if force_index > 0 else 'bearish'} pressure.")

        adx = analysis['ADX'].values[0]
        if adx > 25:
            interpretations.append(f"ADX at {adx:.2f} suggests a strong trend.")
        else:
            interpretations.append(f"ADX at {adx:.2f} suggests a weak trend or ranging market.")

        adxr = analysis['ADXR'].values[0]
        interpretations.append(f"ADXR at {adxr:.2f} confirms the ADX reading.")

        interpretations.append(f"Parabolic SAR suggests a {analysis['Parabolic_SAR'].values[0]} trend.")

        interpretations.append(f"Ichimoku Cloud indicates a {analysis['Ichimoku_Trend'].values[0]} trend.")

        interpretations.append(f"10-day Disparity Index suggests {analysis['Disparity'].values[0]} conditions.")

        roc = analysis['Price_ROC'].values[0]
        interpretations.append(f"Price Rate of Change is {roc:.2f}%, indicating {'accelerating' if roc > 0 else 'decelerating'} momentum.")

        ar = analysis['AR'].values[0]
        br = analysis['BR'].values[0]
        interpretations.append(f"AR at {ar:.2f} and BR at {br:.2f} suggest {'strong' if ar > 100 and br > 100 else 'weak'} buying pressure.")

        trix = analysis['TRIX'].values[0]
        interpretations.append(f"TRIX at {trix:.2f} indicates a {'bullish' if trix > 0 else 'bearish'} trend.")

        mass_index = analysis['Mass_Index'].values[0]
        if mass_index > 27:
            interpretations.append(f"Mass Index at {mass_index:.2f} suggests a potential trend reversal.")
        else:
            interpretations.append(f"Mass Index at {mass_index:.2f} does not indicate a likely trend reversal.")

        interpretations.append(f"Three Line Break chart indicates a {analysis['Three_Line_Break'].values[0]} trend.")

        return interpretations