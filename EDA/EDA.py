import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
from minisom import MiniSom
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #FinancialAnalyzer WARNINGS
from FinancialAnalyzer import FinancialAnalyzer

# GLOBAL SETTINGS
START_DATE = "1991-01-01"
END_DATE = "2024-12-31"
index_tickers = ['^HSI', '^KS11', '^GSPC', '^N225']

# HELPER FUNCTIONS - Extraneous
def get_top_strategies(perf_df, metric='Annualized Return', top_n=5):
    sorted_df = perf_df.sort_values(by=metric, ascending=False)
    return list(sorted_df.head(top_n).index)

def order_legend_by_performance(ax, ranked_strategies, baseline_label='Buy & Hold', best_combined_label='Best Combined Signal'):
    handles, labels = ax.get_legend_handles_labels()
    label_to_rank = {}
    for i, strat in enumerate(ranked_strategies):
        label_to_rank[strat] = i
    label_to_rank[best_combined_label] = len(ranked_strategies)
    label_to_rank[baseline_label] = len(ranked_strategies) + 1
    def get_rank(label):
        return label_to_rank.get(label, len(ranked_strategies) + 2)
    sorted_items = sorted(zip(handles, labels), key=lambda x: get_rank(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_items)
    ax.legend(sorted_handles, sorted_labels)

def assign_colors(all_strategies, top_strategies):
    n = len(top_strategies)
    cmap = plt.cm.tab20
    colors = [cmap(i / n) for i in range(n)]
    color_map = {}
    for strat in all_strategies:
        if strat in top_strategies:
            idx = top_strategies.index(strat)
            color_map[strat] = colors[idx]
        else:
            color_map[strat] = 'gray'
    return color_map


# SIGNAL GENERATION FUNCTIONS
def generate_ma_crossover_signal(data):
    df = data.copy()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Signal'] = 0
    df.loc[df['SMA_50'] > df['SMA_200'], 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_rsi_signal(data, lower=30, upper=70):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['RSI'] < lower, 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_macd_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['MACD'] > df['MACD_Signal'], 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_bollinger_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['Close'] < df['BB_Lower'], 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_trend_signal(data):
    df = data.copy()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Signal'] = 0
    df.loc[df['Close'] > df['SMA_200'], 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_stochastic_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['K_Fast'] < 20, 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_ichimoku_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[(df['Close'] > df['Leading Span A']) & (df['Close'] > df['Leading Span B']), 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_obv_signal(data):
    df = data.copy()
    if 'OBV' not in df.columns:
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    df['OBV_Diff'] = df['OBV'].diff()
    df['Signal'] = 0
    df.loc[df['OBV_Diff'] > 0, 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_cmf_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['CMF'] > 0, 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_force_index_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['Force_Index'] > 0, 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_parabolic_sar_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['Parabolic_SAR'] < df['Close'], 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_disparity_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['Disparity_10'] < 95, 'Signal'] = 1
    df.loc[df['Disparity_10'] > 105, 'Signal'] = 0
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_trix_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['TRIX'] > 0, 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df

def generate_three_line_break_signal(data):
    df = data.copy()
    df['Signal'] = 0
    df.loc[df['Three_Line_Break'] > 0, 'Signal'] = 1
    df['Daily_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'].shift(1) * df['Daily_Return']
    return df


# Helper functions for plotting
def calculate_performance_metrics(df): # Helper function for plot_combined_strategy_performance
    df = df.copy()
    df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
    total_days = len(df)
    annualized_return = (df['Cumulative_Return'].iloc[-1])**(252/total_days) - 1
    annualized_volatility = df['Strategy_Return'].std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - 0.02) / annualized_volatility if annualized_volatility != 0 else np.nan
    max_drawdown = (df['Cumulative_Return'] / df['Cumulative_Return'].cummax() - 1).min()
    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

def rolling_window_metrics(data, window=252): # Helper function for plot_rolling_cumulative_returns
    metrics = []
    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i]
        cum_return = (1 + window_data['Strategy_Return']).prod() - 1
        volatility = window_data['Strategy_Return'].std() * np.sqrt(252)
        sharpe = (window_data['Strategy_Return'].mean() / window_data['Strategy_Return'].std() * np.sqrt(252)
                  if window_data['Strategy_Return'].std() != 0 else 0)
        metrics.append({
            'End_Date': window_data.index[-1],
            'Cumulative_Return': cum_return,
            'Volatility': volatility,
            'Sharpe': sharpe
        })
    return pd.DataFrame(metrics)

def apply_som(rolling_df, som_x=3, som_y=3, num_iterations=1000): # Helper function for rolling_window_metrics - Use Mini SOM to cluster the data
    features = rolling_df[['Cumulative_Return', 'Volatility', 'Sharpe']].values
    norm_features = (features - features.mean(axis=0)) / features.std(axis=0)
    
    som = MiniSom(som_x, som_y, norm_features.shape[1], sigma=1.0, learning_rate=0.5)
    som.random_weights_init(norm_features)
    som.train_random(norm_features, num_iterations)
    
    winner_coords = np.array([som.winner(x) for x in norm_features])
    cluster_indices = np.ravel_multi_index(winner_coords.T, (som_x, som_y))
    rolling_df['Cluster'] = cluster_indices
    return rolling_df, som

def plot_signal_persistence(df, max_horizon=5): # Helper function for plot_combined_signal_persistence
    horizons = list(range(1, max_horizon+1))
    avg_cum_returns = []
    for horizon in horizons:
        returns_list = []
        for i in range(len(df) - horizon):
            if df['Signal'].iloc[i] == 1:
                cum_return = (1 + df['Daily_Return'].iloc[i+1:i+horizon+1]).prod() - 1
                returns_list.append(cum_return)
        avg_cum_returns.append(np.mean(returns_list) if returns_list else np.nan)
    return horizons, avg_cum_returns

def majority_vote_signal_for_subset(strategy_data_dict, subset_keys): # Helper function for plot_combined_strategy_with_best
    if not subset_keys:
        return None
    first_key = subset_keys[0]
    ref_df = strategy_data_dict[first_key]
    df_combined = pd.DataFrame(index=ref_df.index)
    df_combined['Daily_Return'] = ref_df['Daily_Return']
    signals = [strategy_data_dict[k]['Signal'] for k in subset_keys]
    df_combined['Avg_Signal'] = np.mean(signals, axis=0)
    df_combined['Combined_Signal'] = (df_combined['Avg_Signal'] >= 0.5).astype(int)
    df_combined['Strategy_Return'] = df_combined['Combined_Signal'].shift(1).fillna(0) * df_combined['Daily_Return']
    df_combined['Cumulative_Return'] = (1 + df_combined['Strategy_Return']).cumprod()
    return df_combined

def find_best_subset(strategy_data_dict): # Helper function for plot_combined_strategy_with_best
    all_keys = list(strategy_data_dict.keys())
    best_final_value = -np.inf
    best_subset_keys = None
    best_df = None
    for r in range(1, len(all_keys) + 1):
        for subset in combinations(all_keys, r):
            combined_df = majority_vote_signal_for_subset(strategy_data_dict, subset)
            if combined_df is None:
                continue
            final_value = combined_df['Cumulative_Return'].iloc[-1]
            if final_value > best_final_value:
                best_final_value = final_value
                best_subset_keys = subset
                best_df = combined_df
    return best_subset_keys, best_df


# PLOTTING FUNCTIONS
def plot_combined_strategy_performance(strategy_data_dict, ticker, ranked_strategies):
    color_map = assign_colors(strategy_data_dict.keys(), ranked_strategies)
    _, ax = plt.subplots(figsize=(12, 6))
    for strat_name in ranked_strategies:
        df = strategy_data_dict[strat_name]
        if 'Cumulative_Return' not in df.columns:
            df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        ax.plot(df.index, df['Cumulative_Return'], label=strat_name, color=color_map[strat_name])
    for strat_name in [k for k in strategy_data_dict if k not in ranked_strategies]:
        df = strategy_data_dict[strat_name]
        if 'Cumulative_Return' not in df.columns:
            df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        ax.plot(df.index, df['Cumulative_Return'], label=strat_name, color='gray')
    # Buy & Hold baseline
    first_data = list(strategy_data_dict.values())[0]
    buy_hold = (1 + first_data['Daily_Return']).cumprod()
    ax.plot(first_data.index, buy_hold, label='Buy & Hold', color='plum', linestyle='--')
    ax.fill_between(first_data.index, buy_hold, 1, color='plum', alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle=':')
    ax.set_title(f'{ticker} - Strategy Cumulative Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    order_legend_by_performance(ax, ranked_strategies, baseline_label='Buy & Hold', best_combined_label='Best Combined Signal')
    plt.tight_layout()
    ticker_filename = ticker.replace('^', '')
    plt.savefig(f'EDA/Visualizations/{ticker_filename}-StrategyCumulativeReturns.png')
    plt.close()

def plot_combined_strategy_with_best(strategy_data_dict, best_combined_df, ticker, ranked_strategies, best_subset_keys):
    color_map = assign_colors(strategy_data_dict.keys(), ranked_strategies)
    _, ax = plt.subplots(figsize=(12, 6))
    for strat_name in ranked_strategies:
        df = strategy_data_dict[strat_name]
        if 'Cumulative_Return' not in df.columns:
            df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        ax.plot(df.index, df['Cumulative_Return'], label=strat_name, color=color_map[strat_name])
    for strat_name in [k for k in strategy_data_dict if k not in ranked_strategies]:
        df = strategy_data_dict[strat_name]
        if 'Cumulative_Return' not in df.columns:
            df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()
        ax.plot(df.index, df['Cumulative_Return'], label=strat_name, color='gray')
    # Buy & Hold baseline
    first_data = list(strategy_data_dict.values())[0]
    buy_hold = (1 + first_data['Daily_Return']).cumprod()
    ax.plot(first_data.index, buy_hold, label='Buy & Hold', color='plum', linestyle='--')
    ax.fill_between(first_data.index, buy_hold, 1, color='plum', alpha=0.3)
    # Best Combined Signal
    ax.plot(best_combined_df.index, best_combined_df['Cumulative_Return'], label='Best Combined Signal',
            color='black', linewidth=2.5, linestyle='-.')
    ax.axhline(y=1.0, color='gray', linestyle=':')
    ax.set_title(f'{ticker} - Combined Strategy {best_subset_keys}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    order_legend_by_performance(ax, ranked_strategies, baseline_label='Buy & Hold', best_combined_label='Best Combined Signal')
    plt.tight_layout()
    ticker_filename = ticker.replace('^', '')
    plt.savefig(f'EDA/Visualizations/{ticker_filename}-CombinedStrategyWithBest.png')
    plt.close()


def plot_rolling_cumulative_returns(strategy_rolling_dict, ticker, ranked_strategies):
    color_map = assign_colors(strategy_rolling_dict.keys(), ranked_strategies)
    _, ax = plt.subplots(figsize=(12, 6))
    for strat_name in [k for k in strategy_rolling_dict if k not in ranked_strategies]:
        rolling_df = strategy_rolling_dict[strat_name]
        ax.plot(rolling_df['End_Date'], rolling_df['Cumulative_Return'], label=strat_name, color='gray')
    for strat_name in ranked_strategies:
        rolling_df = strategy_rolling_dict[strat_name]
        ax.plot(rolling_df['End_Date'], rolling_df['Cumulative_Return'], label=strat_name, color=color_map[strat_name])
    ax.axhline(y=0.0, color='black', linestyle=':')
    ax.set_title(f'{ticker} - Rolling Window Cumulative Returns')
    ax.set_xlabel('End Date')
    ax.set_ylabel('Cumulative Return')
    order_legend_by_performance(ax, ranked_strategies, baseline_label='Buy & Hold', best_combined_label='Best Combined Signal')
    plt.tight_layout()
    ticker_filename = ticker.replace('^', '')
    plt.savefig(f'EDA/Visualizations/{ticker_filename}-RollingCumulativeReturns.png')
    plt.close()

def plot_rolling_sharpe_ratios(strategy_rolling_dict, ticker, ranked_strategies):
    color_map = assign_colors(strategy_rolling_dict.keys(), ranked_strategies)
    _, ax = plt.subplots(figsize=(12, 6))
    all_sharpe = []
    for strat_name in ranked_strategies:
        rolling_df = strategy_rolling_dict[strat_name]
        ax.plot(rolling_df['End_Date'], rolling_df['Sharpe'], label=strat_name, color=color_map[strat_name])
        all_sharpe.extend(rolling_df['Sharpe'].values)
    for strat_name in [k for k in strategy_rolling_dict if k not in ranked_strategies]:
        rolling_df = strategy_rolling_dict[strat_name]
        ax.plot(rolling_df['End_Date'], rolling_df['Sharpe'], label=strat_name, color='gray')
    ax.axhline(y=0.0, color='black', linestyle=':')
    ax.set_ylim(np.nanmin(all_sharpe) - 0.5, np.nanmax(all_sharpe) + 0.5)
    ax.set_title(f'{ticker} - Rolling Window Sharpe Ratios')
    ax.set_xlabel('End Date')
    ax.set_ylabel('Sharpe Ratio')
    order_legend_by_performance(ax, ranked_strategies, baseline_label='Buy & Hold', best_combined_label='Best Combined Signal')
    plt.tight_layout()
    ticker_filename = ticker.replace('^', '')
    plt.savefig(f'EDA/Visualizations/{ticker_filename}-RollingSharpeRatios.png')
    plt.close()

def plot_combined_signal_persistence(strategy_data_dict, ticker, max_horizon=5):
    plt.figure(figsize=(12, 6))
    num_strategies = len(strategy_data_dict)
    color_palette = plt.cm.tab20(np.linspace(0, 1, num_strategies))
    for (strat_name, df), color in zip(strategy_data_dict.items(), color_palette):
        horizons, avg_returns = plot_signal_persistence(df, max_horizon)
        plt.plot(horizons, avg_returns, marker='o', label=strat_name, color=color)
    plt.axhline(y=0.0, color='black', linestyle=':')
    plt.title('Combined Signal Persistence Analysis')
    plt.xlabel('Horizon (days)')
    plt.ylabel('Average Future Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ticker_filename = ticker.replace('^', '')
    plt.savefig(f'EDA/Visualizations/{ticker_filename}-CombinedSignalPersistence.png')
    plt.close()

# EVALUATION FUNCTION
def evaluate_indicator_performance(ticker):
    print(f"Evaluating indicator performance for {ticker} ...")
    analyzer = FinancialAnalyzer(ticker)
    data = analyzer.get_data(START_DATE, END_DATE)
    if data.empty:
        print(f"No data available for {ticker}")
        return None, None, None
    analyzer.data = data.copy()
    analyzer.calculate_all_indicators()
    data = analyzer.data.copy()
    strategies = {
        "MA Crossover": generate_ma_crossover_signal,
        "RSI": generate_rsi_signal,
        "MACD": generate_macd_signal,
        "Bollinger": generate_bollinger_signal,
        "Trend": generate_trend_signal,
        "Stochastic": generate_stochastic_signal,
        "Ichimoku": generate_ichimoku_signal,
        "OBV": generate_obv_signal,
        "CMF": generate_cmf_signal,
        "Force Index": generate_force_index_signal,
        "Parabolic SAR": generate_parabolic_sar_signal,
        "Disparity": generate_disparity_signal,
        "TRIX": generate_trix_signal,
        "Three Line Break": generate_three_line_break_signal
    }
    strategy_data_dict = {}
    strategy_rolling_dict = {}
    performance_results = {}
    for strat_name, signal_func in strategies.items():
        strat_data = signal_func(data)
        perf = calculate_performance_metrics(strat_data)
        performance_results[strat_name] = perf
        strategy_data_dict[strat_name] = strat_data
        rolling_df = rolling_window_metrics(strat_data, window=252)
        rolling_df, _ = apply_som(rolling_df)
        strategy_rolling_dict[strat_name] = rolling_df
    results_df = pd.DataFrame(performance_results).T
    print("Performance Metrics for Each Strategy:")
    print(results_df)
    return strategy_data_dict, strategy_rolling_dict, results_df

def main():
    overall_results = {}
    for ticker in index_tickers:
        print("\n" + "="*50)
        print(f"Processing {ticker}")
        strat_data_dict, strat_rolling_dict, perf_df = evaluate_indicator_performance(ticker)
        if strat_data_dict is None:
            continue
        perf_sorted = perf_df.sort_values(by='Annualized Return', ascending=False)
        ranked_strategies = list(perf_sorted.index)[:10] # Top 10 strategies
        
        plot_combined_strategy_performance(strat_data_dict, ticker, ranked_strategies)
        
        best_subset_keys, best_subset_df = find_best_subset(strat_data_dict)
        print(f"Best subset of indicators for {ticker}: {best_subset_keys}")
        plot_combined_strategy_with_best(strat_data_dict, best_subset_df, ticker, ranked_strategies, best_subset_keys)
        
        plot_rolling_cumulative_returns(strat_rolling_dict, ticker, ranked_strategies)
        plot_rolling_sharpe_ratios(strat_rolling_dict, ticker, ranked_strategies)
        
        plot_combined_signal_persistence(strat_data_dict, ticker, max_horizon=10)
        
        overall_results[ticker] = {
            'Performance': perf_df,
            'Strategy Data': strat_data_dict,
            'Rolling Data': strat_rolling_dict,
            'Best Subset': best_subset_keys,
            'Best Subset DF': best_subset_df
        }
    return overall_results

if __name__ == "__main__":
    results = main()
