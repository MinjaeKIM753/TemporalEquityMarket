import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests
import yfinance as yf
import os
import warnings
import matplotlib.gridspec as gridspec

warnings.simplefilter(action='ignore', category=FutureWarning)

tickers = ["^HSI", "^KS11", "SPY", "^N225"]
start_date = "1991-01-01"
end_date = "2024-12-31"
df_market = yf.download(tickers, start=start_date, end=end_date)['Close']

df_market_monthly = df_market.resample('M').last()
returns = np.log(df_market_monthly / df_market_monthly.shift(1)).dropna()
returns.columns = ["HSI_Return", "KS11_Return", "SPY_Return", "N225_Return"]

monthly_returns = df_market_monthly.pct_change().dropna()
monthly_returns.columns = ["HSI_Monthly_Return", "KS11_Monthly_Return", "SPY_Monthly_Return", "N225_Monthly_Return"]

volatility = monthly_returns.rolling(window=12).std() * np.sqrt(12)
volatility.columns = ["HSI_Volatility", "KS11_Volatility", "SPY_Volatility", "N225_Volatility"]

index_file = "Policy/datasets/CPU index.csv"
df_cpu_index = pd.read_csv(index_file)
df_cpu_index.rename(columns={df_cpu_index.columns[0]: 'Date'}, inplace=True)
df_cpu_index['Date'] = pd.to_datetime(df_cpu_index['Date'], errors='coerce')
df_cpu_index.dropna(subset=['Date'], inplace=True)
df_cpu_index['Date'] = df_cpu_index['Date'].dt.to_period('M').dt.to_timestamp()
df_cpu_index.set_index('Date', inplace=True)
df_cpu_index = df_cpu_index.resample('ME').last()

def load_uncertainty_file(filename, new_col_name):
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filename)
    elif ext == ".csv":
        df = pd.read_csv(filename)
    else:
        raise ValueError("Unsupported file format for " + filename)

    if 'Date' not in df.columns:
        date_col_name = df.columns[0]
        df.rename(columns={date_col_name: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df.set_index('Date', inplace=True)
    index_col_name = df.columns[0]
    df.rename(columns={index_col_name: new_col_name}, inplace=True)
    return df[[new_col_name]].resample('ME').last()

epu_files = {
    'HK': ("Policy/datasets/HK EPU.xlsx", "EPU_HK"),
    'KR': ("Policy/datasets/KR EPU.xls", "EPU_KR"),
    'US': ("Policy/datasets/US EPU.xlsx", "EPU_US"),
    'JP': ("Policy/datasets/JP EPU.xlsx", "EPU_JP")
}
epu_data = {country: load_uncertainty_file(f, c) for country, (f, c) in epu_files.items()}
global_epu = load_uncertainty_file("Policy/datasets/Global EPU.xlsx", "Global_EPU")
tpu_data = {
    'CN': load_uncertainty_file("Policy/datasets/CN TPU.xlsx", "TPU_CN"),
    'US': load_uncertainty_file("Policy/datasets/US TPU.xlsx", "TPU_US")
}
uct = load_uncertainty_file("Policy/datasets/UCT.csv", "UCT")

def merge_data_for_country(country, return_col):
    df = returns[[return_col]].copy()
    df = df.join(df_cpu_index, how='left')
    df = df.join(epu_data.get(country, pd.DataFrame()), how='inner')
    df = df.join(global_epu, how='left')
    if country in ['HK', 'KR', 'JP']:
        df = df.join(tpu_data['CN'], how='left')
    if country == 'US':
        df = df.join(tpu_data['US'], how='left')
    df = df.join(uct, how='left')
    vol_col = f"{return_col.split('_')[0]}_Volatility"
    if vol_col in volatility.columns:
        df = df.join(volatility[[vol_col]], how='left')
    else:
        print(f"Warning: Volatility column {vol_col} not found.")
    uncertainty_cols = [col for col in df.columns if 'EPU' in col or 'TPU' in col or 'UCT' in col or 'CPU' in col]
    df['Mean_Uncertainty_Change'] = df[uncertainty_cols].pct_change().mean(axis=1)
    return df.dropna(subset=[return_col])

country_return_map = {
    'HK': "HSI_Return",
    'KR': "KS11_Return",
    'US': "SPY_Return",
    'JP': "N225_Return"
}

merged_data = {country: merge_data_for_country(country, col) 
               for country, col in country_return_map.items()}

window_size_top = 12
window_sizes_bottom = [3, 6, 12]

for country, df in merged_data.items():
    rolling_means_top = df.rolling(window=window_size_top).mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    fig.suptitle(f"{country} Market: Rolling Mean Analysis (Window: {window_size_top} months)",
                y=0.95, fontsize=12, fontweight='bold')
    
    ax1.set_ylabel("Rolling Mean Value")
    ax1.set_xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2023-12-31"))
    colors_top = plt.get_cmap('tab10')
    uncertainty_cols_plot = [col for col in rolling_means_top.columns if ('EPU' in col or 'TPU' in col or 'UCT' in col or 'CPU' in col)]

    lns1 = []
    for i, col in enumerate(uncertainty_cols_plot):
        lns1.extend(ax1.plot(rolling_means_top.index, rolling_means_top[col], label=col, color=colors_top(i)))
    ax1.tick_params(axis='y')

    ax1_twin = ax1.twinx()
    vol_col = f"{country_return_map[country].split('_')[0]}_Volatility"
    if vol_col in rolling_means_top.columns:
        lns2 = ax1_twin.plot(rolling_means_top.index, rolling_means_top[vol_col], label=f"{vol_col} (Rolling Mean)", color='purple', linestyle='--')
        ax1_twin.set_ylabel(f"{country} Volatility (Rolling Mean)")
        ax1_twin.tick_params(axis='y', labelcolor='purple')
    else:
        lns2 = []

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rolling Mean Pct Change")
    colors_bottom_uncertainty = plt.get_cmap('Oranges')
    colors_bottom_volatility = plt.get_cmap('Purples')

    lns3 = []
    for idx, window_size in enumerate([6, 12]):
        rolling_means_bottom = df.rolling(window=window_size).mean()
        if 'Mean_Uncertainty_Change' in rolling_means_bottom.columns:
            lns3.extend(ax2.plot(rolling_means_bottom.index, rolling_means_bottom['Mean_Uncertainty_Change'],
                                label=f"Mean Uncert. Change (Roll {window_size})"))
    ax2.tick_params(axis='y')
    ax2.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ylim = ax2.get_ylim()
    max_abs_y = max(abs(ylim[0]), abs(ylim[1]))
    if max_abs_y > 0:
      ax2.set_ylim(bottom=-max_abs_y, top=max_abs_y)

    ax2_twin = ax2.twinx()
    lns4 = []
    if vol_col in df.columns:
        for idx, window_size in enumerate([6, 12]):
            rolling_vol_pct_change = df[vol_col].rolling(window=window_size).mean().pct_change()
            lns4.extend(ax2_twin.plot(rolling_vol_pct_change.index, rolling_vol_pct_change,
                                    label=f"{vol_col} Pct Change (Roll {window_size})", linestyle='--'))
        ax2_twin.set_ylabel(f"{country} Volatility Pct Change (Rolling Mean)")
        ax2_twin.tick_params(axis='y')
        ax2_twin.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ylim_twin = ax2_twin.get_ylim()
        max_abs_y_twin = max(abs(ylim_twin[0]), abs(ylim_twin[1]))
        if max_abs_y_twin > 0:
             ax2_twin.set_ylim(bottom=-max_abs_y_twin, top=max_abs_y_twin)
    else:
        lns4 = []

    lns_b = lns3 + lns4
    labs_b = [l.get_label() for l in lns_b]
    ax2.legend(lns_b, labs_b, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.subplots_adjust(hspace=0.3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(f"Policy/Visualizations/{country}_Rolling_Window.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

for country, df in merged_data.items():
    ret_var = country_return_map[country]
    relevant_vars = []
    if country == 'HK':
        relevant_vars = ['EPU_HK', 'Global_EPU', 'TPU_CN', 'CPU']
    elif country == 'KR':
        relevant_vars = ['EPU_KR', 'Global_EPU', 'TPU_CN', 'CPU']
    elif country == 'US':
        relevant_vars = ['EPU_US', 'Global_EPU', 'TPU_US', 'CPU']
    elif country == 'JP':
        relevant_vars = ['EPU_JP', 'Global_EPU', 'TPU_CN', 'CPU']
    
    if len(relevant_vars) > 0:
        formula = ret_var + ' ~ ' + ' + '.join(relevant_vars)
        model = smf.ols(formula, data=df).fit()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        coefficients = model.params[1:]
        p_values = model.pvalues[1:]
        
        y_pos = np.arange(len(coefficients))
        colors = ['darkblue' if p < 0.05 else 'lightblue' for p in p_values]
        
        bars = ax1.barh(y_pos, coefficients, color=colors)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(relevant_vars)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title(f'{country} Market: Regression Coefficients (OLS)')
        
        for i, p in enumerate(p_values):
            if p < 0.01:
                ax1.text(coefficients[i], i, '***', ha='left' if coefficients[i] >= 0 else 'right', va='center')
            elif p < 0.05:
                ax1.text(coefficients[i], i, '**', ha='left' if coefficients[i] >= 0 else 'right', va='center')
            elif p < 0.1:
                ax1.text(coefficients[i], i, '*', ha='left' if coefficients[i] >= 0 else 'right', va='center')
        
        performance_metrics = [
            f"R-squared: {model.rsquared:.3f}",
            f"Adj. R-squared: {model.rsquared_adj:.3f}",
            f"F-statistic p-value: {model.f_pvalue:.3e}",
            f"Durbin-Watson: {sm.stats.stattools.durbin_watson(model.resid):.3f}"
        ]
        
        ax2.axis('off')
        y_pos = 0.8
        for metric in performance_metrics:
            ax2.text(0.1, y_pos, metric, fontsize=12)
            y_pos -= 0.2
        
        plt.suptitle(f'{country} Market: Regression Analysis Results (OLS)')
        plt.tight_layout()
        plt.savefig(f"Policy/Visualizations/{country}_Regression_Summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n[{country}] Regression Summary:")
        print(model.summary())
    else:
        print(f"\n[{country}] No independent variables available for regression.")

maxlag = 3
colors_granger = plt.get_cmap('Set2')

for country, df in merged_data.items():
    ret_var = country_return_map[country]
    
    relevant_vars = []
    if country == 'HK':
        relevant_vars = ['EPU_HK', 'Global_EPU', 'TPU_CN', 'CPU']
    elif country == 'KR':
        relevant_vars = ['EPU_KR', 'Global_EPU', 'TPU_CN', 'CPU']
    elif country == 'US':
        relevant_vars = ['EPU_US', 'Global_EPU', 'TPU_US', 'CPU']
    elif country == 'JP':
        relevant_vars = ['EPU_JP', 'Global_EPU', 'TPU_CN', 'CPU']
    
    granger_results = {}
    
    for var in relevant_vars:
        if var in df.columns:
            test_data = df[[ret_var, var]].dropna()
            if len(test_data) > maxlag:
                results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                granger_results[var] = [results[i+1][0]['ssr_chi2test'][1] for i in range(maxlag)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.25
    index = np.arange(len(relevant_vars))
    
    for lag in range(maxlag):
        p_values = [granger_results[var][lag] if var in granger_results else np.nan for var in relevant_vars]
        ax.bar(index + lag * bar_width, 
               p_values,
               bar_width,
               label=f'Lag {lag+1}',
               color=colors_granger(lag),
               alpha=0.7)
    
    ax.axhline(y=0.05, color='r', linestyle='--', label='5% Significance Level')
    
    ax.set_xlabel('Uncertainty Measures')
    ax.set_ylabel('P-value')
    ax.set_title(f'{country} Market: Granger Causality Test P-values (Max Lag: {maxlag})')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(relevant_vars)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"Policy/Visualizations/{country}_Granger_Test_Results.png", dpi=300, bbox_inches='tight')
    plt.close()

regression_results = {}
for country, df in merged_data.items():
    ret_var = country_return_map[country]
    relevant_vars = []
    if country == 'HK':
        relevant_vars = ['EPU_HK', 'Global_EPU', 'TPU_CN', 'CPU']
    elif country == 'KR':
        relevant_vars = ['EPU_KR', 'Global_EPU', 'TPU_CN', 'CPU']
    elif country == 'US':
        relevant_vars = ['EPU_US', 'Global_EPU', 'TPU_US', 'CPU']
    elif country == 'JP':
        relevant_vars = ['EPU_JP', 'Global_EPU', 'TPU_CN', 'CPU']
    
    if len(relevant_vars) > 0:
        formula = ret_var + ' ~ ' + ' + '.join(relevant_vars)
        regression_results[country] = smf.ols(formula, data=df).fit()

granger_results = {}
for country, df in merged_data.items():
    ret_var = country_return_map[country]
    relevant_vars = []
    if country == 'HK':
        relevant_vars = ['EPU_HK', 'Global_EPU', 'TPU_CN', 'CPU']
    elif country == 'KR':
        relevant_vars = ['EPU_KR', 'Global_EPU', 'TPU_CN', 'CPU']
    elif country == 'US':
        relevant_vars = ['EPU_US', 'Global_EPU', 'TPU_US', 'CPU']
    elif country == 'JP':
        relevant_vars = ['EPU_JP', 'Global_EPU', 'TPU_CN', 'CPU']
    
    granger_results[country] = {}
    for var in relevant_vars:
        if var in df.columns:
            test_data = df[[ret_var, var]].dropna()
            if len(test_data) > maxlag:
                results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                granger_results[country][var] = {
                    f'Lag {i+1}': results[i+1][0]['ssr_chi2test'][1] 
                    for i in range(maxlag)
                }

def plot_combined_regression_results(results_dict):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    markets = ['HK', 'KR', 'US', 'JP']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for idx, market in enumerate(markets):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        model = results_dict[market]
        coeffs = model.params[1:]
        p_values = model.pvalues[1:]
        var_names = coeffs.index
        
        bars = ax.barh(range(len(coeffs)), coeffs, 
                      color=[colors[idx] if p < 0.05 else f'{colors[idx]}50' for p in p_values],
                      alpha=0.7)
        
        for i, p in enumerate(p_values):
            if p < 0.01: marker = '***'
            elif p < 0.05: marker = '**'
            elif p < 0.1: marker = '*'
            else: marker = ''
            if coeffs[i] != 0:
                ax.text(coeffs[i], i, marker, 
                       ha='left' if coeffs[i] >= 0 else 'right',
                       va='center')
        
        metrics_text = (
            f'R² = {model.rsquared:.3f}\n'
            f'Adj.R² = {model.rsquared_adj:.3f}\n'
            f'Durbin-Watson = {sm.stats.stattools.durbin_watson(model.resid):.3f}'
        )
        ax.text(0.95, 0.95, metrics_text,
                transform=ax.transAxes,
                ha='right', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_yticks(range(len(coeffs)))
        ax.set_yticklabels(var_names)
        ax.set_title(f'{market} Market: Regression Coefficients (OLS)', 
                    pad=20, fontsize=12, fontweight='bold')
    
    significance_text = (
        "Significance levels:\n"
        "*** p < 0.01\n"
        "**  p < 0.05\n"
        "*   p < 0.10\n"
    )
    fig.text(0.98, 0.02, significance_text,
             fontsize=10, ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.suptitle('Regression Analysis (OLS) Results Across Markets', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Policy/Visualizations/Combined_Regression_Results.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_granger_results(granger_results):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    markets = ['HK', 'KR', 'US', 'JP']
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    for idx, market in enumerate(markets):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        results = granger_results[market]
        variables = list(results.keys())
        x = np.arange(len(variables))
        width = 0.25
        
        for lag_idx in range(3):
            lag_values = [results[var][f'Lag {lag_idx+1}'] for var in variables]
            ax.bar(x + width*(lag_idx-1), lag_values, width, 
                  label=f'Lag {lag_idx+1}', 
                  color=colors[lag_idx], 
                  alpha=0.7)
        
        ax.axhline(y=0.05, color='red', linestyle='--', 
                  alpha=0.5, label='5% Significance Level')
        
        ax.set_title(f'{market} Market: Granger Causality Test P-values (Max Lag: {maxlag})', 
                    pad=20, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45)
        ax.set_ylabel('P-value')
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    significance_text = "* Asterisk indicates significant Granger causality (p < 0.05)"
    fig.text(0.98, 0.02, significance_text,
             fontsize=10, ha='right', va='bottom',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.suptitle('Granger Causality Test Results Across Markets', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('Policy/Visualizations/Combined_Granger_Results.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

plot_combined_regression_results(regression_results)
plot_combined_granger_results(granger_results)
