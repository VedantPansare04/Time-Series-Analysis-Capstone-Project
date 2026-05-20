"""
================================================================================
TSA CAPSTONE PROJECT — Complete Time Series Analysis & Portfolio Construction
================================================================================
End-to-end pipeline for 5 NSE stocks: data collection, preprocessing,
stationarity testing, ARIMA & Holt-Winters forecasting, volatility analysis,
STL decomposition, correlation analysis, portfolio construction, and reporting.
================================================================================
"""

# #============================================================================#
# #  SECTION 1 — IMPORTS & CONFIGURATION                                      #
# #============================================================================#

import os
import warnings
import json
import textwrap
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima
from arch import arch_model

warnings.filterwarnings('ignore')

# ── Global Plot Style ──
plt.rcParams.update({
    'figure.figsize': (14, 7),
    'axes.titlesize': 16,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})
sns.set_style('whitegrid')

# ── Master Configuration ──
CONFIG = {
    'stocks': [
        'TCS.NS',
        'HDFCBANK.NS',
        'RELIANCE.NS',
        'SUNPHARMA.NS',
        'MARUTI.NS'
    ],
    'sectors': {
        'TCS.NS': 'IT',
        'HDFCBANK.NS': 'Banking',
        'RELIANCE.NS': 'Energy',
        'SUNPHARMA.NS': 'Pharma',
        'MARUTI.NS': 'Auto'
    },
    'start_date': '2021-01-01',
    'end_date': '2025-12-31',
    'forecast_horizon': 5,
    'train_split_date': '2025-06-30',
    'capital': 1000000,
    'base_dir': os.path.dirname(os.path.abspath(__file__))
}

print("=" * 80)
print("  TSA CAPSTONE PROJECT — Time Series Analysis & Portfolio Construction")
print("=" * 80)
print(f"\nStocks : {CONFIG['stocks']}")
print(f"Period : {CONFIG['start_date']} to {CONFIG['end_date']}")
print(f"Train  : up to {CONFIG['train_split_date']}")
print(f"Capital: Rs.{CONFIG['capital']:,.0f}")
print("=" * 80)

# ── Paths ──
DATA_RAW      = os.path.join(CONFIG['base_dir'], 'data', 'raw')
DATA_PROC     = os.path.join(CONFIG['base_dir'], 'data', 'processed')
DATA_RETURNS  = os.path.join(CONFIG['base_dir'], 'data', 'returns')
DATA_FCAST    = os.path.join(CONFIG['base_dir'], 'data', 'forecasts')
OUT_PLOTS     = os.path.join(CONFIG['base_dir'], 'outputs', 'plots')
OUT_METRICS   = os.path.join(CONFIG['base_dir'], 'outputs', 'metrics')
OUT_DECOMP    = os.path.join(CONFIG['base_dir'], 'outputs', 'decomposition')
OUT_FCAST     = os.path.join(CONFIG['base_dir'], 'outputs', 'forecasts')
OUT_PORT      = os.path.join(CONFIG['base_dir'], 'outputs', 'portfolio')
RPT_FIGS      = os.path.join(CONFIG['base_dir'], 'report', 'figures')

for d in [DATA_RAW, DATA_PROC, DATA_RETURNS, DATA_FCAST,
          OUT_PLOTS, OUT_METRICS, OUT_DECOMP, OUT_FCAST, OUT_PORT, RPT_FIGS]:
    os.makedirs(d, exist_ok=True)


# #============================================================================#
# #  SECTION 2 — REUSABLE HELPER FUNCTIONS                                    #
# #============================================================================#

def fetch_stock_data(stock, start, end):
    """Download daily OHLCV data from Yahoo Finance."""
    print(f"\n  [DL] Downloading {stock} ...")
    df = yf.download(stock, start=start, end=end, interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) == 0:
        print(f"     WARNING: No data returned for {stock}")
        return df
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    print(f"     -> {len(df)} rows  |  {df.index.min().date()} to {df.index.max().date()}")
    return df


def clean_data(df):
    """Handle missing values via forward-fill, then drop remaining NaNs."""
    before = df.isnull().sum().sum()
    df = df.ffill()
    df = df.dropna()
    after = df.isnull().sum().sum()
    if before > 0:
        print(f"     → Cleaned {before} missing values (ffill + dropna)")
    return df


def run_adf_test(series, name=''):
    """Augmented Dickey-Fuller stationarity test."""
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat   = result[0]
    p_value    = result[1]
    stationary = p_value < 0.05
    print(f"     ADF({name}): statistic={adf_stat:.4f}, p-value={p_value:.6f} → {'Stationary [Y]' if stationary else 'Non-stationary [N]'}")
    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'is_stationary': stationary,
        'critical_values': result[4]
    }


def perform_differencing(series):
    """First-order differencing: Y'_t = Y_t - Y_{t-1}."""
    return series.diff().dropna()


def plot_acf_pacf(series, stock, save_dir):
    """Plot and save ACF & PACF for parameter identification."""
    clean_name = stock.replace('.NS', '')
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_acf(series.dropna(), ax=axes[0], lags=40, title=f'ACF — {clean_name}')
    plot_pacf(series.dropna(), ax=axes[1], lags=40, title=f'PACF — {clean_name}',
              method='ywm')

    plt.tight_layout()
    path = os.path.join(save_dir, f'{clean_name}_acf_pacf.png')
    plt.savefig(path)
    plt.close()
    return path


def train_arima_model(train_series, test_series, stock):
    """Train ARIMA using auto_arima for best (p,d,q), then forecast."""
    print(f"     [FIT] Fitting auto_arima for {stock} ...")
    stepwise = auto_arima(
        train_series, seasonal=False, stepwise=True,
        suppress_warnings=True, trace=False,
        error_action='ignore', max_p=5, max_q=5, max_d=2
    )
    order = stepwise.order
    print(f"        Best order: ARIMA{order}  |  AIC={stepwise.aic():.2f}")

    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_series))
    forecast.index = test_series.index

    summary_str = str(model_fit.summary())

    return {
        'order': order,
        'aic': stepwise.aic(),
        'forecast': forecast,
        'summary': summary_str,
        'model': model_fit
    }


def train_holt_winters(train_series, test_series, stock):
    """Train Holt-Winters Exponential Smoothing (additive trend, no season)."""
    print(f"     [FIT] Fitting Holt-Winters for {stock} ...")
    model = ExponentialSmoothing(
        train_series,
        trend='add',
        seasonal=None,
        initialization_method='estimated'
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(len(test_series))
    forecast.index = test_series.index
    return {
        'forecast': forecast,
        'model': fit
    }


def compute_metrics(actual, predicted, model_name=''):
    """RMSE, MAPE, MAE, and Directional Accuracy."""
    actual_vals  = np.array(actual).flatten()
    pred_vals    = np.array(predicted).flatten()

    rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
    mae  = mean_absolute_error(actual_vals, pred_vals)
    mape = np.mean(np.abs((actual_vals - pred_vals) / actual_vals)) * 100

    # Directional accuracy (skip first value since diff is NaN)
    actual_dir = np.sign(np.diff(actual_vals))
    pred_dir   = np.sign(np.diff(pred_vals))
    min_len    = min(len(actual_dir), len(pred_dir))
    dir_acc    = np.mean(actual_dir[:min_len] == pred_dir[:min_len]) * 100

    return {
        'Model': model_name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'MAPE': round(mape, 4),
        'Directional_Accuracy': round(dir_acc, 2)
    }


def compute_volatility(df):
    """Add log returns and rolling 30-day volatility."""
    df = df.copy()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rolling_Volatility'] = df['Log_Returns'].rolling(window=30).std()
    return df


def perform_stl_decomposition(series, period, stock, save_dir):
    """STL decomposition into trend, seasonal, residual."""
    clean_name = stock.replace('.NS', '')
    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(series, color='#2196F3')
    axes[0].set_title(f'{clean_name} — Original', fontsize=14)
    axes[1].plot(result.trend, color='#FF5722')
    axes[1].set_title('Trend', fontsize=14)
    axes[2].plot(result.seasonal, color='#4CAF50')
    axes[2].set_title('Seasonal', fontsize=14)
    axes[3].plot(result.resid, color='#9C27B0')
    axes[3].set_title('Residual', fontsize=14)
    plt.suptitle(f'STL Decomposition — {clean_name}', fontsize=16, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{clean_name}_stl.png')
    plt.savefig(path)
    plt.close()
    return result, path


def short_name(stock):
    return stock.replace('.NS', '')


# #============================================================================#
# #  SECTION 3 — STOCK SELECTION & JUSTIFICATION                              #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 3 — STOCK SELECTION")
print("=" * 80)

stock_table = pd.DataFrame([
    {'Stock': s, 'Sector': CONFIG['sectors'][s]}
    for s in CONFIG['stocks']
])
print("\n  Selected Stocks:")
print(stock_table.to_string(index=False))

justifications = {
    'TCS.NS': 'India\'s largest IT services company with global revenue streams. Provides defensive growth and strong earnings visibility.',
    'HDFCBANK.NS': 'India\'s largest private-sector bank. Core banking sector exposure with consistent loan-book growth.',
    'RELIANCE.NS': 'Diversified conglomerate spanning energy, telecom, and retail. Captures India\'s energy & digital transformation.',
    'SUNPHARMA.NS': 'India\'s largest pharma company with strong domestic and export presence. Adds healthcare sector diversification.',
    'MARUTI.NS': 'India\'s largest passenger vehicle manufacturer with dominant market share. Provides auto sector exposure with consistent volume growth.'
}
print("\n  Justifications:")
for s, j in justifications.items():
    print(f"  - {short_name(s)}: {j}")


# #============================================================================#
# #  SECTION 4 — DATA COLLECTION                                              #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 4 — DATA COLLECTION")
print("=" * 80)

raw_datasets = {}
failed_stocks = []
for stock in CONFIG['stocks']:
    df = fetch_stock_data(stock, CONFIG['start_date'], CONFIG['end_date'])
    if len(df) == 0:
        print(f"     SKIPPING {stock} — no data available")
        failed_stocks.append(stock)
        continue
    raw_path = os.path.join(DATA_RAW, f'{short_name(stock)}.csv')
    df.to_csv(raw_path)
    raw_datasets[stock] = df
    print(f"     [SAVE] Saved -> {raw_path}")

# Remove failed stocks from CONFIG
for fs in failed_stocks:
    CONFIG['stocks'].remove(fs)
if failed_stocks:
    print(f"\n  Removed unavailable stocks: {failed_stocks}")
    print(f"  Continuing with: {CONFIG['stocks']}")


# #============================================================================#
# #  SECTION 5 — DATA PREPROCESSING                                           #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 5 — DATA PREPROCESSING")
print("=" * 80)

processed_datasets = {}
train_sets = {}
test_sets  = {}

for stock in CONFIG['stocks']:
    print(f"\n  > Preprocessing {stock}")
    df = raw_datasets[stock].copy()

    if len(df) == 0:
        print(f"     SKIPPING {stock} — empty dataset")
        continue

    # Step 3.1-3.3: Clean
    print(f"     Missing before: {df.isnull().sum().sum()}")
    df = clean_data(df)

    # Step 3.4: Ensure 'Close' column
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    # Save processed
    proc_path = os.path.join(DATA_PROC, f'{short_name(stock)}.csv')
    df.to_csv(proc_path)
    processed_datasets[stock] = df

    # Step 3.5: Train / Test split
    train = df.loc[:CONFIG['train_split_date']]
    test  = df.loc[CONFIG['train_split_date']:]
    # Remove the split date from test if it's in train
    test = test.loc[test.index > pd.Timestamp(CONFIG['train_split_date'])]

    train_sets[stock] = train
    test_sets[stock]  = test
    print(f"     Train: {len(train)} rows ({train.index.min().date()} → {train.index.max().date()})")
    print(f"     Test : {len(test)} rows ({test.index.min().date()} → {test.index.max().date()})")


# #============================================================================#
# #  SECTION 6 — RAW PRICE VISUALIZATION                                      #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 6 — RAW PRICE VISUALIZATION")
print("=" * 80)

# Individual stock plots
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    df = processed_datasets[stock]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], color='#1976D2', linewidth=1.5, label='Close')
    ax.axvline(pd.Timestamp(CONFIG['train_split_date']), color='red',
               linestyle='--', alpha=0.7, label='Train/Test Split')
    ax.fill_between(df.index, df['Close'].min(), df['Close'],
                    alpha=0.08, color='#1976D2')
    ax.set_title(f'{cn} — Daily Close Price ({CONFIG["start_date"]} to {CONFIG["end_date"]})', fontsize=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (Rs.)')
    ax.legend()
    path = os.path.join(OUT_PLOTS, f'{cn}_close_price.png')
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] Saved: {path}")

# Combined overlay plot
fig, ax = plt.subplots(figsize=(16, 8))
colors = ['#1976D2', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
for i, stock in enumerate(CONFIG['stocks']):
    cn = short_name(stock)
    df = processed_datasets[stock]
    if len(df) == 0:
        continue
    # Normalize to 100 for comparison
    normalized = df['Close'] / df['Close'].iloc[0] * 100
    ax.plot(df.index, normalized, linewidth=1.5, label=cn, color=colors[i % len(colors)])
ax.axvline(pd.Timestamp(CONFIG['train_split_date']), color='gray',
           linestyle='--', alpha=0.7, label='Train/Test Split')
ax.set_title('All Stocks -- Normalized Close Price (Base=100)', fontsize=16)
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Price')
ax.legend(loc='upper left')
path = os.path.join(OUT_PLOTS, 'all_stocks_normalized.png')
plt.savefig(path)
plt.close()
print(f"  [PLOT] Saved: {path}")


# #============================================================================#
# #  SECTION 7 — STATIONARITY ANALYSIS (ADF)                                  #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 7 — STATIONARITY ANALYSIS (ADF)")
print("=" * 80)

adf_results = {}
for stock in CONFIG['stocks']:
    print(f"\n  > {stock}")
    adf = run_adf_test(train_sets[stock]['Close'], name='Close')
    adf_results[stock] = adf

adf_table = pd.DataFrame([
    {
        'Stock': short_name(s),
        'ADF Statistic': r['adf_statistic'],
        'p-value': r['p_value'],
        'Stationary': '[Y] Yes' if r['is_stationary'] else '[N] No'
    }
    for s, r in adf_results.items()
])
print("\n  Summary of ADF Tests:")
print(adf_table.to_string(index=False))


# #============================================================================#
# #  SECTION 8 — DIFFERENCING                                                 #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 8 — DIFFERENCING")
print("=" * 80)

differenced_series = {}
adf_diff_results   = {}
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    if not adf_results[stock]['is_stationary']:
        print(f"\n  > {cn}: Non-stationary → applying first differencing")
        diff = perform_differencing(train_sets[stock]['Close'])
        differenced_series[stock] = diff
        adf_diff = run_adf_test(diff, name='Differenced')
        adf_diff_results[stock] = adf_diff
    else:
        print(f"\n  > {cn}: Already stationary → no differencing needed")
        differenced_series[stock] = train_sets[stock]['Close']
        adf_diff_results[stock] = adf_results[stock]


# #============================================================================#
# #  SECTION 9 — ACF & PACF ANALYSIS                                          #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 9 — ACF & PACF ANALYSIS")
print("=" * 80)

for stock in CONFIG['stocks']:
    cn = short_name(stock)
    path = plot_acf_pacf(differenced_series[stock], stock, OUT_PLOTS)
    print(f"  [PLOT] {cn} ACF/PACF → {path}")


# #============================================================================#
# #  SECTION 10 — ARIMA MODELING                                               #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 10 — ARIMA MODELING")
print("=" * 80)

arima_results = {}
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    print(f"\n  > {cn}")
    try:
        res = train_arima_model(
            train_sets[stock]['Close'],
            test_sets[stock]['Close'],
            stock
        )
        arima_results[stock] = res

        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(test_sets[stock].index, test_sets[stock]['Close'],
                label='Actual', color='#1976D2', linewidth=1.5)
        ax.plot(test_sets[stock].index, res['forecast'],
                label=f'ARIMA{res["order"]} Forecast', color='#F44336',
                linewidth=1.5, linestyle='--')
        ax.fill_between(test_sets[stock].index,
                        test_sets[stock]['Close'], res['forecast'],
                        alpha=0.1, color='red')
        ax.set_title(f'{cn} — ARIMA{res["order"]} Forecast vs Actual', fontsize=15)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (Rs.)')
        ax.legend()
        path = os.path.join(OUT_PLOTS, f'{cn}_arima_forecast.png')
        plt.savefig(path)
        plt.close()
        print(f"     [PLOT] Plot → {path}")

        # Save forecast CSV
        fcast_df = pd.DataFrame({
            'Date': test_sets[stock].index,
            'Actual': test_sets[stock]['Close'].values,
            'ARIMA_Forecast': res['forecast'].values
        })
        fcast_path = os.path.join(OUT_FCAST, f'{cn}_arima_forecast.csv')
        fcast_df.to_csv(fcast_path, index=False)

    except Exception as e:
        print(f"     [WARN] ARIMA failed for {cn}: {e}")
        arima_results[stock] = None


# #============================================================================#
# #  SECTION 11 — EXPONENTIAL SMOOTHING (HOLT-WINTERS)                        #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 11 — HOLT-WINTERS EXPONENTIAL SMOOTHING")
print("=" * 80)

hw_results = {}
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    print(f"\n  > {cn}")
    try:
        res = train_holt_winters(
            train_sets[stock]['Close'],
            test_sets[stock]['Close'],
            stock
        )
        hw_results[stock] = res

        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(test_sets[stock].index, test_sets[stock]['Close'],
                label='Actual', color='#1976D2', linewidth=1.5)
        ax.plot(test_sets[stock].index, res['forecast'],
                label='Holt-Winters Forecast', color='#4CAF50',
                linewidth=1.5, linestyle='--')
        ax.fill_between(test_sets[stock].index,
                        test_sets[stock]['Close'], res['forecast'],
                        alpha=0.1, color='green')
        ax.set_title(f'{cn} — Holt-Winters Forecast vs Actual', fontsize=15)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (Rs.)')
        ax.legend()
        path = os.path.join(OUT_PLOTS, f'{cn}_hw_forecast.png')
        plt.savefig(path)
        plt.close()
        print(f"     [PLOT] Plot → {path}")

        # Save forecast CSV
        fcast_df = pd.DataFrame({
            'Date': test_sets[stock].index,
            'Actual': test_sets[stock]['Close'].values,
            'HW_Forecast': res['forecast'].values
        })
        fcast_path = os.path.join(OUT_FCAST, f'{cn}_hw_forecast.csv')
        fcast_df.to_csv(fcast_path, index=False)

    except Exception as e:
        print(f"     [WARN] Holt-Winters failed for {cn}: {e}")
        hw_results[stock] = None


# #============================================================================#
# #  SECTION 12 — MODEL EVALUATION                                            #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 12 — MODEL EVALUATION")
print("=" * 80)

all_metrics = []
best_model_per_stock = {}

for stock in CONFIG['stocks']:
    cn = short_name(stock)
    actual = test_sets[stock]['Close']

    stock_metrics = []

    # ARIMA metrics
    if arima_results.get(stock) is not None:
        m = compute_metrics(actual, arima_results[stock]['forecast'], 'ARIMA')
        m['Stock'] = cn
        stock_metrics.append(m)
        all_metrics.append(m)

    # Holt-Winters metrics
    if hw_results.get(stock) is not None:
        m = compute_metrics(actual, hw_results[stock]['forecast'], 'Holt-Winters')
        m['Stock'] = cn
        stock_metrics.append(m)
        all_metrics.append(m)

    # Determine best model for this stock (lowest RMSE)
    if stock_metrics:
        best = min(stock_metrics, key=lambda x: x['RMSE'])
        best_model_per_stock[stock] = best['Model']
        print(f"  {cn}: Best model = {best['Model']} (RMSE={best['RMSE']})")

    # Save per-stock metrics
    if stock_metrics:
        mdf = pd.DataFrame(stock_metrics)
        mdf.to_csv(os.path.join(OUT_METRICS, f'{cn}_metrics.csv'), index=False)

# Master Metrics Table
metrics_df = pd.DataFrame(all_metrics)
cols_order = ['Stock', 'Model', 'RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']
metrics_df = metrics_df[cols_order]
print("\n  ┌─────────────────── MASTER METRICS TABLE ───────────────────┐")
print(metrics_df.to_string(index=False))
print("  └───────────────────────────────────────────────────────────────┘")
metrics_df.to_csv(os.path.join(OUT_METRICS, 'master_metrics.csv'), index=False)


# #============================================================================#
# #  SECTION 13 — VOLATILITY ANALYSIS                                         #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 13 — VOLATILITY ANALYSIS")
print("=" * 80)

volatility_data = {}
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    df = compute_volatility(processed_datasets[stock])
    volatility_data[stock] = df

    # Save returns
    ret_path = os.path.join(DATA_RETURNS, f'{cn}_returns.csv')
    df[['Log_Returns', 'Rolling_Volatility']].to_csv(ret_path)

    # Plot rolling volatility
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df.index, df['Log_Returns'], color='#1976D2', alpha=0.6, linewidth=0.7)
    axes[0].set_title(f'{cn} — Daily Log Returns', fontsize=14)
    axes[0].set_ylabel('Log Return')
    axes[0].axhline(0, color='gray', linewidth=0.5)

    axes[1].plot(df.index, df['Rolling_Volatility'], color='#F44336', linewidth=1.2)
    axes[1].fill_between(df.index, 0, df['Rolling_Volatility'],
                          alpha=0.15, color='#F44336')
    axes[1].set_title(f'{cn} — 30-Day Rolling Volatility', fontsize=14)
    axes[1].set_ylabel('Volatility (Std Dev)')
    axes[1].set_xlabel('Date')

    plt.suptitle(f'{cn} — Volatility Analysis', fontsize=16, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_PLOTS, f'{cn}_volatility.png')
    plt.savefig(path)
    plt.close()
    print(f"  [PLOT] {cn} volatility → {path}")

# GARCH(1,1) for each stock
print("\n  Fitting GARCH(1,1) models ...")
garch_results = {}
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    try:
        log_ret = volatility_data[stock]['Log_Returns'].dropna() * 100  # scale
        gm = arch_model(log_ret, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
        gm_fit = gm.fit(disp='off')
        garch_results[stock] = {
            'omega': gm_fit.params.get('omega', None),
            'alpha': gm_fit.params.get('alpha[1]', None),
            'beta': gm_fit.params.get('beta[1]', None),
            'aic': gm_fit.aic,
            'bic': gm_fit.bic
        }
        print(f"  {cn}: GARCH(1,1) AIC={gm_fit.aic:.2f}")
    except Exception as e:
        print(f"  [WARN] GARCH failed for {cn}: {e}")


# #============================================================================#
# #  SECTION 14 — STL DECOMPOSITION                                           #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 14 — STL DECOMPOSITION")
print("=" * 80)

stl_results = {}
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    result, path = perform_stl_decomposition(
        processed_datasets[stock]['Close'], period=30, stock=stock, save_dir=OUT_DECOMP
    )
    stl_results[stock] = result
    print(f"  [PLOT] {cn} STL → {path}")


# #============================================================================#
# #  SECTION 15 — CORRELATION ANALYSIS                                        #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 15 — CORRELATION ANALYSIS")
print("=" * 80)

# Build returns DataFrame
returns_dict = {}
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    returns_dict[cn] = volatility_data[stock]['Log_Returns']

returns_df = pd.DataFrame(returns_dict).dropna()
corr_matrix = returns_df.corr()

print("\n  Correlation Matrix:")
print(corr_matrix.round(3).to_string())

# Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap=cmap, center=0,
            mask=mask, square=True, linewidths=1, cbar_kws={'shrink': 0.8},
            ax=ax, vmin=-1, vmax=1)
ax.set_title('Cross-Stock Return Correlation Heatmap', fontsize=16)
plt.tight_layout()
path = os.path.join(OUT_PLOTS, 'correlation_heatmap.png')
plt.savefig(path)
plt.close()
print(f"  [PLOT] Heatmap → {path}")

corr_matrix.to_csv(os.path.join(OUT_METRICS, 'correlation_matrix.csv'))


# #============================================================================#
# #  SECTION 16 — PORTFOLIO CONSTRUCTION                                      #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 16 — PORTFOLIO CONSTRUCTION (Rs.10,00,000)")
print("=" * 80)

portfolio_rows = []
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    sector = CONFIG['sectors'][stock]

    # Current price = last training price
    current_price = train_sets[stock]['Close'].iloc[-1]

    # Choose best model forecast
    best = best_model_per_stock.get(stock, 'ARIMA')
    if best == 'ARIMA' and arima_results.get(stock):
        fcast = arima_results[stock]['forecast']
    elif hw_results.get(stock):
        fcast = hw_results[stock]['forecast']
    elif arima_results.get(stock):
        fcast = arima_results[stock]['forecast']
    else:
        continue

    forecast_price = fcast.iloc[-1]
    expected_return = (forecast_price - current_price) / current_price

    # Annualized volatility (from 30-day rolling std of log returns)
    vol = volatility_data[stock]['Rolling_Volatility'].dropna().iloc[-1]
    if vol == 0 or np.isnan(vol):
        vol = 0.01  # guard

    # Score = expected_return / volatility
    score = expected_return / vol

    portfolio_rows.append({
        'Stock': cn,
        'Sector': sector,
        'Current_Price': round(float(current_price), 2),
        'Forecast_Price': round(float(forecast_price), 2),
        'Expected_Return': round(float(expected_return) * 100, 2),
        'Volatility': round(float(vol), 6),
        'Score': round(float(score), 4)
    })

portfolio_df = pd.DataFrame(portfolio_rows)

# Normalize scores to weights (handle negative scores by shifting)
scores = portfolio_df['Score'].values
if scores.min() < 0:
    scores = scores - scores.min() + 0.01  # shift all positive
# Also add inverse-volatility weighting
inv_vol = 1.0 / portfolio_df['Volatility'].values
inv_vol_weights = inv_vol / inv_vol.sum()

# Blend: 50% forecast-score + 50% inverse-volatility
score_weights = scores / scores.sum()
blended_weights = 0.5 * score_weights + 0.5 * inv_vol_weights
blended_weights = blended_weights / blended_weights.sum()  # re-normalize

portfolio_df['Weight'] = np.round(blended_weights, 4)
portfolio_df['Capital_Allocated'] = np.round(blended_weights * CONFIG['capital'], 2)
portfolio_df['Shares_Approx'] = np.floor(
    portfolio_df['Capital_Allocated'] / portfolio_df['Current_Price']
).astype(int)

print("\n  ┌────────────────── PORTFOLIO ALLOCATION TABLE ──────────────────┐")
print(portfolio_df[['Stock', 'Sector', 'Expected_Return', 'Volatility',
                     'Weight', 'Capital_Allocated', 'Shares_Approx']].to_string(index=False))
print("  └─────────────────────────────────────────────────────────────────┘")
print(f"\n  Total Capital Allocated: Rs.{portfolio_df['Capital_Allocated'].sum():,.2f}")

portfolio_df.to_csv(os.path.join(OUT_PORT, 'portfolio_allocation.csv'), index=False)

# ── Pie Chart ──
fig, ax = plt.subplots(figsize=(10, 10))
colors_pie = ['#1976D2', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
explode = [0.03] * len(portfolio_df)
wedges, texts, autotexts = ax.pie(
    portfolio_df['Weight'], labels=portfolio_df['Stock'],
    autopct='%1.1f%%', colors=colors_pie, explode=explode,
    startangle=140, textprops={'fontsize': 13},
    pctdistance=0.82
)
for t in autotexts:
    t.set_fontweight('bold')
ax.set_title('Portfolio Allocation (by Weight)', fontsize=16)
path = os.path.join(OUT_PLOTS, 'portfolio_pie_chart.png')
plt.savefig(path)
plt.close()
print(f"  [PLOT] Pie chart → {path}")

# ── Sector Bar Chart ──
sector_alloc = portfolio_df.groupby('Sector')['Capital_Allocated'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(sector_alloc['Sector'], sector_alloc['Capital_Allocated'],
              color=colors_pie[:len(sector_alloc)], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, sector_alloc['Capital_Allocated']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'Rs.{val:,.0f}', ha='center', fontsize=11, fontweight='bold')
ax.set_title('Capital Allocation by Sector', fontsize=16)
ax.set_ylabel('Capital (Rs.)')
ax.set_xlabel('Sector')
path = os.path.join(OUT_PLOTS, 'sector_allocation_bar.png')
plt.savefig(path)
plt.close()
print(f"  [PLOT] Sector bar → {path}")


# #============================================================================#
# #  SECTION 17 — FORECAST NEXT 2–5 DAYS                                      #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 17 — FORECAST NEXT 2–5 DAYS")
print("=" * 80)

future_forecasts = {}
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    full_close = processed_datasets[stock]['Close']

    # Refit ARIMA on full data and forecast 5 days
    try:
        order = arima_results[stock]['order'] if arima_results.get(stock) else (1, 1, 1)
        model = ARIMA(full_close, order=order)
        fit = model.fit()
        fcast_5 = fit.forecast(steps=5)

        last_date = full_close.index[-1]
        future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=5)
        fcast_df = pd.DataFrame({
            'Date': future_dates,
            f'{cn}_Forecast': fcast_5.values
        })
        future_forecasts[stock] = fcast_df
        fcast_df.to_csv(os.path.join(DATA_FCAST, f'{cn}_forecast.csv'), index=False)
        print(f"  {cn} 5-day forecast:")
        for _, row in fcast_df.iterrows():
            print(f"     {row['Date'].strftime('%Y-%m-%d')}:  Rs.{row[f'{cn}_Forecast']:.2f}")
    except Exception as e:
        print(f"  [WARN] Future forecast failed for {cn}: {e}")


# #============================================================================#
# #  SECTION 18 — MODEL COMPARISON                                            #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 18 — MODEL COMPARISON")
print("=" * 80)

print("\n  ┌────────────────── MODEL COMPARISON TABLE ──────────────────┐")
print(metrics_df.to_string(index=False))
print("  └───────────────────────────────────────────────────────────────┘")

# Bar chart comparison
comparison_stocks = metrics_df['Stock'].unique()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics_list = ['RMSE', 'MAPE', 'Directional_Accuracy']
titles = ['RMSE (Lower is Better)', 'MAPE % (Lower is Better)',
          'Directional Accuracy % (Higher is Better)']

for idx, (metric, title) in enumerate(zip(metrics_list, titles)):
    ax = axes[idx]
    pivot = metrics_df.pivot(index='Stock', columns='Model', values=metric)
    pivot.plot(kind='bar', ax=ax, color=['#1976D2', '#4CAF50'], edgecolor='white')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    ax.legend(title='Model')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('ARIMA vs Holt-Winters — Model Comparison', fontsize=16, y=1.02)
plt.tight_layout()
path = os.path.join(OUT_PLOTS, 'model_comparison.png')
plt.savefig(path)
plt.close()
print(f"  [PLOT] Comparison chart → {path}")


# #============================================================================#
# #  SECTION 19 — PREDICTION VS ACTUAL COMPARISON                             #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 19 — PREDICTION VS ACTUAL COMPARISON")
print("=" * 80)

comparison_rows = []
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    actual = test_sets[stock]['Close']

    # Use best model
    best = best_model_per_stock.get(stock, 'ARIMA')
    if best == 'ARIMA' and arima_results.get(stock):
        pred = arima_results[stock]['forecast']
    elif hw_results.get(stock):
        pred = hw_results[stock]['forecast']
    else:
        continue

    # Last predicted vs last actual
    last_actual = float(actual.iloc[-1])
    last_pred   = float(pred.iloc[-1])
    error_pct   = abs(last_actual - last_pred) / last_actual * 100

    # Direction correct on last move?
    if len(actual) > 1 and len(pred) > 1:
        actual_dir = 'Up' if actual.iloc[-1] > actual.iloc[-2] else 'Down'
        pred_dir   = 'Up' if pred.iloc[-1] > pred.iloc[-2] else 'Down'
        dir_correct = actual_dir == pred_dir
    else:
        dir_correct = None

    comparison_rows.append({
        'Stock': cn,
        'Best_Model': best,
        'Predicted': round(last_pred, 2),
        'Actual': round(last_actual, 2),
        'Error_%': round(error_pct, 2),
        'Direction_Correct': '[Y]' if dir_correct else '[N]'
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n  Prediction vs Actual (End of Test Period):")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(OUT_METRICS, 'prediction_vs_actual.csv'), index=False)

# Portfolio return
initial_value = portfolio_df['Capital_Allocated'].sum()
# Simulate: for each stock, compute return over test period
final_value = 0
for _, row in portfolio_df.iterrows():
    stock_full = row['Stock'] + '.NS'
    if stock_full in test_sets and len(test_sets[stock_full]) > 0:
        test_start_price = test_sets[stock_full]['Close'].iloc[0]
        test_end_price   = test_sets[stock_full]['Close'].iloc[-1]
        stock_return = (test_end_price - test_start_price) / test_start_price
        final_value += row['Capital_Allocated'] * (1 + stock_return)

portfolio_return = (final_value - initial_value) / initial_value * 100
print(f"\n  Portfolio Return (simulated over test period): {portfolio_return:.2f}%")
print(f"  Initial: Rs.{initial_value:,.2f}  →  Final: Rs.{final_value:,.2f}")


# #============================================================================#
# #  SECTION 20 — COMPREHENSIVE VISUALIZATIONS                                #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 20 — COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

# Combined Forecast vs Actual for all stocks (2x3 grid)
fig, axes = plt.subplots(3, 2, figsize=(18, 18))
axes = axes.flatten()
for idx, stock in enumerate(CONFIG['stocks']):
    cn = short_name(stock)
    ax = axes[idx]
    actual = test_sets[stock]['Close']
    ax.plot(actual.index, actual.values, label='Actual', color='#1976D2', linewidth=1.5)

    if arima_results.get(stock):
        ax.plot(actual.index, arima_results[stock]['forecast'].values,
                label='ARIMA', color='#F44336', linewidth=1.2, linestyle='--')
    if hw_results.get(stock):
        ax.plot(actual.index, hw_results[stock]['forecast'].values,
                label='Holt-Winters', color='#4CAF50', linewidth=1.2, linestyle='-.')

    ax.set_title(f'{cn}', fontsize=14)
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=30)
# Hide unused subplot
if len(CONFIG['stocks']) < len(axes):
    for j in range(len(CONFIG['stocks']), len(axes)):
        axes[j].set_visible(False)
plt.suptitle('Forecast vs Actual — All Stocks', fontsize=18, y=1.01)
plt.tight_layout()
path = os.path.join(OUT_PLOTS, 'all_forecast_vs_actual.png')
plt.savefig(path)
plt.close()
print(f"  [PLOT] All forecasts → {path}")

# Combined volatility
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
axes = axes.flatten()
for idx, stock in enumerate(CONFIG['stocks']):
    cn = short_name(stock)
    ax = axes[idx]
    vol = volatility_data[stock]['Rolling_Volatility']
    ax.plot(vol.index, vol.values, color='#F44336', linewidth=1)
    ax.fill_between(vol.index, 0, vol.values, alpha=0.15, color='#F44336')
    ax.set_title(f'{cn} — Rolling Volatility', fontsize=13)
if len(CONFIG['stocks']) < len(axes):
    for j in range(len(CONFIG['stocks']), len(axes)):
        axes[j].set_visible(False)
plt.suptitle('30-Day Rolling Volatility — All Stocks', fontsize=16, y=1.01)
plt.tight_layout()
path = os.path.join(OUT_PLOTS, 'all_volatility.png')
plt.savefig(path)
plt.close()
print(f"  [PLOT] All volatility → {path}")

# Copy key plots to report/figures
import shutil
key_plots = ['all_stocks_normalized.png', 'all_forecast_vs_actual.png',
             'correlation_heatmap.png', 'portfolio_pie_chart.png',
             'sector_allocation_bar.png', 'model_comparison.png',
             'all_volatility.png']
for p in key_plots:
    src = os.path.join(OUT_PLOTS, p)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(RPT_FIGS, p))

# Also copy STL plots
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    src = os.path.join(OUT_DECOMP, f'{cn}_stl.png')
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(RPT_FIGS, f'{cn}_stl.png'))


# #============================================================================#
# #  SECTION 21 — FINAL REPORT GENERATION                                     #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 21 — FINAL REPORT GENERATION")
print("=" * 80)

report_lines = []
report_lines.append("# TSA Capstone Project — Final Report\n")
report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

report_lines.append("\n## 1. Introduction\n")
report_lines.append("This project implements a complete end-to-end time series analysis pipeline ")
report_lines.append("for forecasting stock prices of 5 NSE-listed companies across diversified sectors. ")
report_lines.append("The goal is to build reliable forecasting models (ARIMA, Holt-Winters), evaluate ")
report_lines.append("their accuracy, analyze volatility and trends, and construct a data-driven ")
report_lines.append(f"portfolio allocation of Rs.{CONFIG['capital']:,.0f}.\n")

report_lines.append("\n## 2. Data Collection\n")
report_lines.append(f"- **Source**: Yahoo Finance (yfinance library)\n")
report_lines.append(f"- **Period**: {CONFIG['start_date']} to {CONFIG['end_date']}\n")
report_lines.append(f"- **Interval**: Daily\n")
report_lines.append(f"- **Stocks**: {', '.join([short_name(s) for s in CONFIG['stocks']])}\n")
report_lines.append("\n| Stock | Sector | Train Rows | Test Rows |\n")
report_lines.append("|-------|--------|-----------|----------|\n")
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    report_lines.append(f"| {cn} | {CONFIG['sectors'][stock]} | {len(train_sets[stock])} | {len(test_sets[stock])} |\n")

report_lines.append("\n## 3. Preprocessing\n")
report_lines.append("- Missing values handled via forward-fill (ffill)\n")
report_lines.append("- Date index parsed and set as DateTimeIndex\n")
report_lines.append(f"- Train/Test split at {CONFIG['train_split_date']}\n")

report_lines.append("\n### ADF Stationarity Tests\n")
report_lines.append("\n| Stock | ADF Statistic | p-value | Stationary |\n")
report_lines.append("|-------|--------------|---------|------------|\n")
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    r = adf_results[stock]
    stat = '[Y] Yes' if r['is_stationary'] else '[N] No'
    report_lines.append(f"| {cn} | {r['adf_statistic']:.4f} | {r['p_value']:.6f} | {stat} |\n")
report_lines.append("\nAll non-stationary series were first-differenced before ARIMA modeling.\n")

report_lines.append("\n## 4. Forecasting Models\n")
report_lines.append("\n### ARIMA\n")
report_lines.append("Auto-ARIMA (pmdarima) was used to find optimal (p,d,q) orders:\n\n")
report_lines.append("| Stock | Order | AIC |\n")
report_lines.append("|-------|-------|-----|\n")
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    if arima_results.get(stock):
        r = arima_results[stock]
        report_lines.append(f"| {cn} | ARIMA{r['order']} | {r['aic']:.2f} |\n")

report_lines.append("\n### Holt-Winters Exponential Smoothing\n")
report_lines.append("Additive trend model with no seasonal component.\n")

report_lines.append("\n## 5. Model Evaluation\n")
report_lines.append("\n| Stock | Model | RMSE | MAE | MAPE (%) | Dir. Accuracy (%) |\n")
report_lines.append("|-------|-------|------|-----|----------|------------------|\n")
for _, row in metrics_df.iterrows():
    report_lines.append(f"| {row['Stock']} | {row['Model']} | {row['RMSE']} | {row['MAE']} | {row['MAPE']} | {row['Directional_Accuracy']} |\n")

report_lines.append("\n![Model Comparison](figures/model_comparison.png)\n")

report_lines.append("\n## 6. Volatility & Trend Analysis\n")
report_lines.append("- **Log returns** computed: `log(P_t / P_{t-1})`\n")
report_lines.append("- **30-day rolling std** of log returns used as volatility proxy\n")
report_lines.append("- **GARCH(1,1)** fitted to model conditional heteroskedasticity\n")
report_lines.append("- **STL decomposition** (period=30) extracts trend, seasonal, and residual\n")
report_lines.append("\n![Volatility](figures/all_volatility.png)\n")

report_lines.append("\n## 7. Portfolio Construction\n")
report_lines.append(f"\nTotal Capital: Rs.{CONFIG['capital']:,.0f}\n")
report_lines.append("\n**Strategy**: 50% Forecast-Guided + 50% Inverse-Volatility Weighted\n")
report_lines.append("\n| Stock | Sector | Expected Return (%) | Weight | Capital (Rs.) |\n")
report_lines.append("|-------|--------|-------------------|--------|------------|\n")
for _, row in portfolio_df.iterrows():
    report_lines.append(f"| {row['Stock']} | {row['Sector']} | {row['Expected_Return']} | {row['Weight']:.2%} | {row['Capital_Allocated']:,.0f} |\n")
report_lines.append(f"\n**Simulated Portfolio Return**: {portfolio_return:.2f}%\n")
report_lines.append("\n![Portfolio Allocation](figures/portfolio_pie_chart.png)\n")
report_lines.append("\n![Sector Allocation](figures/sector_allocation_bar.png)\n")

report_lines.append("\n## 8. Correlation Analysis\n")
report_lines.append("Low inter-stock correlation aids diversification.\n")
report_lines.append("\n![Correlation Heatmap](figures/correlation_heatmap.png)\n")

report_lines.append("\n## 9. Prediction vs Actual\n")
report_lines.append("\n| Stock | Model | Predicted | Actual | Error (%) | Direction Correct |\n")
report_lines.append("|-------|-------|-----------|--------|-----------|------------------|\n")
for _, row in comparison_df.iterrows():
    report_lines.append(f"| {row['Stock']} | {row['Best_Model']} | {row['Predicted']} | {row['Actual']} | {row['Error_%']} | {row['Direction_Correct']} |\n")

report_lines.append("\n![All Forecasts](figures/all_forecast_vs_actual.png)\n")

report_lines.append("\n## 10. Reflection\n")
report_lines.append("\n### What Worked\n")
report_lines.append("- Auto-ARIMA effectively identifies optimal parameters per stock\n")
report_lines.append("- Holt-Winters provides smooth, interpretable trend forecasts\n")
report_lines.append("- Diversified stock selection reduces portfolio risk\n")
report_lines.append("- Volatility-weighted allocation protects against high-risk assets\n")
report_lines.append("\n### Limitations\n")
report_lines.append("- ARIMA assumes linear relationships; may miss nonlinear patterns\n")
report_lines.append("- Stock markets are influenced by sentiment, news, and macroeconomic factors not captured here\n")
report_lines.append("- Test period may not represent future market conditions\n")
report_lines.append("\n### Future Improvements\n")
report_lines.append("- SARIMA for capturing seasonality\n")
report_lines.append("- LSTM/GRU deep learning models for nonlinear patterns\n")
report_lines.append("- Ensemble forecasting combining multiple models\n")
report_lines.append("- News sentiment analysis as exogenous features\n")
report_lines.append("- Reinforcement learning for dynamic portfolio optimization\n")

report_path = os.path.join(CONFIG['base_dir'], 'report', 'final_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)
print(f"  [DOC] Final report → {report_path}")


# #============================================================================#
# #  SECTION 22 — MASTER METADATA TABLE                                       #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 22 — MASTER METADATA TABLE")
print("=" * 80)

metadata_rows = []
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    row = {
        'Stock': cn,
        'Sector': CONFIG['sectors'][stock],
        'Dataset_Path': os.path.join(DATA_RAW, f'{cn}.csv'),
        'Model_Used': best_model_per_stock.get(stock, 'N/A'),
        'Stationary': adf_results[stock]['is_stationary'],
    }
    # Get best RMSE
    stock_metrics_rows = metrics_df[metrics_df['Stock'] == cn]
    if len(stock_metrics_rows) > 0:
        row['Best_RMSE'] = stock_metrics_rows['RMSE'].min()
    metadata_rows.append(row)

metadata_df = pd.DataFrame(metadata_rows)
print("\n  Master Metadata:")
print(metadata_df.to_string(index=False))
metadata_df.to_csv(os.path.join(OUT_METRICS, 'master_metadata.csv'), index=False)


# #============================================================================#
# #  SECTION 23 — JUPYTER NOTEBOOK GENERATION                                 #
# #============================================================================#

print("\n" + "=" * 80)
print("  SECTION 23 — JUPYTER NOTEBOOK GENERATION")
print("=" * 80)

try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

    nb = new_notebook()
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }

    # Read this script and split into sections
    script_path = os.path.abspath(__file__)
    with open(script_path, 'r', encoding='utf-8') as f:
        script_content = f.read()

    # Split by section markers
    sections = script_content.split('# #=')
    
    for i, section in enumerate(sections):
        if i == 0:
            # First chunk: docstring + imports
            nb.cells.append(new_markdown_cell("# TSA Capstone Project\n## Complete Time Series Analysis & Portfolio Construction\n\n---"))
            nb.cells.append(new_code_cell(section.strip()))
        else:
            # Extract section title from the box header
            lines = section.split('\n')
            title = ''
            code_lines = []
            past_header = False
            for line in lines:
                if '#=' in line:
                    past_header = True
                    continue
                if not past_header:
                    if '#' in line:
                        clean = line.replace('#', '').replace('=', '').strip()
                        if clean:
                            title = clean
                    continue
                code_lines.append(line)

            if title:
                nb.cells.append(new_markdown_cell(f"## {title}\n---"))
            
            code = '\n'.join(code_lines).strip()
            if code:
                # Split very large code blocks
                if len(code) > 5000:
                    # Split at double newlines
                    chunks = code.split('\n\n\n')
                    for chunk in chunks:
                        if chunk.strip():
                            nb.cells.append(new_code_cell(chunk.strip()))
                else:
                    nb.cells.append(new_code_cell(code))

    nb_path = os.path.join(CONFIG['base_dir'], 'notebooks', 'TSA_Capstone_Project.ipynb')
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"  [NB] Notebook → {nb_path}")

except Exception as e:
    print(f"  [WARN] Notebook generation failed: {e}")


# #============================================================================#
# #  FINAL SUMMARY                                                            #
# #============================================================================#

print("\n" + "=" * 80)
print("  ✅ TSA CAPSTONE PROJECT COMPLETE")
print("=" * 80)
print(f"""
  [DIR] Outputs Generated:
     data/raw/          — {len(CONFIG['stocks'])} raw CSVs
     data/processed/    — {len(CONFIG['stocks'])} cleaned CSVs
     data/returns/      — {len(CONFIG['stocks'])} return CSVs
     data/forecasts/    — {len(CONFIG['stocks'])} future forecast CSVs
     outputs/plots/     — All visualization PNGs
     outputs/metrics/   — Evaluation metrics CSVs
     outputs/decomposition/ — STL decomposition PNGs
     outputs/forecasts/ — Forecast vs actual CSVs
     outputs/portfolio/ — Portfolio allocation CSV
     report/            — Final report (Markdown)
     notebooks/         — Jupyter Notebook

  [PLOT] Key Results:
     Portfolio Return (simulated): {portfolio_return:.2f}%
     Total Capital: Rs.{CONFIG['capital']:,.0f}
""")
for stock in CONFIG['stocks']:
    cn = short_name(stock)
    best = best_model_per_stock.get(stock, 'N/A')
    best_rmse = metrics_df[metrics_df['Stock'] == cn]['RMSE'].min() if len(metrics_df[metrics_df['Stock'] == cn]) > 0 else 'N/A'
    print(f"     {cn:15s}  Best: {best:15s}  RMSE: {best_rmse}")

print("\n" + "=" * 80)
