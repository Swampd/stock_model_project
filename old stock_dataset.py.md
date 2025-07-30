# python3 stock_dataset.py | tee runlog.txt
# pip freeze > requirements.txt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import pandas_ta as pta
import ta
from event_flags import add_event_flags
from feature_engineering import add_all_features, sector_etfs
from model_training import run_model_training
from data_loader import load_all_stocks, download_etfs, merge_etf_prices, scrape_nasdaq_earnings, scrape_nasdaq_dividends, scrape_nasdaq_splits
from config import tickers, start_date, end_date, cull_ratio, sector_etfs, scrape_delay_nasdaq, features_list, FINNHUB_APIKEY
from fundamental_scraper import (
    get_pe_ratios_for_tickers,
    get_eps_surprises_for_tickers,
    get_finviz_upgrades_for_tickers,
    get_short_interest_for_tickers,
    get_insider_transactions_for_tickers,
    get_inst_holdings_for_tickers,
    get_news_count_for_tickers,
    get_news_count_for_tickers_finnhub
)

def load_or_update_calendar_csv(csv_path, scrape_func, tickers, start_date, end_date, event_name, sleep_sec):
    if os.path.exists(csv_path):
        cached = pd.read_csv(csv_path)
        date_col = next((col for col in cached.columns if col.lower().endswith('date')), None)
        if date_col is not None:
            cached[date_col] = pd.to_datetime(cached[date_col], errors='coerce')
        else:
            print("[ERROR] Could not find a date column in cached CSV:", csv_path)
            return pd.DataFrame()
        requested_dates = pd.date_range(start=start_date, end=end_date)
        cached_dates = pd.to_datetime(cached[date_col]).unique()
        missing_dates = sorted(set(requested_dates) - set(cached_dates))
        missing_tickers = set(tickers) - set(cached['ticker'].unique())
        has_all_dates = len(missing_dates) == 0
        has_all_tickers = len(missing_tickers) == 0
        if has_all_dates and has_all_tickers:
            print(f"Loaded full {event_name} calendar from cache.")
            return cached
        print(f"Some dates/tickers missing in cached {event_name} calendar.")
        print(f"Missing dates: {missing_dates[:5]}... (and {len(missing_dates)} total)")
        print(f"Missing tickers: {list(missing_tickers)}")
        action = input(f"Would you like to scrape missing dates ({len(missing_dates)} days) or drop those rows? [scrape/drop]: ").strip().lower()
        if action == "scrape":
            print(f"Scraping {event_name} for missing dates only...")
            new_rows = scrape_func(tickers, missing_dates[0].strftime('%Y-%m-%d'), missing_dates[-1].strftime('%Y-%m-%d'), sleep_sec)
            updated = pd.concat([cached, new_rows]).drop_duplicates()
            updated.to_csv(csv_path, index=False)
            print(f"Updated {event_name} calendar CSV.")
            return updated
        else:
            print(f"Dropping missing dates/rows from DataFrame.")
            filtered = cached[cached[date_col].isin(requested_dates) & cached['ticker'].isin(tickers)]
            return filtered
    else:
        action = input(f"No cached {event_name} calendar found. Scrape now? [Y/n]: ").strip().lower()
        if action in ["", "y", "yes"]:
            print(f"Scraping full {event_name} calendar from {start_date} to {end_date}...")
            new_calendar = scrape_func(tickers, start_date, end_date, sleep_sec)
            new_calendar.to_csv(csv_path, index=False)
            return new_calendar
        else:
            print(f"Skipping {event_name} calendar—rows will be missing.")
            return pd.DataFrame()  # Or handle as needed

FINNHUB_API_KEY = FINNHUB_APIKEY   

# --- 1. SCRAPE ALL EXTERNAL FEATURES UP FRONT (DON'T MERGE YET) ---
pe_df       = get_pe_ratios_for_tickers(tickers)
short_df    = get_short_interest_for_tickers(tickers)
insider_df  = get_insider_transactions_for_tickers(tickers)
inst_df     = get_inst_holdings_for_tickers(tickers)

if os.path.exists('news_counts.csv'):
    print("Loading news counts from disk...")
    news_df = pd.read_csv('news_counts.csv')
else:
    print("Scraping news counts from Finnhub...")
    news_df = get_news_count_for_tickers_finnhub(tickers, start_date, end_date, FINNHUB_API_KEY)
    news_df.to_csv('news_counts.csv', index=False)
    print("Saved new news_counts.csv")

FEATURES_PATH = "features_data.parquet"

# --- 2. LOAD OR BUILD MAIN DATAFRAME ---
if os.path.exists(FEATURES_PATH):
    print("Loading features from disk (test)...")
    data = pd.read_parquet(FEATURES_PATH)
else:
    print("Running full feature engineering (test)...")
    data = load_all_stocks(tickers, start_date, end_date)
    unique_etfs = set(sector_etfs.values())
    sector_etf_data = download_etfs(unique_etfs, start_date, end_date)
    data = merge_etf_prices(data, tickers, sector_etfs, sector_etf_data)
    data = data.groupby("ticker").apply(add_all_features).reset_index(drop=True)
    data.to_parquet(FEATURES_PATH)

data['Date'] = pd.to_datetime(data['Date'])
news_df['Date'] = pd.to_datetime(news_df['Date'])

# --- 3. MERGE ALL EXTERNAL FEATURES ---
data = data.merge(pe_df, on="ticker", how="left")
data = data.merge(short_df, on="ticker", how="left")
data = data.merge(insider_df, on="ticker", how="left")
data = data.merge(inst_df, on="ticker", how="left")

# --- 4. ADD MARKET CONTEXT (VIX, SP500, QQQ) ---
vix = yf.download('^VIX', start=start_date, end=end_date)
vix = vix.reset_index()[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
vix.index.name = None

sp500 = yf.download("^GSPC", start=start_date, end=end_date)
sp500 = sp500.reset_index()[["Date", "Close"]].rename(columns={"Close": "SP500_Close"})
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
sp500.index.name = None

qqq = yf.download("QQQ", start=start_date, end=end_date)
qqq = qqq.reset_index()[["Date", "Close"]].rename(columns={"Close": "QQQ_Close"})
if isinstance(qqq.columns, pd.MultiIndex):
    qqq.columns = qqq.columns.get_level_values(0)
qqq.index.name = None

data = data.merge(vix, on='Date', how='left')
data = data.merge(sp500, on='Date', how='left')
data = data.merge(qqq, on='Date', how='left')
data["sp500_return_3d"] = data.groupby("ticker")["SP500_Close"].pct_change(periods=3)
data["qqq_return_3d"] = data.groupby("ticker")["QQQ_Close"].pct_change(periods=3)
data["vix_return_3d"] = data.groupby("ticker")["VIX_Close"].pct_change(periods=3)

# --- 5. ADD DERIVED FEATURES ---
data['rel_strength'] = data['return_3d'] - data['sp500_return_3d']
data['corr_sp500_10d'] = data['Close'].rolling(10).corr(data['SP500_Close'])

# --- 6. ADD EVENT FLAGS (AFTER DATA IS BUILT) ---
earnings_calendar = load_or_update_calendar_csv(
    csv_path='earnings_calendar.csv',
    scrape_func=scrape_nasdaq_earnings,
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    event_name="earnings",
    sleep_sec=scrape_delay_nasdaq
)
dividends_calendar = load_or_update_calendar_csv(
    csv_path='dividends_calendar.csv',
    scrape_func=scrape_nasdaq_dividends,
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    event_name="dividends",
    sleep_sec=scrape_delay_nasdaq
)
splits_calendar = load_or_update_calendar_csv(
    csv_path='splits_calendar.csv',
    scrape_func=scrape_nasdaq_splits,
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    event_name="splits",
    sleep_sec=scrape_delay_nasdaq
)
data = add_event_flags(data, earnings_calendar, event_type='earnings')
data = add_event_flags(data, dividends_calendar, event_type='dividend')
data = add_event_flags(data, splits_calendar, event_type='split')

# --- 7. LABELS - CALCULATE LAST! ---
def calc_label(df, pct=0.02, days=3):
    df = df.sort_values("Date").reset_index(drop=True)
    df["future_close"] = df["Close"].shift(-days)
    df["future_up_3d"] = (df["future_close"] >= df["Close"] * (1 + pct)).astype(int)
    return df

data = data.groupby("ticker").apply(calc_label).reset_index(drop=True)
print("\nSAMPLE WITH LABELS:")
print(data[["Date", "ticker", "Close", "future_close", "future_up_3d"]].tail(10))

# --- 8. OPTIONAL: Analyst Upgrades/Downgrades ---
if os.path.exists('finviz_upgrades.csv'):
    print("Loading Finviz upgrades from disk...")
    upgrades_df = pd.read_csv('finviz_upgrades.csv')
else:
    print("Scraping Finviz upgrades (slow)...")
    upgrades_df = get_finviz_upgrades_for_tickers(tickers)
    upgrades_df.to_csv('finviz_upgrades.csv', index=False)
    print("Saved new finviz_upgrades.csv")

if upgrades_df.empty or not all(x in upgrades_df.columns for x in ['ticker', 'upgrade_flag', 'downgrade_flag', 'upgrade_date']):
    print("WARNING: No analyst upgrade data found! Skipping upgrade/downgrade merge.")
    data['upgrade_flag'] = None
    data['downgrade_flag'] = None
else:
    upgrades_df = upgrades_df.sort_values('upgrade_date').drop_duplicates('ticker', keep='last')
    data = data.merge(upgrades_df[['ticker', 'upgrade_flag', 'downgrade_flag']], on='ticker', how='left')

# --- 9. FEATURE LIST ---
features = features_list

# --- 10. DEBUGGING ---
label_counts = data["future_up_3d"].value_counts(normalize=True)
print("\nLabel Distribution (what % of cases are 'up'):")
print(label_counts)
print("\nAverage indicators by label:")
print(data.groupby("future_up_3d")[["RSI", "MACD", "MACD_signal", "MACD_hist"]].mean())

# Guarantee all features exist in data
for col in features:
    if col not in data.columns:
        data[col] = None

# --- DEBUG: Check which features are NaN before dropping ---
print("\nNaN count for each feature before model training:")
print(data[features].isnull().sum())

# --- PUT THIS CODE BLOCK RIGHT HERE ---
all_nan_features = [col for col in features if data[col].isnull().all()]
if all_nan_features:
    print(f"Warning: Dropping {len(all_nan_features)} features with 100% NaN: {all_nan_features}")
    features = [f for f in features if f not in all_nan_features]
# --- END ---
# --- 9. FEATURE LIST (ABSOLUTE MINIMUM SAFETY PATCH) ---
features = [
    "RSI", "MACD", "MACD_signal", "is_new_high_5d", "above_ma20",
    "volume_spike", "return_3d", "volatility_5d", "up_days_5d", "pe_ratio"
]

for col in features:
    if col not in data.columns:
        data[col] = None

print("\nNaN count for each feature before model training:")
print(data[features].isnull().sum())

# If literally ALL features are NaN, stop
working_features = [col for col in features if data[col].notnull().sum() > 0]
if not working_features:
    print("\n[ERROR] No features have any real data. Model cannot train. Check your CSVs, scrapes, and merges.")
    exit(1)
features = working_features
print(f"\n[INFO] Using {len(features)} features with data: {features}")

# --- 11. RUN MODEL TRAINING ---
run_model_training(data, features, label="future_up_3d", tied_thresh=cull_ratio)
