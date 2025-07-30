# PC  - python stock_dataset.py 
# MAC - python3 stock_dataset.py | tee runlog.txt
# pip freeze > requirements.txt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import yfinance as yf
import pandas as pd
import pandas_ta as pta
import ta
from event_flags import add_event_flags
from feature_engineering import add_all_features, sector_etfs
from model_training import run_model_training
from data_loader import load_all_stocks, download_etfs, merge_etf_prices
from config import (
    tickers, start_date, end_date, cull_ratio, sector_etfs,
    scrape_delay_nasdaq, features_list, FINNHUB_APIKEY,
    drop_column_null_threshold, ALWAYS_KEEP, TIINGO_APIKEY, api_key
)

from fundamental_scraper import (
    get_pe_ratios_for_tickers,
    get_eps_surprises_for_tickers,
    get_finviz_upgrades_for_tickers,
    get_short_interest_for_tickers,
    get_insider_transactions_for_tickers,
    get_inst_holdings_for_tickers,
    get_news_count_for_tickers,
    get_news_count_for_tickers_finnhub,
    get_finnhub_earnings_calendar,
    get_finnhub_dividends,
    get_finnhub_splits, get_news_count_for_tickers_finnhub, get_news_sentiment_for_tickers_finnhub,
    get_tiingo_news_for_tickers,
    get_tiingo_news_for_ticker
)

#########################################################################################################################
###################################################### LET'S BEGIN ######################################################
#########################################################################################################################

# --- NEW FUNCTION: Place THIS RIGHT AFTER imports ---
def fetch_and_update_news_cache(
    tickers, start_date, end_date, api_key,
    archive_dir="news_archive",  # where timestamped files are stored
    


    latest_file="tiingo_news_latest.csv",
    master_file="tiingo_news_master.csv"
):
    print("api_key value:", api_key)
    os.makedirs(archive_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    archive_filename = os.path.join(archive_dir, f"tiingo_news_{timestamp}.csv")

    # 1. Load master if exists, else create empty DataFrame
    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
    else:
        master_df = pd.DataFrame(columns=[
            "ticker", "Date", "news_count", "sentiment_mean", 
            "sentiment_median", "sentiment_max", "sentiment_min"
        ])

    # 2. Determine what to fetch
    existing_keys = set(zip(master_df["ticker"], master_df["Date"])) if not master_df.empty else set()

    new_data_frames = []
    first = True
    for ticker in tickers:
        print(f"[NEWS] Checking {ticker} for missing dates...")
        df_new = get_tiingo_news_for_ticker(ticker, start_date, end_date, api_key)
        if not df_new.empty and not master_df.empty:
            df_new = df_new[~df_new.apply(lambda x: (x["ticker"], str(x["Date"])) in existing_keys, axis=1)]
        if not df_new.empty:
            print(f"[NEWS] Found {len(df_new)} new rows for {ticker}")
            new_data_frames.append(df_new)
            # Append or write after each ticker
            mode = 'w' if first else 'a'
            header = first
            df_new.to_csv(archive_filename, mode=mode, header=header, index=False)
            first = False

    if new_data_frames:
        new_news_df = pd.concat(new_data_frames, ignore_index=True)
    else:
        new_news_df = pd.DataFrame(columns=master_df.columns)

    new_news_df.to_csv(archive_filename, index=False)
    new_news_df.to_csv(latest_file, index=False)
    print(f"[NEWS] Saved {len(new_news_df)} rows to {archive_filename} and {latest_file}")

    combined_df = pd.concat([master_df, new_news_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["ticker", "Date", "news_count", "sentiment_mean"], keep='last')
    combined_df.to_csv(master_file, index=False)
    print(f"[NEWS] Updated {master_file} ({len(combined_df)} total rows)")

    return new_news_df, combined_df, archive_filename
# --- END OF NEW FUNCTION ---

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
        print(f"Scraping {event_name} for all missing dates/tickers...")
        new_rows = scrape_func(
            tickers,
            missing_dates[0].strftime('%Y-%m-%d') if missing_dates else start_date,
            missing_dates[-1].strftime('%Y-%m-%d') if missing_dates else end_date,
            sleep_sec
        )
        updated = pd.concat([cached, new_rows]).drop_duplicates()
        updated.to_csv(csv_path, index=False)
        print(f"Updated {event_name} calendar CSV.")
        return updated
    else:
        print(f"Scraping {event_name} for all dates/tickers (no cache)...")
        new_rows = scrape_func(tickers, start_date, end_date, sleep_sec)
        new_rows.to_csv(csv_path, index=False)
        print(f"Saved new {event_name} calendar CSV.")
        return new_rows

FINNHUB_API_KEY = FINNHUB_APIKEY   

# --- 1. SCRAPE ALL EXTERNAL FEATURES UP FRONT (DON'T MERGE YET) ---
pe_df = get_pe_ratios_for_tickers(tickers)
# (Broken features, not used for now)
# short_df    = get_short_interest_for_tickers(tickers)
# insider_df  = get_insider_transactions_for_tickers(tickers)
# inst_df     = get_inst_holdings_for_tickers(tickers)

news_df, master_df, archive_filename = fetch_and_update_news_cache(
    tickers, start_date, end_date, TIINGO_APIKEY
)

# For sentiment (this is ticker-level, not date-level)
#sentiment_df = get_news_sentiment_for_tickers_finnhub(tickers, FINNHUB_APIKEY)
#sentiment_df.to_csv('news_sentiment.csv', index=False)

# Merge both as you see fit

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

# Merge in news count by ticker and date
data = data.merge(master_df, on=["ticker", "Date"], how="left")

#if not sentiment_df.empty:
#    data = data.merge(sentiment_df, on="ticker", how="left")
print(data[["ticker", "Date", "news_count", "sentiment_mean", "sentiment_median", "sentiment_max", "sentiment_min"]].tail(20))

# --- 3. MERGE ALL EXTERNAL FEATURES ---
data = data.merge(pe_df, on="ticker", how="left")
# data = data.merge(short_df, on="ticker", how="left")
# data = data.merge(insider_df, on="ticker", how="left")
# data = data.merge(inst_df, on="ticker", how="left")

print(data.isnull().sum().sort_values())

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

# --- 6. ADD EVENT FLAGS (DYNAMIC FEATURE DROPPING) ---
earnings_calendar = load_or_update_calendar_csv(
    csv_path='earnings_calendar.csv',
    scrape_func=lambda t, s, e, d: get_finnhub_earnings_calendar(t, s, e, FINNHUB_API_KEY),
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    event_name="earnings",
    sleep_sec=0
)

if earnings_calendar is None or earnings_calendar.empty:
    print("Dropping earnings event features: is_earnings_week, days_to_earnings (no data)")
    features_list = [f for f in features_list if f not in ["is_earnings_week", "days_to_earnings"]]
else:
    data = add_event_flags(data, earnings_calendar, event_type='earnings')

dividends_calendar = load_or_update_calendar_csv(
    csv_path='dividends_calendar.csv',
    scrape_func=lambda t, s, e, d: get_finnhub_dividends(t, s, e, FINNHUB_API_KEY),
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    event_name="dividends",
    sleep_sec=0
)

if dividends_calendar is None or dividends_calendar.empty:
    print("Dropping dividend event features: is_dividend_week, days_to_dividend (no data)")
    features_list = [f for f in features_list if f not in ["is_dividend_week", "days_to_dividend"]]
else:
    data = add_event_flags(data, dividends_calendar, event_type='dividend')

splits_calendar = load_or_update_calendar_csv(
    csv_path='splits_calendar.csv',
    scrape_func=lambda t, s, e, d: get_finnhub_splits(t, s, e, FINNHUB_API_KEY),
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    event_name="splits",
    sleep_sec=0
)

if splits_calendar is None or splits_calendar.empty:
    print("Dropping split event features: is_split_week, days_to_split (no data)")
    features_list = [f for f in features_list if f not in ["is_split_week", "days_to_split"]]
else:
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
def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        print(f"[WARN] {path} is empty. Returning empty DataFrame.")
        return pd.DataFrame()

if os.path.exists('finviz_upgrades.csv'):
    print("Loading Finviz upgrades from disk...")
    upgrades_df = safe_read_csv('finviz_upgrades.csv')
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

# --- AUTOMATIC FEATURE CLEANING ---

features = features_list  # Start from your intended feature list

# Guarantee all features exist in data (add as None if missing)
for col in features:
    if col not in data.columns:
        data[col] = None

# Parameters for feature cleaning:
MIN_COVERAGE = drop_column_null_threshold  # Minimum percent of non-null rows required (3%)
BLACKLIST = set()  # Optionally add any features you always want to exclude

# Compute non-null rate for each feature
feature_coverage = 1 - data[features].isnull().mean()

# Drop features below MIN_COVERAGE unless in ALWAYS_KEEP
to_drop = [col for col, cov in feature_coverage.items() if (cov < MIN_COVERAGE and col not in ALWAYS_KEEP)]
if to_drop:
    print(f"Dropping {len(to_drop)} features with <{int(MIN_COVERAGE*100)}% non-null: {to_drop}")
    features = [f for f in features if f not in to_drop]

# Manual blacklist drop
if BLACKLIST:
    print(f"Dropping blacklisted features: {BLACKLIST}")
    features = [f for f in features if f not in BLACKLIST]

print(f"\n[INFO] Using {len(features)} features: {features}")
print("\nFeature null rates (0=good, 1=all missing):")
print(data[features].isnull().mean().sort_values())

print(data[["ticker", "Date", "news_count", "sentiment_mean", "sentiment_median", "sentiment_max", "sentiment_min"]].tail(20))

# --- ENSURE NUMERIC DATA FOR ALL FEATURES (required by xgboost/lgbm/sklearn) ---
data[features] = data[features].apply(pd.to_numeric, errors="coerce")

# --- 11. RUN MODEL TRAINING ---
run_model_training(data, features, label="future_up_3d", tied_thresh=cull_ratio)
