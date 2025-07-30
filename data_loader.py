# data_loader.py

import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

# ---------- Core Stock & ETF Loaders ----------

def get_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["ticker"] = ticker
    df = df.reset_index()
    return df[["Date", "Open", "High", "Low", "Close", "Volume", "ticker"]]

def load_all_stocks(tickers, start_date, end_date):
    frames = [get_stock_data(t, start_date, end_date) for t in tickers]
    data = pd.concat(frames, ignore_index=True)
    data.index.name = None
    return data

def download_etfs(unique_etfs, start_date, end_date):
    sector_etf_data = {}
    for etf in unique_etfs:
        df = yf.download(etf, start=start_date, end=end_date).reset_index()[["Date", "Close"]]
        df = df.rename(columns={"Close": f"{etf}_Close"})
        sector_etf_data[etf] = df
    return sector_etf_data

def merge_etf_prices(data, tickers, sector_etfs, sector_etf_data):
    for ticker in tickers:
        etf = sector_etfs.get(ticker)
        if etf:
            merge_df = sector_etf_data[etf]
            mask = data["ticker"] == ticker
            data_dates = data.loc[mask, ["Date"]].reset_index(drop=True)
            # Defensive against weird pandas objects
            merge_df.columns = merge_df.columns.get_level_values(0) if isinstance(merge_df.columns, pd.MultiIndex) else merge_df.columns
            data_dates.columns = data_dates.columns.get_level_values(0) if isinstance(data_dates.columns, pd.MultiIndex) else data_dates.columns
            merged = pd.merge(
                data_dates,
                merge_df,
                on="Date",
                how="left"
            )
            data.loc[mask, f"{etf}_Close"] = merged[f"{etf}_Close"].values
    return data

# ---------- Robust Nasdaq API Scrapers ----------

def _force_columns(df, columns):
    """Ensures that df has all columns, even if empty."""
    if df.empty:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df

#def scrape_nasdaq_earnings(tickers, start_date, end_date, sleep_sec=1.5):
#    """
#    Scrape earnings dates for each ticker from the Nasdaq earnings calendar API.
 #   Returns a DataFrame with columns: ['ticker', 'Earnings Date']
  #  """
   # all_results = []
#    headers = {
#        'User-Agent': 'Mozilla/5.0',
#        'Accept': 'application/json'
#    }
#    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
#    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
#    delta = timedelta(days=1)
#    tickers_set = set([t.upper() for t in tickers])
#    current = start_dt
#    while current <= end_dt:
#        date_str = current.strftime('%Y-%m-%d')
#        url = f"https://api.nasdaq.com/api/calendar/earnings?date={date_str}"
#        try:
#            resp = requests.get(url, headers=headers, timeout=10)
#            if resp.status_code != 200:
#                print(f"[WARN] {date_str}: HTTP {resp.status_code}")
#                time.sleep(sleep_sec)
#                current += delta
#                continue
#            data = resp.json()
#            rows = data.get('data', {}).get('rows', [])
#            for row in rows:
#                symbol = row.get('symbol', '').upper()
#                if symbol in tickers_set:
#                    all_results.append({'ticker': symbol, 'Earnings Date': current})
#        except Exception as e:
#v            print(f"Failed on {date_str}: {e}")
#        time.sleep(sleep_sec)   # Delay to avoid being blocked
#        current += delta
#    df = pd.DataFrame(all_results)
#    df = _force_columns(df, ["ticker", "Earnings Date"])
#    df = df.drop_duplicates().sort_values(['ticker', 'Earnings Date'])
#    print(f"Scraped {len(df)} earnings dates for {df['ticker'].nunique()} tickers.")
#    return df

#def scrape_nasdaq_dividends(tickers, start_date, end_date, sleep_sec=1.5):
 #   """
  #  Scrape dividend dates for each ticker from the Nasdaq API.
   # Returns DataFrame with columns: ['ticker', 'Dividend Date']
   # """
#    all_results = []
 #   headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
  #  start_dt = datetime.strptime(start_date, "%Y-%m-%d")
   # end_dt = datetime.strptime(end_date, "%Y-%m-%d")
#    delta = timedelta(days=1)
 #   tickers_set = set([t.upper() for t in tickers])
  #  current = start_dt
   # while current <= end_dt:
    #    date_str = current.strftime('%Y-%m-%d')
#        url = f"https://api.nasdaq.com/api/calendar/dividends?date={date_str}"
 #       try:
  #vv          resp = requests.get(url, headers=headers, timeout=10)
   #         rows = resp.json().get('data', {}).get('rows', [])
    #        for row in rows:
     #           symbol = row.get('symbol', '').upper()
      #          if symbol in tickers_set:
       #             all_results.append({'ticker': symbol, 'Dividend Date': current})
        #except Exception as e:
        #    print(f"Failed on {date_str}: {e}")
#        current += delta
 #       time.sleep(sleep_sec)
  #  df = pd.DataFrame(all_results)
   # df = _force_columns(df, ["ticker", "Dividend Date"])
    #df = df.drop_duplicates().sort_values(['ticker', 'Dividend Date'])
    #print(f"Scraped {len(df)} dividend dates for {df['ticker'].nunique()} tickers.")
    #return df

#def scrape_nasdaq_splits(tickers, start_date, end_date, sleep_sec=1.5):
 #   """
  #  Scrape split dates for each ticker from the Nasdaq API.
   # Returns DataFrame with columns: ['ticker', 'Split Date']
    #"""
#    all_results = []
 #   headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
  #  start_dt = datetime.strptime(start_date, "%Y-%m-%d")
   # end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    #delta = timedelta(days=1)
#    tickers_set = set([t.upper() for t in tickers])
 #   current = start_dt
  #  while current <= end_dt:
   #     date_str = current.strftime('%Y-%m-%d')
    #    url = f"https://api.nasdaq.com/api/calendar/splits?date={date_str}"
     #   try:
      #      resp = requests.get(url, headers=headers, timeout=10)
       #     rows = resp.json().get('data', {}).get('rows', [])
        #    for row in rows:
         #       symbol = row.get('symbol', '').upper()
          #      if symbol in tickers_set:
           #         all_results.append({'ticker': symbol, 'Split Date': current})
#        except Exception as e:
 #           print(f"Failed on {date_str}: {e}")
  #      current += delta
   #     time.sleep(sleep_sec)
    #df = pd.DataFrame(all_results)
#    df = _force_columns(df, ["ticker", "Split Date"])
 #   df = df.drop_duplicates().sort_values(['ticker', 'Split Date'])
  #  print(f"Scraped {len(df)} split dates for {df['ticker'].nunique()} tickers.")
   # return df
