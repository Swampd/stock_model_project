# python3 fundamental_scraper.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import time
from datetime import datetime
from config import scrape_delay_global, scrape_delay_FINNHUB, scrape_delay_finVIZ, FINNHUB_APIKEY, TIINGO_APIKEY, timeout_time
import traceback

# | Scraper                    | Sleep Duration      | Unit           | Where                    |
# | -------------------------- | ------------------- | -------------- | ------------------------ |
# | Nasdaq Earnings/Div/Splits | 1 sec (`config.py`) | **per day**    | Main pipeline            |
# | Finviz Upgrades            | 10 sec (hardcoded)  | **per ticker** | fundamental_scraper.py   |
# | Short Interest             | 3 sec (hardcoded)   | per ticker     | fundamental_scraper.py   |
# | Insider Transactions       | 3 sec (hardcoded)   | per ticker     | fundamental_scraper.py   |
# | Institutional Holdings     | 3 sec (hardcoded)   | per ticker     | fundamental_scraper.py   |
# | Yahoo News                 | 2 sec (hardcoded)   | per ticker     | fundamental_scraper.py   |
# | Finnhub News               | 1.5 sec (hardcoded) | per ticker     | fundamental_scraper.py   |


def get_yahoo_pe_ratio(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/key-statistics?p={ticker}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"{ticker}: Status {resp.status_code} | {resp.url}")
    if resp.status_code != 200:
        print(resp.text[:300])
    soup = BeautifulSoup(resp.text, "html.parser")
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) == 2 and "Trailing P/E" in cells[0].text:
            try:
                value = float(cells[1].text.replace(',', '').strip())
                return value
            except:
                return None
    return None

def get_pe_ratios_for_tickers(tickers):
    records = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            pe = info.get("trailingPE", None)
        except Exception as e:
            pe = None
        records.append({"ticker": ticker, "pe_ratio": pe})
        print(f"{ticker}: PE = {pe}")
    return pd.DataFrame(records)

def get_yahoo_eps_surprise(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/earnings?p={ticker}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"{ticker}: Status {resp.status_code} | {resp.url}")
    if resp.status_code != 200:
        print(resp.text[:300])
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find('table', attrs={'data-test': 'earnings-history'})
    records = []
    if table:
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) >= 5:
                try:
                    date = cells[0].text.strip()
                    eps_est = float(cells[1].text.replace(',', '').strip())
                    eps_actual = float(cells[2].text.replace(',', '').strip())
                    surprise = eps_actual - eps_est
                    records.append({
                        'ticker': ticker,
                        'earnings_date': date,
                        'eps_surprise': surprise
                    })
                except Exception:
                    continue
    return pd.DataFrame(records)

def get_eps_surprises_for_tickers(tickers):
    all_recs = []
    for ticker in tickers:
        df = get_yahoo_eps_surprise(ticker)
        all_recs.append(df)
    if all_recs:
        result = pd.concat(all_recs, ignore_index=True)
        return result
    else:
        return pd.DataFrame(columns=["ticker", "earnings_date", "eps_surprise"])

def get_finviz_upgrades(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"{ticker}: Status {resp.status_code} | {resp.url}")
    if resp.status_code != 200:
        print(resp.text[:300])
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find('table', class_='fullview-ratings-outer')
    records = []
    if table:
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) >= 5:
                try:
                    date = cells[0].text.strip()
                    action = cells[1].text.strip().lower()
                    is_upgrade = int('upgrade' in action)
                    is_downgrade = int('downgrade' in action)
                    records.append({
                        'ticker': ticker,
                        'upgrade_flag': is_upgrade,
                        'downgrade_flag': is_downgrade,
                        'upgrade_date': date
                    })
                except Exception:
                    continue
    return pd.DataFrame(records)

def get_finviz_upgrades_for_tickers(tickers):
    all_recs = []
    for ticker in tickers:
        df = get_finviz_upgrades(ticker)
        all_recs.append(df)
        time.sleep(scrape_delay_finVIZ)  # <-- SLEEP 10 SECONDS BETWEEN REQUESTS
    if all_recs:
        result = pd.concat(all_recs, ignore_index=True)
        return result
    else:
        return pd.DataFrame(columns=["ticker", "upgrade_flag", "downgrade_flag", "upgrade_date"])

def get_nasdaq_short_interest(ticker):
    url = f"https://www.nasdaq.com/market-activity/stocks/{ticker}/short-interest"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"{ticker}: Status {resp.status_code} | {resp.url}")
    time.sleep(scrape_delay_global)
    if resp.status_code != 200:
        print(resp.text[:300])
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    try:
        table = soup.find('table', {'class': 'market-calendar-table__table'})
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2 and 'Short Interest' in cells[0].text:
                si = cells[1].text.replace(',', '').strip()
                return float(si)
    except Exception as e:
        print(f"Parse error for {ticker}: {e}")
        return None

def get_short_interest_for_tickers(tickers):
    results = []
    for ticker in tickers:
        si = get_nasdaq_short_interest(ticker)
        results.append({'ticker': ticker, 'short_interest': si})
    return pd.DataFrame(results)

def get_openinsider_transactions(ticker):
    url = f"https://openinsider.com/screener?s={ticker}"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        print(f"{ticker}: Status {resp.status_code} | {resp.url}")
        time.sleep(scrape_delay_global)
        if resp.status_code != 200:
            print(f"{ticker}: OpenInsider status {resp.status_code}")
            return 0, 0
        soup = BeautifulSoup(resp.text, "html.parser")
        buys, sells = 0, 0
        table = soup.find('table', {'id': 'screener-table'})
        if table:
            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if len(cells) >= 6:
                    trans_type = cells[5].text.strip().lower()
                    if "buy" in trans_type:
                        buys += 1
                    elif "sell" in trans_type:
                        sells += 1
        return buys, sells
    except Exception as e:
        print(f"OpenInsider scrape failed for {ticker}: {e}")
        return 0, 0

def get_insider_transactions_for_tickers(tickers):
    results = []
    for ticker in tickers:
        buys, sells = get_openinsider_transactions(ticker)
        results.append({
            'ticker': ticker,
            'insider_buy_count': buys,
            'insider_sell_count': sells
        })
    return pd.DataFrame(results)

def get_nasdaq_inst_holdings(ticker):
    url = f"https://www.nasdaq.com/market-activity/stocks/{ticker}/institutional-holdings"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"{ticker}: Status {resp.status_code} | {resp.url}")
    time.sleep(scrape_delay_global)
    if resp.status_code != 200:
        print(resp.text[:300])
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    try:
        pct = None
        txt = soup.get_text()
        for line in txt.splitlines():
            if "Institutional Ownership" in line:
                pct_str = ''.join([x for x in line if x.isdigit() or x == '.'])
                if pct_str:
                    pct = float(pct_str)
                    break
        return pct
    except Exception as e:
        print(f"Parse error for {ticker}: {e}")
        return None

def get_inst_holdings_for_tickers(tickers):
    results = []
    for ticker in tickers:
        pct = get_nasdaq_inst_holdings(ticker)
        results.append({'ticker': ticker, 'inst_ownership_pct': pct})
    return pd.DataFrame(results)

def get_yahoo_news_count(ticker, start_date, end_date):
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    time.sleep(scrape_delay_global)
    if resp.status_code != 200:
        print(f"{ticker} news: Status {resp.status_code}")
        return pd.DataFrame()
    soup = BeautifulSoup(resp.text, "html.parser")
    articles = soup.find_all('li', attrs={'data-test-loc': 'Mega'})
    news_data = []
    for article in articles:
        try:
            date_tag = article.find('span')
            headline = article.find('h3').text.strip()
            date = datetime.now().strftime('%Y-%m-%d')
            news_data.append({"ticker": ticker, "Date": date, "headline": headline})
        except Exception:
            continue
    return pd.DataFrame(news_data)

def get_news_count_for_tickers(tickers, start_date, end_date):
    frames = []
    for ticker in tickers:
        df = get_yahoo_news_count(ticker, start_date, end_date)
        frames.append(df)
    all_news = pd.concat(frames, ignore_index=True)
    if not all_news.empty:
        news_counts = all_news.groupby(['ticker', 'Date']).size().reset_index(name='news_count')
    else:
        news_counts = pd.DataFrame(columns=['ticker', 'Date', 'news_count'])
    return news_counts

def get_finnhub_news_count(ticker, start_date, end_date, FINNHUB_APIKEY):
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_APIKEY}"
    resp = requests.get(url)
    time.sleep(scrape_delay_FINNHUB)
    if resp.status_code != 200:
        print(f"{ticker} news: Status {resp.status_code}")
        return pd.DataFrame()
    articles = resp.json()
    df = pd.DataFrame(articles)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "Date", "news_count"])
    df['Date'] = pd.to_datetime(df['datetime'], unit='s').dt.strftime('%Y-%m-%d')
    counts = df.groupby('Date').size().reset_index(name='news_count')
    counts['ticker'] = ticker
    return counts[['ticker', 'Date', 'news_count']]

def get_news_count_for_tickers_finnhub(tickers, start_date, end_date, FINNHUB_APIKEY):
    frames = []
    for ticker in tickers:
        try:
            frame = get_finnhub_news_count(ticker, start_date, end_date, FINNHUB_APIKEY)
            frames.append(frame)
        except Exception as e:
            print(f"News scrape failed for {ticker}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "Date", "news_count"])

def get_finnhub_news_sentiment(ticker, FINNHUB_APIKEY):
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={FINNHUB_APIKEY}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"{ticker} sentiment: Status {resp.status_code}")
        return None
    data = resp.json()
    if not data or "buzz" not in data:
        return None
    sentiment = data.get("sentiment", {})
    out = {
        "ticker": ticker,
        "buzz": data.get("buzz", {}).get("articlesInLastWeek", None),
        "sentiment_bullish": sentiment.get("bullishPercent", None),
        "sentiment_bearish": sentiment.get("bearishPercent", None),
        "sentiment_score": sentiment.get("companyNewsScore", None),
    }
    if "companyNewsScore" not in sentiment:
        print(f"[WARN] No sentiment score for {ticker}")
    return out

def get_news_sentiment_for_tickers_finnhub(tickers, FINNHUB_APIKEY):
    rows = []
    for ticker in tickers:
        sent = get_finnhub_news_sentiment(ticker, FINNHUB_APIKEY)
        if sent:
            rows.append(sent)
    return pd.DataFrame(rows)

def get_finnhub_earnings_calendar(tickers, start_date, end_date, FINNHUB_APIKEY):
    all_data = []
    for ticker in tickers:
        url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_APIKEY}"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Finnhub earnings failed for {ticker}: {resp.status_code}")
            continue
        rows = resp.json().get('earningsCalendar', [])
        for row in rows:
            all_data.append({
                "ticker": ticker,
                "Earnings Date": row.get("date", ""),
                "epsEstimate": row.get("epsEstimate"),
                "epsActual": row.get("epsActual"),
                "revenueEstimate": row.get("revenueEstimate"),
                "revenueActual": row.get("revenueActual")
            })
        time.sleep(0.1)
    return pd.DataFrame(all_data)

def get_finnhub_dividends(tickers, start_date, end_date, FINNHUB_APIKEY):
    all_data = []
    for ticker in tickers:
        url = f"https://finnhub.io/api/v1/stock/dividend?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_APIKEY}"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Finnhub dividends failed for {ticker}: {resp.status_code}")
            continue
        for row in resp.json():
            all_data.append({
                "ticker": ticker,
                "Dividend Date": row.get("paymentDate", row.get("date", "")),
                "amount": row.get("amount")
            })
        time.sleep(0.1)
    return pd.DataFrame(all_data)

def get_finnhub_splits(tickers, start_date, end_date, FINNHUB_APIKEY):
    all_data = []
    for ticker in tickers:
        url = f"https://finnhub.io/api/v1/stock/split?symbol={ticker}&from={start_date}&to={end_date}&token={FINNHUB_APIKEY}"
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Finnhub splits failed for {ticker}: {resp.status_code}")
            continue
        for row in resp.json():
            all_data.append({
                "ticker": ticker,
                "Split Date": row.get("date", ""),
                "fromFactor": row.get("fromFactor"),
                "toFactor": row.get("toFactor")
            })
        time.sleep(0.1)
    return pd.DataFrame(all_data)

# --- TIINGO NEWS SCRAPER ---
def get_tiingo_news_for_ticker(ticker, start_date, end_date, api_key, sleep_sec=0.5):
    """
    Fetch Tiingo news for a ticker and date range.
    Returns DataFrame: [ticker, Date, news_count, sentiment_mean, sentiment_median, sentiment_max, sentiment_min]
    """
    url = "https://api.tiingo.com/tiingo/news"
    headers = {"Authorization": f"Token {api_key}"}
    params = {
        "tickers": ticker,
        "startDate": start_date,
        "endDate": end_date,
        "limit": 1000,
        "sortBy": "publishedDate"
    }
    all_articles = []
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=timeout_time)
        if "application/json" not in resp.headers.get("Content-Type", ""):
            print(f"[TIINGO] Non-JSON response for {ticker}! Dumping first 500 chars:")
            print(resp.text[:500])
            break

        if resp.status_code != 200:
            print(f"[TIINGO] {ticker} {params['startDate']} - {params['endDate']}: {resp.status_code}")
            break
        articles = resp.json()
        if not articles:
            break
        all_articles.extend(articles)
        if len(articles) < params["limit"]:
            break
        last_date = articles[-1]['publishedDate']
        params["startDate"] = last_date[:10]
        time.sleep(sleep_sec)
    if not all_articles:
        return pd.DataFrame(columns=["ticker", "Date", "news_count", "sentiment_mean", "sentiment_median", "sentiment_max", "sentiment_min"])
    df = pd.DataFrame(all_articles)
    df['Date'] = pd.to_datetime(df['publishedDate'], format='ISO8601', utc=True, errors='coerce').dt.date

    if 'sentimentScore' in df.columns:
        df['sentiment'] = df['sentimentScore']
    else:
        df['sentiment'] = pd.NA

    agg = df.groupby('Date').agg(
        news_count=('sentiment', 'count'),
        sentiment_mean=('sentiment', 'mean'),
        sentiment_median=('sentiment', 'median'),
        sentiment_max=('sentiment', 'max'),
        sentiment_min=('sentiment', 'min'),
    ).reset_index()
    agg['ticker'] = ticker
    return agg[['ticker', 'Date', 'news_count', 'sentiment_mean', 'sentiment_median', 'sentiment_max', 'sentiment_min']]

def get_tiingo_news_for_tickers(tickers, start_date, end_date, api_key):
    frames = []
    for ticker in tickers:
        try:
            df = get_tiingo_news_for_ticker(ticker, start_date, end_date, api_key)
            frames.append(df)
        except Exception as e:
            print(f"Tiingo news failed for {ticker}: {e}")
            traceback.print_exc()
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out['Date'] = pd.to_datetime(out['Date'])
        return out
    else:
        return pd.DataFrame(columns=["ticker", "Date", "news_count", "sentiment_mean", "sentiment_median", "sentiment_max", "sentiment_min"])

