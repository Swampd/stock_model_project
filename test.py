# python3 test.py
# python test.py
import requests
from config import TIINGO_APIKEY, timeout_time


import pandas as pd


headers = {"Authorization": f"Token {TIINGO_APIKEY}"}
url = "https://api.tiingo.com/tiingo/news"
params = {
    "tickers": "AAPL",
    "startDate": "2024-07-02",
    "endDate": "2024-07-04",  # Short window, should always have news
    "limit": 100,  # 100 is safe, you can reduce if you want
    "sortBy": "publishedDate"
}
resp = requests.get(url, headers=headers, params=params, timeout=timeout_time)
print("Requested:", url)
print("Params:", params)
print("Response code:", resp.status_code)
if "application/json" not in resp.headers.get("Content-Type", ""):
    print("Non-JSON response! Dumping first 500 chars:")
    print(resp.text[:500])
    exit(1)

print("Status:", resp.status_code)


if resp.status_code == 200:
    news = resp.json()
    if news:
        df = pd.DataFrame(news)
        df.to_csv("test_aapl_news.csv", index=False)
        print(f"Wrote {len(df)} articles to test_aapl_news.csv")
    else:
        print("No news found in this period.")
else:
    print("Failed to fetch news:", resp.text)
