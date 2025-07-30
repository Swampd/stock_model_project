# python3 merge_tiingo_news.py

import glob
import pandas as pd

# Find all timestamped news files (in current dir)
files = glob.glob("tiingo_news_*.csv")

# Load and combine
df_list = [pd.read_csv(f) for f in files]
all_news = pd.concat(df_list, ignore_index=True)

# Drop duplicates - pick your columns (edit as needed!)
all_news = all_news.drop_duplicates(subset=["ticker", "Date", "news_count", "sentiment_mean"], keep='last')

# Save merged file
all_news.to_csv("tiingo_news_merged.csv", index=False)
print(f"[NEWS] Merged {len(files)} files, {len(all_news)} unique rows saved to tiingo_news_merged.csv")
