





Handling of “ALWAYS_KEEP” and “MIN_COVERAGE” (drop threshold)

Resolved: You’ve now set these correctly in your model_training.

403 errors and which endpoints are blocked

Resolved: Finnhub splits endpoint is blocked for your account.

Feature coverage, news_count, and other “dropped” features

Resolved: You know you can get more coverage by fixing scrapers or using another data provider.




df['atr_pct'] = df['atr'] / df['Close']
df['return_10d'] = df['Close'].pct_change(periods=10)
df['return_21d'] = df['Close'].pct_change(periods=21)
df['pct_up_days_10d'] = df['Close'].diff().gt(0).rolling(window=10).mean()
df['rel_strength'] = df['return_3d'] - df['sp500_return_3d']


PROBLEM FEATURE LIST
short_interest
insider_buy_count
insider_sell_count
inst_ownership_pct
news_count
is_dividend_week
days_to_dividend
is_split_week
days_to_split
is_earnings_week
days_to_earnings
upgrade_flag
downgrade_flag
eps_surprise
adx_slope_10d
adx_slope_5d
corr_sp500_10d
corr_sp500_20d
corr_sp500_60d
bb_width
donchian_width
high_252d
low_252d
vol_252d
sector_rel_return_3d
outperform_sector_today
sp500_return_3d
qqq_return_3d
vix_return_3d
