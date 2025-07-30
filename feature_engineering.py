# feature_engineering.py

import pandas_ta as pta
import ta
import numpy as np
from config import sector_etfs


def add_all_features(df):
    df["RSI"] = pta.rsi(df["Close"], length=14)
    macd = pta.macd(df["Close"])
    if macd is not None:
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_hist"] = macd["MACDh_12_26_9"]
    else:
        df["MACD"] = df["MACD_signal"] = df["MACD_hist"] = None
    df["daily_return"] = df["Close"].pct_change()
    df["max_close_5d"] = df["Close"].rolling(window=5).max()
    df["is_new_high_5d"] = (df["Close"] >= df["max_close_5d"]).astype(int)
    df["ma20"] = df["Close"].rolling(window=20).mean()
    df["above_ma20"] = (df["Close"] > df["ma20"]).astype(int)
    df["vol_avg20"] = df["Volume"].rolling(window=20).mean()
    df["volume_spike"] = (df["Volume"] > df["vol_avg20"] * 1.5).astype(int)
    df["return_3d"] = df["Close"].pct_change(periods=3)
    df["return_5d"] = df["Close"].pct_change(periods=5)
    df["volatility_5d"] = df["Close"].rolling(window=5).std()
    df["volatility_10d"] = df["Close"].rolling(window=10).std()
    df["up_days_5d"] = df["Close"].diff().gt(0).rolling(window=5).sum()
    df["relative_vol_20d"] = df["Volume"] / df["Volume"].rolling(window=20).mean()
    df["high_252d"] = df["Close"].rolling(window=252, min_periods=1).max()
    df["low_252d"] = df["Close"].rolling(window=252, min_periods=1).min()
    df["dist_from_high"] = (df["Close"] - df["high_252d"]) / df["high_252d"]
    df["dist_from_low"] = (df["Close"] - df["low_252d"]) / df["low_252d"]
    df["bullish_engulf"] = (
        (df["Open"] < df["Close"]) &
        (df["Close"].shift(1) < df["Open"].shift(1)) &
        (df["Open"] < df["Close"].shift(1)) &
        (df["Close"] > df["Open"].shift(1))
    ).astype(int)
    adx = pta.adx(df['High'], df['Low'], df['Close'])
    df['adx'] = adx['ADX_14'] if adx is not None and 'ADX_14' in adx else None
    df['cci'] = pta.cci(df['High'], df['Low'], df['Close'])
    df['williams_r'] = pta.willr(df['High'], df['Low'], df['Close'])
    stoch = pta.stoch(df['High'], df['Low'], df['Close'])
    df['stoch_k'] = stoch['STOCHk_14_3_3'] if stoch is not None and 'STOCHk_14_3_3' in stoch else None
    df['stoch_d'] = stoch['STOCHd_14_3_3'] if stoch is not None and 'STOCHd_14_3_3' in stoch else None
    df['atr'] = pta.atr(df['High'], df['Low'], df['Close'])
    df['obv'] = pta.obv(df['Close'], df['Volume'])
    df['cmf'] = pta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
    pta.bbands(df["Close"], append=True)
    if all(col in df.columns for col in ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']):
        df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
    else:
        df['bb_width'] = None
    donch = pta.donchian(df['High'], df['Low'])
    df['donchian_width'] = donch['DONCHIAND_20'] - donch['DONCHIANU_20'] if donch is not None and 'DONCHIAND_20' in donch and 'DONCHIANU_20' in donch else None
    df['force_idx'] = pta.efi(df['Close'], df['Volume'])
    df['emv'] = ta.volume.EaseOfMovementIndicator(
        high=df['High'],
        low=df['Low'],
        volume=df['Volume'],
        window=14,
        fillna=True
    ).ease_of_movement()

        # === New advanced features ===

    # ATR% (normalized ATR)
    df['atr_pct'] = df['atr'] / df['Close']

    # 10-day and 21-day returns
    df['return_10d'] = df['Close'].pct_change(periods=10)
    df['return_21d'] = df['Close'].pct_change(periods=21)

    # Percent up days (trend quality)
    df['pct_up_days_10d'] = df['Close'].diff().gt(0).rolling(window=10).mean()

    
    # High-low range (5 days)
    df['high_low_range_5d'] = (df['High'].rolling(5).max() - df['Low'].rolling(5).min()) / df['Close']

    # Gap (overnight move)
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Intraday volatility
    df['intraday_volatility'] = (df['High'] - df['Low']) / df['Open']

    # Rolling correlation to SP500 over 10 days
    if 'SP500_Close' in df.columns:
        df['corr_sp500_10d'] = df['Close'].rolling(10).corr(df['SP500_Close'])
    else:
        df['corr_sp500_10d'] = None

    # Volume ratio vs 5-day average
    df['volume_ratio_5d'] = df['Volume'] / df['Volume'].rolling(5).mean()

    # Momentum (simple 10-day price move)
    df['momentum_10d'] = df['Close'] - df['Close'].shift(10)

    # Z-score of price over 10 days
    df['zscore_10d'] = (df['Close'] - df['Close'].rolling(10).mean()) / df['Close'].rolling(10).std()

    # === New 3-day Features ===

    # 1. 3d Return vs. 10d MA (mean reversion measure)
    df['return_3d_ma_diff'] = df['return_3d'] - df['Close'].pct_change(periods=10).rolling(window=10).mean()

    # 2. RSI Slope (change over last 3 days)
    df['rsi_slope_3d'] = df['RSI'].diff(periods=3)

    # 3. Volatility Ratio (3d / 10d)
    df['vol_ratio_3d_10d'] = df['Close'].rolling(3).std() / df['Close'].rolling(10).std()

    # 4. Inside Bar / Outside Bar (last day compared to prior day)
    df['inside_bar'] = ((df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))).astype(int)
    df['outside_bar'] = ((df['High'] > df['High'].shift(1)) & (df['Low'] < df['Low'].shift(1))).astype(int)

    # 5. 3-day up streak (3 closes up in a row)
    df['three_up_streak'] = (df['Close'].diff().gt(0).rolling(3).sum() == 3).astype(int)
    df['three_down_streak'] = (df['Close'].diff().lt(0).rolling(3).sum() == 3).astype(int)

    # 6. 3d Return vs. Sector ETF (if sector ETF is merged; example with XLK for tech stocks)
    # Requires sector ETF data to be merged in advance as 'XLK_Close'
    #if 'XLK_Close' in df.columns:
    #    df['xlk_return_3d'] = df['XLK_Close'].pct_change(periods=3)
    #    df['rel_strength_xlk_3d'] = df['return_3d'] - df['xlk_return_3d']
    #else:
    #    df['xlk_return_3d'] = None
    #    df['rel_strength_xlk_3d'] = None

    # 7. OBV Change over 3 days
    df['obv_change_3d'] = df['obv'].diff(periods=3)

    # 8. Volume surge last 3 days (any day in last 3 > 2x 20d avg)
    df['volume_surge_3d'] = (
        df['Volume'].rolling(3).max() > df['Volume'].rolling(20).mean() * 2
    ).astype(int)

    # 9. Consecutive gaps in last 3 days
    df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
    df['gap_up_streak_3d'] = df['gap_up'].rolling(3).sum()

    # 10. Day of Week (one-hot, if you want)
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday
    # Optionally: pd.get_dummies(df['day_of_week'], prefix='dow') and concat if you want one-hot

    # 11. 3-day High/Low Break
    df['break_3d_high'] = (df['Close'] > df['Close'].rolling(3).max().shift(1)).astype(int)
    df['break_3d_low'] = (df['Close'] < df['Close'].rolling(3).min().shift(1)).astype(int)

        # --- Sector-based features ---
    etf = sector_etfs.get(df['ticker'].iloc[0])
    if etf and f"{etf}_Close" in df.columns:
        etf_col = f"{etf}_Close"
        # Sector-relative return (3d)
        df[f'{etf}_return_3d'] = df[etf_col].pct_change(periods=3)
        df['sector_rel_return_3d'] = df['return_3d'] - df[f'{etf}_return_3d']
        # Sector ETF momentum (3d, 5d, 10d returns)
        for n in [3, 5, 10]:
            df[f'{etf}_return_{n}d'] = df[etf_col].pct_change(n)
        # Binary indicator: outperforming sector today
        df['stock_return_1d'] = df['Close'].pct_change(1)
        df[f'{etf}_return_1d'] = df[etf_col].pct_change(1)
        df['outperform_sector_today'] = (df['stock_return_1d'] > df[f'{etf}_return_1d']).astype(int)
    else:
        # If for some reason sector ETF column is missing, fill with NaN
        df['sector_rel_return_3d'] = np.nan
        df['outperform_sector_today'] = np.nan
        for n in [3, 5, 10]:
            df[f'{etf}_return_{n}d'] = np.nan

    # After you calculate df['adx'] (should be a float column, not all NaN)
    df['adx_slope_5d'] = df['adx'].diff(5)
    df['adx_slope_10d'] = df['adx'].diff(10)

    if 'SP500_Close' in df.columns:
        df['corr_sp500_20d'] = df['Close'].rolling(20).corr(df['SP500_Close'])
        df['corr_sp500_60d'] = df['Close'].rolling(60).corr(df['SP500_Close'])
    else:
        df['corr_sp500_20d'] = None
        df['corr_sp500_60d'] = None

    df['ma50'] = df['Close'].rolling(50).mean()
    df['trend_regime'] = (df['Close'] > df['ma50']).astype(int)  # 1 = uptrend, 0 = down/sideways

    df['vol_10d'] = df['Close'].rolling(10).std()
    df['vol_252d'] = df['Close'].rolling(252, min_periods=20).std()
    df['vol_regime'] = (df['vol_10d'] > df['vol_252d']).astype(int)  # 1 = high vol, 0 = normal/low vol

    # Gap up: today's open > yesterday's close by 1% (change 1.01 to your threshold)
    df['gap_up'] = (df['Open'] > df['Close'].shift(1) * 1.01).astype(int)
    df['num_gap_ups_10d'] = df['gap_up'].rolling(10).sum()

    # 20-day rolling count of days where today's close is the highest in 20 days
    df['is_new_high_20d'] = (df['Close'] >= df['Close'].rolling(20, min_periods=1).max()).astype(int)
    df['num_new_highs_20d'] = df['is_new_high_20d'].rolling(20).sum()

    # New low in last 20 days
    df['is_new_low_20d'] = (df['Close'] <= df['Close'].rolling(20, min_periods=1).min()).astype(int)
    df['num_new_lows_20d'] = df['is_new_low_20d'].rolling(20).sum()

    df['consecutive_up'] = df['Close'].diff().gt(0).groupby((df['Close'].diff().gt(0) != df['Close'].diff().gt(0).shift()).cumsum()).cumcount() + 1

    df['big_move'] = (df['daily_return'].abs() > 0.03).astype(int)
    df['num_big_moves_10d'] = df['big_move'].rolling(10).sum()

    return df
