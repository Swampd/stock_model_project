# event_flags.py

import pandas as pd

def add_event_flags(data, event_calendar, event_type='event'):
    """
    Adds week/event flags to your DataFrame.
    - data: your main DataFrame (must include Date, ticker columns)
    - event_calendar: DataFrame with columns ['ticker', '<Event> Date'] (datetime64)
    - event_type: string, e.g. 'earnings', 'dividend', 'split'
    """
    # Handle missing/empty event_calendar
    if event_calendar is None or event_calendar.empty:
        print(f"[WARN] {event_type.capitalize()} calendar is empty. Skipping flag creation.")
        return data  # Just return original data

    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'])

    event_col = f'{event_type.capitalize()} Date'
    if event_col not in event_calendar.columns:
        print(f"[WARN] Expected column '{event_col}' not found in calendar. Skipping {event_type} flag creation.")
        return data

    event_calendar[event_col] = pd.to_datetime(event_calendar[event_col])

    # For each row, check if in event week (Mon-Sun including event day)
    def is_event_week(row):
        edates = event_calendar[event_calendar['ticker'] == row['ticker']][event_col]
        return int(((edates >= row['Date']) & (edates <= row['Date'] + pd.Timedelta(days=6))).any())
    data[f'is_{event_type}_week'] = data.apply(is_event_week, axis=1)

    # Days to next event (min positive difference)
    def days_to_event(row):
        edates = event_calendar[event_calendar['ticker'] == row['ticker']][event_col]
        future_edates = edates[edates >= row['Date']]
        if future_edates.empty:
            return None
        return int((future_edates.iloc[0] - row['Date']).days)
    data[f'days_to_{event_type}'] = data.apply(days_to_event, axis=1)
    return data
