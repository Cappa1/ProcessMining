import datetime

def remove_ms(df):
    """Removes milliseconds"""
    df['event time:timestamp'] = df['event time:timestamp'].apply(lambda x: x.split('.')[0])
    return df


def f_memoize_dt(s):
    """
    Memoization technique to convert to datetime
    """
    dates = {date:datetime.datetime.strptime(date,"%d-%m-%Y %H:%M:%S") for date in s.unique()}
    return s.map(dates)