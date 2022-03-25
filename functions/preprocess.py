import datetime
import pandas as pd
import time

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



def UnixTime(df, col="event time:timestamp", newcol="Unix"):
    """Adds a new column to the dataframe containing the UNIX time of the timestamp"""
    cop = df.copy()
    unixTransform = lambda x: time.mktime(x.timetuple())
    df[newcol] = cop[col].apply(unixTransform)

def dropper(df, lbls=["eventID", "event EventID", "timestamp"]):
    df.drop(labels=lbls, axis=1, inplace=True)

    
def data_split(df):
    """returns 10% of the data"""
    return df[: int((len(df)/10))]