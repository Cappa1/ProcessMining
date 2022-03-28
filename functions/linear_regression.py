import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn import linear_model


def one_hot_encode(df, column, prefix):
    """pandas one hot encoder

    Args:
        df (pandas dataframe): input dataframe
        column (str): column to be encoded
        prefix (str): prefix for encoded column names

    Returns:
        pd.DataFrame: one hot encoded dataframe
    """
    df_dummies_name = pd.get_dummies(
        df[column], prefix=prefix, drop_first=True)
    df.drop(column, axis=1, inplace=True)
    return df.join(df_dummies_name)


def encode(df):
    """One hot encodes columns"""
    # df = one_hot_encode(df, 'event org:resource', '')
    df = one_hot_encode(df, 'event Action', 'action')
    df = one_hot_encode(df, 'case LoanGoal', 'loangoal')
    df = one_hot_encode(df, 'case ApplicationType', 'appl_type')
    df = one_hot_encode(df, 'event concept:name', 'eventname')
    df = one_hot_encode(df, 'event EventOrigin', 'origin')
    df = one_hot_encode(df, 'event lifecycle:transition', 'lifecycle')
    return df


def time_diff(df):
    """Calculates time difference between i and i+1 
    within a trace and converts the value to log"""
    df['time_diff'] = (
        df['nextTime'] - df['event time:timestamp']).dt.total_seconds()
    df = df.dropna().reset_index(drop=True)

    df = df[df['time_diff'] >= 0]
    df['time_diff'] = np.log(df['time_diff'].replace(0, np.nan))
    df['time_diff'] = df['time_diff'].replace(np.nan, 0)

    return df.drop(['event time:timestamp', 'nextTime'], axis=1)


def cross_validate(X, Y):
    """Creates a timesseries split and calculates 
    cross validation error fitted on a given estimator

    Args:
        X (array): input array 
        Y (array): output array

    Returns:
        output, model (tuple): list of true y and predicted + model
    """
    output = []
    ts = TimeSeriesSplit(gap=0, max_train_size=None,
                         n_splits=5, test_size=None)

    folds = list(ts.split(X))
    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = linear_model.LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        output.append((y_test, y_pred))

    return output, model
