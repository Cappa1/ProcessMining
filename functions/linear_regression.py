import pandas as pd

def one_hot_encode(df, column, prefix):
    """pandas one hot encoder

    Args:
        df (pandas dataframe): input dataframe
        column (str): column to be encoded
        prefix (str): prefix for encoded column names

    Returns:
        pd.DataFrame: one hot encoded dataframe
    """
    df_dummies_name = pd.get_dummies(df[column], prefix=prefix, drop_first=True)
    df.drop(column, axis=1, inplace=True)
    return df.join(df_dummies_name)



def sliding_window(window_size, df):
    """transforms df_data into supervised form 
    with rolling window implementations

    Args:
        window_size (int): size of rolling window

    Returns:
        (X, Y): tuple of input and output arrays
    """

    windows = list(df.rolling(window=window_size))
    for i in windows[window_size-1:]:    
        # split into X and Y
        temp = i.to_numpy()
        temp = [item for sublist in temp for item in sublist]
        Y.append(temp.pop(-1))
        X.append(temp[1:])
    return None


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
    ts = TimeSeriesSplit(gap=175, max_train_size=None, n_splits=5, test_size=None)

    for train_index, test_index in ts.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        model = linear_model.LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        output.append((y_test, y_pred))
        
    return output, model

        
    

 