import numpy as np


def MAPE(y_true, y_hat):
    return np.mean(np.abs((y_hat - y_true) / y_true)) * 100


def RMSE(y_true, y_hat):
    return np.sqrt(np.mean((y_hat - y_true)**2))


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    # [0, 1, 2, 3, ...] ---(lag=2)--> [NaN, NaN, 0, 1, 2, ...]
    if wrap:  # shift 하면서 비는 위치에 NaN 값 대신 반대편 데이터로 채워 넣음
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))

