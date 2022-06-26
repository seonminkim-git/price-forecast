"""
SK C&C 데이터분석1팀 김선민 작성
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mrmr import mrmr_regression

# plt.rc('font', size=20)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결
pd.set_option('display.max_rows', 500)


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


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


def MAPE(y_true, y_hat):
    return np.mean(np.abs((y_hat - y_true) / y_true)) * 100


def RMSE(y_true, y_hat):
    return np.sqrt(np.mean((y_hat - y_true)**2))


""" Preprocessing """


def func_pca(fv, scaler=None, pca=None):
    """
    Standard scaling & PCA (covering 80% of the variance)
    :param fv: do not contain any nans
    :param scaler: None if it's training phase, scaler variable if it's test phase
    :param pca: None if it's training phase, pca variable if it's test phase
    :return: transformed feature vector, scaler, pca
    """
    # Scaling
    if scaler is None:
        scaler = StandardScaler()
        fv_scaled = scaler.fit_transform(fv)
    else:
        fv_scaled = scaler.transform(fv)

    # PCA
    if pca is None:
        # tmp = PCA(n_components='mle')
        tmp = PCA(n_components=min(fv.shape))
        tmp.fit(fv_scaled)
        n_pca = np.where(np.cumsum(tmp.explained_variance_ratio_) > 0.8)[0][0] + 1  # 분산 coverage > 80% 되는 주성분 개수
        pca = PCA(n_components=n_pca)
        fv_pca = pca.fit_transform(fv_scaled)
    else:
        fv_pca = pca.transform(fv_scaled)

    return pd.DataFrame(fv_pca, columns=['pca'+str(p) for p in range(fv_pca.shape[1])]), scaler, pca


def func_xlag(fv, bestlag):
    fv_xlag = pd.DataFrame(columns=fv.columns)
    for col in fv.columns:
        blc = bestlag[col]
        lag = 0 if (blc >= 0) or (blc < -6) else int(blc)  # target인 철근의 lag
        fv_xlag[col] = fv[col].shift(-lag)
    return fv_xlag


def func_mrmr(X, y, n=10):
    if X.shape[1] < n:
        n = X.shape[1]
    selected_features = mrmr_regression(X, y, K=X.shape[1], return_scores=False)
    return X[selected_features[:n]]


""" Plots """


def plot_pred(y_true, y_hat, ax=None, x=None, titlestr='', savefilename=None, close=False):
    if x is None:
        x = range(len(y_true))
    mape = MAPE(y_true, y_hat)
    rmse = RMSE(y_true, y_hat)

    if ax is None:
        _, ax = plt.subplots()
    plt.plot(x, y_hat, 'r', label='prediction')
    plt.plot(x, y_true, 'b', label='true')
    plt.text(0.05, 0.7, f"MAPE: {np.round(mape, 2)}\nRMSE: {np.round(rmse, 2)}", transform=ax.transAxes, fontsize='large')
    plt.title(titlestr)
    plt.legend()
    if savefilename is not None:
        plt.savefig(savefilename)
    if close:
        plt.close()

    return ax


def plot_two_lines(y1, y2, ax=None, x=None, xlabel='date', ylabel=None):
    if x is None:
        x = range(len(y1))
    ylab1 = 'y1' if ylabel is None else ylabel[0]
    ylab2 = 'y2' if ylabel is None else ylabel[1]

    if ax is None:
        _, ax = plt.subplots()
    ax1 = ax
    ax2 = ax1.twinx()
    l1, = ax1.plot(x, y1, color='b', label=ylab1)
    l2, = ax2.plot(x, y2, color='r', label=ylab2)
    ax1.legend([l1, l2], [l.get_label() for l in [l1, l2]])
    ax1.set_ylabel(ylab1)
    ax1.set_xlabel(xlabel)
    ax2.set_ylabel(ylab2)

    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    ax2.spines.visible = False
    ax2.spines["right"].set_visible(True)
    ax1.yaxis.label.set_color(l1.get_color())
    ax2.yaxis.label.set_color(l2.get_color())
    ax1.spines["left"].set_edgecolor(l1.get_color())
    ax2.spines["right"].set_edgecolor(l2.get_color())
    ax1.tick_params(axis='y', colors=l1.get_color())
    ax2.tick_params(axis='y', colors=l2.get_color())


def plot_scatter(x, y, ax=None, xlabel='', ylabel='', text=''):
    if ax is None:
        _, ax = plt.subplots()
    validx = np.isfinite(x) & np.isfinite(y)
    slp, itc = np.polyfit(x[validx], y[validx], 1)

    ax.scatter(x, y, color='k')
    ax.plot(x, x * slp + itc, 'k')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.text(0.05, 0.5, text + '\ny = %.2f x + %.1f' % (slp, itc), transform=ax.transAxes, fontsize='x-large')


def plot_xce(x, y, timelag, ax=None, text=''):
    xce = [crosscorr(x, y, lag) for lag in timelag]
    peak = timelag[np.argmax(xce)]

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(timelag, xce, 'o-', label='Correlation coefficient')
    ax.axvline(0, color='k', linewidth=0.3)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.axvline(peak, color='r', linestyle='--', label='Peak synchrony')
    ax.set(xlabel='Time lag', ylabel='Pearson r', ylim=[-0.2, 1])
    ax.legend(loc='lower right')
    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize='x-large')

