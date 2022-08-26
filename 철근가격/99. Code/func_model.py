import os
import numpy as np
import pandas as pd
from fbprophet import Prophet as fbProphet
from pmdarima.arima import ndiffs
import pmdarima as pm
import xgboost as xgb
from sklearn.linear_model import ElasticNet as f_ElasticNet
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def Prophet(x_train, y_train, x_test, y_test):
    y_train = pd.DataFrame([y_train.index, y_train], index=['ds', 'y']).T
    y_test = pd.DataFrame([y_test.index, y_test], index=['ds', 'y']).T

    with suppress_stdout_stderr():
        model = fbProphet()
        model.fit(y_train)
        forecast = model.predict(y_test)
        yhat_test = forecast.yhat.values

    return model, yhat_test


def ARIMAX(x_train, y_train, x_test, y_test):
    tsival = int(np.round((y_test.index[0]-y_train.index[-1])/np.timedelta64(1, 'M')))
    y_train = y_train.values

    kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=2)
    adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=2)
    n_diffs = max(adf_diffs, kpss_diffs)  # 추정된 차수

    model = pm.auto_arima(y=y_train,  # 데이터
                          exogenous=None if x_train is None else x_train,
                          d=n_diffs,  # 차분 차수
                          start_p=0,
                          max_p=3,
                          start_q=0,
                          max_q=3,
                          m=12 if len(y_train) > 12 else 1,  # The period for seasonal differencing
                          # m=4이면 분기별, m=12면 월별, m=1이면 계절적 특징을 띠지 않는 데이터 (m=1 -> sesasonal=False)
                          seasonal=True,
                          stepwise=True,
                          # 최적의 모수를 찾기 위해 쓰는 힌드만-칸다카르 알고리즘을 사용할지의 여부, False면 모든 모수 조합으로 모형을 적합
                          trace=False  # stepwise로 모델을 적합할 때마다 결과를 프린트하고 싶은지
                          )
    model.fit(y_train)

    yhat_test = [model.predict(n_periods=tsival)[-1]]  # 한 포인트마다 predict
    return model, yhat_test


def XGB(x_train, y_train, x_test, y_test):
    model = xgb.XGBRegressor(random_state=1)
    model.fit(x_train, y_train)

    yhat_test = model.predict(x_test.to_frame().T.values)
    return model, yhat_test


def ElasticNet(x_train, y_train, x_test, y_test):
    elasticnet = f_ElasticNet()
    parameters = {'alpha': np.logspace(-4, 0, 200)}
    model = GridSearchCV(elasticnet, parameters, scoring='neg_mean_squared_error', cv=5)
    model.fit(x_train, y_train)
    yhat_test = model.predict(x_test.values)
    return model, yhat_test


def LGBM(x_train, y_train, x_test, y_test):
    model = LGBMRegressor()
    model.fit(x_train.values, y_train.values)
    yhat_test = model.predict(x_test.values)
    return model, yhat_test


def TabNet(x_train, y_train, x_test, y_test):
    x_train = x_train.values
    x_test = x_test.values if type(x_test).__name__ == 'DataFrame' else x_test.values.reshape(1, x_test.shape[0])  # Series
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    with suppress_stdout_stderr():
        model = TabNetRegressor(
            seed=42,
            verbose=False
        )

        model.fit(X_train=x_train, y_train=y_train,
                  eval_set=[(x_train, y_train), (x_test, y_test)],
                  eval_name=['train', 'test'],
                  eval_metric=['rmse'],
                  max_epochs=50, patience=20,
                  drop_last=False,
                  )

        yhat_test = [y[0] for y in model.predict(x_test)]

    return model, yhat_test


def DataRobot(x_train, y_train, x_test, y_test):
    """ Light GBM Regressor with GBDT with Boosting on Residuals """
    model, yhat_test = [], []
    return model, yhat_test


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

