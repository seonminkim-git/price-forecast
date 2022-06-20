"""
SK C&C 데이터분석1팀 김선민 작성
"""

from os.path import exists
import pickle
import datetime
import numpy as np
import pandas as pd
from fbprophet import Prophet
from pmdarima.arima import ndiffs
import pmdarima as pm
from func import MAPE, RMSE, suppress_stdout_stderr, func_pca, func_xlag, plot_pred
import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결

import warnings
warnings.simplefilter("ignore")


do_Prophet = False
do_ARIMAX = True

opt = ''
do_PCA = True
do_mrmr = False
do_xlag = True
if do_PCA:
    opt += '_pca'
if do_xlag:
    opt += '_xlag'

train_ival = [3, 12, 36, 60, 120]  # (train_ival) months 데이터로 학습
test_ival = [1, 3, 6, 12]  # (test_ival) month 후 값을 예측


with open('../data/data-raw,corr.pkl', 'rb') as file:
    y, x, corrcoef = pickle.load(file)

y.index = y.index.to_timestamp()
x.index = x.index.to_timestamp()
bestlag = corrcoef['best lag']

idx2 = [a1 or a2 or a3 for a1, a2, a3 in zip(['철강원자재 가격' in c for c in x.columns], ['철강생산량-철근' in c for c in x.columns], ['품목별 수출액, 수입액-철및강-수입액' in c for c in x.columns])]
idx3 = [a1 or a2 or a3 or a4 for a1, a2, a3, a4 in zip(idx2, ['철강생산량' in c for c in x.columns], ['원유 가격' in c for c in x.columns], ['원달러 환율' in c for c in x.columns])]
idx5 = idx3 or []
feature_set = [
    # ['ARIMA', '단변량', None],
    ['ARIMAX', '공급_small', x[x.columns[idx2]]],  # y.to_frame().join(x)
    ['ARIMAX', '공급_large', x[x.columns[idx3]]],
    # ['ARIMAX', '수요+공급', x[x.columns[idx5]]],
    ['ARIMAX', '전부 다', x]
]

test_idx = np.where(y.index >= datetime.datetime(2021, 1, 1))[0]  # 2021년 1월 ~ 12월 12 points를 테스트 데이터로 사용
result_file = '../result/02. modelling/performance.csv'
if not exists(result_file):
    result = pd.DataFrame(columns=['algo', 'input', '학습기간', '예측시점', 'opt', 'MAPE', 'RMSE'] + ['m+'+str(m+1) for m in range(12)])
    cnt = -1
else:
    result = pd.read_csv(result_file)
    cnt = result.shape[0]-1

# plot_acf(y)
# plot_pacf(y)
# plt.show()

# plot_acf(np.diff(y))
# plot_pacf(np.diff(y))
# plt.show()

""" Prophet """
if do_Prophet:
    algo = 'Prophet'
    ip = '단변량'
    for trival in train_ival:
        for tsival in test_ival:
            cnt += 1
            yhat = []
            with suppress_stdout_stderr():  # suppress prophet output
                for i in test_idx:
                    train_idx = list(range(i-trival-tsival+1, i-tsival+1))

                    y_train = pd.DataFrame([y.index[train_idx], y.values[train_idx]], index=['ds', 'y']).T
                    y_test = pd.DataFrame([y.index[i], y.values[i]], index=['ds', 'y']).T

                    model = Prophet()
                    model.fit(y_train)
                    forecast = model.predict(y_test)
                    # model.plot(forecast)
                    # model.plot_components(forecast)
                    yhat += [forecast.yhat.values[0]]

            mape = MAPE(y.values[test_idx], yhat)
            rmse = RMSE(y.values[test_idx], yhat)
            result.loc[cnt] = np.array([algo, ip, trival, tsival, opt[1:], mape, rmse] + yhat)

            print(f'\n{algo}, {ip}, 학습기간 {trival}개월 ({trival/12}년), 예측시점 {tsival}개월 후')
            print('MAPE:', mape)
            print('RMSE:', rmse)

            fig, ax = plt.subplots()
            ax.plot(y, 'k')
            plot_pred(y.values[test_idx], yhat, x=y.index[test_idx], ax=ax, close=True,
                      titlestr=f'{algo}, {ip}\n학습기간 {trival}개월 ({trival/12}년), 예측시점 {tsival}개월 후',
                      savefilename=f'../result/02. modelling/{algo}_{ip}_학습기간{trival}M_예측시점{tsival}M후{opt}.png')

    # plt.draw_all()
    # plt.show()


""" ARIMAX """
if do_ARIMAX:
    for algo, ip, fv in feature_set:
        for trival in np.array(train_ival)[[x > 3 for x in train_ival]]:
            for tsival in test_ival:
                cnt += 1
                yhat = []
                for i in test_idx:
                    train_idx = list(range(i - trival - tsival + 1, i - tsival + 1))

                    y_train = y.values[train_idx]
                    y_test = y.values[i]

                    ''' 모델 fitting '''
                    kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=2)
                    adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=2)
                    n_diffs = max(adf_diffs, kpss_diffs)  # 추정된 차수

                    if fv is not None:  # exogenous feature(X)만
                        fv2 = fv.iloc[train_idx]
                        if do_xlag:
                            fv2 = func_xlag(fv2, bestlag)
                            rmidx = max([fv2[c].isnull().tolist().index(False) for c in fv2.columns])
                            fv2 = fv2.iloc[rmidx:]
                            y_train = y_train[rmidx:]

                    if fv is None or not any(np.where(fv.iloc[train_idx].isnull())[0]):
                        if fv is not None:  # exogenous feature(X)만
                            if do_PCA:
                                fv2, _, _ = func_pca(fv2)

                        model = pm.auto_arima(y=y_train,  # 데이터
                                              exogenous=None if fv is None else fv2,
                                              d=n_diffs,  # 차분 차수
                                              start_p=0,
                                              max_p=3,
                                              start_q=0,
                                              max_q=3,
                                              m=12 if trival > 12 else 1,  # The period for seasonal differencing
                                              # m=4이면 분기별, m=12면 월별, m=1이면 계절적 특징을 띠지 않는 데이터 (m=1 -> sesasonal=False)
                                              seasonal=True,
                                              stepwise=True,
                                              # 최적의 모수를 찾기 위해 쓰는 힌드만-칸다카르 알고리즘을 사용할지의 여부, False면 모든 모수 조합으로 모형을 적합
                                              trace=False  # stepwise로 모델을 적합할 때마다 결과를 프린트하고 싶은지
                                              )
                        model.fit(y_train)
                        # print(model.summary())
                        # print(model.pvalues())

                        ''' 잔차 검정 '''
                        # print(model.summary())
                        # # 1) Ljung-Box (Q): 잔차가 백색잡음인지 검정 (H0 = "잔차(residual)가 백색잡음(white noise) 시계열을 따른다")
                        # # 2) Jarque-Bera (JB): 잔차가 정규성을 띠는지 검정
                        # # 3) Heteroskedasticity (H): 잔차가 이분산을 띠지 않는지 검정
                        # # 4) (경험적으로) 잔차가 정규분포를 따른다면 비대칭도 (Skew)는 0에 가까워야 하고, 첨도 (Kurtosis)는 3에 가까워야 함
                        # model.plot_diagnostics(figsize=(16, 8))
                        # plt.show()

                        ''' 모델 refresh & predict '''
                        yhat += [model.predict(n_periods=tsival)[-1]]  # 하나씩 예측

                    else:
                        test_idx = np.delete(test_idx, np.where(test_idx == i))

                mape = MAPE(y.values[test_idx], yhat)
                rmse = RMSE(y.values[test_idx], yhat)
                result.loc[cnt] = np.array([algo, ip, trival, tsival, opt[1:], mape, rmse] + yhat)

                print(f'\n{algo}, {ip}, 학습기간 {trival}개월 ({trival / 12}년), 예측시점 {tsival}개월 후')
                print('MAPE:', mape)
                print('RMSE:', rmse)

                fig, ax = plt.subplots()
                ax.plot(y, 'k')
                plot_pred(y.values[test_idx], yhat, x=y.index[test_idx], ax=ax, close=True,
                          titlestr=f'{algo}, {ip}\n학습기간 {trival}개월 ({trival/12}년), 예측시점 {tsival}개월 후',
                          savefilename=f'../result/02. modelling/{algo}_{ip}_학습기간{trival}M_예측시점{tsival}M후{opt}.png')

                result.to_csv(result_file, index=False, encoding='utf-8-sig')

result.to_csv(result_file, index=False, encoding='utf-8-sig')

