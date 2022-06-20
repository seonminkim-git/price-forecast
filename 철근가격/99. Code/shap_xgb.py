
'''

# Based on 2022/05/25 result file from google sheet '철근가격예측' as 'result_0525.csv'
# shap values in terms of lowest XGB's MAPE

algorithm : xgboost
사용 input : 공급_small
MAPE = 7.601586
'''

import os
import sys

import pickle
import datetime
from typing import List, Any

import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBRegressor
# from fbprophet import Prophet
# from pmdarima.arima import ndiffs
# import pmdarima as pm
# from func import MAPE, RMSE, suppress_stdout_stderr, func_pca, plot_results

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결

import shap
shap.initjs()

# Parameters
do_Prophet = False
do_ARIMAX = False
do_xgb = True

opt = ''
do_PCA = False
do_xlag = False
if do_PCA:
    opt += '_pca'
if do_xlag:
    opt += '_xlag'

train_ival = [3, 12, 36, 60, 120]  # (train_ival) months 데이터로 학습
test_ival = [1]#[1, 3, 6, 12]  # (test_ival) month 후 값을 예측

# ## Load Data
with open('./data-raw,corr.pkl', 'rb') as file:
    y, x, corrcoef = pickle.load(file)


y.index = y.index.to_timestamp()
x.index = x.index.to_timestamp()

test_idx = np.where(y.index >= datetime.datetime(2021, 1, 1))[0]  # 2021년 1월 ~ 12월 12 points를 테스트 데이터로 사용
result = pd.DataFrame(columns=['algo', 'input', '학습기간', '예측시점', 'opt', 'MAPE', 'RMSE'])
cnt = -1

idx2 = [a1 or a2 or a3 for a1, a2, a3 in zip(['철강원자재 가격' in c for c in x.columns], ['철강생산량-철근' in c for c in x.columns], ['품목별 수출액, 수입액-철및강-수입액' in c for c in x.columns])]
idx3 = [a1 or a2 or a3 or a4 for a1, a2, a3, a4 in zip(idx2, ['철강생산량' in c for c in x.columns], ['원유 가격' in c for c in x.columns], ['원달러 환율' in c for c in x.columns])]
idx5 = idx3 or []
feature_set = [
#     ['xgb', '단변량', None],
    ['xgb', '공급_small', x[x.columns[idx2]]],  # y.to_frame().join(x)
    ['xgb', '공급_large', x[x.columns[idx3]]],
#     ['xgb', '수요+공급', x[x.columns[idx5]]],
    ['xgb', '전부 다', x]
]

### based on result file
result = pd.read_csv('result_0525.csv')
result = pd.DataFrame(result)
result = result.iloc[1: , :]
result['테스트 성능 (2021년 1~12월 데이터)'] = result['테스트 성능 (2021년 1~12월 데이터)'].astype(float)
result = result.dropna()

min_algo = pd.DataFrame(result.groupby(['algorithm','사용 input'])['테스트 성능 (2021년 1~12월 데이터)'].min()).reset_index()
min_algo.rename(columns={'테스트 성능 (2021년 1~12월 데이터)': 'MAPE'}, inplace= True)
min_algo['MAPE'] = min_algo['MAPE'].astype(float)
min_algo = min_algo.loc[(min_algo['사용 input']!= '단변량')]
min_value = min_algo.loc[min_algo['MAPE']==min_algo['MAPE'].min(),:]

min_values = result.loc[result['테스트 성능 (2021년 1~12월 데이터)']==min_value['MAPE'].values[0]]
tr_min = [int(min_values['학습 기간'].values.astype(str)[0][0])]
t_min = [int(min_values['예측 시점'].values[0])]

feature_set_xgb_lowest = [
['xgb', '공급_small', x[x.columns[idx2]]],  # y.to_frame().join(x)
]

tmp_shap = pd.DataFrame()

for algo, ip, fv in feature_set_xgb_lowest:
    for trival in tr_min:
        for tsival in t_min:
            cnt += 1
            yhat = []
            for i in test_idx:
                train_idx = list(range(i - trival - tsival + 1, i - tsival + 1))
                #                     print(train_idx)
                #                     print(i)
                #                     print(fv.values[train_idx])
                #                     print(fv.values[i])

                x_train = fv.values[train_idx]
                x_test = fv.values[i]
                y_train = y.values[train_idx]
                y_test = y.values[i]

                x_train_ = x_train
                #                     y_train_ = y_train.reshape(-1, )
                y_train_ = y_train.reshape(-1, 1)
                x_test_ = x_test.reshape(1, x_test.shape[0])
                y_test_ = y_test.reshape(-1, 1)
                #                     y_test_ = y_test.reshape(1, )

                model = xgb.XGBRegressor(random_state=1)
                model.fit(x_train_, y_train_)

                yhat = model.predict(x_test_)

                # shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x_test_)
                shap_df = pd.DataFrame(shap_values)
                shap_df.columns = fv.columns
                #                     print(shap_df)
                tmp_shap = pd.concat([tmp_shap, shap_df], axis=0)
                print(tmp_shap)
                tmp_shap.to_csv(
                    './result_0620/shap/shap_' + str(algo) + '_변수_' + str(ip) + '_학습기간_' + str(trival) + '_예측기간_' + str(
                        tsival) + '.csv', index=True, encoding='utf-8-sig')



