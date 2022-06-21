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

# functions

def save_plot(trival, tsival):
    fig, ax = plt.subplots()

    plt.rcParams["figure.figsize"] = [10, 5]

    y_df = pd.DataFrame(y)
    result_df = pd.merge(predict_table, y_df, left_index = True, right_index= True, how='outer')

    plt.plot(result_df['철근(천원_톤)'].values, label = 'original')
    plt.plot(result_df['y_actual'].values, label = 'actual')
    plt.plot(result_df['y_pred'].values, label = 'pred')
    plt.legend(loc='lower right')

    plt.title(f'변수: {ip}, 학습기간: {trival}, 예측기간: {tsival}')

    # plt.show()
    return plt.savefig('./plot_' + str(algo) +'_변수_' + str(ip) + '_학습기간_' + str(trival) + '_예측기간_' + str(tsival) + '.png')

do_xgb = True

opt = ''
do_PCA = False
do_xlag = False
if do_PCA:
    opt += '_pca'
if do_xlag:
    opt += '_xlag'

train_ival = [120] #[3, 12, 36, 60, 120]  # (train_ival) months 데이터로 학습
test_ival = [1]#[1, 3, 6, 12]  # (test_ival) month 후 값을 예측

with open('./data-raw,corr.pkl', 'rb') as file:
    y, x, corrcoef = pickle.load(file) # load data

y.index = y.index.to_timestamp()
x.index = x.index.to_timestamp()

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

test_idx = np.where(y.index >= datetime.datetime(2021, 1, 1))[0]  # 2021년 1월 ~ 12월 12 points를 테스트 데이터로 사용
result = pd.DataFrame(columns=['algo', 'input', '학습기간', '예측시점', 'opt', 'MAPE', 'RMSE'])
cnt = -1

predict_table = pd.DataFrame()

if do_xgb:
    for algo, ip, fv in feature_set:
        for trival in train_ival:
            for tsival in test_ival:
                cnt += 1
                yhat = []
                for i in test_idx:
                    train_idx = list(range(i - trival - tsival + 1, i - tsival + 1))

                    x_train = fv.values[train_idx]
                    x_test = fv.values[i]
                    y_train = y.values[train_idx]
                    y_test = y.values[i]

                    # print(f'\n------------- shape of {i} -------------')
                    # print('x_train shape ---------', x_train.shape)
                    # print('x_test shape----------', x_test.shape)
                    # print('y_train shape----------', y_train.shape)
                    # print('y_test shape----------', y_test.shape)

                    if fv is None or not any(np.where(fv.iloc[train_idx].isnull())[0]):
                        model = xgb.XGBRegressor(random_state=1)
                        model.fit(x_train, y_train)

                        yhat = model.predict(np.asarray([x_test]))

                        tmp = pd.DataFrame({'algo': algo,
                                            'ip': ip,
                                            '학습기간(개월)': trival,
                                            '예측기간(개월)': tsival,
                                            'Y_Actual': y_test,
                                            'Y_Predict': yhat})

                        predict_table = pd.concat([predict_table, tmp], axis=0)

                        predict_table.to_csv('./' + str(algo) + '_변수_' + str(ip) + '_학습기간_' + str(trival) + '_예측기간_' + str(tsival) +' .csv', index=True, encoding='utf-8-sig')

                        save_plot(tsival, trival)

                    else:
                        test_idx = np.delete(test_idx, np.where(test_idx == i))

                mape = np.mean(
                    np.abs((predict_table['Y_Actual'] - predict_table['Y_Predict']) / predict_table['Y_Actual'])) * 100
                rmse = np.sqrt(mean_squared_error(predict_table['Y_Actual'], predict_table['Y_Predict']))
                result.loc[cnt] = np.array([algo, ip, trival, tsival, opt[1:], mape2, rmse2])
                result.columns = ['algo', 'input', '학습기간', '예측시점', 'opt', 'MAPE', 'RMSE']

                print(f'\n{algo}, {ip}, 학습기간 {trival}개월 ({trival / 12}년), 예측시점 {tsival}개월 후')
                print('MAPE:', mape)
                print('RMSE:', rmse)


            result.to_csv('./final_' + str(algo) + '_학습기간_' + str(trival) + '_예측기간_' + str(tsival) + '_performance.csv', index=False, encoding='utf-8-sig')