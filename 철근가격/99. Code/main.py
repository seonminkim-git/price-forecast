"""
철근가격예측 통합코드
"""

import datetime
import numpy as np
import pandas as pd
from func_data import load_data
from func_model import Prophet, ARIMAX, TabNet
from func_evaluation import MAPE, RMSE
import func_plot as plot
from os.path import exists
import warnings
warnings.simplefilter("ignore")

""" setting """
target = '철강원자재 가격-철근(천원_톤)'
datadir = '../01. Data/'
input_type = 2
algo = 'TabNet'
opt = ''
do_PCA = False
do_mrmr = False
do_xlag = False
train_size = [3, 12, 36, 60, 120]  # (train_ival) months 데이터로 학습
test_size = [1, 3, 6, 12]  # (test_ival) month 후 값을 예측
# test_ival = [datetime.datetime(2020, 1, 1), datetime.datetime(2021, 1, 1)]  # 2020년 1월 이후 2021년 1월 이전
test_ival = [datetime.datetime(2021, 1, 1), None]  # 2021년 1월 이후 데이터를 테스트 데이터로 사용
result_file = '../03. Modelling/performance.csv'
plot_result = False

""" . """
input_type = 1 if algo == 'Prophet' else input_type
input_dict = {0: '전부 다', 1: '단변량', 2: '공급_small', 3: '공급_large'}
input_txt = input_dict[input_type] if type(input_type).__name__ == 'int' else input_type

# f_algo = lambda xtr, ytr, xts, yts: exec(algo+'(xtr, ytr, xts, yts)')
f_algo = globals()[algo]
if do_PCA:
    opt += '_pca'
if do_xlag:
    opt += '_xlag'

if (algo == 'ARIMAX') & any([ts <= 3 for ts in train_size]):
    train_size = np.array(train_size)[[x > 3 for x in train_size]]

if not exists(result_file):
    result = pd.DataFrame(columns=['algo', 'input', '학습기간', '예측시점', 'opt', 'MAPE', 'RMSE'] + ['m+'+str(m+1) for m in range(12)])
    cnt = -1
else:
    result = pd.read_csv(result_file)
    cnt = result.shape[0]-1

""" load data """
# index: time 정보, 오름차순
dobj = load_data(datadir)
data = dobj.select_data(input_type=input_type)

""" X & y """
y = data[target]
X = data.drop(target, axis='columns')

if test_ival[0] is None:
    test_ival[0] = y.index[0]
if test_ival[1] is None:
    test_ival[1] = y.index[-1] + datetime.timedelta(days=1)
test_idx = np.where((y.index >= test_ival[0]) & (y.index < test_ival[1]))[0]

""" 조건 별 학습 & 테스트 결과 저장 """
for trsz in train_size:  # 학습 데이터 사이즈. 현 시점 기준 몇 달 전 데이터까지 사용할지
    for tssz in test_size:  # 테스트 데이터 ?!?!. 현 시점 기준 몇 달 후 값을 예측할지
        yhat = []
        cnt += 1
        for i in test_idx:  # 테스트 기간 한 포인트마다 windowing하며 학습 & 테스트
            train_idx = list(range(max(0, i-trsz-tssz+1), i-tssz+1))  # 해당 시점 테스트 시 학습 데이터 구간

            """ split into train/test set """
            x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            x_test, y_test = X.iloc[i], pd.Series(y.iloc[i], index=[y.index[i]])

            """ pre-processing """

            """ training """
            model, yhat_ = f_algo(x_train, y_train, x_test, y_test)
            yhat = np.concatenate([yhat, yhat_])

        """ evaluation """
        mape = MAPE(y.values[test_idx], yhat)
        rmse = RMSE(y.values[test_idx], yhat)

        result.loc[cnt] = np.concatenate(([algo, input_txt, trsz, tssz, opt[1:],
                                           np.round(mape, 6), np.round(rmse, 6)], np.round(np.array(yhat), 6)))

        """ 결과 plot """
        print(f'algo: {algo}')
        print(f'input: {input_txt}')
        print(f'train_size: {trsz}')
        print(f'test_size: {tssz}')
        print(f"option: {opt.split('_')}")
        print(f'MAPE: {np.round(mape, 1)}, RMSE: {np.round(rmse, 1)}')

        plot.plot_pred(y.values[test_idx], yhat, x=y.index[test_idx], close=not plot_result,
                       text=f"MAPE: {np.round(mape, 2)}\nRMSE: {np.round(rmse, 2)}")
        print()

result.drop_duplicates(inplace=True, ignore_index=True, subset=result.columns.difference(['opt']))  # opt가 NaN인 경우 때문
result.to_csv(result_file, index=False, encoding='utf-8-sig')

