
# !conda install pytorch torchvision -c pytorch
# !pip3 install pytorch_tabnet

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
plt.rcParams['axes.unicode_minus'] = False

import torch
print(torch.__version__)
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sklearn.model_selection import StratifiedKFold

from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import gc
gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
else:
  DEVICE = torch.device('cpu')

do_Prophet = False
do_ARIMAX = False
do_xgb = False
do_tabnet = True

opt = ''
do_PCA = False
do_xlag = False
if do_PCA:
    opt += '_pca'
if do_xlag:
    opt += '_xlag'

train_ival = [3, 12, 36, 60, 120]# [3, 12, 36, 60, 120]  # (train_ival) months 데이터로 학습
test_ival = [1]#[1, 3, 6, 12]  # (test_ival) month 후 값을 예측

with open('data-raw,corr.pkl', 'rb') as file:
    y, x, corrcoef = pickle.load(file)

y.index = y.index.to_timestamp()
x.index = x.index.to_timestamp()

test_idx = np.where(y.index >= datetime.datetime(2021, 1, 1))[0]  # 2021년 1월 ~ 12월 12 points를 테스트 데이터로 사용
result = pd.DataFrame(columns=['algo', 'input', '학습기간', '예측시점', 'opt', 'MAPE', 'RMSE'])
cnt = -1

for i in x.columns[x.isnull().any(axis=0)]:
    x[i].fillna(x[i].mean(), inplace=True)

idx2 = [a1 or a2 or a3 for a1, a2, a3 in zip(['철강원자재 가격' in c for c in x.columns], ['철강생산량-철근' in c for c in x.columns], ['품목별 수출액, 수입액-철및강-수입액' in c for c in x.columns])]
idx3 = [a1 or a2 or a3 or a4 for a1, a2, a3, a4 in zip(idx2, ['철강생산량' in c for c in x.columns], ['원유 가격' in c for c in x.columns], ['원달러 환율' in c for c in x.columns])]
idx5 = idx3 or []
feature_set = [
#     ['tabnet', '단변량', None],
    ['tabnet', '공급_small', x[x.columns[idx2]]],  # y.to_frame().join(x)
    ['tabnet', '공급_large', x[x.columns[idx3]]],
#     ['tabnet', '수요+공급', x[x.columns[idx5]]],
    ['tabnet', '전부 다', x]
]

# tabnet feature importance

max_epochs = 50  # pretraining
tabnet_table = pd.DataFrame()

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

                x_train_ = x_train
                y_train_ = y_train.reshape(-1, 1)
                x_test_ = x_test.reshape(1, x_test.shape[0])
                y_test_ = y_test.reshape(-1, 1)

                # print('x_train shape ---------', x_train_.shape)
                # print('y_train shape----------', y_train_.shape)
                # print('x_test shape----------', x_test_.shape)
                # print('y_test shape----------', y_test_.shape)

                if fv is None or not any(np.where(fv.iloc[train_idx].isnull())[0]):
                    model = TabNetRegressor(
                        seed=42,
                        verbose=0
                    )

                    model.fit(X_train=x_train_, y_train=y_train_,
                              eval_set=[(x_train_, y_train_), (x_test_, y_test_)],
                              eval_name=['train', 'test'],
                              eval_metric=['rmse'],
                              max_epochs=50, patience=20,
                              drop_last=False,
                              )
                    yhat = model.predict(x_test_)

                    # feature importance
                    tabnet_feature_imp = pd.DataFrame(model.feature_importances_, index=fv.columns)
                    tabnet_feature_imp = tabnet_feature_imp.T
                    tabnet_feature_imp['feature'] = ip
                    tabnet_feature_imp['학습기간'] = trival
                    tabnet_table = pd.concat([tabnet_table, tabnet_feature_imp], axis=0)
                    print(tabnet_table.head())
                    tabnet_table.to_csv(
                        './feature_imp_' + str(algo) + '_변수_' + str(ip) + '_예측기간_' + str(
                            tsival) + '.csv', index=True, encoding='utf-8-sig')

                else:
                    test_idx = np.delete(test_idx, np.where(test_idx == i))