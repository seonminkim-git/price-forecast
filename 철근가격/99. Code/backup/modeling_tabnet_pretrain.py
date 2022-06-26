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

max_epochs = 100  # pretraining
predict_table = pd.DataFrame()

if do_tabnet:
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

                    # reshape
                    x_train_ = x_train
                    y_train_ = y_train.reshape(-1, 1)
                    x_test_ = x_test.reshape(1, x_test.shape[0])
                    y_test_ = y_test.reshape(-1, 1)

                    # print('x_train shape ---------', x_train_.shape)
                    # print('y_train shape----------', y_train_.shape)
                    # print('x_test shape----------', x_test_.shape)
                    # print('y_test shape----------', y_test_.shape)

                    if fv is None or not any(np.where(fv.iloc[train_idx].isnull())[0]):
                        # Pretraining
                        pretrainer = TabNetPretrainer(
                            # cat_idxs=cat_idxs, # cat_idx : 범주형 변수 인덱스 (제외)
                            # cat_dims=cat_dims, # cat_dim : 범주형 변수 리스트 (제외)
                            # cat_emb_dim=3, #cat_emb_dim : 범주형 변수 임베딩 사이즈 (제외)
                            # n_d = 2, # decision prediction layer
                            # n_a = 2, # attention embedding layer (usually n_a = n_d)
                            optimizer_fn=torch.optim.Adam, # optimizer : adam
                            optimizer_params=dict(lr=2e-2), # optimizer : adam 일경우, 정의해야하는 초기 학습
                            mask_type='entmax',  # "sparsemax": sparsity에 강한 활성화함수, "entmax": 일반화 버전
                            n_steps=4, # n_steps : model architecture의 steps
                            n_shared=2, # n_shared : 공유 GLU
                            n_independent=2, # n_independent : 독립 GLU
                            scheduler_params=dict(mode="min",  # scheduler_fn의 parameter
                                                  patience=5,
                                                  min_lr=1e-5,
                                                  factor=0.9, ),
                            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,  # learning rate를 조절하는 pytorch scheduler
                            verbose=10,
                            seed=42
                        )

                        pretrainer.fit(
                            X_train=x_train_,
                            eval_set=[x_test_],
                            # max_epochs=max_epochs,
                            # patience=20, # patience : early stopping 전 모델 성능이 개선되지 않는 연속 epoch (if patience = 0; no early stopping)
                            # batch_size=64, # batch size : batch 마다 사용되는 데이터 수
                            # virtual_batch_size=32,  # virtual_batch_size : 미니배치에 사용되는 데이터 수 (batch size / virtual batch size = type(int))
                            # num_workers=0, # num_workers : torch.utils.data.Dataloader에 사용된 숫자 (제외)
                            # drop_last=False, # drop_last : training 중 완료되지 않은 경우 마지막 배치 삭제 여
                            pretraining_ratio=0.8, # pretraining_ratio : masking 할 input feature의 비율 (0~1)
                        )

                        reconstructed_X, embedded_X = pretrainer.predict(x_test_)

                        # print(reconstructed_X, embedded_X)
                        # modeling
                        model = TabNetRegressor(
                            optimizer_fn=torch.optim.Adam,
                            optimizer_params=dict(lr=2e-2),
                            scheduler_params={"step_size": 10,
                                              "gamma": 0.9},
                            scheduler_fn=torch.optim.lr_scheduler.StepLR,
                            mask_type='sparsemax',  # pretrain model 사용시 'entmax' 사용
                            seed=42,
                            verbose=1
                        )

                        model.fit(X_train=x_train_, y_train=y_train_,
                                  eval_set=[(x_train_, y_train_), (x_test_, y_test_)],
                                  eval_name=['train', 'test'],
                                  eval_metric=['rmse'],
                                  max_epochs=max_epochs, patience=20,
                                  # batch_size=64, virtual_batch_size=32,
                                  # num_workers=0,
                                  drop_last=False,
                                  from_unsupervised=pretrainer) #pretrainer 사용

                        yhat = model.predict(x_test_)

                        tmp = pd.DataFrame.from_dict([{'algo': algo,
                                                       'ip': ip,
                                                       '학습기간(개월)': trival,
                                                       '예측기간(개월)': tsival,
                                                       'Y_Actual': y_test_.reshape(1, ),
                                                       'Y_Predict': yhat.reshape(1, )}])

                        predict_table = pd.concat([predict_table, tmp], axis=0)
                        #                         print(predict_table)
                        predict_table['Y_Actual'] = [float(s) for s in predict_table['Y_Actual']]
                        predict_table['Y_Predict'] = [float(s) for s in predict_table['Y_Predict']]

                        predict_table.to_csv(
                            './pretrain_' + str(algo) + '_prediction_table.csv',
                            index=False, encoding='utf-8-sig')

                    else:
                        test_idx = np.delete(test_idx, np.where(test_idx == i))

                    mape2 = np.mean(np.abs(
                        (predict_table['Y_Actual'] - predict_table['Y_Predict']) / predict_table['Y_Actual'])) * 100
                    rmse2 = np.sqrt(mean_squared_error(predict_table['Y_Actual'], predict_table['Y_Predict']))

                result.loc[cnt] = np.array([algo, ip, trival, tsival, opt[1:], mape2, rmse2])
                result.columns = ['algo', 'input', '학습기간', '예측시점', 'opt', 'MAPE', 'RMSE']

                result['MAPE'] = [float(s) for s in result['MAPE']]

                print(f'\n{algo}, {ip}, 학습기간 {trival}개월 ({trival / 12}년), 예측시점 {tsival}개월 후')
                print('MAPE:', mape2)
                print('RMSE:', rmse2)

                #                 print(result)
                result.to_csv('./tabnet_pretrain/final_' + str(algo) + '_.csv', index=False,
                              encoding='utf-8-sig')
