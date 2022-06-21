# %% 
# Import library and data
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import time
import pickle
import os
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import font_manager, rc

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error as MSE
import scipy.stats as stats

warnings.filterwarnings('ignore')
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# %%
current_path = os.getcwd()
os.chdir(current_path)
with open('../01. Data/data-raw,corr.pkl', 'rb') as file:
    y, x, corrcoef = pickle.load(file)
    
y.index = y.index.to_timestamp()
x.index = x.index.to_timestamp()

train_ival = [3, 12, 36, 60, 120]  # (train_ival) months 데이터로 학습
test_ival = [1]#[1, 3, 6, 12]  # (test_ival) month 후 값을 예측


idx2 = [a1 or a2 or a3 for a1, a2, a3 in zip(['철강원자재 가격' in c for c in x.columns], ['철강생산량-철근' in c for c in x.columns], ['품목별 수출액, 수입액-철및강-수입액' in c for c in x.columns])]
idx3 = [a1 or a2 or a3 or a4 for a1, a2, a3, a4 in zip(idx2, ['철강생산량' in c for c in x.columns], ['원유 가격' in c for c in x.columns], ['원달러 환율' in c for c in x.columns])]

test_idx = np.where(y.index >= datetime(2021, 1, 1))[0]  # 2021년 1월 ~ 12월 12 points를 테스트 데이터로 사용
result = pd.DataFrame(columns=['algo', 'input', '학습기간', '예측시점', 'opt', 'MAPE', 'RMSE'])


# %%
# define function
def MAPE(y_test, y_pred):
	return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def RMSE(y_true, y_hat):
    return np.sqrt(np.mean((y_hat - y_true)**2))

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

# %%
feature_set = [
    ['Linear', '단변량', None],
    ['Linear', '공급_small', x[x.columns[idx2]]], 
    ['Linear', '공급_large', x[x.columns[idx3]]],
    # ['Linear', '수요+공급', x[x.columns[idx5]]],
    ['Linear', '전부 다', x[x.columns[x.isna().sum() == 0]]]
]

y_1 = y.shift(-1).copy()
do_PCA = False
do_linear = True

# %%
make_yhat = pd.DataFrame()
if do_linear:
    for algo, ip, fv in feature_set:
        for trival in train_ival[:-1]:
            for tsival in test_ival:
                # make train/test dataset                
                xs = []
                ys = []
                yhat = []
                
                # 단변량 X
                if fv is not None:
                    '''
                    [[1,2,3], 
                     [2,3,4],
                      ...] 를 하나의 X dataset으로 만들어서 학습
                    '''
                    for i in range(0, test_idx[0] - trival):
                        idx_tr = list(range(i, i + trival- tsival + 1))

                        y_tmp = y_1.values[idx_tr[-1]] 
                        x_tmp = np.array(pd.concat([fv.iloc[idx_tr], y[idx_tr]], axis=1)).reshape(-1)
                        xs.append(x_tmp)
                        ys.append(y_tmp)
                    
                    df_train = pd.DataFrame(xs)
                    df_y_train = pd.DataFrame({'y': ys})
                    
                    # Do PCA == True 인 경우, x_train dataset을 PCA 적용
                    if do_PCA:
                        df_train, scaler_, pca_ = func_pca(df_train)  
                            
                    # linear model fitting
                    model = LinearRegression()
                    model.fit(df_train, df_y_train)
                    
                    # Check feature importance if do_PCA=False / column 생성
                    if not do_PCA:
                        lst = []
                        for i in range(0, trival):
                            x_y = pd.concat([fv.iloc[idx_tr], y[idx_tr]], axis=1)
                            t = [col+"_t+"+ str(i) for col in x_y.columns]
                            lst.extend(t)
                            
                        feat_coef = pd.DataFrame(model.coef_).T
                        feat_coef.index = lst
                        feat_coef.sort_values(by=0, key=abs, ascending=False, inplace=True)
                        
                    # Model Test
                    for j in range(len(test_idx)):
                        idx_ts = range((test_idx[j]-trival), test_idx[j])
                        x_y = pd.concat([fv.iloc[idx_ts], y[idx_ts]], axis=1)
                        x_tmp = np.array(x_y).reshape(1,-1)
                        
                        # do_PCA==True -> x data PCA transform
                        if do_PCA:
                            x_tmp = scaler_.transform(x_tmp)
                            x_tmp = pca_.transform(x_tmp)
                            
                        pred = model.predict(pd.DataFrame(x_tmp)) # 하나씩 예측
                        yhat.append(pred[0][0])
                        
                # 단변량 O
                else:
                    for i in range(0, test_idx[0]-trival):
                        '''
                        [[1,2,3],
                         [2,3,4],
                         ...] 형태로 하나의 X dataset 만들어서 한 번에 fit 
                        '''
                        idx_tr = list(range(i, i + trival- tsival + 1))
                        x_tmp = y.values[idx_tr]
                        y_tmp = y.values[idx_tr[-1] + 1]
                        xs.append(x_tmp)
                        ys.append(y_tmp)
                    df_train = pd.DataFrame(xs)
                    df_y_train = pd.DataFrame({'y': ys})
                    
                    model = LinearRegression()
                    model.fit(df_train, df_y_train)
                    
                    yhat = y[[i+1 for i in idx_tr]].values
                    yhat = yhat.astype(np.float)
                    for j in range(len(test_idx)):
                        xs = yhat[j:j+trival]            
                        pred = model.predict(np.array(xs).reshape(1,-1))
                        yhat = np.append(yhat, pred.reshape(-1))
                    
                    yhat = yhat[-12:]
                    
                print(f'\n{algo}, {ip}, 학습기간 {trival}개월 ({trival/12}년), 예측시점 {tsival}개월 후 (PCA: {do_PCA})')
                print('MAPE:', MAPE(yhat, y[-12:].values))
                print('RMSE:', RMSE(yhat, y[-12:].values))
                            
                t = pd.DataFrame({'value':yhat}, index=y[-12:].index)       
                plt.plot(t, label='predicted')
                plt.plot(y, label='y')
                plt.axvline(x=datetime(2021, 1, 1), color='green')
                plt.legend()
                plt.show()
                
                colname = '_'.join([str(ip), str(trival),'개월',str(do_PCA)])
                make_yhat = pd.concat([make_yhat,pd.DataFrame({colname:yhat})], axis=1)
                # plot feature importance if do_PCA=False
                if fv is not None and not do_PCA:
                    display(feat_coef)
                    sns.barplot(y=feat_coef[:10].index, x=feat_coef[:10].values.reshape(-1), orient='h')
                    plt.show()

                                
# %%
