"""
SK C&C 데이터분석1팀 김선민
2022.06.13 ~ 6.17 작성

"""
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import shap
from func import MAPE, plot_shap


""" setting
ㅁ 학습 데이터
ㅁ 모델 업데이트 여부 (True: stacking, False: learn all) / if nmodel==1, should be False
ㅁ 테스트셋으로 사용할 기간 (분기 단위) : ex) 최근 3년 데이터 = 12개 
ㅁ 타겟의 스케일 여부 (Close or y)
ㅁ 모델 종류 - 
    0: xgboost (선민)
    1: xgboost (민예 / X windowing & 통으로 학습-one model)
    2: gradient boosting (if fv_dr, datarobot feature)
    3: random forest (if fv_dr, datarobot feature)
ㅁ 다우지수/코스피 추가 여부 + SHAP plot 할 지 여부
ㅁ data robot feature 사용할지 (False)
ㅁ (임시) 삼성SDS 2015.06~2020.12 기간만 one model 확인
"""

file = 'ACN.csv'
# file = 'samsungSDS.csv'
update_model = True
test_nquarter = 40
target_scale = True
nmodel = 0
add_dow = False
plot_dow = False
rm_eco = False  # 경제지표 category 변수 제외

fv_dr = False  # datarobot feature
temp = False
plot = True


""" load data """
data = pd.read_csv(file)
if temp:
    idx = list(range(2, 25))
    data = data.iloc[idx]
    data.reset_index(inplace=True, drop=True)

if 'ACN' in file:
    data['Date_quarter'] = data['Date_quarter'].apply(lambda x: pd.to_datetime(str(x), format='%Y.%m.%d'))
    X = data.iloc[:, 2:-3]
elif 'SDS' in file:
    data['Date_quarter'] = data['Date_quarter'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m'))
    X = data.iloc[:, 1:-4]
    X.columns = [c.split('(')[0] for c in X.columns]

if target_scale:
    y = data.y
else:
    y = data.Close

X.columns = [c.lstrip().rstrip() for c in X.columns]  # 공유_ACN_정제_전략만_fnal2.xlsx
X = X.T.drop_duplicates().T  # ACN: '총자산회전율'은 '자본총계'와 동일해 삭제, 'BPS증가율'은 'BPS'와 동일해 삭제
if ('ACN' in file) & add_dow:
    X['Close_dow'] = data.Close_dow
elif ('SDS' in file) & add_dow:
    X['Close_kospi'] = data.Close_kospi


if update_model:
    test_ival = list(range(X.shape[0]-test_nquarter, X.shape[0]))
else:
    test_ival = list(range(X.shape[0]))

fv_datarobot_gb = ['매출액', '이자발생부채', '비유동자산', '자기자본비율', '매출원가', '매출총이익', '부채비율', '이익잉여금', '매입채무회전율',
                   '유동비율', '유동자산', '부채총계', '매출총이익률', '이자보상율', '자본총계', 'ROIC', '비유동비율', '매출채권회전율',
                   'ROA', 'BPS', '당기순이익', '운전자본증감', 'EPS', 'EBITDA2', '재고자산회전율', 'ROE', '순부채', '영업이익률',
                   '매출액증가율', '순이익률', '영업이익', 'CAPEX', 'EBITDA2증가율', 'Free Cash Flow2', '영업이익증가율', '순이익증가율',
                   'EPS증가율', 'EBITDA2마진율', '현금배당성향']
fv_datarobot_rf = ['매출액', '부채비율', '이자발생부채', '이익잉여금', '매입채무회전율', '당기순이익', '유동비율', '매출액증가율',
                   'ROIC', '매출채권회전율', 'BPS']


""" feature information """
info = pd.read_excel('../자료/해외기업 재무index 분류(Master)_0614.xlsx', skiprows=1)
info = info[['한글명', '구분']]

xcol, xgroup = [], []
for fn, xt in zip(info['한글명'], info['구분']):
    i = np.where([fn == c for c in X.columns])[0]
    if any(i):
        xcol += list(X.columns[i].values)
        xgroup += [xt] * len(i)

idx = np.argsort(xgroup)
info = pd.DataFrame([np.array(xcol)[idx], np.array(xgroup)[idx]], index=['feature', 'group']).T
info.drop_duplicates(inplace=True)
info.reset_index(inplace=True, drop=True)
info.to_csv('info.csv', index=False, encoding='utf-8-sig')

if rm_eco:
    if not add_dow:
        rmidx = np.where(info.group == '경제지표')[0]
    else:
        if 'ACN' in file:
            rmidx = np.where((info.group == '경제지표') & (info.feature != 'Close_dow'))[0]
        elif 'SDS' in file:
            rmidx = np.where((info.group == '경제지표') & (info.feature != 'Close_kospi'))[0]
    info.drop(rmidx, axis=0, inplace=True)
    info.reset_index(inplace=True, drop=True)

# assert(all([c in info.feature.values for c in X.columns]))
X = X[info.feature]
X = X.T.drop_duplicates().T

nocol = np.where([c not in info.feature.values for c in X.columns])[0]
print('info에 없는 X columns:', X.columns[nocol])
nocol = np.where([c not in X.columns for c in info.feature.values])[0]
print('X에 없는 info columns:', X.columns[nocol])


""" SHAP """
if fv_dr:
    if nmodel == 2:
        SHAP = pd.DataFrame(columns=fv_datarobot_gb)
        info = info.iloc[np.where([f in fv_datarobot_gb for f in info.feature])[0]].reset_index()
    elif nmodel == 3:
        SHAP = pd.DataFrame(columns=fv_datarobot_rf)
        info = info.iloc[np.where([f in fv_datarobot_rf for f in info.feature])[0]].reset_index()
    else:
        SHAP = pd.DataFrame(columns=X.columns)
else:
    SHAP = pd.DataFrame(columns=X.columns)

""" windowing - one by one """
# test 시점 이전 데이터는 전부 학습에 사용
pred_test = []
for test_idx in test_ival:
    # split train & test
    if update_model:
        train_idx = list(range(test_idx))
    else:
        train_idx = list(range(X.shape[0]))  # 이 경우 test_ival의 for loop 안도는게 맞지만 수정 귀찮아서 그대로.

    if nmodel != 1:
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx].to_frame().T
        y_test = y.iloc[test_idx]
    else:
        X_train, X_test = [], []
        # 과거 2 시점의 X, y 정보를 사용합니다.  --> 3 시점 flatten
        for ix in train_idx[2:]:
            xx = X.iloc[ix-2:ix+1, :].values.flatten()
            X_train.append(xx)
        for ix in test_ival[2:]:
            xx = X.iloc[ix-2:ix+1, :].values.flatten()
            X_test.append(xx)
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = y.iloc[train_idx[2:]]
        y_test = y.iloc[test_idx]

    # 모델 생성
    if nmodel == 0:  # xgboost
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test)
        fit = xgb.train(params={}, dtrain=dtrain, num_boost_round=400)
        pred_test += list(fit.predict(dtest))
        pred_train = fit.predict(dtrain)

    elif nmodel == 1:  # xgboost (민예)
        fit = XGBRegressor(random_state=0, objective='reg:squarederror', subsample=0.9, max_depth=4,
                           reg_alpha=0.1, colsample_bytree=1.0, colsample_bylevel=1.0, min_child_weight=2,
                           gamma=0.0)
        fit.fit(X_train, y_train)
        pred_test += list(fit.predict(X_test))
        pred_train = fit.predict(X_train)

    elif nmodel == 2:  # gradient boosting (datarobot)
        if fv_dr:
            X_train = X_train[fv_datarobot_gb]
            X_test = X_test[fv_datarobot_gb]

        fit = GradientBoostingRegressor(learning_rate=0.005, n_estimators=500, max_depth=3, max_leaf_nodes=None,
                                        min_samples_split=2, min_samples_leaf=1, subsample=1.0, max_features=0.3,
                                        alpha=0.9, )
        fit.fit(X_train, y_train)
        pred_test += list(fit.predict(X_test))
        pred_train = fit.predict(X_train)

    elif nmodel == 3:  # random forest (datarobot)
        if fv_dr:
            X_train = X_train[fv_datarobot_rf]
            X_test = X_test[fv_datarobot_rf]

        fit = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=None, max_leaf_nodes=10, min_samples_split=2, min_samples_leaf=5, max_features=0.3, bootstrap=False)
        fit.fit(X=X_train, y=y_train)
        pred_test += list(fit.predict(X_test))
        pred_train = fit.predict(X_train)

    # SHAP explainer 생성
    explainer = shap.TreeExplainer(fit)

    # SHAP value 계산
    shap_values = explainer.shap_values(X_test)[0]
    # # 하나의 instance에 대해 변수 별 Shapley value
    # shap.initjs()
    # shap.force_plot(explainer.expected_value, shap_values, X)

    SHAP = SHAP.append(pd.DataFrame([shap_values], columns=X_train.columns, index=[data['Date_quarter'][test_idx]]))

    if nmodel == 1: break

SHAP.to_csv('shap.csv', index=False, encoding='utf-8-sig')

date_train = data['Date_quarter'].iloc[train_idx]
date_test = data['Date_quarter'].iloc[test_ival]
true_train = y.iloc[train_idx].values
true_test = y.iloc[test_ival].values
if nmodel == 1:
    date_train = date_train[2:]
    date_test = date_test[2:]
    true_train = true_train[2:]
    true_test = true_test[2:]
MAPE_train = MAPE(true_train, np.array(pred_train))
MAPE_test = MAPE(true_test, np.array(pred_test))

# plt.figure(figsize=(18, 5))
# plt.subplot(1, 2, 1)
# plt.plot(date_train, true_train, 'b')
# plt.plot(date_train, pred_train, 'r')
# plt.title('train - MAPE %.2f%%' % MAPE_train)
# plt.subplot(1, 2, 2)
# plt.plot(date_test, true_test, 'b')
# plt.plot(date_test, pred_test, 'r')
# plt.title('test - MAPE %.2f%%' % MAPE_test)
# plt.show()

print()
print('model', nmodel)
print('test mape', np.round(MAPE_test, 2))

if ('ACN' in file) & add_dow & ~plot_dow:
    SHAP.drop(['Close_dow'], axis='columns', inplace=True)
    info.drop(np.where(info.feature == 'Close_dow')[0], axis=0, inplace=True)
    info.reset_index(inplace=True, drop=True)
elif ('SDS' in file) & add_dow & ~plot_dow:
    SHAP.drop(['Close_kospi'], axis='columns', inplace=True)
    info.drop(np.where(info.feature == 'Close_kospi')[0], axis=0, inplace=True)
    info.reset_index(inplace=True, drop=True)

if plot:
    # plotname = f"{file.split('.')[0]}_{'stacking' if update_model else 'learnAll'}_targetScaled{'O' if target_scale else 'X'}_model{nmodel}.png"
    # plot_shap(date_test, true_test, pred_test, SHAP, feature_group=info.group, save_plot=plotname)
    plot_shap(date_test, true_test, pred_test, SHAP, feature_group=info.group)

