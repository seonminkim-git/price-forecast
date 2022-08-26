"""
SK C&C 데이터분석1팀 김선민 작성
"""

import func
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결

do_yCheck = False
do_xyCorr = False
do_GCTest = True

timelag = range(3, -14, -1)  # 철근이 3달 빠른 때부터 13달 느린 때까지


with open('../01. Data/data.pkl', 'rb') as file:
    data1, data2, data3, data4, data5, data6, data7, data8, data9 = pickle.load(file)
with open('../01. Data/datainfo.pkl', 'rb') as file:
    filename = pickle.load(file)

datalist = [data1, data2, data3, data4, data5, data6, data7, data8, data9]

for i, d in enumerate(datalist):
    d.name = filename[i].split('.')[0]
    d.index = d.index.to_timestamp(how='e')

# target
yname = '철근(천원_톤)'
y = data1[yname]
data1.drop(yname, axis='columns', inplace=True)

# 공통기간 데이터
dlist = []
idx = data1.index
for i, d in enumerate(datalist):
    tmp = d.loc[idx].copy()
    tmp.columns = [filename[i].split('.')[0] + '-' + c for c in tmp.columns]
    dlist += [tmp]
data_all = pd.concat(dlist, axis=1).copy()


""" y Analysis """
if do_yCheck:
    dy = np.diff(y)
    dyp = dy/y[:-1]*100
    N = len(dy)
    n = []
    rng = [-100, -5, -2, 0, 2, 5, 100]
    for i in range(len(rng)-1):
        n += [sum((dyp >= rng[i]) & (dyp < rng[i+1]))]
    diffcnt = pd.DataFrame(np.round(np.array(n)/N*100, 1), columns=['데이터 비율 (%)'],
                           index=['-5% 미만']+[f'{rng[i]}% 이상, {rng[i+1]}% 미만' for i in range(1, len(rng)-2)]+['5% 이상'])
    diffcnt.index.name = 'differential (%, 1 month)'

    _, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(y)
    ax[0].set(ylabel=yname, title='Target')

    ax[1].plot(dyp, '.-')
    ax[1].axhline(0, color='k', linewidth=0.3)
    ax[1].axhline(5, color='r', linestyle='--')
    ax[1].axhline(-5, color='r', linestyle='--')
    ax[1].axhline(3, color='g', linestyle='--')
    ax[1].axhline(-3, color='g', linestyle='--')
    ax[1].set(ylabel='Difference (%)', title='1 Month Difference (%)')

    ax[2].hist(dyp, bins=50, density=True)
    ax[2].axvline(0, color='k')
    ax[2].axvline(5, color='r', linestyle='--')
    ax[2].axvline(-5, color='r', linestyle='--')
    ax[2].axvline(3, color='g', linestyle='--')
    ax[2].axvline(-3, color='g', linestyle='--')
    ax[2].table(cellText=diffcnt.values,
                colLabels=diffcnt.columns,
                rowLabels=diffcnt.index,
                colWidths=[0.3],
                loc='right',
                )
    ax[2].set(title='1 Month Difference', xlabel='difference (%)', ylabel='density')
    plt.savefig(f'../02. EDA/Target {yname}.png')
    # plt.show()
    plt.close()


""" X & y Relation - Cross Correlation """
if do_xyCorr:
    corrcoef = pd.DataFrame(
        columns=['파일', '항목', '기간', 'N'] + ['lag='+str(t) for t in timelag] + ['highest corr', 'best lag (highest corr)'])

    cnt = -1
    for d in datalist:
        print(f'\n[{d.name}]')
        idx = y.index.intersection(d.index)
        d_ = d.loc[idx]
        y_ = y.loc[idx]

        for i in range(d.shape[1]):
            cnt += 1
            print(f'- {d.columns[i]}')
            x = d_.iloc[:, i]

            d1 = idx[0].strftime('%Y/%m')
            d2 = idx[-1].strftime('%Y/%m')
            duration = f'{d1} ~ {d2}'

            ce = x.corr(y_)  # np.corrcoef(x, y_)[0][1]
            xce = [func.crosscorr(x, y_, lag) for lag in timelag]
            peak = timelag[np.argmax(np.abs(xce))]
            peak2 = -1 if peak == 0 else peak
            y_sft = y_.shift(peak2)
            validx = np.isfinite(y_) & np.isfinite(x)
            slp, itc = np.polyfit(x[validx], y_[validx], 1)
            validx_ = np.isfinite(y_sft) & np.isfinite(x)
            slp_, itc_ = np.polyfit(x[validx_], y_sft[validx_], 1)
            corrcoef.loc[cnt] = [d.name, d.columns[i], duration, len(idx)] + xce + [np.max(xce), peak]

            _, ax = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

            """ Raw Data - line plot """
            ax1 = ax[0][0]
            ax2 = ax1.twinx()
            l1, = ax1.plot(y_, color='b', label=yname)
            l2, = ax2.plot(x, color='r', label=d.columns[i])
            # l3, = ax1.plot(y_sft, color='c', label=d.columns[i]+'(best lag)')
            ax1.legend([l1, l2], [l.get_label() for l in [l1, l2]])
            # ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]])
            ax1.set_ylabel(yname)
            ax2.set_ylabel(d.columns[i])
            plt.title(d.columns[i] + ' - CORR: ' + str(np.round(ce, 4)) + ' (time lag = 0)')

            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)
            plt.setp(ax2.spines.values(), visible=False)
            ax2.spines["right"].set_visible(True)
            ax1.yaxis.label.set_color(l1.get_color())
            ax2.yaxis.label.set_color(l2.get_color())
            ax1.spines["left"].set_edgecolor(l1.get_color())
            ax2.spines["right"].set_edgecolor(l2.get_color())
            ax1.tick_params(axis='y', colors=l1.get_color())
            ax2.tick_params(axis='y', colors=l2.get_color())

            """ Time Lagged Cross Correlation """
            ax3 = ax[0][1]
            ax3.plot(timelag, xce, 'o-', label='Correlation coefficient')
            ax3.axvline(0, color='k', linewidth=0.3)
            ax3.axhline(0, color='k', linewidth=0.3)
            ax3.axvline(peak, color='r', linestyle='--', label='Peak synchrony')
            titstr = f"'{d.columns[i]}'이/가 '{yname}'보다 {np.abs(peak)}개월 {'늦게' if peak>0 else '빨리'} 움직임"
            ax3.set(title=titstr, xlabel='Time lag', ylabel='Pearson r', ylim=[-0.2, 1])
            ax3.legend(loc='lower right')

            """ Raw Data - scatter plot """
            ax4 = ax[1][0]
            ax4.scatter(x, y_, c='k')
            ax4.plot(x, x * slp + itc, 'k')
            ax4.set(xlabel=d.columns[i], ylabel=yname, title=f'{d.columns[i]} (time lag = 0) - CORR: {np.round(ce, 4)}')

            """ Time Shifted Data - scatter plot """
            ax5 = ax[1][1]
            ax5.scatter(x, y_sft, c='k')
            ax5.plot(x, x * slp_ + itc_, 'k')
            ax5.set(xlabel=d.columns[i], ylabel='Shifted '+yname, xlim=ax4.get_xlim(), ylim=ax4.get_ylim(),
                    title=d.columns[i] + f" ({yname.split('(')[0]} time lag = {peak2}) - CORR: {np.round(xce[timelag.index(peak2)], 4)}")

            plt.savefig('../02. EDA/Correlation/image/CORR ' + '%.4f' % np.round(ce, 4) + '-' + d.name + '-' + d.columns[i] + '.png')
            plt.close()

    corrcoef.to_csv('../02. EDA/Correlation/Pearson correlation coefficient.csv', index=False, encoding='utf-8-sig')


""" Granger Causality Test """
if do_GCTest:
    mlag = 13
    # granger_causality = pd.DataFrame(columns=['항목']+[f'lag={t+1}M' for t in range(mlag)])
    col1 = ['항목 (X)'] + [c2 for c1 in [[f'lag={t}M', '', ''] for t in range(mlag+1)] for c2 in c1]
    col2 = [''] + ['X->Y', 'Y->X', '']*(mlag+1)
    granger_causality = pd.DataFrame(columns=[col1, col2])
    cnt = -1
    for ic in data_all.columns:
        cnt += 1

        # X -> Y
        xg = pd.concat([y.to_frame(), data_all[ic]], axis=1)  # Does 'data_all[ic]]' causes 'y'?
        xg.dropna(axis=0, inplace=True)
        gr = grangercausalitytests(xg, maxlag=mlag, verbose=False)
        p = [gr[m+1][0]['ssr_ftest'][1] for m in range(mlag)]
        # granger_causality.loc[cnt] = [ic] + p

        xg0 = pd.concat([y.to_frame().shift(1), data_all[ic]], axis=1)
        xg0.dropna(axis=0, inplace=True)
        gr0 = grangercausalitytests(xg0, maxlag=1, verbose=False)
        p0 = gr0[1][0]['ssr_ftest'][1]

        # Y -> X
        xgr = pd.concat([data_all[ic], y.to_frame()], axis=1)  # Does 'y' causes 'data_all[ic]]'?
        xgr.dropna(axis=0, inplace=True)
        gr_r = grangercausalitytests(xgr, maxlag=mlag, verbose=False)
        p_r = [gr_r[m+1][0]['ssr_ftest'][1] for m in range(mlag)]

        xgr0 = pd.concat([data_all[ic], y.to_frame().shift(-1)], axis=1)
        xgr0.dropna(axis=0, inplace=True)
        gr_r0 = grangercausalitytests(xgr0, maxlag=1, verbose=False)
        p_r0 = gr_r0[1][0]['ssr_ftest'][1]

        # conclusion
        con = ['O' if (p[i] <= 0.05) & (p_r[i] > 0.05) else 'X' for i in range(mlag)]

        granger_causality.loc[cnt] = [ic] + [p0, p_r0, 'O' if (p0 <= 0.05) & (p_r0 > 0.05) else 'X'] + \
                                     [c2 for c1 in [[p[i], p_r[i], con[i]] for i in range(mlag)] for c2 in c1]

    granger_causality.to_csv('../02. EDA/Granger causality/Granger causality test.csv', index=False, encoding='utf-8-sig')

