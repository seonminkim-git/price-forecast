"""
SK C&C 데이터분석1팀 김선민
2022.06.13 ~ 6.17 작성

참고: /Users/seonmin/Desktop/SKCC/3. 예측값 설명/color_explain2_4_report.py

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결


def MAPE(y_true, y_hat):
    return np.mean(np.abs((y_hat - y_true) / y_true)) * 100


def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos-.15, xpos], [ypos, ypos],
                      transform=ax.transAxes, color='black', lw=0.7)
    line.set_clip_on(False)
    ax.add_line(line)


def plot_shap(x, y_true, y_hat, shap_value, feature_group=None, save_plot=None):
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(6, 30, figure=fig)
    ax1 = fig.add_subplot(gs[0, :29])
    ax3 = fig.add_subplot(gs[1:, :29])
    ax4 = fig.add_subplot(gs[1:, 29])

    ax1.plot(x, y_true, c='b', label='True')
    ax1.plot(x, y_hat, 'r', label='Prediction')
    # ax1.set_xlim(x[0], x[-1])
    ax1.legend(loc='upper left')
    ax1.set_ylabel('Price')

    nmax = np.max(np.abs(shap_value.values))
    # im = ax3.imshow(shap_value.T, interpolation='none', aspect='auto', cmap='jet', vmax=nmax, vmin=-nmax)
    cmap = mcolors.ListedColormap(np.concatenate([[(1-cx/255, 1-cx/255, 1) for cx in range(255, -1, -1)],
                                                  [(1, 1-cx/255, 1-cx/255) for cx in range(256)]]))
    im = ax3.imshow(shap_value.T, interpolation='none', aspect='auto', cmap=cmap, vmax=nmax, vmin=-nmax)
    xt = np.array(ax3.get_xticks()[1:-1]).astype(int)
    # xtl = np.datetime_as_string(x.iloc[xt], unit='h')
    xtl = x.iloc[xt].astype(str)
    ax3.set_xticks(xt)
    ax3.set_xticklabels(xtl)
    yt = list(range(shap_value.shape[1]))
    ytl = shap_value.columns
    ax3.set_yticks(yt)
    ax3.set_yticklabels(ytl)

    if feature_group is not None:
        print(pd.concat([pd.Series(shap_value.columns.values), pd.Series(feature_group)], axis=1))
        diffloc = [i if feature_group[i-1] != feature_group[i] else None for i in range(1, len(feature_group))]
        diffloc = list(filter(None, diffloc))
        add_line(ax3, 0, 1)
        N = len(feature_group)
        # p = 0
        for dl in diffloc+[N]:
            add_line(ax3, 0, 1-dl/N)
            # ax3.text(-2.2, (p+dl+1)/2, feature_group[dl], fontdict={'size': 'x-large', 'weight': 'bold'})
            ax3.text(-.15, 1-dl/N, feature_group[dl-1], transform=ax3.transAxes,
                     fontdict={'size': 'x-large', 'weight': 'bold', 'verticalalignment': 'bottom'})
            # p = dl
    cbar = plt.colorbar(im, cax=ax4)
    cbar.set_label('shap')

    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
    if save_plot is None:
        plt.show()
    else:
        plt.savefig(save_plot)

