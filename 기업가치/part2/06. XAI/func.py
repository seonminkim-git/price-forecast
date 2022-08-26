import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pygad
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결
plt.rc('font', size=13)
# plt.rc('axes', titlesize=SMALL_SIZE)
# plt.rc('axes', labelsize=MEDIUM_SIZE)
# plt.rc('xtick', labelsize=SMALL_SIZE)
# plt.rc('ytick', labelsize=SMALL_SIZE)
# plt.rc('legend', fontsize=SMALL_SIZE)
# plt.rc('figure', titlesize=BIGGER_SIZE)

# VIF (Variance Inflation Factor) 높은 태그 삭제
def remove_high_vif(dat, th=10, verbose=True):
    dat = dat.copy()

    rm_col = []
    while True:
        vif = pd.DataFrame()
        vif["features"] = dat.columns
        vif["VIF"] = [variance_inflation_factor(dat.values, i) for i in range(dat.shape[1])]
        # print(vif.sort_values(by='VIF', ascending=False))
        # display(vif.style.hide_index())

        # xcorr = dat.corr()

        col_vif10 = vif.features[vif.VIF > th]
        if col_vif10.empty:
            break
        else:
            col_drop = vif.features[vif.VIF == max(vif.VIF)].values
            # col_drop = [col_drop[0]]
            rm_col = np.concatenate((rm_col, col_drop))
            dat.drop(columns=col_drop, inplace=True)
            if verbose:
                print('VIF>'+str(th)+':', col_vif10.values)
                print('drop:', col_drop, '\n')

    return dat, rm_col


def remove_high_vif_2(dat, th=10, verbose=True, keep_feature=None):
    dat = dat.copy()

    rm_col = []
    while True:
        vif = pd.DataFrame()
        vif["features"] = dat.columns
        vif["VIF"] = [variance_inflation_factor(dat.values, i) for i in range(dat.shape[1])]
        # print(vif.sort_values(by='VIF', ascending=False))
        # display(vif.style.hide_index())

        col_vif10 = vif.features[vif.VIF > th]  # candidates
        # col_vif10 = col_vif10[[f for f in col_vif10 if f not in keep_feature]]
        if col_vif10.empty:
            break
        else:
            corr = dat[col_vif10].corr()
            for i in range(corr.shape[0]): corr.iloc[i, i] = 0
            while True:
                maxval = np.max(np.abs(corr.values))
                idx = np.where(np.abs(corr.values) == maxval)
                try:
                    f1, f2 = corr.columns[idx[0][0]], corr.columns[idx[0][1]]
                except:
                    print(dat)
                col_drop = [f for f in [f1, f2] if f not in keep_feature]
                col_drop = [col_drop[0]]
                if col_drop:
                    rm_col = np.concatenate((rm_col, col_drop))
                    dat.drop(columns=col_drop, inplace=True)
                    if verbose:
                        print('VIF>' + str(th) + ':', col_vif10.values)
                        print('drop:', col_drop, '\n')
                    break
                else:
                    corr.values[idx[0][0], idx[0][1]] = 0
                    continue

    return dat, rm_col


# def genetic_algorithm(function_inputs, desired_output):
#
#     def fitness_func(solution, solution_idx):
#         output = np.sum(solution * function_inputs)
#         fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
#         return fitness
#
#     fitness_function = fitness_func
#
#     def on_start(ga_instance):
#         print("on_start()")
#
#     def on_fitness(ga_instance, population_fitness):
#         print("on_fitness()")
#
#     def on_parents(ga_instance, selected_parents):
#         print("on_parents()")
#
#     def on_crossover(ga_instance, offspring_crossover):
#         print("on_crossover()")
#
#     def on_mutation(ga_instance, offspring_mutation):
#         print("on_mutation()")
#
#     def on_generation(ga_instance):
#         print("on_generation()")
#
#     def on_stop(ga_instance, last_population_fitness):
#         print("on_stop()")
#
#     num_generations = 100  # Number of generations.
#     num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
#     sol_per_pop = 10  # Number of solutions in the population.
#     num_genes = len(function_inputs)
#
#     ga_instance = pygad.GA(num_generations=num_generations,
#                            num_parents_mating=num_parents_mating,
#                            fitness_func=fitness_function,
#                            sol_per_pop=sol_per_pop,
#                            num_genes=num_genes,
#                            on_start=on_start,
#                            on_fitness=on_fitness,
#                            on_parents=on_parents,
#                            on_crossover=on_crossover,
#                            on_mutation=on_mutation,
#                            on_generation=on_generation,
#                            on_stop=on_stop)
#
#     ga_instance.run()
#
#     print()


def MAPE(y_true, y_hat):
    return np.mean(np.abs((y_hat - y_true) / y_true)) * 100


def RMSE(y_true, y_hat):
    return np.sqrt(np.mean((y_hat - y_true)**2))


def R2(y_true, y_hat):
    return r2_score(y_true, y_hat)


def adj_R2(y_true, y_hat, k, n=None):
    rs = R2(y_true, y_hat)
    if n is None:  # 소희M test adj_R2 의 n=#train
        n = len(y_true)
    return 1-(1-rs)*(n-1)/(n-k-1)


def plot_line(x, y_true, y_pred, i, test_index=None, title=''):
    plt.figure(figsize=(18, 4))
    plt.plot(x, y_true, 'k', label='true')
    if i != 0:
        plt.plot(x[:i], y_pred[:i], 'b', label='train')
        plt.plot(x[i:], y_pred[i:], 'r', label='test')
    else:
        plt.plot(x, y_pred, 'b', label='predict')
        if test_index is not None:
            plt.plot(x[test_index], y_pred[test_index], 'ro', label='test')
    plt.legend()
    plt.title(title)
    plt.show()


def plot_scatter(y_true, y_pred, i, title=''):
    plt.figure()
    plt.plot(y_true[:i], y_pred[:i], 'b.', label='train')
    plt.plot(y_true[i:], y_pred[i:], 'r.', label='test')
    plt.plot(y_true, y_true, 'k')
    plt.legend()
    plt.xlabel('true')
    plt.ylabel('prediction')
    plt.title(title)
    plt.show()


def plot_box(dat):
    col = dat.columns
    nrow = len(col)//3
    if nrow*3 != len(col):
        nrow += 1

    for r in range(nrow):
        plt.figure(figsize=(18, 4))
        for c in range(3):
            plt.subplot(1, 3, c+1)
            if r*3+c < len(col):
                boxplot = dat.boxplot(column=col[r*3+c])
                boxplot.plot()
                plt.title(col[r*3+c])
        plt.show()


def plot_shap(x, y_true, y_hat, shap_value, feature_group=None, save_plot=None, figsize=(18, 9)):

    def add_line(ax, xpos, ypos):
        line = plt.Line2D([xpos - .15, xpos], [ypos, ypos],
                          transform=ax.transAxes, color='black', lw=0.7)
        line.set_clip_on(False)
        ax.add_line(line)

    fig = plt.figure(figsize=figsize)
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
    # im = ax3.imshow(shap_value.T, interpolation='none', aspect='auto', cmap=cmap)
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
        # print(pd.concat([pd.Series(shap_value.columns.values), pd.Series(feature_group)], axis=1))
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

