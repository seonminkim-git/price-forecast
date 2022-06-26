import numpy as np
from func_evaluation import crosscorr
import matplotlib.pyplot as plt


def plot_pred(y_true, y_hat, ax=None, x=None, text='', title='', savefilename=None, close=False):
    if x is None:
        x = range(len(y_true))

    if ax is None:
        _, ax = plt.subplots()

    plt.plot(x, y_hat, 'r', label='prediction')
    plt.plot(x, y_true, 'b', label='true')
    plt.text(0.05, 0.7, text, transform=ax.transAxes, fontsize='large')
    plt.title(title)
    plt.legend()

    if savefilename is not None:
        plt.savefig(savefilename)

    if close:
        plt.close()
    else:
        plt.show()

    return ax


def plot_two_lines(y1, y2, ax=None, x=None, xlabel='date', ylabel=None):
    if x is None:
        x = range(len(y1))
    ylab1 = 'y1' if ylabel is None else ylabel[0]
    ylab2 = 'y2' if ylabel is None else ylabel[1]

    if ax is None:
        _, ax = plt.subplots()
    ax1 = ax
    ax2 = ax1.twinx()
    l1, = ax1.plot(x, y1, color='b', label=ylab1)
    l2, = ax2.plot(x, y2, color='r', label=ylab2)
    ax1.legend([l1, l2], [l.get_label() for l in [l1, l2]])
    ax1.set_ylabel(ylab1)
    ax1.set_xlabel(xlabel)
    ax2.set_ylabel(ylab2)

    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    ax2.spines.visible = False
    ax2.spines["right"].set_visible(True)
    ax1.yaxis.label.set_color(l1.get_color())
    ax2.yaxis.label.set_color(l2.get_color())
    ax1.spines["left"].set_edgecolor(l1.get_color())
    ax2.spines["right"].set_edgecolor(l2.get_color())
    ax1.tick_params(axis='y', colors=l1.get_color())
    ax2.tick_params(axis='y', colors=l2.get_color())


def plot_scatter(x, y, ax=None, xlabel='', ylabel='', text=''):
    if ax is None:
        _, ax = plt.subplots()
    validx = np.isfinite(x) & np.isfinite(y)
    slp, itc = np.polyfit(x[validx], y[validx], 1)

    ax.scatter(x, y, color='k')
    ax.plot(x, x * slp + itc, 'k')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.text(0.05, 0.5, text + '\ny = %.2f x + %.1f' % (slp, itc), transform=ax.transAxes, fontsize='x-large')


def plot_xce(x, y, timelag, ax=None, text=''):
    xce = [crosscorr(x, y, lag) for lag in timelag]
    peak = timelag[np.argmax(xce)]

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(timelag, xce, 'o-', label='Correlation coefficient')
    ax.axvline(0, color='k', linewidth=0.3)
    ax.axhline(0, color='k', linewidth=0.3)
    ax.axvline(peak, color='r', linestyle='--', label='Peak synchrony')
    ax.set(xlabel='Time lag', ylabel='Pearson r', ylim=[-0.2, 1])
    ax.legend(loc='lower right')
    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize='x-large')

