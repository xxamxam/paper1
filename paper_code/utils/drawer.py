import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from decimal import Decimal     
from copy import deepcopy
import seaborn as sns
import os
DPI = 200

def get_fig_set_style(shape=(1, 1), figsize=None):
    params = {
        "legend.fontsize": 10,
        "lines.markersize": 10,
        "axes.labelsize": 15,
        "axes.titlesize": 18,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "font.size": 15,
        #  "text.usetex": True
    }
    sns.set_context("paper", rc=params)
    # sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2.5})
    if figsize is None:
        fig, ax = plt.subplots(*shape, dpi=DPI)
    else:
        fig, ax = plt.subplots(*shape, dpi=DPI, figsize=figsize)
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.unicode'] = True
    # plt.grid(which="both")
    return fig, ax


def plot_ratios(ratios, gammas, max_budgets, x1, y1, l1, ratios_procecced = False, used_budgets = None, **kwargs):
    # ratios = 1/ ratios
    if ratios_procecced:
        ratios_stat = ratios
    else:
        ratios_stat = np.array([(np.max(r), np.std(r)) for r in ratios]).T
    gm = (1 + np.array(gammas)) #1/(1 - np.array(gammas))

    fig, ax = plt.subplots(1,3, figsize = (16, 4))
    colors = np.linspace(0, 2., len(gm))
    ax[0].scatter(gm , ratios_stat[0], c = colors, ec = 'k')
    ax[0].fill_between(gm , ratios_stat[0] - ratios_stat[1], ratios_stat[0] + ratios_stat[1], alpha = 0.2)

    ax[0].set_xlabel(l1, fontsize = 15)
    ax[0].set_ylabel(y1, fontsize = 15)

    ax[1].scatter(max_budgets, ratios_stat[0], c = colors, ec = 'k')
    ax[1].fill_between(max_budgets, ratios_stat[0] - ratios_stat[1], ratios_stat[0] + ratios_stat[1], alpha = 0.2)
    
    if used_budgets is not None:
        ax[1].axvline(used_budgets.min(),linewidth=2, color='r')
        # ax[1].axvline(used_budgets.max(),linewidth=2, color='r')

    ax[1].set_xlabel(x1, fontsize = 14)
    ax[1].set_ylabel(y1, fontsize = 14)
    
    # ax[2].plot(max_costs, gammas)
    ax[2].scatter(max_budgets, gm , c= colors, ec='k')

    ax[2].set_xlabel(x1, fontsize = 15)
    ax[2].set_ylabel(l1, fontsize = 15)
    for i in range(3):
        ax[i].grid()
    return fig, ax

def save_figures(dct: dict, path):
    for name, fig in dct.items():
        # fig.savefig(str(path / f"{name}.png"))
        p = path / f"{name}.pdf"
        if p.exists():
            os.remove(p)
        fig.savefig(str(p), bbox_inches="tight", dpi = DPI)

        data = np.array(fig.canvas.buffer_rgba())
        weights = [0.2989, 0.5870, 0.1140]
        data = np.dot(data[..., :-1], weights)
        # plt.imsave(str(path / f"{name}_image_gray.png"), data, cmap="gray")
        plt.imsave(str(path / f"{name}_gray.pdf"), data, cmap="gray")
        plt.close(fig)