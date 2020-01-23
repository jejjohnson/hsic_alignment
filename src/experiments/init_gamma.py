import sys, os
import warnings
import tqdm
import random
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# Insert path to model directory,.
cwd = os.getcwd()
path = f"{cwd}/../../src"
sys.path.insert(0, path)

# toy datasets
from data.toy import generate_dependence_data

# Kernel Dependency measure
from models.dependence import HSIC, train_hsic
from models.kernel import estimate_sigma, sigma_to_gamma, gamma_to_sigma, get_param_grid

import matplotlib.pyplot as plt


def plot_max_hsic(
    results_list,
    scorer="hsic",
    function="line",
    sigma_est="mean",
    save=True,
    plot_legend=True,
):

    save_path = f"{cwd}/../../results/figures/init_gamma/"

    fig, ax = plt.subplots(nrows=1, figsize=(7, 5))

    # MAX HSIC Values
    max_idx = np.argmax(scorers[scorer])
    hsic_max = scorers[scorer][max_idx]
    gamma_max = gammas[max_idx]
    print(f"Max {scorer.upper()}: {hsic_max:.4f}")
    print(f"Max Gamma ({scorer.upper()}): {gamma_max:.4f}\n")

    # init HSIC Values
    init_median, _ = get_init(method="median", factor=1)
    median_hsic = HSIC(gamma_X=init_median, scorer=scorer).fit(X, Y).score(X)
    #     print(init_median, median_hsic)

    init_mean, _ = get_init(method="mean", factor=1)
    mean_hsic = HSIC(gamma_X=init_mean, scorer=scorer).fit(X, Y).score(X)
    #     print(init_mean, mean_hsic)

    init_silv, _ = get_init(method="silverman", factor=1)
    silv_hsic = HSIC(gamma_X=init_silv, scorer=scorer).fit(X, Y).score(X)
    #     print(init_silv, silv_hsic)

    ax.set_xscale("log")
    ax.plot(gammas, scorers[scorer], color="red", linewidth=10, zorder=0)
    ax.scatter(gamma_max, hsic_max, s=300, c="yellow", label=f"Maximum", zorder=1)
    ax.scatter(init_median, median_hsic, s=300, c="black", label="Median", zorder=1)
    ax.scatter(init_mean, mean_hsic, s=300, c="green", label="Mean", zorder=1)
    ax.scatter(init_silv, silv_hsic, s=300, c="blue", label="Silverman", zorder=1)
    ax.set_xlabel("$\gamma$", fontsize=20)
    ax.set_ylabel(f"{scorer.upper()}", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=10)

    if scorer in ["ctka", "tka"]:
        ax.set_ylim([0.0, 1.1])
    elif scorer in ["hsic"]:
        ax.set_ylim([0.0, 0.11])
    else:
        raise ValueError(f"Unrecognized scorer: {scorer}")
    if not plot_legend:
        plt.legend(fontsize=20)
    plt.xticks(fontsize=20)

    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

    #     if save:
    save_name = f"demo_{function}_{sigma_est}_{scorer}"
    fig.savefig(save_path + save_name + ".png")

    # plot legend
    # colors = [c for c in handles.]
    handles, labels = ax.get_legend_handles_labels()
    if plot_legend:
        plot_legend_alone(handles, labels, save_name)


def plot_legend_alone(handles, labels, save_name: Optional[str] = None):
    fig_legend = plt.figure(constrained_layout=True)
    ax = fig_legend.add_subplot(111)
    fig_legend.legend(
        handles, labels, loc="upper center", frameon=False, ncol=len(labels)
    )
    # ax.axis("off")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig_legend.canvas.draw()
    plt.tight_layout()
    # bbox_inches = "tight"
    # plt.tight_layout()

    if save_name is not None:
        fig_legend.savefig(FIG_PATH + save_name + "_legend.png", bbox_inches="tight")
    fig_legend.show()


class DataParams:
    datasets = ["sine", "line", "circle", "random"]
    num_points = 1_000
    noise_x = 0.1
    noise_y = 0.1


class ExperimentParams:
    seeds = np.linspace(1, 10, 10, dtype=int)
    sigma_est = "mean"
    factor = 1
    n_gammas = 100
    percent = 0.5
    # hsic model params
    kernel = "rbf"
    subsample = None
    bias = True
    scorers = ["hsic", "tka", "ctka"]


def experiment_step(scorer, gamma, seed, exp_params):
    # Get HSIC Score
    clf_hsic = HSIC(
        kernel=exp_params.kernel,
        scorer=scorer,
        gamma_X=gamma,
        gamma_Y=gamma,
        subsample=exp_params.subsample,
        bias=exp_params.bias,
    )

    # calculate HSIC return scorer
    clf_hsic.fit(X, Y)

    # hsic value and kernel alignment score
    hsic_score = clf_hsic.score(X)

    results_df = pd.DataFrame(
        {"scorer": scorer, "gamma": gamma, "hsic_score": hsic_score, "trial": seed},
        index=[0],
    )
    return results_df
