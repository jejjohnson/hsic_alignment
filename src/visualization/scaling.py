from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use(["ggplot", "seaborn-paper"])


FIG_PATH = "/home/emmanuel/projects/2019_hsic_align/results/figures/scaling_experiment/"


# plot the results


def plot_scorer_scale_norm(
    df: pd.DataFrame,
    scorer: str,
    dataset: Optional[str] = None,
    omit_methods: Optional[Tuple[str, List[str]]] = None,
    omit_samples: Optional[Tuple[str, List[str]]] = None,
    style: Optional[List[str]] = None,
    log_scale: bool = False,
    log_score: bool = False,
    save: bool = False,
    title: Optional[str] = None,
    plot_legend: bool = True,
    normalized: bool = True,
) -> None:

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # extract dataset
    if dataset is not None:
        df = df[df["dataset"] == dataset]

    # if normalized == True:
    #     df = df[df["normalized"] == True]
    #     normalized = 1
    # elif normalized == False:
    #     df = df[df["normalized"] == False]
    #     normalized = 0
    # else:
    #     raise ValueError("Unrecognized normalized:", normalized)
    # omit methods
    if omit_methods is not None:
        df = df[~df[omit_methods[0]].isin(omit_methods[1])]
        # omit methods
        if omit_samples is not None:
            df = df[~df[omit_samples[0]].isin(omit_samples[1])]

    if log_scale:
        df["scale"] = np.log10(df["scale"])
    if log_score:
        df["hsic_value"] = np.log(df["hsic_value"])
    if style is not None:
        plt.style.use(style)

    # plot
    lines = sns.scatterplot(
        x="hsic_value",
        y="mi",
        hue="scaled",
        style="normalized",
        data=df[df["scorer"] == scorer],
        ax=ax,
    )
    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("Score")
    ax.legend("")
    # else:
    #     ax.legend(prop={"size": 9})
    # if scorer == "hsic":
    #     ax.set_ylim([-0.001, 0.105])
    # pass
    # else:
    #     ax.set_ylim([-0.01, 1.05])
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(scorer.upper())

    # Force Integer Values for x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
    if save:
        save_name = f"{scorer}_{dataset}_n{normalized}"
        fig.savefig(FIG_PATH + save_name + ".png")

    # plot legend
    # colors = [c for c in handles.]
    handles, labels = ax.get_legend_handles_labels()
    if plot_legend:
        plot_legend_alone(handles, labels, save_name)


def plot_scorer_scale(
    df: pd.DataFrame,
    scorer: str,
    dataset: Optional[str] = None,
    omit_methods: Optional[Tuple[str, List[str]]] = None,
    omit_samples: Optional[Tuple[str, List[str]]] = None,
    style: Optional[List[str]] = None,
    log_scale: bool = False,
    log_score: bool = False,
    save: bool = False,
    title: Optional[str] = None,
    plot_legend: bool = True,
    normalized: bool = True,
) -> None:

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # extract dataset
    if dataset is not None:
        df = df[df["dataset"] == dataset]

    if normalized == True:
        df = df[df["normalized"] == True]
        normalized = 1
    elif normalized == False:
        df = df[df["normalized"] == False]
        normalized = 0
    else:
        raise ValueError("Unrecognized normalized:", normalized)
    # omit methods
    if omit_methods is not None:
        df = df[~df[omit_methods[0]].isin(omit_methods[1])]
        # omit methods
        if omit_samples is not None:
            df = df[~df[omit_samples[0]].isin(omit_samples[1])]

    if log_scale:
        df["scale"] = np.log10(df["scale"])
    if log_score:
        df["hsic_value"] = np.log(df["hsic_value"])
    if style is not None:
        plt.style.use(style)

    # plot
    lines = sns.scatterplot(
        x="scale",
        y="hsic_value",
        hue="gamma_method",
        data=df[df["scorer"] == scorer],
        ax=ax,
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Scale (log10)")
    ax.legend("")
    # else:
    #     ax.legend(prop={"size": 9})
    if scorer == "hsic":
        ax.set_ylim([-0.001, 0.105])
        # pass
    else:
        ax.set_ylim([-0.01, 1.05])
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(scorer.upper())

    # Force Integer Values for x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()
    if save:
        save_name = f"{scorer}_{dataset}_n{normalized}"
        fig.savefig(FIG_PATH + save_name + ".png")

    # plot legend
    # colors = [c for c in handles.]
    handles, labels = ax.get_legend_handles_labels()
    if plot_legend:
        plot_legend_alone(handles, labels, save_name)


def plot_legend_alone(handles, labels, save_name: Optional[str] = None):
    fig_legend = plt.figure(constrained_layout=True)
    ax = fig_legend.add_subplot(111)
    fig_legend.legend(
        handles[1:], labels[1:], loc="upper center", frameon=False, ncol=len(labels[1:])
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
