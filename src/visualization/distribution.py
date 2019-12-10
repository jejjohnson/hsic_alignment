from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use(["ggplot", "seaborn-paper"])


FIG_PATH = (
    "/home/emmanuel/projects/2019_hsic_align/results/figures/distribution_experiment/"
)


def plot_scorer(results_df: pd.DataFrame, scorer: str) -> None:

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    sns.scatterplot(
        x="gamma_init",
        y="hsic_value",
        hue="gamma_method",
        data=results_df[results_df["scorer"] == scorer],
        ax=ax,
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Gamma Initialization")
    ax.legend(prop={"size": 9})
    ax.set_title(scorer.upper())
    plt.show()
    return None


# plot the results


def plot_scorer_mi(
    df: pd.DataFrame,
    scorer: str,
    dataset: Optional[str] = None,
    hue: str = "gamma_method",
    omit_methods: Optional[Tuple[str, List[str]]] = None,
    omit_samples: Optional[Tuple[str, List[str]]] = None,
    style: Optional[List[str]] = None,
    log_mi: bool = True,
    log_score: bool = True,
    save: bool = False,
    title: Optional[str] = None,
    plot_legend: bool = False,
) -> None:

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # extract dataset
    if dataset is not None:
        df = df[df["dataset"] == dataset]

    # omit methods
    if omit_methods is not None:
        df = df[~df[omit_methods[0]].isin(omit_methods[1])]
        # omit methods
        if omit_samples is not None:
            df = df[~df[omit_samples[0]].isin(omit_samples[1])]

    if log_mi:
        df["mi_value"] = np.log2(df["mi_value"] + 1)
    if log_score:
        df["hsic_value"] = np.log(df["hsic_value"] + 1)
    if style is not None:
        plt.style.use(style)

    # plot
    sns.scatterplot(
        x="hsic_value", y="mi_value", hue=hue, data=df[df["scorer"] == scorer], ax=ax
    )
    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("Score")
    if not plot_legend:
        ax.legend(prop={"size": 9})
    else:
        ax.legend([])
    #     if scorer == 'hsic':
    #         ax.set_xlim([-0.01, 0.03])
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(scorer.upper())

    plt.tight_layout()
    plt.show()
    if save:
        save_name = f"summary_{scorer}_{dataset}_c{hue}"
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
