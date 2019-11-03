from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use(["ggplot", "seaborn-paper"])


FIG_PATH = "/home/emmanuel/projects/2019_hsic_align/results/figures/scaling_experiment/"


# plot the results


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

    if log_scale:
        df["scale"] = np.log(df["scale"])
    if log_score:
        df["hsic_value"] = np.log(df["hsic_value"])
    if style is not None:
        plt.style.use(style)

    # plot
    sns.scatterplot(
        x="scale",
        y="hsic_value",
        hue="gamma_method",
        data=df[df["scorer"] == scorer],
        ax=ax,
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Scale")
    ax.legend(prop={"size": 9})
    if scorer == "hsic":
        ax.set_ylim([-0.001, 0.11])
    else:
        ax.set_ylim([-0.01, 1.1])
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(scorer.upper())

    plt.tight_layout()
    plt.show()
    if save:
        save_name = f"{scorer}_{dataset}.png"
        fig.savefig(FIG_PATH + save_name)
