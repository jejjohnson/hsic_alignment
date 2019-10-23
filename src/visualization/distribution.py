from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
    dataset: str,
    omit_methods: Optional[Tuple[str, List[str]]] = None,
) -> None:

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    # extract dataset
    df = df[df["dataset"] == dataset]

    # omit methods
    if omit_methods is not None:
        df = df[~df[omit_methods[0]].isin(omit_methods[1])]

    # plot
    sns.scatterplot(
        x="hsic_value",
        y="mi_value",
        hue="gamma_method",
        data=df[df["scorer"] == scorer],
        ax=ax,
    )
    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("Score")
    ax.legend(prop={"size": 9})
    ax.set_title(scorer.upper())
    plt.show()
