import itertools
from collections import namedtuple
from typing import List, Optional, Union

import pandas as pd
from scipy import stats


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))


corr_stats = namedtuple("corr_stats", ["pearson", "spearman"])

df_query = namedtuple("df_query", ["name", "elements"])


def subset_dataframe(df: pd.DataFrame, queries: List[df_query],) -> pd.DataFrame:

    # copy dataframe to prevent overwriting
    sub_df = df.copy()

    #
    for iquery in queries:
        sub_df = sub_df[sub_df[iquery.name].isin(iquery.elements)]

    return sub_df


def get_correlations(df: pd.DataFrame):
    """Inputs a dataframe and outputs the correlation between
    the mutual information and the score.

    Requires the 'mutual_info' and 'score' columns."""

    # check that columns are in dataframe
    msg = "No 'mutual_info'  and/or 'score' column(s) found in dataframe"
    assert {"mutual_info", "score"}.issubset(df.columns), msg

    # get pearson correlation
    corr_pear = stats.pearsonr(df.score, df.mutual_info)[0]

    # get spearman correlation
    corr_spear = stats.spearmanr(df.score, df.mutual_info)[0]

    return corr_stats(corr_pear, corr_spear)
