import sys, os
import warnings
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import argparse
from sklearn.utils import check_random_state

# toy datasets
from data.it_data import MIData

# Kernel Dependency measure
from models.train_models import get_gamma_init
from models.train_models import get_hsic

# RBIG IT measures
from models.ite_algorithms import run_rbig_models

warnings.filterwarnings("ignore")  # get rid of annoying warnings

SAVE_PATH = "/home/emmanuel/projects/2019_hsic_align/results/distribution/"


from typing import Tuple


def get_gamma_name(gamma_method: Tuple[str, str, str]) -> str:
    if gamma_method[1] is None and gamma_method[2] is None:
        gamma_name = gamma_method[0]
    elif gamma_method[1] is not None and gamma_method[2] is None:
        gamma_name = f"{gamma_method[0]}_p{gamma_method[1]}"
    elif gamma_method[1] is None and gamma_method[2] is not None:
        gamma_name = f"{gamma_method[0]}_s{gamma_method[2]}"
    elif gamma_method[1] is not None and gamma_method[2] is not None:
        gamma_name = f"{gamma_method[0]}_s{gamma_method[1]}_s{gamma_method[2]}"
    else:
        raise ValueError("Unrecognized Combination...")
    return gamma_name


def main(args):

    # clf_exp = DistributionExp(
    #     seed=args.seed,
    #     factor=args.factor,
    #     sigma_est=args.sigma,
    #     n_gamma=args.gamma,
    #     save_path=SAVE_PATH,
    #     save_name=args.save,
    # )

    # # run full experiment
    # clf_exp.run_experiment()

    # dataset params
    datasets = ["tstudent", "gauss"]
    samples = [50, 100, 500, 1_000, 5_000]
    dimensions = [2, 3, 10, 50, 100]
    trials = np.linspace(1, 5, 5, endpoint=True, dtype=int)

    # max params
    n_gamma = args.gamma
    factor = args.factor

    # experimental parameters
    scorers = ["hsic", "tka", "ctka"]
    gamma_methods = [
        ("silverman", None, None),
        ("scott", None, None),
        ("median", 0.2, None),
        ("median", 0.4, None),
        ("median", None, None),
        ("median", 0.6, None),
        ("median", 0.8, None),
        ("median", None, 0.01),
        ("median", None, 0.1),
        ("median", None, 10),
        ("median", None, 100),
        #     ('max', None, None)
    ]
    std_params = np.linspace(1, 11, 11, endpoint=True, dtype=int)
    nu_params = np.linspace(1, 9, 9, endpoint=True, dtype=int)

    # results dataframe
    results_df = pd.DataFrame()

    # loop through datasets
    for idataset in datasets:
        # run experiment
        with tqdm(gamma_methods) as gamma_bar:

            # Loop through Gamma params
            for imethod in gamma_bar:
                # Loop through samples
                for isample in samples:

                    # Loop through dimensions
                    for idim in dimensions:

                        # Loop through trials
                        for itrial in trials:

                            # Loop through HSIC scorers
                            for iscorer in scorers:

                                # extract dataset
                                if idataset == "gauss":
                                    dof_params = std_params
                                elif idataset == "tstudent":
                                    dof_params = nu_params
                                else:
                                    raise ValueError(
                                        f"Unrecognized dataset: {idataset}"
                                    )

                                # Loop through dof params
                                for idof in dof_params:

                                    X, Y, mi_val = MIData(idataset).get_data(
                                        samples=isample,
                                        dimensions=idim,
                                        std=idof,
                                        nu=idof,
                                        trial=itrial,
                                    )

                                    # initialize gamma
                                    if imethod[0] == "max":
                                        hsic_value = get_hsic(
                                            X,
                                            Y,
                                            iscorer,
                                            gamma_init,
                                            maximum=True,
                                            n_gamma=n_gamma,
                                            factor=factor,
                                        )
                                    else:
                                        gamma_init = get_gamma_init(
                                            X, Y, imethod[0], imethod[1], imethod[2]
                                        )
                                        hsic_value = get_hsic(X, Y, iscorer, gamma_init)

                                    gamma_name = get_gamma_name(imethod)

                                    # append results to dataframe
                                    results_df = results_df.append(
                                        {
                                            "dataset": idataset,
                                            "samples": isample,
                                            "dimensions": idim,
                                            "trial": itrial,
                                            "scorer": iscorer,
                                            "gamma_method": gamma_name,
                                            "gamma_init": gamma_init,
                                            "hsic_value": hsic_value,
                                            "dof": idof,
                                            "mi_value": mi_val,
                                        },
                                        ignore_index=True,
                                    )

                                    # save results
                                    results_df.to_csv(f"{SAVE_PATH}{args.save}.csv")
                        postfix = dict(Samples=f"{isample}", Dimensions=f"{idim}")
                        gamma_bar.set_postfix(postfix)
    results_df.head()
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HSIC Measures Experiment")
    parser.add_argument(
        "--seed", type=int, default=123, help="The random state for data generation."
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=50,
        help="Number of points in gamma parameter grid.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Factor to be used for bounds for gamma parameter.",
    )
    parser.add_argument(
        "--save", type=str, default="dist_v2_gamma", help="Save name for final data."
    )

    args = parser.parse_args()

    main(args)
