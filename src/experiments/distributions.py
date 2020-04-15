import os, sys

# Insert path to model directory...
cwd = os.getcwd()
project_path = f"/home/emmanuel/projects/2019_hsic_align/src"
sys.path.insert(0, project_path)

# Insert path to package,.
pysim_path = f"/home/emmanuel/code/pysim/"
sys.path.insert(0, pysim_path)

import warnings
from typing import Optional, Tuple, Dict
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import argparse
import random

random.seed(123)

# toy datasets
from data.distribution import DataParams, Inputs

# Kernel Dependency measure
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from models.dependence import HSICModel
from pysim.kernel.utils import estimate_sigma


# RBIG IT measures
# from models.ite_algorithms import run_rbig_models

# Plotting
from visualization.distribution import plot_scorer

# experiment helpers
from experiments.utils import dict_product, run_parallel_step
from tqdm import tqdm

# Plotting Procedures
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

RES_PATH = (
    "/home/emmanuel/projects/2019_hsic_align/data/results/distributions/mutual_info/"
)


def get_parameters(dataset: str = "gauss", shuffle: bool = True) -> Dict:
    # initialize parameters
    params = {
        # dataset modification params
        "standardize": [True, False],
        "separate_scales": [True, False],
        "per_dimension": [True, False],
        # HSIC method params
        "scorer": ["hsic", "ka", "cka"],
        # Sigma estimation parameters
        "sigma_estimator": [
            ("silverman", None),
            ("scott", None),
            *[("median", x) for x in np.arange(0.1, 1.0, 0.1, dtype=np.float64)],
        ],
    }

    # add specific params
    if dataset == "gauss":
        # standard dataset parameters
        data_params = {
            "trial": [1, 2, 3, 4, 5],
            "samples": [50, 100, 500, 1_000, 5_000],
            "dimensions": [2, 3, 10, 50, 100],
            "dataset": ["gauss"],
            "std": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "nu": [1],
        }
    elif dataset == "tstudent":
        # standard dataset parameters
        data_params = {
            "trial": [1, 2, 3, 4, 5],
            "samples": [50, 100, 500, 1_000, 5_000],
            "dimensions": [2, 3, 10, 50, 100],
            "dataset": ["tstudent"],
            "std": [1],
            "nu": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    else:
        raise ValueError("Unrecognized dataset: ", {dataset})

    # shuffle parameters
    params = list(dict_product(params))
    data_params = list(dict_product(data_params))

    params = random.sample(params, len(params))
    data_params = random.sample(data_params, len(data_params))
    return params, data_params


def get_hsic(
    X: np.ndarray,
    Y: np.ndarray,
    scorer: str,
    sigma_X: Optional[float] = None,
    sigma_Y: Optional[float] = None,
) -> float:
    """Estimates the HSIC value given some data, sigma and
    the score."""
    # init hsic model class

    hsic_model = HSICModel()
    # hsic model params
    if sigma_X is not None:

        hsic_model.kernel_X = RBF(sigma_X)
        hsic_model.kernel_Y = RBF(sigma_Y)

    # get hsic score
    hsic_val = hsic_model.get_score(X, Y, scorer)

    return hsic_val


def get_sigma(
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "silverman",
    percent: Optional[float] = None,
    per_dimension: bool = False,
    separate_scales: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    # sigma parameters
    subsample = None
    random_state = 123

    sigma_X = estimate_sigma(
        X,
        subsample=subsample,
        method=method,
        percent=percent,
        random_state=random_state,
        per_dimension=per_dimension,
    )

    sigma_Y = estimate_sigma(
        Y,
        subsample=subsample,
        method=method,
        percent=percent,
        random_state=random_state,
        per_dimension=per_dimension,
    )

    if separate_scales is False:
        sigma_Y = sigma_X = np.mean([sigma_X, sigma_Y])

    return sigma_X, sigma_Y


def standardize_data(
    X: np.ndarray, Y: np.ndarray, standardize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    X = StandardScaler().fit_transform(X)
    Y = StandardScaler().fit_transform(Y)
    return X, Y


def step(params: Dict, data_params: Dict, inputs):

    # ====================
    # Sigma Estimator
    # ====================

    # estimate sigma
    sigma_X, sigma_Y = get_sigma(
        X=inputs.X,
        Y=inputs.Y,
        method=params["sigma_estimator"][0],
        percent=params["sigma_estimator"][1],
        per_dimension=params["per_dimension"],
        separate_scales=params["separate_scales"],
    )

    # ====================
    # HSIC Model
    # ====================
    # get hsic score
    score = get_hsic(inputs.X, inputs.Y, params["scorer"], sigma_X, sigma_Y)

    # ====================
    # Results
    # ====================

    # append results to dataframe
    results_df = pd.DataFrame(
        {
            # Data Params
            "dataset": [data_params["dataset"]],
            "trial": [data_params["trial"]],
            "std": [data_params["std"]],
            "nu": [data_params["nu"]],
            "samples": [data_params["samples"]],
            "dimensions": [data_params["dimensions"]],
            # STANDARDIZE PARSM
            "standardize": [params["standardize"]],
            # SIGMA FORMAT PARAMS
            "per_dimension": [params["per_dimension"]],
            "separate_scales": [params["separate_scales"]],
            # SIGMA METHOD PARAMS
            "sigma_method": [params["sigma_estimator"][0]],
            "sigma_percent": [params["sigma_estimator"][1]],
            "sigma_X": [sigma_X],
            "sigma_Y": [sigma_Y],
            # HSIC Params
            "scorer": [params["scorer"]],
            "score": [score],
            "mutual_info": [inputs.mutual_info],
        }
    )
    return results_df


def main(args):

    # get params
    params, data_params = get_parameters(args.dataset)

    # initialize datast
    header = False
    mode = "w"

    for iparam in tqdm(data_params, total=len(data_params)):

        # ================
        # DATA
        # ================
        dist_data = DataParams(
            dataset=iparam["dataset"],
            trial=iparam["trial"],
            std=iparam["std"],
            nu=iparam["nu"],
            samples=iparam["samples"],
            dimensions=iparam["dimensions"],
        )

        # generate data
        inputs = dist_data.generate_data()

        results_df = run_parallel_step(
            exp_step=step,
            parameters=params,
            n_jobs=args.njobs,
            verbose=args.verbose,
            data_params=iparam,
            inputs=inputs,
        )

        # concat current results
        results_df = pd.concat(results_df, ignore_index=True)

        # save results
        with open(f"{RES_PATH}{args.save}_{args.dataset}.csv", mode) as f:
            results_df.to_csv(f, header=header)

        header = False
        mode = "a"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HSIC Measures Experiment")

    parser.add_argument("--dataset", type=str, default="gauss", help="The dataset")

    parser.add_argument(
        "--save", type=str, default="dist_exp_v1", help="Save name for final data"
    )

    parser.add_argument(
        "--njobs", type=int, default=16, help="number of processes in parallel",
    )

    parser.add_argument(
        "--verbose", type=int, default=1, help="Number of helpful print statements."
    )
    args = parser.parse_args()

    main(args)
