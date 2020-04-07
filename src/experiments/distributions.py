import os

# Insert path to model directory...
cwd = os.getcwd()
project_path = f"/home/emmanuel/projects/2019_hsic_align/src"
sys.path.insert(0, project_path)

# Insert path to package...
pysim_path = f"/home/emmanuel/code/pysim/"
sys.path.insert(0, pysim_path)

import argparse
import collections
import random
import sys
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils import check_random_state
from tqdm import tqdm

# toy datasets
from data.distribution import DataParams, Inputs

# experiment utilities
from experiments.utils import dict_product, run_parallel_step

# Kernel Dependency measure
from models.dependence import HSICModel
from pysim.kernel.utils import GammaParam, SigmaParam


def get_parameters(dataset: str = "gauss") -> Dict:
    # initialize parameters
    parameters = {
        # standard dataset parameters
        "trials": [1, 2, 3, 4, 5],
        "samples": [50, 100, 500, 1_000, 5_000],
        "dimensions": [2, 3, 10, 50, 100],
        # dataset modification params
        "standardize": [True, False],
        # HSIC method params
        "scorers": ["hsic", "ka", "cka"],
        # Sigma estimation parameters
        "sigma_estimators": [
            SigmaParam("silverman", None, None),
            SigmaParam("scott", None, None),
            *[
                SigmaParam("median", x, d)
                for x in np.arange(0.1, 1.0, 0.1, dtype=np.float64)
                for d in [True, False]
            ],
        ],
    }

    # add specific params
    if dataset == "gauss":
        parameters["dataset"] = [
            "gauss",
        ]
        parameters["std"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        parameters["nu"] = [1]
    elif dataset == "tstudent":
        parameters["dataset"] = [
            "tstudent",
        ]
        parameters["nu"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        parameters["std"] = [1]
    else:
        raise ValueError("Unrecognized dataset: ", {dataset})
    return list(dict_product(parameters))


def step(params: Dict):
    # ================
    # DATA
    # ================
    dist_data = DataParams()

    # set params
    dist_data.dataset = params["dataset"]
    dist_data.trials = params["trials"]
    dist_data.std = params["std"]
    dist_data.nu = params["nu"]
    dist_data.samples = params["samples"]
    dist_data.dimensions = params["dimensions"]
    dist_data.standardize = params["standardize"]

    # generate data
    inputs = dist_data.generate_data()

    # ====================
    # Sigma Estimator
    # ====================

    # estimate sigma
    sigma_X = params["sigma_estimators"].estimate_sigma(X=inputs.X,)
    sigma_Y = params["sigma_estimators"].estimate_sigma(X=inputs.Y,)

    # ====================
    # HSIC Model
    # ====================

    # init hsic model class
    hsic_model = HSICModel(kernel_X=RBF(sigma_X), kernel_Y=RBF(sigma_Y))

    # get hsic score
    score = hsic_model.get_score(inputs.X, inputs.Y, params["scorers"])

    # ====================
    # Results
    # ====================

    # create dataframe
    results_df = pd.DataFrame(
        {
            # Data Params
            "dataset": [params["dataset"]],
            "trials": [params["trials"]],
            "std": [params["std"]],
            "nu": [params["nu"]],
            "samples": [params["samples"]],
            "dimensions": [params["dimensions"]],
            "standardize": [params["standardize"]],
            # Sigma Params
            "sigma_method": [params["sigma_estimators"].method],
            "sigma_percent": [params["sigma_estimators"].percent],
            "per_dimensions": [params["sigma_estimators"].per_dimension],
            "sigma_X": [sigma_X],
            "sigma_Y": [sigma_Y],
            # HSIC Params
            "scorer": [params["scorers"]],
            "score": [score],
            "mutual_info": [inputs.mutual_info],
        }
    )
    return results_df


def main(args):

    results_df = run_parallel_step(
        exp_step=step,
        parameters=get_parameters(args.dataset),
        n_jobs=args.njobs,
        verbose=args.verbose,
    )

    # save results
    results_df = pd.concat(results_df, ignore_index=True)
    results_df.to_csv(f"{RES_PATH}{args.save}_{args.dataset}.csv")


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
