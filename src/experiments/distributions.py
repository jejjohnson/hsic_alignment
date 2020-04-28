import argparse
import os
import random
import sys
import warnings
from typing import Dict, Optional, Tuple

# Plotting Procedures
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.gaussian_process.kernels import RBF
# Kernel Dependency measure
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# toy datasets
from data.distribution import DataParams, Inputs
# experiment helpers
from experiments.utils import dict_product, run_parallel_step
from models.dependence import HSICModel
# Plotting
from visualization.distribution import plot_scorer

# Insert path to model directory...
cwd = os.getcwd()
project_path = f"/home/emmanuel/projects/2019_hsic_align/src"
sys.path.insert(0, project_path)

# Insert path to package,.
pysim_path = f"/home/emmanuel/code/pysim/"
sys.path.insert(0, pysim_path)



random.seed(123)



# from pysim.kernel.utils import estimate_sigma


# RBIG IT measures
# from models.ite_algorithms import run_rbig_models




RES_PATH = (
    "/home/emmanuel/projects/2019_hsic_align/data/results/distributions/mutual_info/"
)


def get_parameters(
    dataset: str = "gauss", shuffle: bool = True, njobs: Optional = None
) -> Dict:
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
            *[("median", x) for x in [10, 20, 30, 40, 50, 60, 70, 80, 90]],
        ],
    }

    # add specific params
    if dataset == "gauss":
        # standard dataset parameters
        params["trial"] = [1, 2, 3, 4, 5]
        params["dimensions"] = [2, 3, 10, 50, 100]
        params["dataset"] = ["gauss"]
        params["std"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        params["nu"] = [1]
        # standard dataset parameters

    elif dataset == "tstudent":
        # standard dataset parameters
        params["trial"] = [1, 2, 3, 4, 5]
        params["dimensions"] = [2, 3, 10, 50, 100]
        params["dataset"] = ["tstudent"]
        params["std"] = [1]
        params["nu"] = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    else:
        raise ValueError("Unrecognized dataset: ", {dataset})

    # Loop Params
    loop_params = {}
    loop_params["samples"] = [50, 100, 500, 1_000, 5_000]

    # shuffle parameters
    params = list(dict_product(params))
    loop_params = list(dict_product(loop_params))
    if shuffle:
        params = random.sample(params, len(params))
    return params, loop_params




def scotts_factor(X: np.ndarray) -> float:
    """Scotts Method to estimate the length scale of the 
    rbf kernel.
    
        factor = n**(-1./(d+4))
    
    Parameters
    ----------
    X : np.ndarry
        Input array
    
    Returns
    -------
    factor : float
        the length scale estimated
    
    """
    n_samples, n_features = X.shape

    return np.power(n_samples, -1 / (n_features + 4.0))


def silvermans_factor(X: np.ndarray) -> float:
    """Silvermans method used to estimate the length scale
    of the rbf kernel.
    
    factor = (n * (d + 2) / 4.)**(-1. / (d + 4)).
    
    Parameters
    ----------
    X : np.ndarray,
        Input array
    
    Returns
    -------
    factor : float
        the length scale estimated
    """
    n_samples, n_features = X.shape

    base = (n_samples * (n_features + 2.0)) / 4.0

    return np.power(base, -1 / (n_features + 4.0))


def kth_distance(dists: np.ndarray, percent: float) -> np.ndarray:

    if isinstance(percent, float):
        percent /= 100

    # kth distance calculation (50%)
    kth_sample = int(percent * dists.shape[0])

    # take the Kth neighbours of that distance
    k_dist = dists[:, kth_sample]

    return k_dist


def sigma_estimate(
    X: np.ndarray,
    method: str = "median",
    percent: Optional[int] = None,
    heuristic: bool = False,
) -> float:

    # get the squared euclidean distances
    if method == "silverman":
        return silvermans_factor(X)
    elif method == "scott":
        return scotts_factor(X)
    elif percent is not None:
        kth_sample = int((percent / 100) * X.shape[0])
        dists = np.sort(squareform(pdist(X, "sqeuclidean")))[:, kth_sample]
    else:
        dists = np.sort(pdist(X, "sqeuclidean"))

    if method == "median":
        sigma = np.median(dists)
    elif method == "mean":
        sigma = np.mean(dists)
    else:
        raise ValueError(f"Unrecognized distance measure: {method}")

    if heuristic:
        sigma = np.sqrt(sigma / 2)
    return sigma


def step(
    params: Dict, loop_param: Dict,
):

    # ================
    # DATA
    # ================
    dist_data = DataParams(
        dataset=params["dataset"],
        trial=params["trial"],
        std=params["std"],
        nu=params["nu"],
        samples=loop_param["samples"],
        dimensions=params["dimensions"],
    )

    # generate data
    inputs = dist_data.generate_data()

    # ========================
    # Estimate Sigma
    # ========================
    f_x = lambda x: sigma_estimate(
        x,
        method=params["sigma_estimator"][0],
        percent=params["sigma_estimator"][1],
        heuristic=False,
    )

    # ========================
    # Per Dimension
    # ========================
    if params["per_dimension"]:
        sigma_X = [f_x(ifeature.reshape(-1, 1)) for ifeature in inputs.X.T]
        sigma_Y = [f_x(ifeature.reshape(-1, 1)) for ifeature in inputs.Y.T]

    else:
        sigma_X = f_x(inputs.X)
        sigma_Y = f_x(inputs.Y)

    # ========================
    # Separate Length Scales
    # ========================
    # print(params)
    # print(sigma_X, sigma_Y)
    if params["separate_scales"] is True:
        sigma_X = np.mean([np.mean(sigma_X), np.mean(sigma_Y)])
        sigma_Y = np.mean([np.mean(sigma_X), np.mean(sigma_Y)])

    # =========================
    # Estimate HSIC
    # =========================
    hsic_clf = HSICModel(kernel_X=RBF(sigma_X), kernel_Y=RBF(sigma_Y),)

    score = hsic_clf.get_score(inputs.X, inputs.Y, params["scorer"])

    # ====================
    # Results
    # ====================

    # append results to dataframe
    results_df = pd.DataFrame(
        {
            # Data Params
            "dataset": [params["dataset"]],
            "trial": [params["trial"]],
            "std": [params["std"]],
            "nu": [params["nu"]],
            "samples": [loop_param["samples"]],
            "dimensions": [params["dimensions"]],
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
    params, loop_params = get_parameters(
        args.dataset, njobs=args.njobs, shuffle=args.shuffle
    )

    if args.smoke_test:
        iparams = params[0]
        iloop_param = loop_params[0]
        _ = step(iparams, iloop_param)

    # initialize datast
    else:
        header = True
        mode = "w"
        with tqdm(loop_params) as pbar:
            for iparam in pbar:

                pbar.set_description(
                    f"# Samples: {iparam['samples']}, Tasks: {len(params)}"
                )

                results_df = run_parallel_step(
                    exp_step=step,
                    parameters=params,
                    n_jobs=args.njobs,
                    verbose=args.verbose,
                    loop_param=iparam,
                )

                # concat current results
                results_df = pd.concat(results_df, ignore_index=True)

                # save results
                with open(f"{RES_PATH}{args.save}_{args.dataset}.csv", mode) as f:
                    results_df.to_csv(f, header=header)

                header = False
                mode = "a"
                del results_df


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
    parser.add_argument("-sm", "--smoke_test", action="store_true")
    parser.add_argument("-r", "--shuffle", action="store_true")
    args = parser.parse_args()

    main(args)
