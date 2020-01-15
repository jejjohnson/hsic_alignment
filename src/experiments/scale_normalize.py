import sys, os

# Insert path to model directory,.
cwd = os.getcwd()
path = f"/home/emmanuel/projects/2019_hsic_align/src"
sys.path.insert(0, path)


from typing import Tuple, Type, Optional

# Standard Packages
import pandas as pd
import numpy as np
import argparse
from sklearn.utils import check_random_state

# toy datasets
from data.toy import generate_dependence_data

# Kernel Dependency measure
from models.train_models import get_gamma_init
from models.train_models import get_hsic
from models.kernel import estimate_sigma, sigma_to_gamma, gamma_to_sigma, get_param_grid
from models.ite_algorithms import run_rbig_models
from sklearn.preprocessing import StandardScaler

# Logging
import logging


PROJECT_PATH = "/home/emmanuel/projects/2019_hsic_align/"
LOG_PATH = "src/experiments/logs/"
SAVE_PATH = "data/results/scaling/"


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


def get_params(case: int):

    # Case I - Unscaled, Unnormalized
    if case == 1:

        class DataParams:
            num_points = 2_000
            noise_y = 0.001
            alpha = 1.0
            beta = 1.0

        class ExpParams:
            dataset = [
                "line",
                "sine",
                "circ",
                # "rand"
            ]
            seed = np.linspace(1, 10, 10, dtype=int)
            scale = [1.0]
            normalized = [False]
            noise = np.logspace(-2, 1, 10)
            method = ["hsic", "tka", "ctka"]
            gamma_method = [
                ("median", 0.2, None),
                ("median", 0.4, None),
                ("median", 0.5, None),
                ("median", 0.6, None),
                ("median", 0.8, None),
            ]

    # Case II - Unscaled, Normalized
    elif case == 2:

        class DataParams:
            num_points = 2_000
            noise_y = 0.0
            alpha = 1.0
            beta = 1.0

        class ExpParams:
            dataset = [
                "line",
                "sine",
                "circ",
                # "rand"
            ]
            seed = np.linspace(1, 10, 10, dtype=int)
            scale = [1.0]
            normalized = [True]
            noise = np.logspace(-2, 1, 10)
            method = ["hsic", "tka", "ctka"]
            gamma_method = [
                ("median", 0.2, None),
                ("median", 0.4, None),
                ("median", 0.5, None),
                ("median", 0.6, None),
                ("median", 0.8, None),
            ]

    # Case III - Scaled, Unnormalized
    elif case == 3:

        class DataParams:
            num_points = 2_000
            noise_y = 0.01
            alpha = 1.0
            beta = 1.0

        class ExpParams:
            dataset = [
                "line",
                "sine",
                "circ",
                # "rand",
            ]
            seed = np.linspace(1, 10, 10, dtype=int)
            scale = np.logspace(-2, 2, 10)
            normalized = [False]
            noise = np.logspace(-2, 1, 10)
            method = ["hsic", "tka", "ctka"]
            gamma_method = [
                ("median", 0.2, None),
                ("median", 0.4, None),
                ("median", 0.5, None),
                ("median", 0.6, None),
                ("median", 0.8, None),
            ]

    elif case == 4:

        class DataParams:
            dataset = "line"
            num_points = 2_000
            noise_y = 0.01
            alpha = 1.0
            beta = 1.0

        class ExpParams:
            dataset = [
                "line",
                "sine",
                "circ",
                # "rand"
            ]
            seed = np.linspace(1, 10, 10, dtype=int)
            scale = np.logspace(-2, 2, 10)  # [0.01, 1.0, 100.0]
            normalized = [True]
            noise = np.logspace(-3, 1, 10)
            method = ["hsic", "tka", "ctka"]
            gamma_method = [
                ("median", 0.2, None),
                ("median", 0.4, None),
                ("median", 0.5, None),
                ("median", 0.6, None),
                ("median", 0.8, None),
            ]

    else:
        raise ValueError(f"Unrecognized case: '{case}'")

    return DataParams, ExpParams


class ScaleExperiment:
    def __init__(self, data_params, exp_params):
        self.data_params = data_params
        self.exp_params = exp_params

    def _get_data(
        self, dataset: str, noise: float, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Gathers the raw dependence data"""
        # get dataset
        X, Y = generate_dependence_data(
            dataset=dataset,
            num_points=self.data_params.num_points,
            seed=seed,
            noise_x=noise,
            noise_y=noise,
            alpha=self.data_params.alpha,
            beta=self.data_params.beta,
        )
        return X, Y

    def _apply_noise(
        self, X: np.ndarray, Y: np.ndarray, noise: float, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:

        rng = check_random_state(seed)

        X += rng.randn(X.shape[0], X.shape[1])

        return X, Y

    def _apply_scaling(self, X: np.ndarray, scale: float) -> np.ndarray:
        """The scaling step in our experiment"""
        # apply scaling
        return scale * X

    def _apply_normalization(
        self, X: np.ndarray, Y: np.ndarray, normalize: bool
    ) -> np.ndarray:
        """The normalization step in our experiment."""
        # apply normalization
        if normalize == True:
            X = StandardScaler().fit_transform(X)
            # Y = StandardScaler().fit_transform(Y)
        elif normalize == False:
            pass
        else:
            raise ValueError(f"Unrecognized boolean value for normalize {normalize}")
        return X, Y

    def _apply_mi_estimate(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Apply Mutual Information estimator. 
        We choose to use RBIG as our estimator."""
        # estimate mutual information
        mi, _ = run_rbig_models(X, Y, measure="mi", verbose=None)

        return mi

    def _apply_hsic_estimate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        method: str,
        gamma_init: Tuple[str, Optional[float], Optional[float]],
    ) -> float:
        """Apply HSIC estimator using one of the 3 algorithms:
        * HSIC
        * KA
        * cKA
        """
        # initialize the gamma parameter
        gamma_init = get_gamma_init(X, Y, gamma_init[0], gamma_init[1], gamma_init[2])

        # get hsic_value
        hsic_value = get_hsic(X, Y, method, gamma_init, maximum=False)

        return hsic_value

    def _experiment_step(
        self,
        results_df: pd.DataFrame,
        dataset: str,
        noise: float,
        seed: int,
        scale: float,
        normalize: bool,
        method: str,
        gamma_init: Tuple[str, Optional[float], Optional[float]],
    ) -> pd.DataFrame:

        # Step I - Extract Data
        X, Y = self._get_data(dataset=dataset, noise=noise, seed=seed)

        # Step II - Apply Scaling
        X = self._apply_scaling(X=X, scale=scale)

        # Step III - Apply Normalization
        X, Y = self._apply_normalization(X=X, Y=Y, normalize=normalize)

        # Step IV - Estimate mutual information
        mi = self._apply_mi_estimate(X, Y)

        # Step IV - Estimate HSIC value
        hsic_value = self._apply_hsic_estimate(X, Y, method, gamma_init)

        # Step V - Save Results to dataframe
        results_df = results_df.append(
            {
                "normalized": normalize,
                "trial": seed,
                "dataset": dataset,
                "scale": scale,
                "scorer": method,
                "gamma_method": get_gamma_name(gamma_init),
                "hsic_value": hsic_value,
                "mi": mi,
                "noise": noise,
            },
            ignore_index=True,
        )
        return results_df

    def run_experiment(self):

        results_df = pd.DataFrame()

        # Loop Through Free Parameters
        for idataset in self.exp_params.dataset:

            for iscale in self.exp_params.scale:
                print(f"Case: {args.case} | Dataset: {idataset} | Scale: {iscale}")
                for iseed in self.exp_params.seed:
                    for inoise in self.exp_params.noise:
                        for inormalize in self.exp_params.normalized:
                            for igamma in self.exp_params.gamma_method:
                                for imethod in self.exp_params.method:
                                    results_df = self._experiment_step(
                                        results_df=results_df,
                                        dataset=idataset,
                                        noise=inoise,
                                        seed=iseed,
                                        scale=iscale,
                                        normalize=inormalize,
                                        method=imethod,
                                        gamma_init=igamma,
                                    )

                                results_df.to_csv(
                                    PROJECT_PATH
                                    + SAVE_PATH
                                    + f"exp_scale_c{args.case}_{args.save}.csv"
                                )

        return results_df


def main(args):

    # get parameters
    DataParams, ExpParams = get_params(case=args.case)

    # Initialize Experiment Class
    exp_class = ScaleExperiment(DataParams, ExpParams)

    # Run Experiment
    results_df = exp_class.run_experiment()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="HSIC Align: Scale-Normalize Experiment"
    )
    parser.add_argument(
        "--case", type=int, default=123, help="The case for the experiment."
    )

    parser.add_argument("--save", type=str, default="v1", help="Save name.")

    args = parser.parse_args()

    main(args)
