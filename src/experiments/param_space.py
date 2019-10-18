import sys, os
import warnings
import tqdm
import random
import pandas as pd
import numpy as np
import argparse

# toy datasets
from data.toy import generate_dependence_data

# Kernel Dependency measure
from models.dependence import HSIC, train_rbf_hsic
from models.kernel import estimate_sigma, sigma_to_gamma, gamma_to_sigma, get_param_grid

# RBIG IT measures
from models.ite_algorithms import run_rbig_models

warnings.filterwarnings("ignore")  # get rid of annoying warnings

SAVE_PATH = "/home/emmanuel/projects/2019_hsic_align/results/hsic/"


class ExperimentGamma:
    def __init__(
        self,
        seed=123,
        n_trials=10,
        hsic_points=1000,
        n_noise=10,
        n_gamma=50,
        factor=2,
        mi_points=100_000,
        sigma_est="median",
        save_path=None,
        save_name="test",
    ):

        # fixed experimental params
        self.seed = seed
        self.hsic_points = hsic_points
        self.n_noise = n_noise
        self.n_gamma = n_gamma
        self.factor = factor
        self.mi_points = mi_points
        self.sigma_est = sigma_est
        self.n_trials = n_trials
        self.save_path = save_path
        self.save_name = save_name

        # free experimental params
        self.noise_params = np.logspace(-3, -0.3, n_noise)
        self.func_params = ["line", "sine", "circ", "rand"]
        self.scorers = ["hsic", "tka", "ctka"]
        self.seeds = [i for i in range(1, n_trials + 1)]

        # saved dataframe

        pass

    def run_experiment(self):

        # initialize results dataframe
        self.results_df = self.generate_results_df()

        # Loop through functions
        for ifunction in self.func_params:
            print(f"Function: {ifunction}")

            # Loop through noise parameters
            for inoise in self.noise_params:

                # Loop through random seeds
                for iseed in self.seeds:

                    # generate data for MI measure
                    X, Y = self._generate_data(inoise, ifunction, iseed, dataset="mi")

                    # calculate MI
                    mi_score, _ = run_rbig_models(
                        X, Y, measure="mi", verbose=None, random_state=self.seed
                    )

                    # =======================
                    # HSIC MEASURES
                    # =======================

                    # initialize sigma
                    init_sigma, sigma_params = self._get_init_sigmas(
                        X, Y, method=self.sigma_est
                    )

                    # convert sigma to gamma
                    init_gamma = sigma_to_gamma(init_sigma)
                    gamma_params = sigma_to_gamma(sigma_params)

                    # Loop through HSIC scoring methods
                    for hsic_method in self.scorers:

                        # Loop through gamma parameters
                        for igamma in gamma_params:

                            # generate data for MI measure
                            X, Y = self._generate_data(
                                inoise, ifunction, iseed, dataset="hsic"
                            )

                            # Calculate HSIC
                            hsic_score = self._get_hsic(
                                X, Y, init_gamma=igamma, scorer=hsic_method
                            )

                            # append results to results dataframe
                            self.results_df = self.append_results(
                                self.results_df,
                                ifunction,
                                iseed,
                                inoise,
                                init_gamma,
                                igamma,
                                hsic_method,
                                hsic_score,
                                mi_score,
                            )

                            # save results to csv
                            self.save_data(self.results_df)
        return self

    def _generate_data(self, noise, function, seed, dataset="hsic"):

        if dataset == "hsic":
            num_points = self.hsic_points
        elif dataset == "mi":
            num_points = self.mi_points
        else:
            raise ValueError(f"Unrecoginized dataset: {dataset}")

        # get dataset
        X, Y = generate_dependence_data(
            dataset=function,
            num_points=num_points,
            seed=seed,
            noise_x=noise,
            noise_y=noise,
        )
        return X, Y

    def _get_init_sigmas(self, X, Y, method=None):

        # check override for sigma estimator
        if method in ["belkin20", "belkin40", "belkin60", "belkin80"]:
            percent = float(method[-2:]) / 100
            method = "belkin"

            # estimate initialize sigma
            sigma_x = estimate_sigma(X, method=method, percent=percent)
            sigma_y = estimate_sigma(Y, method=method, percent=percent)

        elif method is None or method == "max":

            method = self.sigma_est
            # estimate initialize sigma
            sigma_x = estimate_sigma(X, method=method)
            sigma_y = estimate_sigma(Y, method=method)

        elif method in ["silverman", "scott", "median", "mean"]:

            sigma_x = estimate_sigma(X, method=method)
            sigma_y = estimate_sigma(Y, method=method)

        else:
            raise ValueError(f"Unrecognized method: {method}")

        # init overall sigma is mean between two
        init_sigma = np.mean([sigma_x, sigma_y])

        # get gamma params
        sigma_grid = get_param_grid(
            init_sigma=init_sigma, factor=self.factor, n_grid_points=self.n_gamma
        )

        return init_sigma, sigma_grid

    def _get_hsic(self, X, Y, scorer, init_gamma=None):

        # cross validated HSIC method for maximization

        clf_hsic = HSIC(
            gamma=init_gamma, kernel="rbf", random_state=self.seed, scorer=scorer
        )

        clf_hsic.fit(X, Y)

        # hsic value and kernel alignment score
        return clf_hsic.hsic_value

    def generate_results_df(self):
        return pd.DataFrame(
            columns=[
                "trial",
                "function",
                "noise",
                "init_gamma",
                "gamma",
                "scorer",
                "value",
                "mi",
            ]
        )

    def append_results(
        self,
        results_df,
        function,
        trial,
        noise,
        init_gamma,
        gamma,
        hsic_method,
        hsic_score,
        mi_score,
    ):
        # append data
        return results_df.append(
            {
                "function": function,
                "trial": trial,
                "noise": noise,
                "init_gamma": init_gamma,
                "gamma": gamma,
                "scorer": hsic_method,
                "value": hsic_score,
                "mi": mi_score,
            },
            ignore_index=True,
        )

    def load_data(self):
        pass

    def save_data(self, results_df):
        results_df.to_csv(f"{self.save_path}{self.save_name}.csv")


def main(args):

    clf_exp = ExperimentGamma(
        seed=args.seed,
        n_trials=args.ntrials,
        hsic_points=args.n_points,
        mi_points=args.n_points,
        n_noise=args.noise,
        n_gamma=args.gamma,
        factor=args.factor,
        sigma_est=args.sigma,
        save_path=SAVE_PATH,
        save_name=args.save,
    )

    # run full experiment
    clf_exp.run_experiment()
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HSIC Measures Experiment")
    parser.add_argument(
        "--seed", type=int, default=123, help="The random state for data generation."
    )
    parser.add_argument(
        "--ntrials", type=int, default=1, help="Number of trials for points generation."
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=10_000,
        help="Number of points for data generation.",
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=50,
        help="Number of points in noise parameter grid.",
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
        "--sigma", type=str, default="median", help="Sigma estimator to be used."
    )
    parser.add_argument(
        "--save", type=str, default="trial_v3", help="Save name for final data."
    )

    args = parser.parse_args()

    main(args)
