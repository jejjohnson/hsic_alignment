import sys, os
import warnings
import tqdm
import random
import pandas as pd
import numpy as np
import argparse
from sklearn.utils import check_random_state

# toy datasets
from data.toy import RBIGData

# Kernel Dependency measure
from models.dependence import HSIC, train_rbf_hsic
from models.kernel import estimate_sigma, sigma_to_gamma, gamma_to_sigma, get_param_grid

# RBIG IT measures
from models.ite_algorithms import run_rbig_models

warnings.filterwarnings("ignore")  # get rid of annoying warnings

SAVE_PATH = "/home/emmanuel/projects/2019_hsic_align/results/hsic/"


class LargeScaleKTA:
    def __init__(
        self,
        seed=123,
        n_gamma=100,
        n_noise=10,
        factor=1,
        sigma_est="silverman",
        save_path=None,
        save_name="scale_test",
    ):

        # fixed experimental params
        self.seed = seed
        self.rng_x = check_random_state(seed)
        self.rng_y = check_random_state(seed + 1)
        self.n_gamma = n_gamma
        self.factor = factor
        self.sigma_est = sigma_est
        self.save_path = save_path
        self.save_name = save_name

        self.d_dimensions = [2, 3, 10, 50, 100]
        # free experimental params
        self.scorers = ["hsic", "tka", "ctka"]
        self.noise_params = np.logspace(-3, -0.3, n_noise)
        self.datasets = ["tstudent", "gauss"]
        self.nus = [1, 2, 3]
        self.trials = [1, 2, 3, 4, 5]
        self.n_samples = [
            500,
            1_000,
            5_000,
            # 10_000,
            # 30_000,
            # 50_000
        ]
        # saved dataframe

        pass

    def run_experiment(self):

        # initialize results dataframe
        self.results_df = self.generate_results_df()

        # Loop through datasets
        for idataset in self.datasets:
            print(f"Function: {idataset}")

            # Loop through samples
            for isample in self.n_samples:
                for idimension in self.d_dimensions:

                    # Loop through random seeds
                    for itrial in self.trials:
                        for inoise in self.noise_params:
                            if idataset == "tstudent":

                                for inu in self.nus:

                                    X, Y = self._get_data(
                                        idataset, "mi", idimension, isample, itrial, inu
                                    )

                                    # Add noise
                                    X += inoise * self.rng_x.rand(
                                        X.shape[0], X.shape[1]
                                    )
                                    Y += inoise * self.rng_x.rand(
                                        Y.shape[0], Y.shape[1]
                                    )

                                    # estimate initial sigmas and save
                                    gamma_median = self._get_init_sigmas(
                                        X, Y, method="median"
                                    )
                                    gamma_silv = self._get_init_sigmas(
                                        X, Y, method="silverman"
                                    )
                                    gamma_scott = self._get_init_sigmas(
                                        X, Y, method="scott"
                                    )

                                    # Loop through HSIC scoring methods
                                    for hsic_method in self.scorers:

                                        # =======================
                                        # HSIC MEASURES
                                        # =======================

                                        # Calculate HSIC
                                        hsic_score, gamma = self._get_hsic(
                                            X, Y, hsic_method
                                        )

                                        # append results to results dataframe
                                        self.results_df = self.append_results(
                                            results_df=self.results_df,
                                            dataset=idataset,
                                            trial=itrial,
                                            n_samples=isample,
                                            d_dimensions=idimension,
                                            noise=inoise,
                                            gamma_median=gamma_median,
                                            gamma_silv=gamma_silv,
                                            gamma_scott=gamma_scott,
                                            nu=inu,
                                            gamma=gamma,
                                            hsic_method=hsic_method,
                                            hsic_score=hsic_score,
                                        )

                                        # save results to csv
                                        self.save_data(self.results_df)

                            elif idataset == "gauss":
                                X, Y = self._get_data(
                                    idataset, "mi", idimension, isample, itrial, None
                                )

                                # =======================
                                # HSIC MEASURES
                                # =======================

                                # Loop through HSIC scoring methods
                                for hsic_method in self.scorers:

                                    # =======================
                                    # HSIC MEASURES
                                    # =======================

                                    # Calculate HSIC
                                    hsic_score, gamma = self._get_hsic(
                                        X, Y, hsic_method
                                    )

                                    # append results to results dataframe
                                    self.results_df = self.append_results(
                                        results_df=self.results_df,
                                        dataset=idataset,
                                        trial=itrial,
                                        n_samples=isample,
                                        d_dimensions=idimension,
                                        noise=inoise,
                                        gamma_median=gamma_median,
                                        gamma_silv=gamma_silv,
                                        gamma_scott=gamma_scott,
                                        nu=np.nan,
                                        gamma=gamma,
                                        hsic_method=hsic_method,
                                        hsic_score=hsic_score,
                                    )

                                    # save results to csv
                                    self.save_data(self.results_df)

                            else:
                                raise ValueError(f"Unrecognized dataset: {idataset}")

        return self

    def _get_data(self, dataset, info_meas, dimensions, samples, trials, nu):

        # initialize dataset Generator
        clf_rbigdata = RBIGData(dataset=dataset, info_meas=info_meas)

        data = clf_rbigdata.get_data(
            d_dimensions=dimensions, n_samples=samples, t_trials=trials, nu=nu
        )
        return data["X"], data["Y"]

    def _get_init_sigmas(self, X, Y, method=None):

        # check override for sigma estimator
        if method is None:
            method = self.sigma_est

        # estimate initialize sigma
        sigma_x = estimate_sigma(X, method=method)
        sigma_y = estimate_sigma(Y, method=method)

        # init overall sigma is mean between two
        init_sigma = np.mean([sigma_x, sigma_y])

        return init_sigma

    def _get_hsic(self, X, Y, scorer):

        # hsic params
        kernel = "rbf"
        subsample = None
        bias = True

        # initialize HSIC calculator
        clf_hsic = HSIC(kernel=kernel, scorer=scorer, subsample=subsample, bias=bias)

        # calculate HSIC return scorer
        clf_hsic = train_rbf_hsic(
            X,
            Y,
            clf_hsic,
            n_gamma=self.n_gamma,
            factor=self.factor,
            sigma_est="mean",
            verbose=0,
            n_jobs=-1,
            cv=3,
        )
        # hsic value and kernel alignment score
        return clf_hsic.hsic_value, clf_hsic.gamma

    def generate_results_df(self):
        return pd.DataFrame(
            columns=[
                "dataset",
                "trial",
                "n_samples",
                "d_dimensions",
                "nu",
                "gamma",
                "scorer",
                "value",
            ]
        )

    def append_results(
        self,
        results_df,
        dataset,
        trial,
        n_samples,
        d_dimensions,
        noise,
        nu,
        gamma,
        gamma_median,
        gamma_silv,
        gamma_scott,
        hsic_method,
        hsic_score,
    ):
        # append data
        return results_df.append(
            {
                "dataset": dataset,
                "trial": trial,
                "n_samples": n_samples,
                "d_dimensions": d_dimensions,
                "noise": noise,
                "nu": nu,
                "gamma": gamma,
                "gamma_median": gamma_median,
                "gamma_silv": gamma_silv,
                "gamma_scott": gamma_scott,
                "scorer": hsic_method,
                "value": hsic_score,
            },
            ignore_index=True,
        )

    def load_data(self):
        pass

    def save_data(self, results_df):
        results_df.to_csv(f"{self.save_path}{self.save_name}.csv")


def main(args):

    clf_exp = LargeScaleKTA(
        seed=args.seed,
        factor=args.factor,
        sigma_est=args.sigma,
        n_noise=args.noise,
        n_gamma=args.gamma,
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
        "--gamma",
        type=int,
        default=50,
        help="Number of points in gamma parameter grid.",
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=20,
        help="Amount of points for the noise parameter.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Factor to be used for bounds for gamma parameter.",
    )
    parser.add_argument(
        "--sigma",
        type=str,
        choices=["median", "mean", "silverman", "scott"],
        default="median",
        help="Sigma estimator to be used.",
    )
    parser.add_argument(
        "--save", type=str, default="dist_v4_median", help="Save name for final data."
    )

    args = parser.parse_args()

    main(args)
