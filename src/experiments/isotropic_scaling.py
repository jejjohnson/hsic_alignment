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
from models.dependence import HSIC
from models.kernel import estimate_sigma, sigma_to_gamma, gamma_to_sigma, get_param_grid

# RBIG IT measures
from models.ite_algorithms import run_rbig_models

warnings.filterwarnings("ignore")  # get rid of annoying warnings

SAVE_PATH = "/home/emmanuel/projects/2019_hsic_align/results/hsic/"


class ExperimentScale:
    def __init__(
        self,
        seed=123,
        n_trials=10,
        hsic_points=1000,
        n_noise=10,
        n_gamma=50,
        n_scale=5,
        factor=2,
        mi_points=100_000,
        sigma_est="silverman",
        save_path=None,
        save_name="test",
    ):

        # fixed experimental params
        self.seed = seed
        self.hsic_points = hsic_points
        self.n_noise = n_noise
        self.n_gamma = n_gamma
        self.n_scale = n_scale
        self.factor = factor
        self.mi_points = mi_points
        self.sigma_est = sigma_est
        self.n_trials = n_trials
        self.save_path = save_path
        self.save_name = save_name

        # free experimental params
        self.noise_params = np.logspace(-3, -0.3, n_noise)
        self.scale_params = np.linspace(1, 100, n_scale)
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

                for iscale in self.scale_params:
                    # Loop through random seeds
                    for iseed in self.seeds:

                        # generate data for MI measure
                        X, Y = self._generate_data(
                            inoise, ifunction, iseed, dataset="mi"
                        )

                        X *= iscale

                        # calculate MI
                        mi_score, _ = run_rbig_models(
                            X, Y, measure="mi", verbose=None, random_state=self.seed
                        )

                        # =======================
                        # HSIC MEASURES
                        # =======================

                        # initialize sigma
                        init_sigma, sigma_params = self._estimate_sigmas(X, Y)

                        # convert to gamma
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
                                    X,
                                    Y,
                                    gamma=igamma,
                                    subsample=None,
                                    scorer=hsic_method,
                                )

                                # append results to results dataframe
                                self.results_df = self.append_results(
                                    results_df=self.results_df,
                                    function=ifunction,
                                    trial=iseed,
                                    noise=inoise,
                                    init_gamma=init_gamma,
                                    gamma=igamma,
                                    hsic_method=hsic_method,
                                    hsic_score=hsic_score,
                                    mi_score=mi_score,
                                    scale=iscale,
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

    def _estimate_sigmas(self, X, Y):

        # estimate initialize sigma
        sigma_x = estimate_sigma(X, method=self.sigma_est)
        sigma_y = estimate_sigma(Y, method=self.sigma_est)

        # init overall sigma is mean between two
        init_sigma = np.mean([sigma_x, sigma_y])

        # get parameter grid
        sigmas = get_param_grid(init_sigma, self.factor, self.n_gamma)

        # convert to gammas

        return init_sigma, sigmas

    def _get_hsic(self, X, Y, gamma, subsample, scorer):

        # initialize hsic
        clf_hsic = HSIC(kernel="rbf", gamma=gamma, scorer=scorer, subsample=subsample)

        # calculate HSIC value
        clf_hsic.fit(X, Y)

        # return HSIC score
        return clf_hsic.score(X)

    def generate_results_df(self):
        return pd.DataFrame(
            columns=[
                "trial",
                "function",
                "noise",
                "init_gamma",
                "gamma",
                "scale",
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
        scale,
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
                "scale": scale,
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

    clf_exp = ExperimentScale(
        seed=args.seed,
        n_trials=args.ntrials,
        hsic_points=args.n_points,
        mi_points=args.n_points,
        n_noise=args.noise,
        n_gamma=args.gamma,
        n_scale=args.scale,
        factor=args.factor,
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
        "--scale", type=int, default=5, help="Number of points in the scaling grid."
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
        "--save", type=str, default="scale_v1", help="Save name for final data."
    )

    args = parser.parse_args()

    main(args)