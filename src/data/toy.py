import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as scio
import pandas as pd
from sklearn.utils import check_random_state
from typing import Tuple
from scipy.stats import norm, uniform, ortho_group, entropy as sci_entropy


class RBIGData(object):
    """This class extracts the toy data that was used to validate the
    credibility of RBIG for information measurements.
    
    The experimental parameters used:
        * n_samples
            [500, 1000, 5000, 10000, 30000, 50000]
        * d_dimensions
            [2, 3, 10, 50, 100]
        * t_trials
            [1, 2, 3, 4, 5]
    
    The Information theory measures:
        * Total Correlation (TC)
        * Entropy 
        * MultiDataset Multidimensional Mutual Information (MSMI)
        * Kullback-Leibler Divergence (KLD)
    
    Distributions uses:
        * Linear
        * Gaussian
        * T-Student
        
    Other IT measures compared:
        * K-Nearest Neighnors (KNN)
        * (KDP)
        * (vME)
        * (expF)
        * (ensemble)
        * MATLAB RBIG (rbig)
    
    Parameters
    ----------
    dataset : str, default = 'gauss',
        The toy data to be used to validate the information theory
        experiments.
        {'gauss', 'linear', 'tstudent'}
        
    info_meas : str, default = 'tc'
        The information theory measurement used on the toy data.
        {'tc', 'entropy', 'mi'}
        
    Information
    -----------
    Author: J. Emmanuel Johnson
    Date  : 11/01/2019
    Email : jemanjohnson34@gmail.com
    """

    def __init__(self, dataset="gauss", info_meas="tc"):
        self.data_location = (
            "/media/disk/erc/papers/2018_RBIG_IT_measures/"
            "2018_RBIG_IT_measures/reproducible_results/DATA/"
        )
        self.results_location = (
            "/media/disk/erc/papers/2018_RBIG_IT_measures/"
            "2018_RBIG_IT_measures/reproducible_results/RES/"
        )

        self.n_samples = [500, 1000, 5000, 10000, 30000, 50000]
        self.d_dimensions = [2, 3, 10, 50, 100]
        self.trials = [1, 2, 3, 4, 5]
        self.nu = [1, 2, 3, 4, 5]

        # Check Dataset
        if dataset in ["gauss"]:
            dataset = "gaus"

        elif dataset in ["linear"]:
            dataset = "lin"

        elif dataset in ["tstudent"]:
            dataset = "tstu"

        elif dataset in ["gauss_vs_gauss"]:
            dataset = "gaus_vs_gaus"

        elif dataset in ["tstudent_vs_gauss", "gauss_vs_tstudent"]:
            dataset = "ts_vs_gaus"

        elif dataset in ["tstudent_vs_tstudent"]:
            dataset = "ts_vs_ts"

        else:
            raise ValueError(f"Unrecognized 'dataset' param: {dataset}")

        self.dataset = dataset

        # Check Information Measure
        if info_meas in ["tc", "entropy", "mi"]:
            self.info_meas = info_meas.upper()

        elif info_meas in ["kld"]:
            self.info_meas = info_meas.upper()

        else:
            raise ValueError(f"Unrecognized 'info_meas' parameter: {info_meas}")

        pass

    def get_data(
        self,
        d_dimensions=2,
        n_samples=500,
        t_trials=1,
        nu=None,
        mu=None,
        return_results=False,
    ):

        # Get the base results
        data_name = self._get_data_name(d_dimensions, n_samples, t_trials, nu=nu, mu=mu)

        file_name = str(f"{self.info_meas}_{self.dataset}")
        full_file = self.data_location + file_name + "/" + data_name
        # check if path exists
        self._check_path(data_name, file_name, full_file)

        if self.info_meas.lower() in ["tc", "entropy"]:
            data = scio.loadmat(full_file)["dat"]
        elif self.info_meas.lower() in ["mi"]:
            data = {
                "X": scio.loadmat(full_file)["X"],
                "Y": scio.loadmat(full_file)["Y"],
            }
        elif self.info_meas.lower() in ["kld"]:
            data = {
                "X": scio.loadmat(full_file)["X"],
                "Y": scio.loadmat(full_file)["Y"],
            }
        else:
            raise ValueError(f"Unrecognized info measure: {self.info_meas}")

        if return_results:

            return data, self.get_results(d_dimensions, n_samples, t_trials)
        else:
            return data

    def _get_data_name(self, d_dimensions, n_samples, t_trials, nu, mu):
        if self.dataset in ["gaus", "lin"]:
            data_name = str(
                f"DATA_{self.info_meas}_{self.dataset}_nd_{d_dimensions}_Ns_{n_samples}_tryal_{t_trials}.mat"
            )

        elif self.dataset in ["gaus_vs_gaus"]:
            data_name = str(
                f"DATA_{self.info_meas}_{self.dataset}_mean_nd_{d_dimensions}_Ns_{n_samples}_tryal_{t_trials}_mu_{mu}.mat"
            )

        elif self.dataset in ["ts_vs_gaus"]:
            data_name = str(
                f"DATA_{self.info_meas}_{self.dataset}_nd_{d_dimensions}_Ns_{n_samples}_tryal_{t_trials}_mu_{mu}.mat"
            )

        elif self.dataset in ["tstu", "ts_vs_ts"]:

            data_name = str(
                f"DATA_{self.info_meas}_{self.dataset}_nd_{d_dimensions}_Ns_{n_samples}_tryal_{t_trials}_nu_{nu}.mat"
            )

        else:
            raise ValueError(f"Unrecognized info measure: {self.dataset}")

        return data_name

    def _check_path(self, data_name, file_name, full_file):

        if not Path(full_file).exists():
            raise ValueError(f"Path: \n{full_file}'\n does not exist.")
        else:
            return self

    def get_results(self, d_dimensions=2, n_samples=500, t_trials=1, nu=None, mu=None):

        results = dict()
        results["it_measure"] = self.info_meas.lower()
        results["dataset"] = self.dataset
        results["dimensions"] = d_dimensions
        results["n_samples"] = n_samples
        results["trials"] = t_trials
        if nu:
            results["nu"] = nu
        if mu:
            results["mu"] = mu

        # get indices for experiments
        dim_idx = self.d_dimensions.index(d_dimensions)
        sample_idx = self.n_samples.index(n_samples)
        trial_idx = self.trials.index(t_trials)

        # Get the base results
        data_name = self._get_data_name(d_dimensions, n_samples, t_trials, nu=nu, mu=mu)
        print(data_name)
        file_name = str(f"{self.info_meas}_{self.dataset}")
        full_file = self.data_location + file_name + "/" + data_name

        # check file
        self._check_path(data_name, file_name, full_file)
        if self.info_meas.lower() in ["tc"]:
            field = "TC"
        elif self.info_meas.lower() in ["entropy"]:
            field = "H"
        elif self.info_meas.lower() in ["mi"]:
            field = "MI"
        elif self.info_meas.lower() in ["kld"]:
            field = "KLD"

        else:
            raise ValueError(f"Unrecognized info_measure:{self.info_meas}")

        results["original"] = float(scio.loadmat(full_file)[f"{field}_ori_nats"])

        # get other methods results
        if self.dataset in ["gaus_vs_gaus"]:
            results_file = str(f"RES_{self.info_meas}_gauss_vs_gauss_mean.mat")
        elif self.dataset in ["ts_vs_gaus"]:
            results_file = str(f"RES_{self.info_meas}_ts_vs_gauss.mat")
        else:
            results_file = str(f"RES_{self.info_meas}_{self.dataset}.mat")

        full_file = self.results_location + results_file
        if self.dataset in ["gaus_vs_gaus", "ts_vs_gaus"]:

            results["knn_k"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_kNN_k"][
                    sample_idx, dim_idx, mu - 1, trial_idx
                ]
            )
            results["knn_kiTi"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_kNN_kiTi"][
                    sample_idx, dim_idx, mu - 1, trial_idx
                ]
            )
            results["vME"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_vME"][
                    sample_idx, dim_idx, mu - 1, trial_idx
                ]
            )
            results["szabo_expF"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_expF"][
                    sample_idx, dim_idx, mu - 1, trial_idx
                ]
            )
            results["rbig"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_rbig_nats"][
                    sample_idx, dim_idx, mu - 1, trial_idx
                ]
            )
            results["original"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_ori_nats"][
                    sample_idx, dim_idx, mu - 1, trial_idx
                ]
            )

        elif self.dataset in ["ts_vs_ts"]:
            results["knn_k"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_kNN_k"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["knn_kiTi"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_kNN_kiTi"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["vME"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_vME"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["szabo_expF"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_expF"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["rbig"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_rbig_nats"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["original"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_ori_nats"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )

        elif self.dataset in ["gaus", "lin"]:
            results["knn"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_kNN_k"][
                    sample_idx, dim_idx, trial_idx
                ]
            )
            results["kdp"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_KDP"][
                    sample_idx, dim_idx, trial_idx
                ]
            )
            results["vme"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_vME"][
                    sample_idx, dim_idx, trial_idx
                ]
            )
            results["expf"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_expF"][
                    sample_idx, dim_idx, trial_idx
                ]
            )
            results["ensemble"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_vME"][
                    sample_idx, dim_idx, trial_idx
                ]
            )
            results["rbig"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_rbig_nats"][
                    sample_idx, dim_idx, trial_idx
                ]
            )
        elif self.dataset in ["tstu"]:
            results["knn"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_kNN_k"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["kdp"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_KDP"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["vme"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_vME"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["expf"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_expF"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["ensemble"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_szabo_vME"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
            results["rbig"] = float(
                scio.loadmat(full_file)["RES"][f"{field}_rbig_nats"][
                    sample_idx, dim_idx, nu - 1, trial_idx
                ]
            )
        # Construct dataframe with all results and stats
        results = pd.DataFrame(results, index=[0])

        return results


def generate_dependence_data(
    dataset: str = "line",
    num_points: int = 1000,
    seed: int = 123,
    noise_x: float = 0.1,
    noise_y: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates sample datasets to go along with a demo for paper.
    Each dataset corresponds to a different measure of correlation
    and dependence.
    
    Parameters
    ----------
    dataset = str, optional (default='line')
        The dataset generated from the function.
        {'line', 'sine', 'circ', 'rand'}
        'line' : High Correlation, High Dependence (linear)
        'sine' : High Correlation, High Dependence (nonlinear)
        'circ' : High Correlation, Low Depedence
        'rand' : Low Correlation, Low Dependence
        
    num_points : int, optional (default=1000)
        Number points per variable generated.

    seed : int, default: 123
        the random seed for the noise

    noise_x : int, default = 0.1
        the amount of noise added to the X variable

    noise_y : int, default = 0.1
        the amount of noise added to the Y variable
    
    Returns
    -------

        
    """
    rng = check_random_state(seed - 1)
    rng_x = check_random_state(seed)
    rng_y = check_random_state(seed + 1)

    # Dataset I: High Correlation, High Depedence
    if dataset.lower() == "line":
        X = rng_x.rand(num_points, 1)
        Y = X + noise_y * rng_y.randn(num_points, 1)
    elif dataset.lower() == "sine":
        X = rng_x.rand(num_points, 1)
        Y = np.sin(2 * np.pi * X) + noise_y * rng_y.randn(num_points, 1)
    elif dataset.lower() == "circ":
        t = 2 * np.pi * rng.rand(num_points, 1)
        X = np.cos(t) + noise_x * rng_x.randn(num_points, 1)
        Y = np.sin(t) + noise_y * rng_y.randn(num_points, 1)
    elif dataset.lower() == "rand":
        X = rng_x.rand(num_points, 1)
        Y = rng_y.rand(num_points, 1)
    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")

    return X, Y


def entropy_marginal(data, bin_est="standard", correction=True):
    """Calculates the marginal entropy (the entropy per dimension) of a
    multidimensional dataset. Uses histogram bin counnts. Also features
    and option to add the Shannon-Miller correction.
    
    Parameters
    ----------
    data : array, (n_samples x d_dimensions)
    
    bin_est : str, (default='standard')
        The bin estimation method.
        {'standard', 'sturge'}
    
    correction : bool, default=True
    
    Returns
    -------
    H : array (d_dimensions)
    
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    """
    n_samples, d_dimensions = data.shape

    n_bins = bin_estimation(n_samples, rule=bin_est)

    H = np.zeros(d_dimensions)

    for idim in range(d_dimensions):
        # Get histogram (use default bin estimation)
        [hist_counts, bin_edges] = np.histogram(
            a=data[:, idim],
            bins=n_bins,
            range=(data[:, idim].min(), data[:, idim].max()),
        )

        # Calculate bin_centers from bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # get difference between the bins
        delta = bin_centers[3] - bin_centers[2]

        # Calculate the marginal entropy
        H[idim] = entropy(hist_counts, correction=correction) + np.log2(delta)

    return H


def entropy(hist_counts, correction=None):

    # MLE Estimator with Miller-Maddow Correction
    if not (correction is None):
        correction = 0.5 * (np.sum(hist_counts > 0) - 1) / hist_counts.sum()
    else:
        correction = 0.0

    # Plut in estimator of entropy with correction
    return sci_entropy(hist_counts, base=2) + correction


def bin_estimation(n_samples, rule="standard"):

    if rule is "sturge":
        n_bins = int(np.ceil(1 + 3.322 * np.log10(n_samples)))

    elif rule is "standard":
        n_bins = int(np.ceil(np.sqrt(n_samples)))

    else:
        raise ValueError(f"Unrecognized bin estimation rule: {rule}")

    return n_bins

