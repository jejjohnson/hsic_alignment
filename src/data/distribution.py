from typing import Optional, NamedTuple
import scipy.io as scio
import collections
from dataclasses import dataclass
import numpy as np


class Inputs(NamedTuple):
    """Helpful data holder which stores:
    
    X : np.ndarray, (n_samples, n_features)
    
    Y : np.ndarray, (n_samples, n_features)
    
    mutual_info : float
        the mutual information value"""

    X: np.ndarray
    Y: np.ndarray
    mutual_info: float


@dataclass
class DataParams:
    """A dataclass which holds all of the options to 
    generate datasets. 

    Parameters
    -------
    trials : int, default=1
        {1, 2, 3, 4, 5}

    samples : int, default=100
        {50, 100, 500, 1_000, 5_000}

    dimensions : int, default = 2
        {2, 3, 10, 50, 100}

    std : int, default=2
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
        
    nu : int, default = 2 
        {1, 2, 3, 4, 5, 6, 7, 8, 9}
    """

    dataset: str = "gauss"
    samples: int = 100
    dimensions: int = 2
    std: int = 1
    trial: int = 1
    nu: int = 1

    def __str__(self):
        return (
            f"Dataset: {self.dataset}"
            f"\nSamples: {self.samples}"
            f"\nDimensions: {self.dimensions}"
            f"\nStandard Deviation: {self.std}"
            f"\nNu: {self.nu}"
            f"\nTrial: {self.trial}"
        )

    def __repr__(self):
        return (
            f"Dataset: {self.dataset}"
            f"\nSamples: {self.samples}"
            f"\nDimensions: {self.dimensions}"
            f"\nStandard Deviation: {self.std}"
            f"\nNu: {self.nu}"
            f"\nTrial: {self.trial}"
        )

    def generate_data(self) -> Inputs:
        """Helper function to generate data using the 
        parameters above."""
        # initialize dataloader
        dataloader = DistributionData(distribution=self.dataset)

        # return dataset
        return dataloader.get_data(
            samples=self.samples,
            dimensions=self.dimensions,
            std=self.std,
            nu=self.nu,
            trial=self.trial,
        )


class DistributionData:
    """MI Data
    
    
    Dataset
    -------
    trials = 1:5
    samples = 50, 100, 500, 1_000, 5_000
    dimensions = 2, 3, 10, 50, 100
    std = 1:11
    nu = 1:9
    """

    def __init__(self, distribution: Optional["gauss"] = None) -> None:

        self.distribution = distribution
        self.data_path = "/media/disk/erc/papers/2019_HSIC_ALIGN/data/mi_distributions/"

        if self.distribution == "gauss":
            self.dist_path = f"{self.data_path}MI_gaus/"
        elif self.distribution == "tstudent":
            self.dist_path = f"{self.data_path}MI_tstu/"
        else:
            raise ValueError(f"Unrecognized Dataset: {distribution}")

    def get_data(self, samples=50, dimensions=2, std=1, trial=1, nu=1):

        if self.distribution == "gauss":
            dat = scio.loadmat(
                f"{self.dist_path}DATA_MI_gaus_nd_{dimensions}_"
                f"Ns_{samples}_std_{std}_tryal_{trial}.mat"
            )

        elif self.distribution == "tstudent":
            dat = scio.loadmat(
                f"{self.dist_path}DATA_MI_tstu_nd_{dimensions}_"
                f"Ns_{samples}_tryal_{trial}_nu_{nu}.mat"
            )

        else:
            raise ValueError(f"Unrecognized distribution '{self.distribution}'")

        return Inputs(
            X=dat["X"], Y=dat["Y"], mutual_info=float(dat["MI_ori_nats"][0][0])
        )
