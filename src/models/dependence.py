import sys, os

# Insert path to package,.
pysim_path = f"/home/emmanuel/code/pysim/"
sys.path.insert(0, pysim_path)

# Kernel Dependency measure
from pysim.kernel.hsic import HSIC
import numpy as np
from typing import Optional, Callable, Union, List, Tuple
from dataclasses import dataclass


@dataclass
class HSICModel:
    """HSICModel as a form of a dataclass with a score method.
    
    This initializes the parameters and then fits to some input
    data and gives out a score.
    
    
    Parameters
    ----------
    kernel : str, default='rbf'
        the kernel function 

    bias : bool, default=True
        option for the bias term (only affects the HSIC method)

    gamma_X : float, (optional, default=None)
        gamma parameter for the kernel matrix of X

    gamma_Y : float, (optional, default=None)
        gamma parameter for the kernel matrix of Y

    subsample : int, (optional, default=None)
        option to subsample the X, Y
    
    Example
    --------
    >> from src.models.dependence import HSICModel
    >> from sklearn import datasets
    >> X, _ = datasets.make_blobs(n_samples=1_000, n_features=2, random_state=123)
    >> Y, _ = datasets.make_blobs(n_samples=1_000, n_features=2, random_state=1)
    >> hsic_model = HSICModel()
    >> hsic_model.gamma_X = 100 
    >> hsic_model.gamma_Y = 0.01
    >> hsic_score = hsic_model.score(X, Y, 'hsic')
    >> print(hsic_score)
    0.00042726396990996684
    """

    kernel: str = "rbf"
    bias: bool = True
    gamma_X: Optional[float] = None
    gamma_Y: Optional[float] = None
    subsample: Optional[int] = None

    def get_score(
        self, X: np.ndarray, Y: np.ndarray, method: str = "hsic", **kwargs
    ) -> float:
        """method to get the HSIC score
        
        Parameters
        ----------
        X : np.ndarray, (n_samples, n_features)
        
        Y : np.ndarray, (n_samples, n_features)
        
        method : str, default = 'hsic'
            {'hsic', 'ka', 'cka'}
        
        kwargs : dict, (optional)
        
        Returns
        -------
        score : float
            the score based on the hsic method proposed above
        """
        if method == "hsic":
            # change params for HSIC
            self.normalize = False
            self.center = True
        elif method == "ka":
            # change params for Kernel Alignment
            self.normalize = True
            self.center = False
        elif method == "cka":
            # change params for centered Kernel Alignment
            self.normalize = True
            self.center = True
        else:
            raise ValueError(f"Unrecognized hsic method: {method}")

        # initialize HSIC model
        clf_hsic = HSIC(
            kernel=self.kernel,
            center=self.center,
            subsample=self.subsample,
            bias=self.bias,
            gamma_X=self.gamma_X,
            gamma_Y=self.gamma_Y,
            **kwargs,
        )

        # calculate HSIC return scorer
        clf_hsic.fit(X, Y)

        # return score
        return clf_hsic.score(X, normalize=self.normalize)
