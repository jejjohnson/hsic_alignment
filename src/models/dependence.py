import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from scipy.spatial.distance import pdist
from scipy import stats


class HSIC(BaseEstimator):
    """Hilbert-Schmidt Independence Criterion (HSIC). This is
    a method for measuring independence between two variables.
    
    Parameters
    ----------
    gamma : float, optional (default=1.0)
        The length scale for the RBF kernel function for the X
        variable (gamma in sklearn)
    
    scorer : str, default='ctka'
        The method to score how well the sigma fits the two datasets.
        
        The following options are:
        * 'ctka': centered target kernel alignment
        * 'tka' : target kernel alignment
        * 'hsic': the hsic value

    random_state : int, optional (default=1234)
  
    Attributes
    ----------
    hsic_value : float
        The HSIC value is scored after fitting.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 14-Feb-2019
    
    Resources
    ---------
    Original MATLAB Implementation : 
        http:// isp.uv.es/code/shsic.zip
    Paper :
        Sensitivity maps of the Hilbert–Schmidt independence criterion
        Perez-Suay et al., 2018
    """

    def __init__(
        self,
        gamma: float = 1.0,
        random_state: Optional[int] = None,
        scorer: str = "tka",
        subsample: Optional[int] = 1000,
    ):
        self.gamma = gamma
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.scorer = scorer
        self.subsample = subsample

    def fit(self, X, Y):

        # Check sizes of X, Y
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)

        # Check samples are the same
        assert X.shape[0] == Y.shape[0]

        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]

        # subsample data if necessary
        if self.subsample is not None:
            X = self.rng.permutation(X)[: self.subsample, :]
            Y = self.rng.permutation(Y)[: self.subsample, :]

        self.X_train_ = X
        self.Y_train_ = Y

        # Calculate Kernel Matrices
        self.K_x = rbf_kernel(X, gamma=self.gamma)
        self.K_y = rbf_kernel(Y, gamma=self.gamma)

        # Center Kernel
        # H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples)
        # K_xc = K_x @ H
        self.K_xc = KernelCenterer().fit_transform(self.K_x)
        self.K_yc = KernelCenterer().fit_transform(self.K_y)

        # Compute HSIC value
        self.hsic_value = (1 / (self.n_samples - 1) ** 2) * np.einsum(
            "ji,ij->", self.K_xc, self.K_yc
        )
        return self

    def score(self, X, y=None):
        """This is not needed. It's only needed to comply with sklearn API.
        
        We will use the target kernel alignment algorithm as a score
        function. This can be used to find the best parameters."""

        if self.scorer == "tka":
            return kernel_alignment(self.K_x, self.K_y, center=False)

        elif self.scorer == "ctka":
            return kernel_alignment(self.K_xc, self.K_yc, center=False)

        elif self.scorer == "hsic":
            return (1 / (self.n_samples - 1) ** 2) * np.einsum(
                "ji,ij->", self.K_xc, self.K_yc
            )

        else:
            return self.hsic_value


def train_hsic(X: np.ndarray, Y: np.ndarray, scorer: str = "tka") -> dict:
    param_grid = {"gamma": np.logspace(-6, 6, 100)}

    clf_hsic = HSIC(scorer=scorer)

    clf_grid = GridSearchCV(clf_hsic, param_grid, iid=False, n_jobs=-1, verbose=0, cv=2)

    clf_grid.fit(X, Y)

    # print results
    print(f"Best {scorer.upper()} score: {clf_grid.best_score_:.3e}")
    print(f"HSIC: {clf_grid.best_estimator_.hsic_value:.3e}")
    print(f"gamma: {clf_grid.best_estimator_.gamma:.3f}")

    return clf_grid


def kernel_alignment(K_x: np.array, K_y: np.array, center: bool = False) -> float:
    """Gives a target kernel alignment score: how aligned the kernels are. Very
    useful for measures which depend on comparing two different kernels, e.g.
    Hilbert-Schmidt Independence Criterion (a.k.a. Maximum Mean Discrepency)
    
    Note: the centered target kernel alignment score is the same function
          with the center flag = True.
    
    Parameters
    ----------
    K_x : np.array, (n_samples, n_samples)
        The first kernel matrix, K(X,X')
    
    K_y : np.array, (n_samples, n_samples)
        The second kernel matrix, K(Y,Y')
        
    center : Bool, (default: False)
        The option to center the kernels (independently) before hand.
    
    Returns
    -------
    kta_score : float,
        (centered) target kernel alignment score.
    """

    # center kernels
    if center:
        K_x = KernelCenterer().fit_transform(K_x)
        K_y = KernelCenterer().fit_transform(K_y)

    # target kernel alignment
    return np.sum(K_x * K_y) / np.linalg.norm(K_x) / np.linalg.norm(K_y)

