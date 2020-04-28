import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

# Kernel Dependency measure
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.metrics.pairwise import linear_kernel, pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import check_array, check_random_state

from src.models.utils import estimate_gamma

# Insert path to package,.
pysim_path = f"/home/emmanuel/code/pysim/"
sys.path.insert(0, pysim_path)



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

    kernel_X: str = "rbf"
    kernel_Y: Optional[str] = "rbf"
    bias: bool = True
    gamma_X: Optional[float] = None
    gamma_Y: Optional[float] = None
    subsample: Optional[int] = None
    kernel_kwargs_X: Optional[Dict] = None
    kernel_kwargs_Y: Optional[Dict] = None

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
            kernel_X=self.kernel_X,
            kernel_Y=self.kernel_Y,
            center=self.center,
            subsample=self.subsample,
            bias=self.bias,
            gamma_X=self.gamma_X,
            gamma_Y=self.gamma_Y,
            kernel_params_X=self.kernel_kwargs_X,
            kernel_params_Y=self.kernel_kwargs_Y,
            **kwargs,
        )

        # calculate HSIC return scorer
        clf_hsic.fit(X, Y)

        # return score
        return clf_hsic.score(X, normalize=self.normalize)






class HSIC(BaseEstimator):
    """Hilbert-Schmidt Independence Criterion (HSIC). This is
    a method for measuring independence between two variables.

    Methods in the Literature

    * HSIC - centered, unnormalized
    * KA - uncentered, normalized
    * CKA - centered, normalized
    
    Parameters
    ----------
    center : bool, default=True
        The option to center the kernel matrices after construction
    
    bias : bool, default=True
        To add the bias for the scaling. Only necessary if calculating HSIC alone.

    subsample : int, default=None
        The option to subsample the data.

    random_state : int, default=123
        The random state for the subsample.

    kernel : string or callable, default="linear"
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number. Set to "precomputed" in
        order to pass a precomputed kernel matrix to the estimator
        methods instead of samples.
    
    gamma_X : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Used only by the X parameter. 
        Interpretation of the default value is left to the kernel; 
        see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.
    
    
    gamma_Y : float, default=None
        The same gamma parameter as the X. If None, the gamma will be estimated.
    
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    
    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.
    
    Attributes
    ----------
    hsic_value : float
        The HSIC value is scored after fitting.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 14-Feb-2019

    Example
    -------
    >> samples, features = 100, 50
    >> X = np.random.randn(samples, features)
    >> A = np.random.rand(features, features)
    >> Y = X @ A
    >> hsic_clf = HSIC(center=True, kernel='linear')
    >> hsic_clf.fit(X, Y)
    >> cka_score = hsic_clf.score(X)
    >> print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {cka_score:.3f}")
    """

    def __init__(
        self,
        gamma_X: Optional[float] = None,
        gamma_Y: Optional[float] = None,
        kernel_X: Optional[Union[Callable, str]] = "linear",
        kernel_Y: Optional[Union[Callable, str]] = "linear",
        degree: float = 3,
        coef0: float = 1,
        kernel_params_X: Optional[dict] = None,
        kernel_params_Y: Optional[dict] = None,
        random_state: Optional[int] = None,
        center: Optional[int] = True,
        subsample: Optional[int] = None,
        bias: bool = True,
    ):
        self.gamma_X = gamma_X
        self.gamma_Y = gamma_Y
        self.center = center
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params_X = kernel_params_X
        self.kernel_params_Y = kernel_params_Y
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.subsample = subsample
        self.bias = bias

    def fit(self, X, Y):

        # Check sizes of X, Y
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)

        # Check samples are the same
        assert (
            X.shape[0] == Y.shape[0]
        ), f"Samples of X ({X.shape[0]}) and Samples of Y ({Y.shape[0]}) are not the same"

        self.n_samples = X.shape[0]
        self.dx_dimensions = X.shape[1]
        self.dy_dimensions = Y.shape[1]

        # subsample data if necessary
        if self.subsample is not None:
            X = self.rng.permutation(X)[: self.subsample, :]
            Y = self.rng.permutation(Y)[: self.subsample, :]

        self.X_train_ = X
        self.Y_train_ = Y

        # Calculate the kernel matrices
        K_x = self.compute_kernel(
            X, kernel=self.kernel_X, gamma=self.gamma_X, params=self.kernel_params_X
        )
        K_y = self.compute_kernel(
            Y, kernel=self.kernel_Y, gamma=self.gamma_Y, params=self.kernel_params_Y
        )

        # Center Kernel
        # H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples)
        # K_xc = K_x @ H
        if self.center == True:
            K_x = KernelCenterer().fit_transform(K_x)
            K_y = KernelCenterer().fit_transform(K_y)

        # Compute HSIC value
        self.hsic_value = np.sum(K_x * K_y)

        # Calculate magnitudes
        self.K_x_norm = np.linalg.norm(K_x)
        self.K_y_norm = np.linalg.norm(K_y)

        return self

    def compute_kernel(self, X, Y=None, kernel="rbf", gamma=None, params=None):
        # check if kernel is callable
        if callable(kernel) and params == None:
            params = {}
        else:
            # estimate the gamma parameter
            if gamma is None:
                gamma = estimate_gamma(X)

            # set parameters for pairwise kernel
            params = {"gamma": gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=kernel, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def score(self, X, y=None, normalize=False):
        """This is not needed. It's only needed to comply with sklearn API.
        
        We will use the target kernel alignment algorithm as a score
        function. This can be used to find the best parameters."""

        if normalize == True:

            return self.hsic_value / self.K_x_norm / self.K_y_norm

        elif normalize == False:

            if self.bias:
                self.hsic_bias = 1 / (self.n_samples ** 2)
            else:
                self.hsic_bias = 1 / (self.n_samples - 1) ** 2

            return self.hsic_bias * self.hsic_value
        else:
            raise ValueError(f"Unrecognized normalize argument: {normalize}")


class RandomizedHSIC(BaseEstimator):
    """Hilbert-Schmidt Independence Criterion (HSIC). This is
    a method for measuring independence between two variables. This method
    uses the Nystrom method as an approximation to the large kernel matrix.
    Typically this works really well as it is data-dependent; thus it will
    converge to the real kernel matrix as the number of components increases.

    Methods in the Literature

    * HSIC - centered, unnormalized
    * KA - uncentered, normalized
    * CKA - centered, normalized
    
    Parameters
    ----------
    center : bool, default=True
        The option to center the kernel matrices after construction
    
    bias : bool, default=True
        To add the bias for the scaling. Only necessary if calculating HSIC alone.

    subsample : int, default=None
        The option to subsample the data.

    random_state : int, default=123
        The random state for the subsample.

    kernel : string or callable, default="linear"
        Kernel mapping used internally. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number. Set to "precomputed" in
        order to pass a precomputed kernel matrix to the estimator
        methods instead of samples.
    
    gamma_X : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Used only by the X parameter. 
        Interpretation of the default value is left to the kernel; 
        see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.
    gamma_Y : float, default=None
        The same gamma parameter as the X. If None, the same gamma_X will be
        used for the Y.
    
    degree : float, default=3
        Degree of the polynomial kernel. Ignored by other kernels.
    
    coef0 : float, default=1
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.
    
    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.
    
    Attributes
    ----------
    hsic_value : float
        The HSIC value is scored after fitting.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 14-Feb-2019

    Example
    -------
    >> samples, features, components = 100, 50, 10
    >> X = np.random.randn(samples, features)
    >> A = np.random.rand(features, features)
    >> Y = X @ A
    >> rhsic_clf = RandomizeHSIC(center=True, n_components=components)
    >> rhsic_clf.fit(X, Y)
    >> cka_score = rhsic_clf.score(X)
    >> print(f"<K_x,K_y> / ||K_xx|| / ||K_yy||: {cka_score:.3f}")
    """

    def __init__(
        self,
        n_components: int = 100,
        gamma_X: Optional[None] = None,
        gamma_Y: Optional[None] = None,
        kernel: Union[Callable, str] = "linear",
        degree: float = 3,
        coef0: float = 1,
        kernel_params: Optional[dict] = None,
        random_state: Optional[int] = None,
        center: Optional[int] = True,
        normalized: Optional[bool] = True,
        subsample: Optional[int] = None,
        bias: bool = True,
    ):
        self.n_components = n_components
        self.gamma_X = gamma_X
        self.gamma_Y = gamma_Y
        self.center = center
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.subsample = subsample
        self.bias = bias

    def fit(self, X, Y):

        # Check sizes of X, Y
        X = check_array(X, ensure_2d=True)
        Y = check_array(Y, ensure_2d=True)

        # Check samples are the same
        assert (
            X.shape[0] == Y.shape[0]
        ), f"Samples of X ({X.shape[0]}) and Samples of Y ({Y.shape[0]}) are not the same"

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
        Zx = self.compute_kernel(X, gamma=self.gamma_X)
        Zy = self.compute_kernel(Y, gamma=self.gamma_Y)

        # Center Kernel
        # H = np.eye(n_samples) - (1 / n_samples) * np.ones(n_samples)
        # K_xc = K_x @ H
        if self.center == True:
            Zx = Zx - Zx.mean(axis=0)
            Zy = Zy - Zy.mean(axis=0)

        # Compute HSIC value
        if self.n_components < self.n_samples:
            Rxy = Zx.T @ Zy
            self.hsic_value = np.sum(Rxy * Rxy)

            # Calculate magnitudes
            self.Zx_norm = np.linalg.norm(Zx.T @ Zx)
            # print("Norm (FxF):", np.linalg.norm(Zx.T @ Zx))
            # print("Norm (NxN):", np.linalg.norm(Zx @ Zx.T))
            self.Zy_norm = np.linalg.norm(Zy.T @ Zy)
            # print("Norm (FxF):", np.linalg.norm(Zy.T @ Zy))
            # print("Norm (NxN):", np.linalg.norm(Zy @ Zy.T))

        else:
            Rxx = Zx @ Zx.T
            Ryy = Zy @ Zy.T
            self.hsic_value = np.sum(Rxx * Ryy)

            # Calculate magnitudes
            self.Zx_norm = np.linalg.norm(Zx @ Zx.T)
            self.Zy_norm = np.linalg.norm(Zy @ Zy.T)

        return self

    def compute_kernel(self, X, Y=None, gamma=None, *args, **kwargs):

        # estimate gamma if None
        if gamma is None:
            gamma = estimate_gamma(X)

        # initialize RBF kernel
        nystrom_kernel = Nystroem(
            gamma=gamma,
            kernel=self.kernel,
            n_components=self.n_components,
            coef0=self.coef0,
            degree=self.degree,
            random_state=self.random_state,
            *args,
            **kwargs,
        )

        # transform data
        return nystrom_kernel.fit_transform(X)

    def score(self, X, y=None, normalize=False):
        """This is not needed. It's only needed to comply with sklearn API.
        
        We will use the target kernel alignment algorithm as a score
        function. This can be used to find the best parameters."""

        if normalize == True:

            return self.hsic_value / self.Zx_norm / self.Zy_norm

        elif normalize == False:

            if self.bias:
                self.hsic_bias = 1 / (self.n_samples ** 2)
            else:
                self.hsic_bias = 1 / (self.n_samples - 1) ** 2

            return self.hsic_bias * self.hsic_value
        else:
            raise ValueError(f"Unrecognized normalize argument: {normalize}")
