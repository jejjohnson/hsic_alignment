from typing import Optional
import numpy as np
from .kernel import estimate_sigma, sigma_to_gamma
from .dependence import HSIC, train_rbf_hsic


def get_hsic(
    X: np.ndarray,
    Y: np.ndarray,
    scorer: str,
    gamma_init_X: float,
    gamma_init_Y: Optional[float] = None,
    maximum: bool = False,
    n_gamma: int = 50,
    factor: int = 1,
) -> float:
    """Gets the HSIC parameters
    
    Parameters
    ----------
    X : np.ndarray, (n_samples, d_dimensions)
        1st input array 
    
    Y : np.ndarray, (n_samples, d_dimensions)
        2nd input array
    
    scorer : str, 
        the scorer to calculate the hsic
        * hsic - HSIC method
        * tka  - kernel tangent alignment
        * ctka - centered kernel tangent alignment
    
    maximum : bool, default=False
        takes the maximum value irregardless of the initialization

    n_gamma : int, default=50
        The number of gamma parameters to use
    
    factor : int, default=1
        the spread of the grid points to search
    gamma_init : float
        the initial gamma parameter
    
    Returns
    -------
    hsic_value : float
        the hsic value calculated from the scorer
    """
    # hsic parameters
    kernel = "rbf"
    subsample = None
    bias = True

    # cross val parameters

    # initialize HSIC model
    if maximum == True and gamma_init_Y is not None:
        raise ValueError("Cannot do maximum of two initializations.")
    elif maximum == True and gamma_init_Y is None:
        clf_hsic = train_rbf_hsic(X, Y, scorer, n_gamma, factor, "median")
    else:
        clf_hsic = HSIC(
            gamma_X=gamma_init_X,
            gamma_Y=gamma_init_Y,
            kernel=kernel,
            scorer=scorer,
            subsample=subsample,
            bias=bias,
        )

        # fit model to data
        clf_hsic.fit(X, Y)

    # get hsic value
    hsic_value = clf_hsic.score(X)

    return hsic_value


def get_gamma_init(
    X: np.ndarray,
    Y: np.ndarray,
    method: str,
    percent: Optional[float] = None,
    scale: Optional[float] = None,
    each_length: bool = False,
) -> float:
    """Get Gamma initializer
    
    Parameters
    ----------
    method : str,
        the initialization method
        
    percent : float
        if using the Belkin method, this uses a percentage
        of the kth nearest neighbour
    
    Returns
    -------
    gamma_init_X : float
        the initial gamma value for X
        (will be used for both X and Y)

    gamma_init_Y : float (Optional)
        the initial gamma value for Y
    """

    # initialize sigma
    sigma_init_X = estimate_sigma(X, method=method, percent=percent, scale=scale)
    sigma_init_Y = estimate_sigma(Y, method=method, percent=percent, scale=scale)

    if each_length == False:
        # mean of the two
        sigma_init = np.mean([sigma_init_X, sigma_init_Y])

        # convert sigma to gamma
        gamma_init_X = sigma_to_gamma(sigma_init)

        # return initial gamma value
        return gamma_init_X
    elif each_length == True:
        # convert sigmas to gammas
        gamma_init_x = sigma_to_gamma(sigma_init_X)
        gamma_init_y = sigma_to_gamma(sigma_init_Y)

        # return initial gamma values
        return gamma_init_x, gamma_init_y
    else:
        raise ValueError(f"Unrecognized option for each_length: {each_length}")
