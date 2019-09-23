import numpy as np
from sklearn.utils import check_random_state
from typing import Tuple


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
