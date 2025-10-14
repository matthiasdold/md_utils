from typing import Optional

import numpy as np
import scipy.stats as st


class AuxModel:
    """Helper to fix X during simulation"""

    def __init__(self, model):
        self.model = model

    def predict(self, y, x=None):
        """Predict using the model with fixed exogenous variables.

        Parameters
        ----------
        y : np.ndarray
            Endogenous variable values.
        x : np.ndarray, optional
            Exogenous variable values, by default None.

        Returns
        -------
        np.ndarray
            Predicted values as a flattened array.
        """
        return self.model.predict(X=x, y=y).flatten()


def simulate_forward(
    model: callable,
    y0: np.ndarray,
    noise_dist: st._distn_infrastructure.rv_continuous_frozen,
    x: Optional[np.ndarray] = None,
    n_steps_pred=1,
    n_sim=100,
    random_state: int = 1337,
):
    """Simulate a model forward in time, making `n_steps_pred` predictions at a time.
    Currently only a single noise distribution for all dimensions of y is supported.

    Parameters
    ----------
    model : callable
        A callable model which has a `predict(y0, ...)` method.
        The predict methods needs to accept y0 as the first argument with the
        potential kwargs for `x`.
    y0 : array-like of shape = n_features
        The initial state of the system.
    noise_dist : scipy.stats._distn_infrastructure.rv_continuous_frozen
        A distribution from scipy.stats.distributions to generate noise from
    x : np.ndarray (optional)
        The input to the model for the stimulation horizon (n_steps_pred * n_stim)
    n_steps_pred : int
        The number of steps to simulate forward at a time
    n_sim : int
        The number of simulation iterations.
    random_state : int
        Random seed for reproducibility.
    Returns
    -------
    x : ndarray of floats
        The simulated state of the system.
    """
    # generate the noise for all steps
    eps = noise_dist.rvs(size=n_steps_pred * n_sim + 1, random_state=random_state)

    ysim = np.zeros((n_sim * n_steps_pred + 1, len(y0)))
    ysim[:, 0] = y0

    for i in range(0, n_sim):
        preds = np.zeros((n_steps_pred, len(y0)))

        # always do one step ahead predictions, recursively to reach n_steps_pred
        for j in range(0, n_steps_pred):
            kwargs = {} if x is None else {"x": x[i * n_steps_pred + j, :]}
            y_j = ysim[i * n_steps_pred, :] if j == 0 else preds[j - 1]
            preds[j] = model.predict(y_j.reshape(1, -1), **kwargs)

        # add noise and replace predictions
        idx_start = i * n_steps_pred + 1
        idx_end = (i + 1) * n_steps_pred + 1
        ysim[idx_start:idx_end] = preds + eps[idx_start:idx_end]

    return ysim
