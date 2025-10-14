import mne
import numpy as np


def combine_epochs(epo: mne.BaseEpochs, combine: str) -> np.ndarray:
    """
    Combine epochs data using specified method.

    Parameters
    ----------
    epo : mne.BaseEpochs
        MNE Epochs object containing the data to combine.
    combine : str
        Method for combining epochs. Options are:
        - 'mean' : Average across channels
        - 'gfp' : Global Field Power (RMS across channels)

    Returns
    -------
    np.ndarray
        Combined epoch data with shape (n_epochs, n_times).
    """
    combine_map = {
        "mean": lambda x: x.get_data().mean(axis=1),
        "gfp": lambda x: np.sqrt((x.get_data() ** 2).mean(axis=1)),
    }
    return combine_map[combine](epo)


def bootstrap(
    arr: np.ndarray,
    ci: list[float, float] = [0.025, 0.975],
    min_max: bool = False,
    nboot: int = 2000,
    rng: np.random.Generator | None = None,
    seed: int = 42,
):
    """
    Compute bootstrap confidence intervals for array data.

    Parameters
    ----------
    arr : np.ndarray
        Input data array with shape (n_samples, ...).
    ci : list of float, default=[0.025, 0.975]
        Confidence interval bounds as percentiles (e.g., [0.025, 0.975] for 95% CI).
    min_max : bool, default=False
        If True, return min/max instead of percentile-based confidence intervals.
    nboot : int, default=2000
        Number of bootstrap iterations.
    rng : np.random.Generator or None, default=None
        Random number generator. If None, creates one with PCG64 algorithm.
    seed : int, default=42
        Random seed for reproducibility when rng is None.

    Returns
    -------
    ci_bounds : np.ndarray
        Confidence interval bounds with shape (2, ...) where first row is lower bound
        and second row is upper bound.
    bd : np.ndarray
        Bootstrap distribution with shape (nboot, ...).
    """
    # confidence intervals
    if rng is None:
        # alternative would be MT19937 (not recommended by numpy)       # noqa

        rng = np.random.Generator(np.random.PCG64(seed))
        # mne uses MT19937 which is a bit faster

    bootstrap_idx = rng.integers(0, arr.shape[0], size=(arr.shape[0], nboot))

    bd = np.asarray([arr[idx].mean(axis=0) for idx in bootstrap_idx.T])

    if min_max:
        bd_min = bd.min(axis=0)
        bd_max = bd.max(axis=0)
        return np.array([bd_min, bd_max]), bd
    else:
        ci_low, ci_up = np.percentile(bd, tuple(np.array(ci) * 100), axis=0)
        return np.array([ci_low, ci_up]), bd
