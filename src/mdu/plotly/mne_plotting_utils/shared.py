import mne
import numpy as np


def combine_epochs(epo: mne.BaseEpochs, combine: str) -> np.ndarray:
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
