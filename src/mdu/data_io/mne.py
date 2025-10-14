from pathlib import Path

import mne


def load_fifraw_data(file: Path) -> mne.io.Raw:
    """
    Load MNE FIF raw data from file.

    Parameters
    ----------
    file : Path
        Path to the FIF raw data file (.fif).

    Returns
    -------
    mne.io.Raw
        MNE Raw object containing the loaded FIF data with preloaded data.
    """
    return mne.io.read_raw_fif(file, preload=True)
