# Loader for brainvisions data
from pathlib import Path

import mne


def load_bv_data(file: Path) -> mne.io.Raw:
    """
    Load BrainVision data from file.

    Parameters
    ----------
    file : Path
        Path to the BrainVision file (.vhdr header file).

    Returns
    -------
    mne.io.Raw
        MNE Raw object containing the loaded BrainVision data with preloaded data.
    """
    return mne.io.read_raw_brainvision(file, preload=True)
