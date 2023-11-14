# Loader for brainvisions data
from pathlib import Path

import mne


def load_bv_data(file: Path) -> mne.io.Raw:
    return mne.io.read_raw_brainvision(file, preload=True)
