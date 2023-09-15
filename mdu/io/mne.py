from pathlib import Path

import mne


def load_fifraw_data(file: Path) -> mne.io.Raw:
    return mne.io.read_raw_fif(file, preload=True)
