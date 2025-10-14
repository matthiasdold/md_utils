# IO for data from alpha omega system
import re
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

AO_STREAM_NAME = "AODataStream"
ECOG_CHANNELS_PATTERN = "CECOG_HF_2___[0-4]*___Array_3___[0-4]*"
LFP_CHANNELS_PATTERN = "CECOG_HF_1___[0-9]*___Array_[12]___[0-9]*"
ECOG_AND_LFP_CHANNELS_PATTERN = "CECOG_HF_[12]___[0-9]*___Array_[123]___[0-9]*"
OLD_LFP_CHANNELS_PATTERN = "CEEG_1___[0-9]*___[LR][0-9]*"


def load_ecog_channel_data_from_mat(file: Path) -> mne.io.Raw:
    """
    Load ECoG channel data from AlphaOmega MAT file.

    Parameters
    ----------
    file : Path
        Path to the AlphaOmega .mat file.

    Returns
    -------
    mne.io.Raw
        MNE Raw object containing ECoG channel data.
    """
    return load_channel_data_from_mat(file, ECOG_CHANNELS_PATTERN)


def load_lfp_channel_data_from_mat(file: Path) -> mne.io.Raw:
    """
    Load LFP (Local Field Potential) channel data from AlphaOmega MAT file.

    Parameters
    ----------
    file : Path
        Path to the AlphaOmega .mat file.

    Returns
    -------
    mne.io.Raw
        MNE Raw object containing LFP channel data.
    """
    return load_channel_data_from_mat(file, LFP_CHANNELS_PATTERN)


def load_ao_data(file: Path) -> mne.io.Raw:
    """
    Load both ECoG and LFP channel data from AlphaOmega MAT file.

    Parameters
    ----------
    file : Path
        Path to the AlphaOmega .mat file.

    Returns
    -------
    mne.io.Raw
        MNE Raw object containing both ECoG and LFP channel data.
    """
    return load_channel_data_from_mat(file, ECOG_AND_LFP_CHANNELS_PATTERN)


def load_channel_data_from_mat(file: Path, pattern: str) -> mne.io.Raw:
    """
    Load channel data from AlphaOmega MAT file using a regex pattern.

    This function reads channel data from an AlphaOmega system MAT file,
    filters channels based on a regex pattern, extracts metadata, converts
    units to microvolts, and creates an MNE Raw object with annotations
    from parallel port markers if available.

    Parameters
    ----------
    file : Path
        Path to the AlphaOmega .mat file.
    pattern : str
        Regular expression pattern to match channel names.

    Returns
    -------
    mne.io.Raw
        MNE Raw object containing the channel data with annotations.
    """
    data = loadmat(file, simplify_cells=True)
    channels = [e for e in data.keys() if re.fullmatch(pattern, e)]

    if channels == [] and pattern == LFP_CHANNELS_PATTERN:
        channels = [
            e for e in data.keys() if re.fullmatch(OLD_LFP_CHANNELS_PATTERN, e)
        ]

    hdr = get_ao_channels_header(data, channels[0])

    info = mne.create_info(
        channels,
        hdr["sfreq"],
        ch_types=["ecog"] * len(channels),
    )

    # Unit conversion to uV
    data_matrix = np.asarray([data[c] for c in channels]) / 10**6
    tmp = mne.io.RawArray(data_matrix, info)

    if "CPORT__1" in data:
        vmrk = data["CPORT__1"]
        vmrk = np.array(vmrk[:, vmrk[1, :] != 0], dtype=float)
        # pdb.set_trace()

        # convert to seconds, substract baseline and convert to data rate
        vmrk[0, :] = (
            vmrk[0, :] / (data["CPORT__1_KHz"] * 1000) - hdr["t_start"]
        ) * hdr["sfreq"]

        # to mne structure
        events = np.ones((vmrk.shape[1], 3), dtype=int)
        events[:, 0] = vmrk[0, :]
        events[:, -1] = vmrk[1, :]

    else:
        # -> no markers from parallel port received -> create one at start
        print("=" * 80)
        print(f"Did not find any CPORT__1 in file: \n{file}")
        print("=" * 80)
        events = np.array([[1, 1, 999]], dtype=int)

    annotations = mne.annotations_from_events(events, sfreq=hdr["sfreq"])
    tmp.set_annotations(annotations)

    return tmp


def get_ao_channels_header(data, prefix):
    """
    Extract header information for AlphaOmega channels.

    Parameters
    ----------
    data : dict
        Loaded MAT file data dictionary.
    prefix : str
        Channel name prefix to extract metadata for.

    Returns
    -------
    dict
        Dictionary containing channel metadata with keys:
        - t_start : Start time in seconds
        - n_samples : Number of samples
        - gain : Signal gain in microvolts
        - bit_res : Bit resolution
        - sfreq : Sampling frequency in Hz
    """
    d = dict(
        t_start=data[prefix + "_TimeBegin"],
        n_samples=data[prefix].shape[0],
        # TODO: Q - This gain is showing 55 - shouldn't it be 110?
        gain=1e6 * data[prefix + "_Gain"],
        bit_res=data[prefix + "_BitResolution"],
        sfreq=data[prefix + "_KHz"] * 1e3,
    )

    return d
