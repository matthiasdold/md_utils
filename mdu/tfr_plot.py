from pathlib import Path

import mne
import numpy as np
import plotly.graph_objects as go
from fire import Fire
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from plotly.subplots import make_subplots

from mdu.io.ao import load_ao_data
from mdu.io.bv import load_bv_data
from mdu.io.mne import load_fifraw_data


def load_raw(file: Path) -> mne.io.Raw:
    if file.suffix == ".mat":
        return load_ao_data(file)
    elif file.suffix == ".vhdr":
        return load_bv_data(file)
    elif file.suffix == ".fif":
        return load_fifraw_data(file)
    else:
        raise ValueError(f"Unknown file type: {file.suffix}")


def calc_tfr(
    raw: mne.io.Raw, freqs: np.ndarray, n_cycles: np.ndarray, decim: int = 100
) -> np.ndarray:
    # tfr calculation
    time_bandwidth = 4
    rd = raw.copy().get_data()
    rd = rd.reshape(1, *rd.shape)
    tfr = mne.time_frequency.tfr_array_multitaper(
        rd,
        sfreq=raw.info["sfreq"],
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        output="power",
        decim=decim,
        n_jobs=-1,
        verbose=True,
    )

    return tfr[0]


def generate_images(data: np.ndarray, vmin: float, vmax: float) -> list[Image]:
    db_data = 10 * np.log10(data * 10**12)
    color_norm = Normalize(vmin=vmin, vmax=vmax)
    scalar_map = cm.ScalarMappable(norm=color_norm, cmap="jet")
    cdata_per_ch = [
        scalar_map.to_rgba(db_data[::-1, i, :]) for i in range(data.shape[1])
    ]
    images = [Image.fromarray(np.uint8(cdata * 255)) for cdata in cdata_per_ch]

    return images


def plot_tfr(
    tfr_array: np.ndarray,
    freqs: list[float] | np.ndarray,
    raw: mne.io.Raw,
    vmin: None | float,
    vmax: None | float,
    ncolwrap: int,
    decim: int,
):
    """
    By default the plots are created as png images which are used as background
    in the plotly plots
    """
    images = generate_images(
        tfr_array.transpose(1, 0, 2), vmin=vmin, vmax=vmax
    )
    ch_names = raw.ch_names
    ncols = min(ncolwrap, len(ch_names))
    nrows = int(np.ceil(len(ch_names) / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"{ch=}" for ch in ch_names],
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    ymin, ymax = freqs[0], freqs[-1]
    times = (raw.times / raw.info["sfreq"]) * decim
    xmin, xmax = times[0], times[-1]

    for i, img in enumerate(images):
        r = i // ncols + 1
        c = i % ncols + 1
        fig.add_scatter(
            x=[xmin, xmax],
            y=[ymin, ymax],
            mode="markers",
            marker={
                "color": [vmin, vmax],
                "colorscale": "jet",
                "showscale": True if r == 1 and c == 1 else False,
                "colorbar": {"title": "Power [DB]", "titleside": "right"},
                "opacity": 0,
            },
            showlegend=False,
            row=r,
            col=c,
        )

        # the actual image as background
        fig.add_layout_image(
            go.layout.Image(
                x=xmin,
                sizex=xmax - xmin,
                y=ymax,
                sizey=ymax - ymin,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=img,
            ),
            row=r,
            col=c,
        )

    fig.update_xaxes(
        title="Time [s]",
        showgrid=False,
        zeroline=False,
        range=[xmin, xmax],
    )

    fig.update_yaxes(
        title="Freq [Hz]",
        showgrid=False,
        zeroline=False,
        range=[ymin, ymax],
    )

    fig.update_layout(
        width=600 * ncols,
        height=500 * nrows,
    )

    return fig


def main(
    file: Path,
    fmin: float = 1,
    fmax: float = 149,
    fstep: float = 1,
    ncycles: str | float | np.ndarray | int = 2.0,
    tmin: None | float = None,
    tmax: None | float = None,
    vmin: None = 10,
    vmax: None = 80,
    channels: None | list[int] = None,
    ref_channels: None | list[int] = None,
    ncolwrap: int = 5,
    resample: None | float = 300,
    decim: int = 100,
):
    """CLI to create a TFR plot of a single file of electrophysiology data.

    Parameters
    ----------
    file : Path
        the file containing the raw data

    fmin : float
        the minimum frequency to plot

    fmax : float
        the maximum frequency to plot

    fstep : float
        the frequency resolution (step size in Hz)

    ncycles : str | float | np.ndarray | int
        the parameter to be used for mne.time_frequency.tfr_array_multitaper as
        `n_cycles`. If a float is provided, it will scale according to the
        frequency, i.e. `n_cycles = np.arange(fmin, fmax, fstep) * ncycles`.

    tmin : None | float
        the minimum time to plot

    tmax : None | float
        the maximum time to plot

    vmin : None | float
        the minimum value for the color scale

    vmax : None | float
        the maximum value for the color scale

    channels : None | list[int]
        indices of the channels to use, if None, all are used

    ref_channels : None | list[int]
        indices of the channels to use as reference, if None, no re-reference

    ncolwrap : int
        number of columns to wrap the channel plots. A individual subplot is
        created for each channel.

    resample : None | float
        the sampling frequency to resample the data to, if None, no resampling

    decim : int = 100
        the decimation factor for the tfr calculation

    """

    # Preparing raw data
    raw = load_raw(Path(file))
    if channels is not None:
        raw.pick_channels([raw.ch_names[int(c)] for c in channels])
    if resample is not None:
        raw.resample(resample, n_jobs=-1)

    raw.filter(fmin, fmax, n_jobs=-1)
    if ref_channels is not None:
        print(f">>> Re-referencing with {ref_channels}")
        if isinstance(ref_channels, str) or isinstance(ref_channels, int):
            ref_channels = [int(ref_channels)]
        else:
            ref_channels = [int(e) for e in ref_channels]
        raw.set_eeg_reference([raw.ch_names[i] for i in ref_channels])

    if tmin is not None or tmax is not None:
        raw.crop(tmin, tmax)

    # time frequency
    freqs = np.arange(fmin, fmax, fstep)
    ncycles = freqs * ncycles if isinstance(ncycles, float) else ncycles
    tfr = calc_tfr(raw, freqs, ncycles, decim=decim)

    # plotting
    fig = plot_tfr(tfr, freqs, raw, vmin, vmax, ncolwrap, decim=decim)
    fig.update_layout(title=str(file))
    fig.show()


if __name__ == "__main__":
    Fire(main)
