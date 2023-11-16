import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import (register_plotly_resampler,
                              unregister_plotly_resampler)

from mdu.plotly.styling import apply_default_styles


class DataShapeError(Exception):
    pass


def plot_ts_resampling(
    data: np.ndarray,
    x: np.ndarray | None = None,
    names: list[str] | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot one ore multiple time series with resampling active

    Parameters
    ----------
    data : np.ndarray
        data array with shape (n_samples, n_features)

    x : np.ndarray | None
        array of x values, by default None. If None, np.arange(data.shape[0])
        is used

    names: list[str] | None
        names to use as labels for the traces, by default None.

    show : bool, optional
        if True, show the figure, by default False


    Returns
    -------
    go.Figure

    """

    register_plotly_resampler()
    fig = plot_ts(data, x, names=names, show=False)
    unregister_plotly_resampler()
    if show:
        fig.show_dash(mode="external")

    return fig


def plot_ts(
    data: np.ndarray,
    x: np.ndarray | None = None,
    names: list[str] | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot one ore multiple time series

    Parameters
    ----------
    data : np.ndarray
        data array with shape (n_samples, n_features)

    x : np.ndarray | None
        array of x values, by default None. If None, np.arange(data.shape[0])
        is used

    names: list[str] | None
        names to use as labels for the traces, by default None.

    show : bool, optional
        if True, show the figure, by default False

    Returns
    -------
    go.Figure

    """

    if len(data.shape) > 2:
        raise DataShapeError(f"{data.shape=}, but should at most 2D")

    x = np.arange(data.shape[0]) if x is None else x

    # Defaults
    nrows = data.shape[1] if data.ndim == 2 else 1
    names = [f"y{i}" for i in range(data.shape[1])] if names is None else names

    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True)

    for iy in range(nrows):
        fig.add_scatter(x=x, y=data[:, iy], name=names[iy], row=iy + 1, col=1)

    fig = apply_default_styles(fig)

    if show:
        fig.show()

    return fig
