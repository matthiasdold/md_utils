import base64
import io

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mdu.plotly.mne_plotting_utils.shared import combine_epochs
from mdu.plotly.styling import apply_default_styles


def plot_epo_image(
    epo: mne.BaseEpochs,
    df: pd.DataFrame,
    sort_by: str = "",
    color_by: str = "",
    combine: str = "mean",
    plot_mode: str = "full",
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    vmin_q: float = 0.01,
    vmax_q: float = 0.99,
    log_vals: bool = False,
    showscale: bool = True,
    show: bool = False,
):
    """Plot the epoch image of given epochs

    Parameters
    ----------
    epo : mne.Epochs
        the epoched time series
    df : pandas.DataFrame, None
        data frame containing meta information about the single epochs
    sort_by : str
        name of column in df to soort the epochs.
    color_by: str
        name of column in df to group the epochs into colors. This will create
        a column left of the y-axis which shows the color
    combine : str
        how to combine accross the channel axis, default is 'mean',
        other options is 'gfp' (global field potential)
        => np.sqrt((x**2).mean(axis=1))
        Note: mne default is gfp
    plot_mode : str
        either `full` or `base64`, if full, creates a full plotly fig,
        else just create axis and fill the background with a matplotlib plot
        which saves a lot of data to be processed by the browser
    fig : go.FigureWidget
        if a figure is provided, plot to this figure using the row and col
        parameters
    row : int
        row parameter used in fig.add_traces to add the plot
    col : int
        col parameter used in fig.add_traces to add the plot
    vmin_q : float
        quantile to limit lower color bound - only used in plot_mode=full
    vmax_q : float
        quantile to limit upper color bound - only used in plot_mode=full
    log_vals : bool
        if true, work on log transformed data
    showscale : bool
        if true, show the colorbar
    show : bool, optional
        if True, fig.show() is called

    Returns
    -------
    fig : go.FigureWidget
        the plotly figure


    Note: visually tested vs
        epo.plot_image(evoked=False)

    """

    if sort_by != "":
        dff = df.copy().reset_index()
        dff = dff.rename(columns={"index": "orig_idx"})
        dff = dff.sort_values(sort_by)

        if color_by == "":
            color_by = sort_by

        wepo = epo.copy()[dff.orig_idx]

    else:
        dff = df
        wepo = epo.copy()

    data = combine_epochs(wepo, combine)

    if log_vals:
        data = np.log(data)
    fig = make_subplots(1, 1) if fig is None else fig

    if plot_mode == "full":
        style = {
            "colorscale": "reds" if combine == "gfp" else "jet",
            "reversescale": False if combine == "gfp" else False,
            "showscale": showscale,
        }
        fig = plot_epoch_image_full_mode(
            data,
            epo.times,
            fig=fig,
            row=row,
            col=col,
            vmin_q=vmin_q,
            vmax_q=vmax_q,
            zero_center=True,
            **style,
        )

    elif plot_mode == "base64":
        # Note, I tried other colormaps, but the contrast only gets worse
        style = {
            "cm": plt.cm.RdPu if combine == "gfp" else plt.cm.bwr,
        }
        fig = plot_epoch_image_base64(
            data, epo.times, fig=fig, row=row, col=col, **style
        )

    if color_by != "":
        grps = list(dff.groupby(color_by))
        colormap = dict(
            zip([g[0] for g in grps], px.colors.qualitative.Plotly)
        )
        cat_number = dict(zip([g[0] for g in grps], range(len(grps))))

        # get a numpy vector
        color_array = dff[color_by].map(cat_number).to_numpy()
        carr_diff = np.diff(color_array)
        carr_split = np.split(
            np.arange(len(epo)), np.where(np.abs(carr_diff) > 0)[0] + 1
        )

        trange = epo.times[-1] - epo.times[0]
        xrange = [
            epo.times[-1] + trange * 0.02,
            epo.times[-1] + trange * 0.05,
        ]  # noqa

        for arr in carr_split:
            gk = dff.iloc[arr, :][color_by].unique()
            assert len(gk) == 1, (
                "Split of arrays for coloring did not work" f" got gk={gk}"
            )
            fig.add_trace(
                go.Scatter(
                    x=np.hstack([xrange, xrange[::-1]]),
                    y=[arr[0], arr[0], arr[-1], arr[-1]],
                    fill="toself",
                    fillcolor=colormap[gk[0]],
                    showlegend=False,
                    mode="lines",
                    line={"width": 2, "color": colormap[gk[0]]},
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(range=[epo.times[0], xrange[-1]], row=row, col=col)
        fig.update_yaxes(range=[0, len(epo)], row=row, col=col)

    fig.update_layout(title=f"Epoch image - {combine}")

    return fig


def plot_epoch_image_base64(data, times, fig=None, row=1, col=1, cm=None):
    cm = plt.cm.bwr if cm is None else cm

    auxfig, aux_ax = plt.subplots()
    aux_ax.axis("off")

    # matplotlib plots from top to bottom, plotly the other way around
    aux_ax.imshow(data[::-1], aspect="auto", cmap=cm)

    base64_data = serialize_matplotlib_figure(auxfig, format="png")
    plt.close()  # clode the matplotlib figure

    fig = make_subplots(1, 1) if fig is None else fig

    # Constants
    scale_factor = 1
    y_range = data.shape[0]

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=times,
            y=[0, y_range * scale_factor],
            mode="markers",
            marker_opacity=0,
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # ensure that axis will be tight with the picture
    fig.update_xaxes(range=[times[0], times[-1]], showgrid=False)
    fig.update_yaxes(range=[0, y_range * scale_factor], showgrid=False)

    # Add image
    fig.add_layout_image(
        dict(
            x=times[0],
            sizex=times[-1] - times[0],
            y=y_range * scale_factor,
            sizey=y_range * scale_factor,
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=f"data:image/png;base64,{base64_data.decode()}",
        ),
        row=row,
        col=col,
    )

    # add a line at zero
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0.05 * data.shape[0], data.shape[0] * 0.95],
            hoverinfo="skip",
            mode="lines",
            showlegend=False,
            line=dict(color="#000000", dash="dash", width=2),
        ),
        row=row,
        col=col,
    )

    fig = apply_default_styles(fig, row=row, col=col, xzero=False, yzero=False)
    fig.update_yaxes(title="Epoch Nbr.", row=row, col=col)
    fig.update_xaxes(title="Time [s]", row=row, col=col)

    return fig


def serialize_matplotlib_figure(
    fig, format="png", bbox_inches="tight", **kwargs
) -> bytes:
    """Serialize by saving to a byte stream
    Note: png seems to work best as a format
    """
    string_io_bytes = io.BytesIO()
    fig.savefig(
        string_io_bytes,
        format=format,
        bbox_inches=bbox_inches,
        pad_inches=0,
        **kwargs,
    )  # just interior
    string_io_bytes.seek(0)
    base64_data = base64.b64encode(string_io_bytes.read())

    return base64_data


def plot_epoch_image_full_mode(
    data: np.ndarray,
    times: np.ndarray,
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    colorscale: str = "picnic",
    showscale: bool = False,
    reversescale: bool = False,
    vmin_q: float = 0.01,
    vmax_q: float = 0.99,
    zero_center: bool = False,
    zscale: bool = False,
) -> go.Figure:
    fig = make_subplots(1, 1) if fig is None else fig

    if zscale:  # per epoch zscaling
        zdata = (data - data.mean(axis=1, keepdims=True)) / data.std(
            axis=1, keepdims=True
        )
        zmin, zmax = zdata.min(), zdata.max()
    else:
        zmin, zmax = np.quantile(data, (vmin_q, vmax_q))
        zdata = data

    scale_kwargs = {}

    if zero_center:
        cm, cb = get_zero_green_JET_scale(zmin, zmax)
        scale_kwargs.update(
            dict(colorscale=cm, colorbar=cb, zmin=zmin, zmax=zmax)
        )
    else:
        scale_kwargs.update(dict(zmin=zmin, zmax=zmax, colorscale=colorscale))

    fig.add_trace(
        go.Heatmap(
            z=zdata,
            x=times,
            showscale=showscale,
            hoverinfo="skip",
            reversescale=reversescale,
            **scale_kwargs,
        ),
        row=row,
        col=col,
    )

    # add a line at zero
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0.05 * data.shape[0], data.shape[0] * 0.95],
            hoverinfo="skip",
            mode="lines",
            showlegend=False,
            line=dict(color="#000000", dash="dash", width=2),
        ),
        row=row,
        col=col,
    )

    fig = apply_default_styles(fig, row=row, col=col, xzero=False, yzero=False)
    fig.update_layout(coloraxis_colorscale=colorscale)
    fig.update_yaxes(title="Epoch Nbr.", row=row, col=col)
    fig.update_xaxes(title="Time [s]", row=row, col=col)

    return fig


def get_zero_green_JET_scale(
    zmin: float, zmax: float
) -> tuple[list[list[float, str]], dict]:
    """
    Given min and max values, create a colormap, which has green at 0, but
    might be asymmetric.
    """

    jscale = px.colors.sequential.Jet
    delta = zmax - zmin
    # values are in 'rgb()' notation, len(jscalel) = 6
    zero_green = "rgb(130, 255, 128)"

    # First handle the odd cases where actually no 0 is needed
    if zmin < zmax <= 0:
        # lower half only
        cm = [[v, c] for v, c in zip([0, 0.5, 1], jscale[:3])]
        cb = {
            "tickvals": np.linspace(zmin, zmax, 5),
            "ticktext": [f"{v}" for v in np.linspace(zmin, zmax, 5)],
        }
    elif zmax > zmin >= 0:
        # upper half only
        cm = [[v, c] for v, c in zip([0, 0.5, 1], jscale[3:])]
        cb = {
            "tickvals": np.linspace(zmin, zmax, 5),
            "ticktext": [f"{v}" for v in np.linspace(zmin, zmax, 5)],
        }
    else:
        # determine where the 0 would be if we map [zmin, zmax] -> [0, 1]
        zeroval = np.abs(zmin) / delta
        topvals = np.linspace(zeroval, 1, 4)
        botvals = np.linspace(0, zeroval, 4)
        vals = np.hstack([botvals[:-1], [zeroval], topvals[1:]])
        modscale = jscale[:3] + [zero_green] + jscale[3:]

        cm = [[v, c] for v, c in zip(vals, modscale)]
        zvals = np.hstack(
            [np.linspace(zmin, 0, 4)[:-1], [0], np.linspace(0, zmax, 4)[1:]]
        )

        cb = {
            "tickvals": zvals,
            "ticktext": [f"{v:.2}" for v in zvals],
            "tickmode": "array",
        }

    return cm, cb
