# Plotting methods to work on MNE objects

import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import (register_plotly_resampler,
                              unregister_plotly_resampler)

from mdu.plotly.mne_plotting_utils.epoch_image import plot_epo_image
from mdu.plotly.mne_plotting_utils.topoplot import create_plotly_topoplot
from mdu.plotly.styling import apply_default_styles


# TODO: [ ] Have a look at the CorticalFootprint/movingdots epo visualisation
# and combine with this here
def plot_raw(raw: mne.io.Raw, show: bool = True, add_events: bool = True):
    """Plot traces for an mne raw object, one row for each channel

    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object
    show : bool, optional
        Show the plot, by default True
    add_events : bool, optional
        Add events from the raw object as vertical lines, by default True

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure
    """
    # Register resampler
    register_plotly_resampler()
    # Create figure
    fig = make_subplots(
        rows=len(raw.ch_names), cols=1, shared_xaxes=True, vertical_spacing=0
    )

    # Add traces
    for ir, (ch, data) in enumerate(zip(raw.ch_names, raw.get_data())):
        fig.add_scatter(
            x=raw.times,
            y=data,
            name=ch,
            row=ir + 1,
            col=1,
        )
        fig.update_yaxes(title_text=ch, row=ir + 1, col=1)

    if add_events:
        ev, evid = mne.events_from_annotations(raw)
        n_colors = len(np.unique(ev[:, 2]))
        colors = px.colors.sample_colorscale(
            "viridis", [n / (n_colors - 1) for n in range(n_colors)]
        )
        cmap = dict(zip(np.unique(ev[:, 2]), colors))

        for e in ev:
            fig.add_vline(
                x=raw.times[e[0]],
                line_width=1,
                line_color=cmap[e[2]],
                opacity=0.5,
                annotation={"text": str(e[2]), "font_size": 12},
            )

    # Apply default styles
    fig.update_xaxes(showticklabels=False)
    apply_default_styles(fig)
    fig.update_layout(
        height=1080,
    )

    if show:
        fig.show_dash(mode="external")

    unregister_plotly_resampler()

    return fig


def plot_topo(
    data: np.ndarray,
    inst: mne.io.Raw | mne.Epochs | mne.Evoked,
    contour_kwargs: dict = {"colorscale": "viridis"},
    show: bool = True,
) -> go.Figure:
    """
    Plot a topoplot from data and an mne instance for meta data information.
    This is a wrapper for ./mne_plotting_utils/topoplot.py::create_plotly_topoplot


    Parameters
    ----------
    data : np.ndarray
        the data for the topoplot, one value for each channel in inst.ch_names

    inst : mne.io.Raw | mne.Epochs | mne.Evoked
        the mne instance to get the channel meta information from

    contour_kwargs : dict, optional
        kwargs for the contour plot, by default {"colorscale": "viridis"}

    Returns
    -------
    go.FigureWidget
        topo plot figure in plotly

    """
    fig = create_plotly_topoplot(
        data, inst, contour_kwargs=contour_kwargs, show=show
    )

    return fig


def plot_variances(
    epo: mne.BaseEpochs,
    df: pd.DataFrame,
    color_by: str = "",
    figs: go.Figure | None = None,
    row: list[int, int] = [1, 1],
    col: list[int, int] = [1, 2],
    show: bool = False,
) -> go.Figure:
    """Plot the variance distribution as scatter along time and
    cumulative densities

    Parameters
    ----------
    epo : mne.Epochs
        the epoched time series
    df : pandas.DataFrame, None
        data frame containing meta information about the single epochs
    color_by: str
        name of column in df to group the epochs into colors. This will create
        a column left of the y-axis which shows the color
    fig : go.FigureWidget
        figure to add the subplots to
    row : list(int, int)
        row parameter used in fig.add_traces to add the plot, for the scatter
        and the hist plot
    col : list(int, int)
        col parameter used in fig.add_traces to add the plot for the scatter
        and the hist plot
    show : bool, optional
        if True, fig.show() is called

    Returns
    -------
    fig : go.FigureWidget
        the plotly figure


    Note: visually tested vs
        epo.plot_image(evoked=False)

    """

    data = epo.get_data()

    if data.shape[1] > 1:
        print("Combining the variance of more than one channel!")

    var_data = np.var(data.mean(axis=1), axis=1)

    grps = list(df.groupby(color_by))
    colormap = dict(zip([g[0] for g in grps], px.colors.qualitative.Plotly))

    dw = pd.DataFrame(var_data, columns=["y"])
    dw[color_by] = df[color_by].to_numpy()
    dw["color"] = df[color_by].map(colormap).to_numpy()
    dw = dw.reset_index().rename(columns={"index": "x"})

    fig = None
    fig = (
        make_subplots(1, 2, column_widths=(0.8, 0.2), horizontal_spacing=0)
        if fig is None
        else fig
    )
    # Scatter

    # Bars
    for i, (ck, color) in enumerate(colormap.items()):
        # scatter
        fig.add_trace(
            go.Scatter(
                x=dw.loc[dw[color_by] == ck, "x"],
                y=dw.loc[dw[color_by] == ck, "y"],
                marker=dict(color=color, size=12),
                name=ck,
                mode="markers",
                opacity=0.8,
            ),
            row=row[0],
            col=col[0],
        )

        # histogram
        fig.add_trace(
            go.Histogram(
                y=dw.loc[dw[color_by] == ck, "y"],
                histnorm="probability",
                marker_color=color,
                showlegend=False,
                bingroup=i,
                opacity=0.3,
            ),
            row=row[1],
            col=col[1],
        )

    fig.update_layout(barmode="overlay", bargap=0.0)

    fig = apply_default_styles(fig, row=row[1], col=col[1])
    fig = apply_default_styles(fig, row=row[0], col=col[0])
    fig.update_yaxes(
        title="Variance [AU²]",
        range=[0, var_data.max() * 1.02],
        row=row[0],
        col=col[0],
    )
    fig.update_xaxes(
        title="Epoch Nbr", row=row[0], col=col[0], range=[0, len(epo) * 1.01]
    )
    fig.update_xaxes(showticklabels=False, row=row[1], col=col[1])
    fig.update_yaxes(
        showticklabels=False,
        row=row[1],
        col=col[1],
        range=[0, var_data.max() * 1.02],
    )
    fig.update_layout(legend=dict(x=0.0, y=1), title="Component variance")

    if show:
        fig.show()

    return fig


# TODO [ ]: -- Consider if wrappers are needed at all
def plot_epoch_image(
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
    """Plot the epoch image of given epochs, wrapper around ./mne_plotting_utils/epoch_image.py::plot_epo_image

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

    return plot_epo_image(
        epo=epo,
        df=df,
        sort_by=sort_by,
        color_by=color_by,
        combine=combine,
        plot_mode=plot_mode,
        fig=fig,
        row=row,
        col=col,
        vmin_q=vmin_q,
        vmax_q=vmax_q,
        log_vals=log_vals,
        showscale=showscale,
        show=show,
    )
