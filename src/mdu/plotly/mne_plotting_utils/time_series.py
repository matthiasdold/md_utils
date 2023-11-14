import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mdu.plotly.mne_plotting_utils.shared import bootstrap, combine_epochs
from mdu.plotly.stats import add_cluster_permut_sig_to_plotly
from mdu.plotly.styling import apply_default_styles


class SignificanceBetweenGroups(ValueError):
    pass


def plot_evoked_ts(
    epo: mne.Epochs,
    df: pd.DataFrame,
    color_by: str = "",
    combine: str = "mean",
    ci: list[float, float] = [0.05, 0.95],
    nboot: int = 2000,
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    showlegend: bool = True,
    show: bool = False,
):
    fig = plot_ts(
        epo,
        df,
        color_by=color_by,
        combine=combine,
        ci=ci,
        nboot=nboot,
        fig=fig,
        row=row,
        col=col,
        showlegend=showlegend,
        show=False,
    )
    fig.update_yaxes(title="Evoked Signal [µV]", row=row, col=col)
    fig.update_layout(
        legend=dict(x=0.7, y=1), title=f"Evoked - combined={combine}"
    )
    if show:
        fig.show()

    return fig


def plot_ts(
    epo: mne.Epochs,
    df: pd.DataFrame,
    color_by: str = "",
    combine: str = "mean",
    ci: list[float, float] = [0.05, 0.95],
    nboot: int = 2000,
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    showlegend: bool = True,
    show: bool = True,
    legendgroup: str | None = None,
    add_p_stats: bool = False,
    add_reaction_time: bool = True,
    envelop: bool = False,
):
    """Plot the analytical envelope of a band filtered signal

    Parameters
    ----------
    epo : mne.Epochs
        the epoched time series
    df : pandas.DataFrame, None
        data frame containing meta information about the single epochs
    color_by : str
        name of column in df to group the epochs in individual colors
    combine : str
        how to combine accross the channel axis, default is 'mean',
        other options is 'gfp' (global field potential)
        => np.sqrt((x**2).mean(axis=1))
    ci : list
        list of lower and upper bound of confidence interval using
        bootstrapping accross the epochs
    nboot : int
        number of bootstrap samples to draw for ci calculation
    fig : go.FigureWidget
        if a figure is provided, plot to this figure using the row and col
        parameters
    row : int
        row parameter used in fig.add_traces to add the plot
    col : int
        col parameter used in fig.add_traces to add the plot
    showlegend : bool
        if true, display the legend
    legendgroup : str
        legendgroud to use
    baseline : tuple
        if provided use the tuple as baseline, else the current baseline of
        the epo input is used
    color_baseline : bool
        if true, color shade the background of the period which is baselined
    add_p_stats : bool
        if true, make a statistical test on each x-axis bin
    add_reaction_time : bool
        if true, add the reaction_time as a histogram to the bottom of the plot
    envelop : bool
        if true, plot the hilbert transform

    Returns
    -------
    fig : go.FigureWidget
        the plotly figure


    Note: visually tested vs
    mne.viz.plot_compare_evokeds({'cond': list(epo.iter_evoked(copy=False))},
                                 combine=pass_combine)

    """

    if color_by != "":
        grps = list(df.groupby(color_by))
    else:
        grps = [("all", df)]

    colormap = dict(zip([g[0] for g in grps], px.colors.qualitative.Plotly))

    if fig is None:
        if not add_reaction_time:
            fig = make_subplots(1, 1)
        else:
            fig = make_subplots(1, 1, specs=[[{"secondary_y": True}]])

    # collect the means for t-testing
    samples = {}

    for gk, dg in grps:
        color = colormap[gk]
        wepo = epo.copy()[dg.index]
        nmean = len(wepo)

        if envelop:
            wepo.apply_hilbert(envelope=True)

        agg_epo = combine_epochs(wepo, combine)
        ci_arr, boot_s = bootstrap(agg_epo, nboot=nboot, ci=ci)
        samples[str(gk)] = agg_epo

        # for c in dg.columns:
        # just to get the domain correct
        fig.add_trace(
            go.Scatter(
                y=[agg_epo.mean(), agg_epo.mean()],
                x=[wepo.times[0], wepo.times[-1]],
                line=dict(width=0.0),
                mode="lines",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        cutoff = int(
            0.02 * len(wepo.times)
        )  # to not show the filter artifacts      # noqa
        # CI
        fig.add_trace(
            go.Scatter(
                y=np.hstack(
                    [
                        ci_arr[0][cutoff:-cutoff],
                        ci_arr[1][cutoff:-cutoff][::-1],
                    ]
                ),
                x=np.hstack(
                    [
                        wepo.times[cutoff:-cutoff],
                        wepo.times[cutoff:-cutoff][::-1],
                    ]
                ),
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line=dict(width=0.1, color=color),
                name=f"ci={ci[0]:.1%}-{ci[1]:.1%}_n={nboot}",
                hoverinfo="skip",
                showlegend=showlegend,
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )

        # mean
        fig.add_trace(
            go.Scatter(
                y=agg_epo.mean(axis=0)[cutoff:-cutoff],
                x=wepo.times[cutoff:-cutoff],
                mode="lines",
                line_color=color,
                name=f"mean_{gk}",
                text=f"n_mean={nmean}",
                showlegend=showlegend,
                legendgroup=legendgroup,
            ),
            row=row,
            col=col,
        )

    if add_p_stats:
        if len(grps) != 2:
            raise SignificanceBetweenGroups(
                f"{color_by=} resulted in {len(grps)=} groups. Can only add"
                " significance values for exactly 2 groups."
            )
        else:
            slist = list(samples.values())
            add_cluster_permut_sig_to_plotly(slist[0], slist[1], fig)

    fig.update_xaxes(title="Time [s]", row=row, col=col)
    fig = apply_default_styles(fig, row=row, col=col)

    if show:
        fig.show()

    return fig
