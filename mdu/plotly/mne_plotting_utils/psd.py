import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mdu.plotly.stats import add_cluster_permut_sig_to_plotly
from mdu.plotly.styling import apply_default_styles


class SignificanceBetweenGroups(ValueError):
    pass


def plot_epo_psd(
    epo: mne.Epochs,
    df: pd.DataFrame,
    color_by: str = "",
    picks: list[str] = [],
    fig: go.Figure | None = None,
    row: int = 1,
    col: int = 1,
    psd_multitaper_kwargs: dict = {"fmax": 50},
    showlegend: bool = True,
    legendgroup: str | None = None,
    add_p_stats: bool = True,
    color_fband: list[float, float] = [],
) -> go.Figure:
    """Plot the psd of given epochs

    Parameters
    ----------
    epo : mne.Epochs
        the epoched time series
    df : pandas.DataFrame, None
        data frame containing meta information about the single epochs
    color_by : str
        name of column in df to group the epochs in individual colors
    picks : list
        list of channel names to be passed to picks if mne.time_frequency.psd_*
    fig : go.FigureWidget
        if a figure is provided, plot to this figure using the row and col
        parameters
    row : int
        row parameter used in fig.add_traces to add the plot
    col : int
        col parameter use
    psd_multitaper_kwargs : dict
        kwargs passed to psd_multitaper
    showlegend : bool
        if true, display the legend
    legendgroup : str
        legendgroud to use
    add_p_stats : bool
        if true, make a statistical test on each x-axis bin
    color_fband : list (freq_start, freq_end)
        if provided, color the background between given freqs

    Returns
    -------
    fig : go.FigureWidget
        the plotly figure


    Note: visually tested vs
        epo.plot_psd(average=True)

    """

    if color_by != "":
        grps = list(df.groupby(color_by))
    else:
        grps = [("all", df)]

    colormap = dict(zip([g[0] for g in grps], px.colors.qualitative.Plotly))
    fig = make_subplots(1, 1) if fig is None else fig

    samples = {}
    for gk, dg in grps:
        color = colormap[gk]
        wepo = epo.copy()[dg.index]
        nmean = len(wepo)

        if picks == []:
            picks = wepo.ch_names

        psds, freqs = mne.time_frequency.psd_multitaper(
            wepo, verbose=False, picks=picks, **psd_multitaper_kwargs
        )
        # scale depending on the channel types
        ch_types = [d["kind"] for d in wepo.info["chs"]]
        assert len(set(ch_types)) == 1, (
            "Received various channel types"
            " no handling for different units implemented - please check."
            " Potentially you want to pick_types(eeg=True)."
        )
        scaling = 10**12 if "FIFF_EEG_CH" in ch_types[0]._name else 1
        psds *= scaling  # As we were dealing with micro Volts

        # NOTE: The actual means value seems to change in mne -> be explicit
        #       here
        # multiple channels -> STD will be the STD of means per channel
        if psds.shape[1] > 1:
            print("Averaging over channels for psd")
            psds = psds.mean(axis=1)
        else:
            psds = psds[:, 0, :]

        psds = np.log10(np.maximum(psds, np.finfo(float).tiny))
        psds *= 10
        samples[gk] = psds

        # average accross epochs
        mpsds = psds.mean(axis=0)
        diffs = psds - mpsds
        spectrum_std = [
            [np.sqrt((d[d < 0] ** 2).mean(axis=0)) for d in diffs.T],
            [np.sqrt((d[d > 0] ** 2).mean(axis=0)) for d in diffs.T],
        ]
        spectrum_std = np.array(spectrum_std)

        # CI -> std
        fig.add_trace(
            go.Scatter(
                y=np.hstack(
                    [mpsds - spectrum_std[0], (mpsds + spectrum_std[1])[::-1]]
                ),
                x=np.hstack([freqs, freqs[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line=dict(width=0.1, color=color),
                name="+/- 1 STD",
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
                y=mpsds,
                x=freqs,
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

    if color_fband != []:
        fig.add_vrect(
            x0=color_fband[0],
            x1=color_fband[1],
            line_width=0,
            fillcolor="#aaaaaa",
            opacity=0.5,
            layer="below",
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

    title = "Spectrum" if picks == [] else f"{picks}"

    fig = apply_default_styles(fig, xzero=False, yzero=False, row=row, col=col)
    fig.update_xaxes(title="Frequency [Hz]", row=row, col=col)
    fig.update_yaxes(title="Average power [dB]", row=row, col=col)
    fig.update_layout(legend=dict(x=0.7, y=1), title=title)

    return fig
