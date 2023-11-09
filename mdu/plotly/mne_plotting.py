# Plotting methods to work on MNE objects

import mne
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from plotly_resampler import (register_plotly_resampler,
                              unregister_plotly_resampler)

from mdu.plotly.styling import apply_default_styles


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
