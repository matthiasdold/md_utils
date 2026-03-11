"""Resampler-based plotting utilities for ICA visualization.

This module contains plotting functions that use FigureResampler for efficient
rendering of large time series data in ICA analysis workflows.
"""

import plotly.graph_objects as go

from mdu.plotly.resampler_compat import FigureResampler


def create_raw_overlay_figure(
    inst_times,
    raw_data,
    filtered_data,
    resampler_fig: FigureResampler | None = None,
    relayout_data: dict | None = None,
) -> FigureResampler:
    """Create a resampled overlay plot of raw and ICA-filtered data.

    Parameters
    ----------
    inst_times : array-like
        Time points for the data
    raw_data : array-like
        Original raw channel data
    filtered_data : array-like
        ICA-filtered channel data
    resampler_fig : FigureResampler, optional
        Existing FigureResampler to update. If None, creates a new one.
    relayout_data : dict, optional
        Plotly relayout data for preserving zoom/pan state

    Returns
    -------
    FigureResampler
        Updated or new FigureResampler with raw and filtered traces
    """
    if resampler_fig is None:
        fig = FigureResampler()
    else:
        fig = resampler_fig

    # Clear existing traces if any
    if len(fig.data):
        fig.replace(go.Figure())

    # Add raw data trace
    fig.add_trace(
        go.Scattergl(name="raw", line=dict(color="#ff5555"), opacity=0.5),
        hf_x=inst_times,
        hf_y=raw_data,
    )

    # Add filtered data trace
    fig.add_trace(
        go.Scattergl(name="filtered", line=dict(color="#111")),
        hf_x=inst_times,
        hf_y=filtered_data,
    )

    # Update layout
    layout_kwargs = {
        "font": dict(size=16),
        "margin": dict(l=10, r=10, t=10, b=10),
    }

    # Preserve zoom/pan state if provided
    if relayout_data:
        layout_kwargs.update(parse_relayout_data(relayout_data))

    fig.update_layout(**layout_kwargs)

    return fig


def parse_relayout_data(layout: dict | None) -> dict:
    """Parse Plotly relayout data to extract axis range information.

    Extracts x-axis and y-axis range values from Plotly's relayoutData callback
    format and converts them to a format suitable for updating figure layout.

    Parameters
    ----------
    layout : dict or None
        The relayoutData dictionary from Plotly Dash callback containing keys like
        'xaxis.range[0]', 'xaxis.range[1]', etc., or None if no layout data exists.

    Returns
    -------
    dict
        Dictionary with 'xaxis_range' and/or 'yaxis_range' keys containing
        [min, max] lists, or empty dict if no range data found.
    """
    if layout is None:
        return {}

    ret_d = {}
    if "xaxis.range[0]" in layout.keys() and "xaxis.range[1]" in layout.keys():
        ret_d["xaxis_range"] = [
            layout["xaxis.range[0]"],
            layout["xaxis.range[1]"],
        ]

    if "yaxis.range[0]" in layout.keys() and "yaxis.range[1]" in layout.keys():
        ret_d["yaxis_range"] = [
            layout["yaxis.range[0]"],
            layout["yaxis.range[1]"],
        ]

    return ret_d
