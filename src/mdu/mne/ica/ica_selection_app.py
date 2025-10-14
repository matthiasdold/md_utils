#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20210722
#
# App for selecting components of EEG ICA projection

from functools import partial
from pathlib import Path

import dash
import mne
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from tqdm import tqdm

from mdu.mne.ica.ica_utils.shared import (attach_callbacks,
                                          create_ica_plot_overlay)
from mdu.plotly.mne_plotting import plot_topo, plot_variances
from mdu.plotly.mne_plotting_utils.epoch_image import plot_epo_image
from mdu.plotly.mne_plotting_utils.psd import plot_epo_psd
from mdu.plotly.mne_plotting_utils.time_series import plot_evoked_ts

# ==============================================================================
# Plotting functions
# ==============================================================================


def create_comp_i_figures(
    ica, ica_epos, epo, df, ncomponent, nth_row=1, color_by="stim"
):
    """Create the plotly figures for the ith ICA component.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        ICA instance to process.
    ica_epos : mne.Epochs
        Epochs in source space (ICA components).
    epo : mne.Epochs
        Regular epochs in sensor space.
    df : pd.DataFrame
        Data frame with epoch labels and behavioral info.
    ncomponent : int
        Number of the ICA component to display.
    nth_row : int, default=1
        Row number used to create odd/even class labels for coloring backgrounds.
    color_by : str, default="stim"
        Column name in df to use for coloring plots by group.

    Returns
    -------
    html.Div
        Dash HTML div for the ith ICA component including plots, buttons, and
        callbacks.
    """

    ch_name = ica_epos.ch_names[ncomponent]
    ica_component = ica_epos.copy().pick_channels([ch_name])
    # Ensure the ICA component is considered as EEG -> relevant for plotting only
    ica_component.set_channel_types({ch_name: "eeg"}, verbose=False)

    # prepare the figures
    figs = {}
    process_map = {
        "topomap": plot_topo,
        "image": partial(
            plot_epo_image,
            # plot_mode="base64",
            plot_mode="full",  # full plot helps with reading the meta data
            combine="mean",
            sort_by="stim",
        ),
        "erp": partial(plot_evoked_ts, combine="mean", color_by=""),
        "spectrum": partial(plot_epo_psd, picks=[ch_name]),
        "variance": plot_variances,
    }

    # create the figures
    for k, plot_func in process_map.items():
        tqdm.write(f"Processing: {k}")
        if k == "topomap":
            # Get the ICA weights
            contour_kwargs = {"colorscale": "Jet"}
            eeg_epo = epo.copy().crop(0, 0.1).pick_types(eeg=True)

            data = np.dot(
                ica.mixing_matrix_.T,
                ica.pca_components_[: ica.n_components_],
            )
            z = data[ncomponent, :]
            zmax = np.max(np.abs(z))
            zmin = -zmax
            dz = (zmax - zmin) / 20
            contour_kwargs["contours"] = dict(start=zmin, end=zmax, size=dz)
            fig = plot_func(z, eeg_epo, contour_kwargs=contour_kwargs)

            # have background white
            fig = fig.update_layout(
                # paper_bgcolor="#fff",
                plot_bgcolor="#fff",
            )
            figs[k] = fig
        elif k == "image":
            fig = plot_func(ica_component, df, color_by=color_by)
            fig = fig.update_yaxes(
                title="Epochs"
            )  # the epochs are sorted by stim --> no longer valid to call it Epcch Nbr.
            figs[k] = fig
        elif k == "erp":
            fig = plot_func(ica_component, df, color_by=color_by)
            fig = fig.update_layout(
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    font=dict(size=10),
                )
            )

            figs[k] = fig
        else:
            figs[k] = plot_func(ica_component, df, color_by=color_by)

    # Position in layout
    radio_value = "reject" if ncomponent in ica.exclude else "accept"
    row_type = "even" if nth_row % 2 == 0 else "odd"
    out_html = html.Div(
        id=f"ica_component_{ncomponent}",
        className=f"ica_component_row ica_component_row_{row_type}",
        children=[
            html.Div(
                className="firstCol",
                children=[
                    html.Div(
                        className="ICA_component_title", children=[ch_name]
                    ),
                    dcc.Graph(
                        id=f"graph_topo_{ncomponent}",
                        className="topoplot",
                        figure=figs["topomap"],
                    ),
                    dcc.RadioItems(
                        id=f"select_radio_{ncomponent}",
                        className="selection_radio",
                        options=[
                            {"label": "Accept", "value": "accept"},
                            {"label": "Reject", "value": "reject"},
                        ],
                        value=radio_value,
                        inputClassName="radio_input",
                        labelClassName="radio_label",
                    ),
                ],
            ),
            html.Div(
                id=f"div_erp_{ncomponent}",
                className="ica_erpplots_div",
                children=[
                    dcc.Graph(
                        id=f"graph_heatmap_{ncomponent}",
                        className="erp_heatmap",
                        figure=figs["image"],
                    ),
                    dcc.Graph(
                        id=f"graph_erp_traces_{ncomponent}",
                        className="erp_traces",
                        figure=figs["erp"],
                    ),
                ],
            ),
            html.Div(
                id=f"div_psd_and_var_{ncomponent}",
                className="ica_psd_and_var_col",
                children=[
                    dcc.Graph(
                        id=f"graph_spectrum_{ncomponent}",
                        className="spectra",
                        figure=figs["spectrum"],
                    ),
                    dcc.Graph(
                        id=f"graph_variance_{ncomponent}",
                        className="varianceplot",
                        figure=figs["variance"],
                    ),
                ],
            ),
        ],
    )
    return out_html


def create_layout_and_figures(
    ica: mne.preprocessing.ICA,
    epo: mne.BaseEpochs,
    nmax: int = -1,
    session: str = "",
) -> dash.html:
    """Create the layout and populate the figures

    Parameters
    ----------
    epo : mne.BaseEpochs
        the BaseEpochs to be filtered by ICA (usually not the same)
        as what is used for training!
    ica : mne.preprocessing.ICA
        the fitted ICA model
    nmax : int
        maximum number of components to include, if -1 (default) -> all
    session : str (optional)
        name of the session -> used for the title


    Returns
    -------
    layout: dash html object including go.Figures

    """

    df = epo.metadata.loc[epo.selection].reset_index(drop=True)
    ica_epos = ica.get_sources(epo)
    nmax = ica.n_components if nmax < 0 else nmax

    assert len(epo) == df.shape[0], "Missmatch <> epoch data and behavioral"

    layout = html.Div(
        id="ica_dash_main",
        children=[
            html.Div(session),
            html.Div(
                id="ica_header_row",
                children=[
                    html.Button(
                        "Save",
                        id="save_btn",
                        n_clicks=0,
                        className="non_saved_btn",
                    ),
                    "ICA Selection - Accepted / Rejected: ",
                    html.Div(f"{nmax}", id="accepted_count_div"),
                    html.Div("0", id="rejected_count_div"),
                    html.Div(
                        id="selection_bar",
                        children=[
                            html.A(
                                "",
                                id=f"selection_bar_{i}",
                                className="selection_bar_cell_green",
                                href=f"#ica_component_{i}",
                            )
                            for i in range(nmax)
                        ],
                    ),
                ],
            ),
            html.Div(
                id="ica_eval_body",
                children=[
                    html.Div(
                        id="ica_plots_div",
                        children=[
                            create_comp_i_figures(
                                ica, ica_epos, epo, df, i, nth_row=i + 1
                            )
                            for i in tqdm(
                                range(nmax), desc="Processing single plot"
                            )
                        ],
                    ),
                    html.Div(id="ica_plot_overlay_div", children=[]),
                ],
            ),
            html.Div(
                id="saved_state",
                children=["non_saved"],
                style={"display": "hidden"},
            ),
        ],
    )
    return layout


def build_ica_app(
    epo: mne.BaseEpochs,
    ica: mne.preprocessing.ICA,
    nmax: int = -1,
    ica_store_file: Path = Path("./wip_ica.fif"),
) -> dash.Dash:
    """Create an app given a session

    Parameters
    ----------
    epo : mne.BaseEpochs
        the BaseEpochs to be filtered by ICA (usually not the same)
        as what is used for training!
    ica : mne.preprocessing.ICA
        the fitted ICA model
    nmax : int
        maximum number of components to include, if -1 (default) -> all
    ica_store_file : Path
        path to the ica fif file to store the model (including the selection)

    Returns
    -------
    app : dash.Dash
        app for selection of components -> will update in the ica containers
        .header['selection']

    """

    n = ica.n_components
    if nmax >= 0:
        n = nmax

    assets_pth = Path(__file__).parent / "assets"
    app = dash.Dash(
        __name__, external_stylesheets=[assets_pth.joinpath("ica_styles.css")]
    )
    app.layout = create_layout_and_figures(ica, epo, nmax=n)
    app = attach_callbacks(app, n, ica, epo, ica_file=ica_store_file)
    return app
