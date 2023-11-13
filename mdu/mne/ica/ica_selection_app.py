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
import matplotlib.pyplot as plt
import mne
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from tqdm.rich import tqdm

from mdu.plotly.mne_plotting import plot_variances
from mdu.plotly.mne_plotting_utils.epoch_image import plot_epo_image
from mdu.plotly.mne_plotting_utils.psd import plot_epo_psd
from mdu.plotly.mne_plotting_utils.time_series import plot_evoked_ts
from mdu.plotly.mne_plotting_utils.topoplot import plot_topo

# ==============================================================================
# Plotting functions
# ==============================================================================


def create_comp_i_figures(
    ica, ica_epos, df, ncomponent, nth_row=1, color_by="stim"
):
    """Create the plotly figures for the ith ICA component

    Parameters
    ----------
    ica : mne.ICA
        ica instance to porcess
    epo : mne.Epochs
        epoch data for psd and time-frequency maps
    ncomponet : int
        number of the component to display
    nth_row : int
        number of rows this will create -> create odd/even
        class labels for coloring the backgrounds
    df : pandas.DataFrame
        data frame with epoch labels and behavioral info

    Returns
    -------
    out_html : html.Div
        html div for the ith ICA component including plots, buttons and
        callbacks
    """

    ch_name = ica_epos.ch_names[ncomponent]
    ica_component = ica_epos.copy().pick_channels([ch_name])

    # prepare the figures
    topo_img_ax = ica.plot_components(picks=[ncomponent], show=False)
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
        # tqdm.write(f"Processing: {k}")
        if k == "topomap":
            figs[k] = plot_func(topo_img_ax.get_axes()[0])
            plt.close()  # close figure that was created in the backgrounds with matplotlib # noqa
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


def create_ica_plot_overlay(ica, epo):
    evk = epo.average()
    evk_clean = ica.apply(evk.copy())

    print(ica.exclude)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=evk.times,
            x=np.zeros(evk.times.shape),
            showlegend=False,
            hoverinfo="skip",
            line={"color": "#0000ff"},
        )
    )

    for i, yi in enumerate(evk.data):
        legend = True if i == 1 else False
        fig.add_trace(
            go.Scatter(
                y=evk.times,
                x=yi,
                name="raw",
                showlegend=legend,
                hoverinfo="skip",
                line={"color": "#ff0000"},
            )
        )

    for i, yi in enumerate(evk_clean.data):
        legend = True if i == 1 else False
        fig.add_trace(
            go.Scatter(
                y=evk.times,
                x=yi,
                name="ica_filtered",
                showlegend=legend,
                hoverinfo="skip",
                line={"color": "#000000"},
            )
        )

    fig.update_layout(
        dict(
            font=dict(
                size=18,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
    )

    return dcc.Graph(id="ica_overlay_graph", figure=fig)


# ==============================================================================
# Dash setup from xileh xPData
# ==============================================================================


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

    df = epo.metadata
    ica_epos = ica.get_sources(epo)

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
                                ica, ica_epos, df, i, nth_row=i + 1
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


def attach_callbacks(
    app: dash.Dash,
    ncomponents: int,
    ica: mne.preprocessing.ICA,
    epo: mne.BaseEpochs,
    ica_file: Path = Path("./wip_ica.fif"),
) -> dash.Dash:
    # dynamic header row
    @app.callback(
        [
            Output("accepted_count_div", "children"),
            Output("rejected_count_div", "children"),
        ]
        + [
            Output(f"selection_bar_{i}", "className")
            for i in range(ncomponents)
        ]
        + [Output("ica_plot_overlay_div", "children")],
        [Input(f"select_radio_{i}", "value") for i in range(ncomponents)],
    )
    def change_accepted_rejected_count(*radios):
        """
        accepted_ and rejected_count_div simply show the total amount of
        accepted and rejected redio boxes.

        The selection_bar_* will be a simple one char box either green or red
        for each component. -> coloring via css, we set the text to 1 or 0
        """

        # nicer to have a binary list for debug printing
        bin_list = [1 if v == "accept" else 0 for v in radios]
        selection_list_str = [
            "selection_bar_cell_green"
            if v == "accept"
            else "selection_bar_cell_red"
            for v in radios
        ]

        # update the exclude list
        ica.exclude = [i for i in range(len(bin_list)) if bin_list[i] == 0]

        return (
            [sum(bin_list), sum([i == 0 for i in bin_list])]
            + selection_list_str
            + [create_ica_plot_overlay(ica, epo)]
        )

    # saving the selection
    @app.callback(
        Output("save_btn", "className"),
        Input("save_btn", "n_clicks"),
        [Input(f"select_radio_{i}", "value") for i in range(ncomponents)],
    )
    def save_to_file_or_change_color(nclick, *radios):
        """Save the ica model with the selection to a fif"""

        save_str = "non_saved_btn"
        # find which was the change -> take just the first as we would not have
        # a simulatneous change of two inputs
        ctx = dash.callback_context.triggered[0]

        if ctx["prop_id"] == "save_btn.n_clicks":
            ica_file = ica.header["file"]
            ica.exclude = [i for i, v in enumerate(radios) if v == "reject"]

            ica.save(ica_file, overwrite=True)

            save_str = "saved_btn"

        return save_str

    # Placeholder --> preview of the projection given the new selection

    return app


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

    app = dash.Dash(__name__, external_stylesheets=["assets/ica_styles.css"])
    app.layout = create_layout_and_figures(epo, nmax=n)
    app = attach_callbacks(app, n, ica, epo, ica_file=ica_store_file)
    return app
