# a simplified version of the selection app
#
# Save button   --   Drop selection chart
# -------------------------------------------------
#
#   Resampler trace of the input raw
#
# -------------------------------------------------
#
#   Overlay view with ica internal chunking of raw
#
# -------------------------------------------------
#
#  Scroll area with component pngs (for speed)
#
#
# TODO:  [ ] - resampler seems not to work properly
#        [ ] - graph for the epoched overlay is very small

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from plotly_resampler import FigureResampler
from tqdm import tqdm

from mdu.mne.ica.ica_utils.shared import attach_callbacks


def matplotlib_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


class SelectionApp:
    def __init__(
        self,
        ica: mne.preprocessing.ICA,
        inst: mne.Epochs | mne.io.BaseRaw,
        save_model_path: Path = Path("./wip_ica.fif"),
    ):
        self.ica = ica
        self.inst = inst
        self.save_path = save_model_path
        self.resampler_fig: FigureResampler = FigureResampler()

        # add the epochs as used by ica plotting
        if not isinstance(inst, mne.Epochs):
            self.epo = mne.epochs.make_fixed_length_epochs(
                self.inst, duration=2.0, preload=True, proj=False
            )
        else:
            self.epo = self.inst

        # Prepare the matplotlib figures
        self.figs = self.ica.plot_properties(
            self.inst,
            show=False,
            picks=range(self.ica.n_components),
            figsize=(20, 10),
        )
        plt.close("all")
        self.convert_fig_to_base64()

        # Prepare the dash app
        try:
            assets_pth = Path(__file__).parent / "assets"
        except NameError:
            assets_pth = Path("./src/mdu/ica/assets").resolve()

        self.app = Dash(
            __name__,
            # assets_folder=assets_pth,
            external_stylesheets=["ica_styles.css"],
        )
        self.create_layout()
        self.app = attach_callbacks(
            self.app,
            self.ica.n_components,
            ica=self.ica,
            epo=self.epo,
            ica_file=self.save_path,
        )
        self.add_resampler_callback()

    def convert_fig_to_base64(self):
        self.figs_base64 = []
        for fig in tqdm(self.figs, desc="Converting figs to base64"):
            self.figs_base64.append(
                f"data:image/png;base64,{matplotlib_to_base64(fig)}"
            )

    def run(self, **kwargs):
        self.app.run_server(**kwargs)

    def create_layout(self):
        layout = html.Div(
            id="selection-app",
            children=[
                html.Div(
                    id="top_segment",
                    children=[
                        html.Div(
                            id="ica_header_row",
                            children=[
                                html.Button(
                                    "Save",
                                    id="save_btn",
                                    n_clicks=0,
                                    className="non_saved_btn",
                                ),
                                dcc.Dropdown(
                                    self.inst.ch_names,
                                    self.inst.ch_names[0],
                                    id="ch_dropdown",
                                ),
                                "ICA Selection - Accepted / Rejected: ",
                                html.Div(
                                    f"{self.ica.n_components}",
                                    id="accepted_count_div",
                                ),
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
                                        for i in range(self.ica.n_components)
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="raw_overlay_div",
                            children=[
                                dcc.Graph(id="graph_raw_overlay"),
                            ],
                        ),
                        html.Div(
                            id="ica_plot_overlay_div",
                            className="ica_overlay_horizontal",
                        ),
                    ],
                ),
                html.Div(id="figs", children=create_figs(self)),
            ],
        )

        self.app.layout = layout

    def add_resampler_callback(self):
        @self.app.callback(
            Output("graph_raw_overlay", "figure"),
            State("graph_raw_overlay", "relayoutData"),
            (
                [Input("ch_dropdown", "value")]
                + [
                    Input(f"select_radio_{i}", "value")
                    for i in range(self.ica.n_components)
                ]
            ),
        )
        def resample_raw(relayout_data: dict, channel: str, *radios):
            # update current exclude
            self.ica.exclude = [i for i, v in enumerate(radios) if v == "reject"]
            yc = self.inst.copy().pick([channel]).get_data()[0]
            yica = self.ica.apply(self.inst.copy()).pick([channel]).get_data()[0]
            fig = self.resampler_fig

            if len(fig.data):
                # Replace the figure with an empty one to clear the graph
                fig.replace(go.Figure())
            fig.add_trace(
                go.Scattergl(name="raw", line=dict(color="#ff5555"), opacity=0.5),
                hf_x=self.inst.times,
                hf_y=yc,
            )
            fig.add_trace(
                go.Scattergl(name="filtered", line=dict(color="#111")),
                hf_x=self.inst.times,
                hf_y=yica,
            )

            fig = fig.update_layout(
                font=dict(
                    size=16,
                ),
                margin=dict(l=10, r=10, t=10, b=10),
                **parse_relayout_data(relayout_data),
            )

            return fig


def parse_relayout_data(layout: dict | None) -> dict:
    """Just parse the x/y ranges"""
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


def create_figs(app: SelectionApp) -> list[html.Div]:
    divs = []
    for i, fig in enumerate(app.figs_base64):
        row_type = "even" if i % 2 == 0 else "odd"
        radio_value = "reject" if i in app.ica.exclude else "accept"
        div = html.Div(
            className=f"ica_component_row ica_component_row_{row_type}",
            children=[
                dcc.RadioItems(
                    id=f"select_radio_{i}",
                    className="selection_radio",
                    options=[
                        {"label": "Accept", "value": "accept"},
                        {"label": "Reject", "value": "reject"},
                    ],
                    value=radio_value,
                    inputClassName="radio_input",
                    labelClassName="radio_label",
                ),
                # the figure
                dcc.Graph(
                    id=f"graph_base64_{i}",
                    className="base64_component_graph",
                    figure=go.Figure(
                        layout=go.Layout(
                            images=[
                                go.layout.Image(
                                    source=fig,
                                    sizex=1,
                                    sizey=1,
                                    yanchor="bottom",
                                )
                            ],
                            template="plotly_white",
                            height=600,
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False, scaleanchor="x"),
                            margin=dict(l=0, r=0, t=0, b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        ),
                    ),
                ),
            ],
        )
        divs.append(div)

    return divs


def test_sum(x):
    print(sum(x))
    a = [
        "A",
    ]
    return a


if __name__ == "__main__":
    nch = 16
    tmax = 10

    sfreq = 100
    times = np.linspace(0, tmax, tmax * sfreq)
    x = np.vstack(
        [np.sin(times * i) + np.random.randn(len(times)) for i in range(1, nch + 1)]
    )

    mnt = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(mnt.ch_names[:nch], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(x, info)
    raw.set_montage(mnt)

    ica = mne.preprocessing.ICA(n_components=nch)
    ica.fit(raw.copy().filter(1, 40))

    ica.plot_properties(raw, picks=range(nch), show=False)

    fig = go.Figure(go.Scattergl(x=times, y=x[0]))
    fig.update_layout({"xaxis_range": [0, 5]})

    __file__ = Path("src/mdu/mne/ica/simple_app.py").resolve()
    self = SelectionApp(ica, raw)

    self.app.run_server(debug=True)
