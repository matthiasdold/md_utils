#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# author: Matthias Dold
# date: 20210616
#
# Functions shared by multiple pipelines

import re
from pathlib import Path

import dash
import mne
import numpy as np
import plotly.graph_objects as go
import yaml
from dash import dcc
from dash.dependencies import Input, Output
from memoization import cached
from xileh.core.pipelinedata import xPData


def load_config():
    conf = yaml.safe_load(open("config.yaml"))
    conf = replace_templates(conf, flatten_dict(conf))
    return conf


def replace_templates(conf, conf_flat):
    """
    Find <[^>]*> in the strings and replace with appropriate key val

    The template <some.thing[0]> might include an index [0] --> assuming
    that the value for some.thing is iterable, choose according to index
    """
    for k, v in conf.items():
        if isinstance(v, str):
            templates = re.findall("<([^>]*)>", v)

            for tmp in templates:
                try:
                    idx = re.findall(r"\[(\d*)\]", tmp)
                    tmp_stump = re.sub(r"\[(\d*)\]", "", tmp)
                    rval = conf_flat[tmp_stump]
                    rval = rval[int(idx[0])] if idx != [] else rval
                    v = v.replace(f"<{tmp}>", str(rval))
                except KeyError:
                    raise KeyError(
                        f"Template str <{tmp}> not pointing to "
                        " a valid key in config. Cannot replace!"
                    )
            conf[k] = v
        elif isinstance(v, dict):
            conf[k] = replace_templates(v, conf_flat)

    return conf


def flatten_dict(d_in):
    """flatten a dict if any of its values is a dict -> use key as prefix"""
    d = d_in.copy()
    # list as we do not want a generator
    kvals = [(k, v) for k, v in d.items()]
    for k, v in kvals:
        if isinstance(v, dict):
            new_d = {
                ".".join([k, kv]): vv for kv, vv in flatten_dict(v).items()
            }
            d.pop(k)
            d.update(new_d)

    return d


def has_config(pdata):
    conf = pdata.get_by_name("config")
    assert conf is not None, "This pipeline requires a 'config' container"
    return pdata


def make_choice(options, allow_multiple=True):
    """Select out of a list of options"""

    choice_index = [str(i) for i in range(len(options))]
    msg = "Please select one"

    if allow_multiple:
        choice_index += ["a"]
        options += ["all"]
        msg += " or multiple (e.g. 1 or 1,2,3)"

    choice_msg = "\n".join(
        [f"{i}: {o}" for i, o in zip(choice_index, options)]
    )

    selection = []
    # Select at least on index
    while set(selection) - set(choice_index) != set() or selection == []:
        selection_str = input(choice_msg + f"\n{msg}: ")

        selection = selection_str.split(",")

    if selection == ["a"]:
        return options[:-1]
    else:
        return [options[int(i)] for i in selection]


def load_epo_fif(pdata, trg_container="epos", filter_exp=None):
    """Read the raw fif from disc"""

    conf = pdata.get_by_name("config").data
    sess_root = Path(conf["data_root"]).joinpath(conf["session"])
    prsd_folder = sess_root.joinpath(conf["processed_folder"])

    epo_fifs = list(prsd_folder.rglob("*epo.fif"))
    ln = len(epo_fifs)

    if filter_exp:
        epo_fifs = [f for f in epo_fifs if re.match(filter_exp, str(f))]

        if epo_fifs == [] and ln > 0:
            raise ValueError(
                f"Expression <{filter_exp}> lead to dropping all"
                f" {ln} potential *epo.fifs in {prsd_folder}"
                " check if expression is complete .*<something>.*"
            )

    if len(epo_fifs) > 1:
        selected_epo_fif = make_choice(epo_fifs, allow_multiple=False)[0]
    else:
        selected_epo_fif = epo_fifs[0]

    epochs = xPData(
        cached_mne_read_epo(selected_epo_fif),
        header={
            "name": trg_container,
            "file": selected_epo_fif,
            "file_stat": selected_epo_fif.stat(),
        },
    )

    pdata.data.append(epochs)
    return pdata


def apply_common_reference(
    pdata, src_container="epos", ref_channels="average"
):
    src = pdata.get_by_name(src_container)
    mne.set_eeg_reference(src.data, ref_channels=ref_channels)

    return pdata


@cached
def cached_mne_read_epo(fpath):
    """Given the fif file path call mne.read_epochs"""
    return mne.read_epochs(fpath, preload=True)


def filter_epo_data(
    pdata,
    src_container="epos",
    trg_container="epos",
    fband=[0.1, 300],
    **kwargs,
):
    epo = pdata.get_by_name(src_container).data

    if (
        trg_container != src_container
        and pdata.get_by_name(trg_container) is None
    ):
        trg_c = xPData(
            [],
            header={
                "name": trg_container,
                "description": "The bandpass filtered epochs",
            },
        )
        pdata.data.append(trg_c)
    else:
        trg_c = pdata.get_by_name(trg_container)

    epo_bf = epo.copy().filter(*fband, **kwargs)
    trg_c.data = epo_bf

    return pdata


def create_ica_plot_overlay(ica, epo):
    evk = epo.average()
    evk_clean = ica.apply(evk.copy())

    print(ica.exclude)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=evk.times,
            y=np.zeros(evk.times.shape),
            showlegend=False,
            hoverinfo="skip",
            line={"color": "#0000ff"},
        )
    )

    for i, yi in enumerate(evk.data):
        legend = True if i == 1 else False
        fig.add_trace(
            go.Scatter(
                x=evk.times,
                y=yi,
                name="raw",
                showlegend=legend,
                hoverinfo="skip",
                line={"color": "#ff0000"},
                opacity=0.5,
            )
        )

    for i, yi in enumerate(evk_clean.data):
        legend = True if i == 1 else False
        fig.add_trace(
            go.Scatter(
                x=evk.times,
                y=yi,
                name="filtered",
                showlegend=legend,
                hoverinfo="skip",
                line={"color": "#000000"},
            )
        )

    fig.update_layout(
        dict(
            font=dict(
                size=16,
            ),
            # margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            # legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            xaxis=dict(title="Time [s]"),
        )
    )

    return dcc.Graph(id="ica_overlay_graph", figure=fig)


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
            (
                "selection_bar_cell_green"
                if v == "accept"
                else "selection_bar_cell_red"
            )
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
            ica.exclude = [i for i, v in enumerate(radios) if v == "reject"]

            ica.save(ica_file, overwrite=True)

            save_str = "saved_btn"

        return save_str

    # Placeholder --> preview of the projection given the new selection

    return app
