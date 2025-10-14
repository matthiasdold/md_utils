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
    """
    Load and process configuration from YAML file.

    Reads the config.yaml file, flattens the configuration dictionary,
    and replaces any template strings with their corresponding values
    from the flattened configuration.

    Returns
    -------
    dict
        The processed configuration dictionary with all templates replaced.

    Notes
    -----
    This function expects a 'config.yaml' file to exist in the current
    working directory. Template strings in the format <key.path> or
    <key.path[index]> will be replaced with their corresponding values.
    """
    conf = yaml.safe_load(open("config.yaml"))
    conf = replace_templates(conf, flatten_dict(conf))
    return conf


def replace_templates(conf, conf_flat):
    """
    Replace template strings in configuration with actual values.

    Recursively searches through a configuration dictionary for template
    strings in the format <key.path> or <key.path[index]> and replaces
    them with the corresponding values from the flattened configuration.

    Parameters
    ----------
    conf : dict
        The configuration dictionary that may contain template strings.
    conf_flat : dict
        A flattened version of the configuration dictionary where nested
        keys are joined with dots (e.g., 'parent.child').

    Returns
    -------
    dict
        The configuration dictionary with all template strings replaced
        with their corresponding values.

    Raises
    ------
    KeyError
        If a template string references a key that doesn't exist in the
        flattened configuration.

    Notes
    -----
    Template strings can include array indexing (e.g., <some.key[0]>).
    The function recursively processes nested dictionaries.
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
    """
    Flatten a nested dictionary into a single-level dictionary.

    Recursively flattens nested dictionaries by joining keys with dots.
    For example, {'a': {'b': 1}} becomes {'a.b': 1}.

    Parameters
    ----------
    d_in : dict
        The input dictionary to flatten, which may contain nested dictionaries.

    Returns
    -------
    dict
        A flattened dictionary where nested keys are joined with dots.
        Non-dictionary values remain unchanged.

    Examples
    --------
    >>> flatten_dict({'a': {'b': 1, 'c': 2}, 'd': 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}
    """
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
    """
    Validate that pipeline data contains a configuration container.

    Checks if the provided pipeline data object has a 'config' container
    and raises an assertion error if it doesn't.

    Parameters
    ----------
    pdata : xPData
        Pipeline data object that should contain a 'config' container.

    Returns
    -------
    xPData
        The same pipeline data object that was passed in, unchanged.

    Raises
    ------
    AssertionError
        If the 'config' container is not found in pdata.

    Notes
    -----
    This function is typically used as a validation step in a pipeline
    to ensure required configuration is present before proceeding.
    """
    conf = pdata.get_by_name("config")
    assert conf is not None, "This pipeline requires a 'config' container"
    return pdata


def make_choice(options, allow_multiple=True):
    """
    Interactively prompt user to select from a list of options.

    Displays a numbered list of options and prompts the user to select
    one or more items via command-line input. Supports single or multiple
    selections, including selecting all options at once.

    Parameters
    ----------
    options : list
        List of options to choose from. Can be any iterable of items
        that can be converted to strings for display.
    allow_multiple : bool, optional
        If True, allows selecting multiple options via comma-separated
        indices (e.g., '1,2,3') or 'a' for all. If False, only single
        selection is allowed. Default is True.

    Returns
    -------
    list
        List of selected options from the input list. Returns all options
        (except 'all' if present) if 'a' was selected.

    Notes
    -----
    The function will continue prompting until valid input is received.
    Valid input consists of comma-separated indices within the range
    of available options, or 'a' for all (if allow_multiple is True).
    """

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
    """
    Load epoch FIF files from disk and add to pipeline data.

    Searches for epoch FIF files (*epo.fif) in the processed data folder
    specified in the configuration, optionally filters them using a regular
    expression, and prompts user selection if multiple files are found.

    Parameters
    ----------
    pdata : xPData
        Pipeline data object containing configuration and where the loaded
        epochs will be added.
    trg_container : str, optional
        Name for the target container to store the loaded epochs.
        Default is 'epos'.
    filter_exp : str or None, optional
        Regular expression pattern to filter the list of found FIF files.
        If None, all *epo.fif files are included. Default is None.

    Returns
    -------
    xPData
        The pipeline data object with the loaded epochs added as a new
        container with the specified name.

    Raises
    ------
    ValueError
        If the filter_exp leads to dropping all available FIF files.

    Notes
    -----
    The function uses cached_mne_read_epo for efficient file loading.
    If multiple files match after filtering, the user is prompted to
    select one interactively. File metadata is stored in the container
    header for tracking purposes.
    """

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
    """
    Apply common reference to EEG epochs data.

    Re-references the EEG data in the specified container using MNE's
    set_eeg_reference function. The operation is performed in-place on
    the source container's data.

    Parameters
    ----------
    pdata : xPData
        Pipeline data object containing the epochs to be re-referenced.
    src_container : str, optional
        Name of the container holding the epochs data to re-reference.
        Default is 'epos'.
    ref_channels : str or list, optional
        Reference channel(s) to use. Can be 'average' for average reference,
        a channel name, or a list of channel names. Default is 'average'.

    Returns
    -------
    xPData
        The pipeline data object with the re-referenced epochs data.

    Notes
    -----
    This function modifies the data in-place. The original unreferenced
    data is not preserved.
    """
    src = pdata.get_by_name(src_container)
    mne.set_eeg_reference(src.data, ref_channels=ref_channels)

    return pdata


@cached
def cached_mne_read_epo(fpath):
    """
    Load MNE epochs from FIF file with caching.

    A cached wrapper around mne.read_epochs that loads epoch data with
    preloading enabled. The @cached decorator ensures that repeated
    calls with the same file path return the cached result.

    Parameters
    ----------
    fpath : str or Path
        Path to the epoch FIF file to load.

    Returns
    -------
    mne.Epochs
        The loaded epochs object with data preloaded into memory.

    Notes
    -----
    The caching mechanism improves performance when the same file is
    read multiple times during a session. Data is always preloaded
    for faster subsequent operations.
    """
    return mne.read_epochs(fpath, preload=True)


def filter_epo_data(
    pdata,
    src_container="epos",
    trg_container="epos",
    fband=[0.1, 300],
    **kwargs,
):
    """
    Apply bandpass filter to epoch data.

    Filters the epochs data from the source container using the specified
    frequency band and stores the result in the target container. If the
    target container doesn't exist, it will be created.

    Parameters
    ----------
    pdata : xPData
        Pipeline data object containing the epochs to filter.
    src_container : str, optional
        Name of the container holding the source epochs data.
        Default is 'epos'.
    trg_container : str, optional
        Name of the container to store the filtered epochs data.
        Can be the same as src_container to replace the original data.
        Default is 'epos'.
    fband : list of float, optional
        Two-element list specifying the lower and upper frequency bounds
        for the bandpass filter in Hz. Default is [0.1, 300].
    **kwargs : dict
        Additional keyword arguments passed to mne.Epochs.filter().

    Returns
    -------
    xPData
        The pipeline data object with the filtered epochs in the target
        container.

    Notes
    -----
    The function creates a copy of the epochs before filtering to avoid
    modifying the original data unintentionally. If trg_container differs
    from src_container and doesn't exist, a new container is created with
    a descriptive header.
    """
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
    """
    Create an overlay plot comparing raw and ICA-filtered EEG data.

    Generates a Plotly graph showing the averaged epochs before and after
    ICA filtering. The plot overlays raw data (in red with transparency)
    and filtered data (in black) to visualize the effect of ICA component
    exclusion.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        The ICA object with components marked for exclusion in ica.exclude.
    epo : mne.BaseEpochs
        The epochs data to average and compare.

    Returns
    -------
    dash.dcc.Graph
        A Dash graph component containing the overlay plot with the ID
        'ica_overlay_graph'.

    Notes
    -----
    The function prints the current ica.exclude list for debugging.
    The plot includes:
    - A zero reference line (blue)
    - Raw evoked data traces (red, 50% opacity)
    - ICA-filtered evoked data traces (black)
    Only the second trace of each type is shown in the legend to avoid
    clutter.
    """
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
    """
    Attach interactive callbacks to a Dash app for ICA component selection.

    Sets up two main callbacks for the ICA selection interface:
    1. Updates component selection display and overlay plot based on radio
       button selections
    2. Saves the ICA model with selected components to a FIF file

    Parameters
    ----------
    app : dash.Dash
        The Dash application to attach callbacks to.
    ncomponents : int
        Number of ICA components to create selection controls for.
    ica : mne.preprocessing.ICA
        The ICA object whose exclude list will be updated based on user
        selections.
    epo : mne.BaseEpochs
        The epochs data used for generating overlay plots.
    ica_file : Path, optional
        Path where the ICA model will be saved when the save button is
        clicked. Default is Path('./wip_ica.fif').

    Returns
    -------
    dash.Dash
        The Dash application with callbacks attached.

    Notes
    -----
    The function creates two callbacks:
    - change_accepted_rejected_count: Updates the acceptance/rejection
      counts, visual indicators, and overlay plot in real-time
    - save_to_file_or_change_color: Saves the ICA model and changes the
      save button color to indicate save status

    The ica.exclude list is updated dynamically based on component
    selections (components marked as 'reject' are added to ica.exclude).
    """
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
