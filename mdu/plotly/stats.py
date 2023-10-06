from functools import partial
from typing import Callable

import numpy as np
import plotly.graph_objs as go
import statsmodels.api as sm

from mdu.utils.converters import ToFloatConverter


def add_statsmodelfit(
    fig: go.Figure,
    x: np.ndarray,
    y: np.ndarray,
    fitfunc: Callable = sm.OLS,
    row: int | None = None,
    col: int | None = None,
    ci_alpha: float = 0.05,
    show_ci: bool = True,
    show_obs_ci: bool = True,
    line_kwargs: dict = {
        "line": {"color": "#222"},
    },
    ci_kwargs: dict = {
        "fill": "toself",
        "fillcolor": "#222",
        "line_color": "#222",
        "opacity": 0.2,
    },
    obs_ci_kwargs: dict = {
        "line": {"dash": "dash", "color": "#222"},
        "opacity": 0.5,
    },
) -> go.Figure:
    """Add a statsmodels fit to a plotly figure

    Parameters
    ----------
    fig : go.Figure
        figure to add to

    x : np.ndarray
        array of endogenous variables (currently only 1D supported)

    y : np.ndarray
        array of exogenous variables to fit to

    fitfunc : Callable
        statsmodels fit function, default is OLS

    row : int | None
        used to add to specific subplot

    col : int | None
        used to add to specific subplot

    ci_alpha : float
        alpha value to use for confidence intervals. Default is 0.05 -> CIs are
        2.5% and 97.5% quantiles

    show_ci : bool
        show the confidence interval

    show_obs_ci : bool
        show the observed confidence interval / prediction interval

    line_kwargs : dict
        options passed to the plotting of the fit line

    ci_kwargs : dict
        options passed to the plotting of the confidence interval

    obs_ci_kwargs : dict
        options passed to the plotting of the observed confidence interval


    Returns
    -------
    go.Figure
        modified figure

    """
    tfc = ToFloatConverter()
    xorig = x.copy()
    x = tfc.to_float(x)

    assert len(x.shape) == 1, "x must be a 1D array - TODO: inplement more"

    # add constant for intercept
    x = sm.add_constant(x)
    model = fitfunc(y, x).fit()
    statframe = model.get_prediction(x).summary_frame(alpha=ci_alpha)
    if show_ci:
        fig.add_scatter(
            x=np.hstack([xorig, xorig[::-1]]),
            y=np.hstack(
                [statframe["mean_ci_upper"], statframe["mean_ci_lower"][::-1]]
            ),
            name=f"{ci_alpha:.0%} fit CI",
            hoverinfo="skip",  # hover with the filled trace is tricky
            mode="lines",
            **ci_kwargs,
            row=row,
            col=col,
        )

    if show_obs_ci:
        fig.add_scatter(
            x=xorig,
            y=statframe["obs_ci_upper"],
            name=f"{ci_alpha:.0%} fit obs CI upper",
            mode="lines",
            legendgroup="obs_ci",
            hovertemplate="Obs CI: %{y}<br>x: %{x}",
            row=row,
            col=col,
            **obs_ci_kwargs,
        )
        fig.add_scatter(
            x=xorig,
            y=statframe["obs_ci_lower"],
            name=f"{ci_alpha:.0%} fit obs CI lower",
            hovertemplate="Obs CI: %{y}<br>x: %{x}",
            row=row,
            col=col,
            mode="lines",
            legendgroup="obs_ci",
            **obs_ci_kwargs,
        )

    stat_text = "<br>".join(f"{model.summary()}".split("\n")[:-5])

    fig.add_scatter(
        x=xorig,
        y=statframe["mean"],
        mode="lines",
        name="fit line",
        hovertemplate="<b>Pred. mean: %{y}, x: %{x}</b><br><br>%{text}",
        hoverlabel=dict(font=dict(family="monospace"), bgcolor="#ccc"),
        text=[stat_text] * len(xorig),
        row=row,
        col=col,
        **line_kwargs,
    )

    return fig
