# plotting founctionality around data distrinution
#
# This builds on statsmodels.api.ProbPlot as it provides convenient
# core attributes and methods
#
#
# TODO:
# [ ] consider adding Anderson-Darling test - https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/regression/supporting-topics/regression-models/what-is-a-confidence-band/
# [ ] adding other distributions, akin to minitab https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/quality-tools/how-to/individual-distribution-identification/methods-and-formulas/probability-plot/
#
from functools import partial
from typing import Callable, Literal, Optional

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# import probscale
import statsmodels.api as sm

# from autograd import jacobian as jac
# from matplotlib import pyplot
# from plotly._subplots import _get_grid_subplot, _get_subplot_ref_for_trace
from plotly.subplots import make_subplots

# from probscale.viz import validate as ax_validate
from scipy import stats

# from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# from mdu.plotly.styling import apply_default_styles


def prepare_subplots(shape: tuple, facet_col_wrap: int = 4) -> go.Figure:
    ncols = min(shape[1], facet_col_wrap)
    nrows = int(np.ceil(shape[1] / ncols))

    # Prepare empty titles for simplified use later
    subplot_titles = [""] * (nrows * ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles)

    return fig


def probplot(
    x: np.ndarray,
    distgen: Callable = stats.distributions.norm,
    distargs: Optional[tuple] = None,
    fit: bool = True,
    facet_col_wrap: int = 4,
    ci: float | None = 0.95,
    invert_axis: bool = False,
) -> go.Figure:
    """Plot probability plot for normal distribution

    Mainly using the following
    https://www.statsmodels.org/stable/_modules/statsmodels/graphics/gofplots.html#ProbPlot

    Axis are sorted same as used by minitab
    https://support.minitab.com/en-us/minitab/help-and-how-to/graphs/probability-plot/interpret-the-results/key-results/

    Parameters
    ----------
    x : np.ndarray
        data array with samples in first dimension, further dimensions are
        considered as different variables

    distgen : Callable (stats.distributions)
        the distribution object to use, by default stats.distributions.norm

    distargs : Optional[tuple]
        additional arguments for the distribution, by default None

    facet_col_wrap : int, optional
        number of columns to wrap the facet grid, by default 4

    ci : float | None
        the confidence interval to add to the plot, by default 0.95

    invert_axis :
        whether to invert the axis, by default False


    Returns
    -------
    go.Figure

    """

    x = x.reshape(-1, 1) if x.ndim == 1 else x
    fig = prepare_subplots(x.shape, facet_col_wrap=facet_col_wrap)
    grid = list(fig._get_subplot_coordinates())

    a = 3 / 8 if len(x) <= 10 else 0.5  # matching pingouin
    # a = 0.3  # matching minitab

    for i, xx in enumerate(x.T):
        row, col = grid[i]
        pp = sm.ProbPlot(xx, dist=distgen, distargs=distargs, fit=True, a=a)
        if invert_axis:
            x = pp.sorted_data
            y = pp.theoretical_quantiles
        else:
            x = pp.theoretical_quantiles
            y = pp.sorted_data

        # Note: the axis are flipped compared to statsmodels standard -> following the minitab convention
        fig = fig.add_scatter(
            x=x,
            y=y,
            mode="markers",
            row=row,
            col=col,
            marker=dict(color="blue"),
        )

        # add CI - following code from pingouin.plotting.qqplot
        if ci is not None:
            fig = add_ci_and_line(
                fig,
                pp=pp,
                y=y,
                x=x,
                invert_axis=invert_axis,
                row=row,
                col=col,
            )

        ax = "y" if invert_axis else "x"
        fig = fmt_probplot_axis(fig, pp, ax=ax, row=row, col=col)

    return fig


def qq_plot(
    x: np.ndarray,
    distgen: Callable = stats.distributions.norm,
    distargs: Optional[tuple] = None,
    fit: bool = True,
    facet_col_wrap: int = 4,
    ci: float | None = 0.95,
) -> go.Figure:

    x = x.reshape(-1, 1) if x.ndim == 1 else x
    fig = prepare_subplots(x.shape, facet_col_wrap=facet_col_wrap)
    grid = list(fig._get_subplot_coordinates())

    a = 3 / 8 if len(x) <= 10 else 0.5  # matching pingouin
    # a = 0.3  # matching minitab

    for i, xx in enumerate(x.T):
        row, col = grid[i]
        pp = sm.ProbPlot(xx, dist=distgen, distargs=distargs, fit=True, a=a)

        fig = fig.add_scatter(
            x=pp.theoretical_quantiles,
            y=pp.sample_quantiles,
            mode="markers",
            row=row,
            col=col,
        )

        # ensure yaxis is symmetric
        pad = 0.05
        max_y = np.max(np.abs(pp.sample_quantiles)) * (1 + pad)
        fig = fig.update_yaxes(range=[-max_y, max_y])

        if ci is not None:
            fig = add_ci_and_line(
                fig,
                pp=pp,
                y=pp.sample_quantiles,
                x=pp.theoretical_quantiles,
                row=row,
                col=col,
            )

    return fig


def pp_plot(
    x: np.ndarray,
    distgen: Callable = stats.distributions.norm,
    distargs: Optional[tuple] = None,
    fit: bool = True,
    facet_col_wrap: int = 4,
    ci: float | None = 0.95,
) -> go.Figure:

    x = x.reshape(-1, 1) if x.ndim == 1 else x
    fig = prepare_subplots(x.shape, facet_col_wrap=facet_col_wrap)
    grid = list(fig._get_subplot_coordinates())

    a = 3 / 8 if len(x) <= 10 else 0.5  # matching pingouin
    # a = 0.3  # matching minitab

    for i, xx in enumerate(x.T):
        row, col = grid[i]
        pp = sm.ProbPlot(xx, dist=distgen, distargs=distargs, fit=True, a=a)

        fig = fig.add_scatter(
            x=pp.theoretical_percentiles,
            y=pp.sample_percentiles,
            mode="markers",
            row=row,
            col=col,
        )

        if ci is not None:
            fig = add_ci_and_line(
                fig,
                pp=pp,
                y=pp.sample_percentiles,
                x=pp.theoretical_percentiles,
                input_type="percentiles",
                row=row,
                col=col,
            )

    return fig


def fwd_transform(x: np.ndarray, pp: sm.ProbPlot, input_type: str) -> np.ndarray:
    if input_type == "quantiles":
        return x
    else:
        return pp.dist.ppf(x)


def back_transform(x: np.ndarray, pp: sm.ProbPlot, input_type: str) -> np.ndarray:
    if input_type == "quantiles":
        return x
    else:
        return pp.dist.cdf(x)


def add_ci_and_line(
    fig: go.Figure,
    pp: sm.ProbPlot,
    x: np.ndarray,
    y: np.ndarray,
    ci: float = 0.95,
    row: int | None = None,
    col: int | None = None,
    input_type: Literal["quantiles", "percentiles"] = "quantiles",
    invert_axis: bool = False,
) -> go.Figure:

    # in the inverted scenario, we still fit from an x == theoretical quantiles
    # perspective
    xw = y if invert_axis else x
    yw = x if invert_axis else y

    # calculations are done in quantiles -> plotting is potentially in percentiles
    n = len(xw)
    # NOTE: the regression result here is slightly different from the
    # what the output of pp.fit_params is --> why is not clear?
    slope, intercept, r, prob, _ = stats.linregress(xw, yw)
    fit_val = slope * xw + intercept

    xp = fit_val if invert_axis else xw
    yp = xw if invert_axis else fit_val
    slope_label = 1 / slope if invert_axis else slope
    intercept_label = -intercept / slope if invert_axis else intercept

    fig.add_scatter(
        x=xp,
        y=yp,
        mode="lines",
        line_color="rgba(0, 0, 0, 0.8)",
        row=row,
        col=col,
        name=(
            "OLS:"
            f"<br>{'slope:':<10}{slope_label:.2f}"
            f"<br>{'interc:':<10}{intercept_label:.2f}"
            f"<br>{'r:':<10}{r:.2f}"
            f"<br>{'pval:':<10}{prob:.4f}"
        ),
    )

    a = 3 / 8 if n <= 10 else 0.5
    # a=0.3  # matching minitab
    prob_points = (np.arange(n) + 1 - a) / (n + 1 - 2 * a)
    crit = stats.norm.ppf(
        1 - (1 - ci) / 2
    )  # crit vals accoring to normal dist --> for standard errors

    fwd_tf = partial(fwd_transform, pp=pp, input_type=input_type)
    back_tf = partial(back_transform, pp=pp, input_type=input_type)

    xq = fwd_tf(xw)
    pdfvals = pp.dist.pdf(xq)
    se = (slope / pdfvals) * np.sqrt(prob_points * (1 - prob_points) / n)

    upper = back_tf(fwd_tf(fit_val) + crit * se)
    lower = back_tf(fwd_tf(fit_val) - crit * se)

    for yvals, name in zip([upper, lower], ["Upper", "Lower"]):
        xp = yvals if invert_axis else xw
        yp = xw if invert_axis else yvals
        fig.add_scatter(
            x=xp,
            y=yp,
            mode="lines",
            line_dash="dash",
            line_color="rgba(0,0,0,0.5)",
            showlegend=False,
            row=row,
            col=col,
            name=f"{name} - {ci=:.1%}",
        )

    return fig


def get_axis_probs(n: int) -> np.ndarray:

    axis_probs = np.linspace(10, 90, 9, dtype=float)
    small = np.array([1.0, 2, 5])
    axis_probs = np.hstack([small, axis_probs, 100 - small[::-1]])

    # different spacing for larger data
    if n >= 50:
        axis_probs = np.hstack([small / 10, axis_probs, 100 - small[::-1] / 10])
    if n >= 500:
        axis_probs = np.hstack([small / 100, axis_probs, 100 - small[::-1] / 100])
    axis_probs /= 100.0

    return axis_probs


def fmt_probplot_axis(
    fig: go.Figure,
    pp: sm.ProbPlot,
    row: int | None = None,
    col: int | None = None,
    ax: Literal["x", "y"] = "y",
) -> go.Figure:
    """Format probability plot axis following along with the statsmodels.api.gofplots._fmt_probplot_axis

    Parameters
    ----------
    fig : go.Figure
        the figure object to format
    pp : sm.ProbPlot
        the probability plot object
    row : int | None
        the row index of the subplot
    col : int | None
        the column index of the subplot
    ax : Literal["x", "y"]
        the axis to format, by default "y"
    Returns
    -------
    go.Figure
        the formatted figure object
    """

    axis_probs = get_axis_probs(pp.nobs)
    axis_qntls = pp.dist.ppf(axis_probs)

    if ax == "y":
        fig = fig.update_yaxes(
            title="Probabilities of theoretical quantiles",
            tickvals=axis_qntls,
            ticktext=[f"{p:.0%}" for p in axis_probs],
            row=row,
            col=col,
        )
    else:
        fig = fig.update_xaxes(
            title="Probabilities of theoretical quantiles",
            tickvals=axis_qntls,
            ticktext=[f"{p:.0%}" for p in axis_probs],
            row=row,
            col=col,
        )

    return fig


def add_ref_line(
    fig: go.Figure,
    line_type: Literal["diag", "standardized", "regression", "quartiles"] = "diag",
    dist: Callable | None = None,
    line_kwargs: dict | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Add reference line to the figure object
    This is akin to statsmodels:
        statsmodels.graphics.gofplots.qqline

    Parameters
    ----------
    fig : go.Figure
        the figure object to add the reference line to. This assumes that there
        is only one scatter trace which contains the data we should fit to

    dist : Callable | None
        the distribution object to use which requires .ppf() to be implemented.
        Only relevant for `quartiles` line_type, by default None

    line_type : Literal["diag", "standardize", "regression", "quartiles"]
        the type of reference line to add. Standardized estimates STD
        from data and adds a line according to the expected order statistics

    row : int | None
        the row index of the subplot

    col : int | None
        the column index of the subplot

    Returns
    -------
    go.Figure
        the figure object with the reference line added
    """

    line_kwargs = line_kwargs or dict()

    # only relevant traces
    traces = list(fig.select_traces(row=row, col=col))

    # axis are only specified for subplot traces
    if traces[0].xaxis and traces[0].yaxis:
        cmap, rmap = create_subplot_axis_map(fig)

    for trace in traces:
        if traces[0].xaxis and traces[0].yaxis:
            row = rmap[trace.xaxis]
            col = cmap[trace.yaxis]
        else:
            row = None
            col = None

        if line_type == "diag":
            # get the ranges or set them according to data, if not specified
            if row is not None and col is not None:
                xrange = fig.get_subplot(row=row, col=col).xaxis.range
                yrange = fig.get_subplot(row=row, col=col).yaxis.range
            else:
                xrange = fig.layout.xaxis.range
                yrange = fig.layout.yaxis.range

            pad = 0.05
            xd = np.asarray([trace.x.min(), trace.x.max()])
            yd = np.asarray([trace.y.min(), trace.y.max()])

            xrange = xrange or xd + np.array([-1, 1]) * pad * np.ptp(xd)
            yrange = yrange or yd + np.array([-1, 1]) * pad * np.ptp(yd)

            # this is following the `45` option from
            # statsmodels.api.gofplots.qqline, which is a diagonal rather than
            # a 45 degree line
            fig.add_scatter(
                x=xrange,
                y=yrange,
                mode="lines",
                row=row,
                col=col,
                name="diag",
                **line_kwargs,
            )
            fig.update_xaxes(range=xrange, row=row, col=col)
            fig.update_yaxes(range=yrange, row=row, col=col)

        elif line_type == "standardized":
            m, b = np.std(trace.y), np.mean(trace.y)
            ref_line = m * trace.x + b
            fig.add_scatter(
                x=trace.x,
                y=ref_line,
                mode="lines",
                row=row,
                col=col,
                name="std(y) * x + mean(y)",
                **line_kwargs,
            )

        elif line_type == "regression":
            ols_model = OLS(trace.y, add_constant(trace.x)).fit()
            preds = ols_model.get_prediction(add_constant(trace.x))
            dfp = preds.summary_frame(alpha=0.05)
            fig.add_scatter(
                x=trace.x,
                y=dfp["mean"],
                mode="lines",
                row=row,
                col=col,
                name="OLS",
                **line_kwargs,
            )

            # These would be CIs for a linear fit
            # fig.add_scatter(
            #     x=trace.x,
            #     y=dfp.mean_ci_lower,
            #     mode="lines",
            #     row=row,
            #     col=col,
            #     name="OLS_lower_ci",
            #     showlegend=False,
            #     line_dash="dash",
            #     line_color="rgba(0,0,0,0.5)",
            # )
            #
            # fig.add_scatter(
            #     x=trace.x,
            #     y=dfp.mean_ci_upper,
            #     mode="lines",
            #     row=row,
            #     col=col,
            #     name="OLS_upper_ci",
            #     showlegend=False,
            #     line_dash="dash",
            #     line_color="rgba(0,0,0,0.5)",
            # )

        elif line_type == "quartiles":
            q25, q75 = np.quantile(trace.y, [0.25, 0.75])
            theo_q25, theo_q75 = dist.ppf([0.25, 0.75])
            m = (q75 - q25) / (theo_q75 - theo_q25)
            b = q25 - m * theo_q25
            ypred = m * trace.x + b
            fig.add_scatter(
                x=trace.x,
                y=ypred,
                mode="lines",
                name="quartiles",
                **line_kwargs,
            )
        else:
            raise ValueError(
                f"line_type {line_type} not recognized. Choose from 'diag',"
                " 'standardized', 'regression', 'quartiles'"
            )

    return fig


def create_subplot_axis_map(fig: go.Figure) -> tuple[dict, dict]:
    """Create a mapping of subplot axis to rows and columns
    Parameters
    ----------
    fig : go.Figure
        the figure object to create the mapping for
    Returns
    -------
    tuple[dict, dict]
        two mapping dictionaries, one for rows and one for columns
    """

    # Axis are created in the make_subplots call -> we assume the domains
    #   have not been tinkered with manually
    subplotgrid = list(fig._get_subplot_coordinates())

    # Note the axis flip as we are interested in the anchors
    cmap = dict(
        zip([a.anchor for a in fig.select_yaxes()], [t[1] for t in subplotgrid])
    )

    rmap = dict(
        zip([a.anchor for a in fig.select_xaxes()], [t[0] for t in subplotgrid])
    )
    return rmap, cmap


def test_against_reliability():
    import matplotlib.pyplot as plt
    from reliability.Distributions import Normal_Distribution
    from reliability.Probability_plotting import Normal_probability_plot

    dist = Normal_Distribution(mu=50, sigma=10)
    failures = dist.random_samples(100, seed=5)
    figr = Normal_probability_plot(failures=failures)  # generates the probability plot
    dist.CDF(
        linestyle="--", label="True CDF"
    )  # this is the actual distribution provided for comparison
    plt.legend()
    plt.show()

    # note: the y positions in reliability are calculated with `a=0.3` instead
    # of `a=0.5` as in pingouin for n>10...
    # for statsmodels, the a value is =0 by default. Here changed to 0.5

    figp = probplot(failures, distgen=stats.distributions.norm, ci=0.95)
    figp = probplot(
        failures, distgen=stats.distributions.norm, ci=0.95, invert_axis=True
    )
    figp.show()


def plot_hist_and_dist(
    x: np.ndarray,
    distgen: stats._continuous_distns = stats.distributions.norm,
    **kwargs,
) -> go.Figure:
    """Plot histogram and the pdf of a fitted distribution

    Parameters
    ----------
    x : np.ndarray
        data for the histogram

    distgen: stats._continuous_distns = stats.distributions.norm,
        distribution for pdf line plot

    **kwargs
        additional kwargs are passed to the histogram plot


    Returns
    -------
    go.Figure

    """

    params = distgen.fit(x)
    args = params[:-2]
    loc = params[0]
    scale = params[1]

    fig = go.Figure()
    fig = fig.add_histogram(
        x=x,
        histnorm="probability density",
        name="hist",
        marker_color="blue",
        **kwargs,
    )

    xx = np.linspace(x.min(), x.max(), 100)
    fig = fig.add_scatter(
        x=xx,
        y=distgen.pdf(xx, *args, loc=loc, scale=scale),
        mode="lines",
        name="pdf",
        line=dict(color="red"),
    )

    return fig


if __name__ == "__main__":
    x = np.random.randn(300)

    x = np.random.randn(300).reshape(50, 6)

    x[:, 4] = np.random.random(50) * 4

    probfig = probplot(x, distgen=stats.distributions.norm, ci=0.95)
    probfig = probfig.update_layout(title="Probability Plot")
    probfig.show()

    qqfig = qq_plot(x, distgen=stats.distributions.norm, ci=0.95)
    qqfig = qqfig.update_layout(title="QQ Plot")
    qqfig.show()

    pp_fig = pp_plot(x, distgen=stats.distributions.norm, ci=0.95)
    pp_fig = pp_fig.update_layout(title="PP Plot")
    pp_fig.show()
