# evaluations around residuals
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy.stats as st
from plotly.subplots import make_subplots

from mdu.plotly.dist_plotting import plot_hist_and_dist, probplot
from mdu.plotly.styling import apply_default_styles


def fit_residual_dist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dist: st._continuous_distns = st.distributions.norm,
    plot: bool = False,
) -> st._distn_infrastructure.rv_continuous_frozen:
    """Fit a distribution to the residuals of two arrays. The default is a
    normal distribution.

    Parameters
    ----------
    y_true : array-like of shape = n_samples
        The true values of the system.
    y_pred : array-like of shape = n_samples
        The predicted values of the system.
    dist : callable
        A distribution from scipy.stats.distributions.
    Returns
    -------
    dist : scipy.stats.distributions
        The fitted distribution.
    """
    resid = y_true - y_pred

    params = dist.fit(resid)
    args = params[:-2]
    loc = params[0]
    scale = params[1]

    # check fit quality be SSE
    nbins = 10 if len(resid) < 100 else 20
    y, x = np.histogram(resid, bins=nbins, density=True)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0
    pdf = dist.pdf(x_mid, *args, loc=loc, scale=scale)
    sse = np.sum((y - pdf) ** 2)
    print(
        f"Fitted distribution: {dist.name} -> {loc=:4f}, {scale=:4f}, "
        f"{args=}, {sse=:.2f}, {nbins=}"
    )
    if plot:
        plt.hist(resid, bins=nbins, density=True)
        plt.plot(x_mid, pdf)
        plt.show()

    d = dist(*args, loc=loc, scale=scale)
    # draw random samples from d via d.rvs(size=1000)

    return d


def residuals_analysis_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distgen: st._continuous_distns = st.distributions.norm,
    show: bool = False,
) -> go.Figure:
    """
    Create an analysis figure with nrows = y_true.shape[1] and ncols=3
    First column contains histogram and fit, second column contains a probplot,
    third column contains the residuals auto correlation plot.
    """

    if y_true.ndim == 1 and y_pred.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    resid = y_true - y_pred

    subplot_titles = [
        f"x_{i + 1} {fig_type}"
        for i in range(y_true.shape[1])
        for fig_type in ["hist", "probplot", "acf"]
    ]

    fig = make_subplots(
        rows=y_true.shape[1],
        cols=3,
        subplot_titles=subplot_titles,
    )

    for ir, res in enumerate(resid.T):
        hfig = plot_hist_and_dist(res, distgen=distgen)
        ppfig = probplot(res, distgen=distgen)
        acffig = plot_acf(res)

        for i, auxf in enumerate([hfig, ppfig, acffig]):
            showlegend = False
            for trace in auxf.data:
                trace.showlegend = showlegend
                fig.add_trace(trace, row=ir + 1, col=i + 1)

            # change the axis ticks for the probplot
            if i == 1:
                tt = [""] * len(auxf.layout["xaxis"].ticktext)
                tt[::5] = auxf.layout["xaxis"].ticktext[::5]
                fig.update_xaxes(
                    ticktext=tt,
                    tickvals=auxf.layout["xaxis"].tickvals,
                    row=ir + 1,
                    col=2,
                )

    fig = fig.update_xaxes(title_text="Residuals", col=1)
    fig = fig.update_xaxes(title_text="Prob. theor. quantiles", col=2)
    fig = fig.update_xaxes(title_text="Lag [#]", col=3)

    fig = fig.update_yaxes(
        title_text="Normalized hist", col=1, title_standoff=5
    )
    fig = fig.update_yaxes(title_text="Residuals", col=2, title_standoff=5)
    fig = fig.update_yaxes(
        title_text="Normalized ACF", col=3, title_standoff=5
    )

    fig = fig.update_layout(height=500 * y_true.shape[1])

    fig = apply_default_styles(fig, xzero=False, yzero=False)
    if show:
        fig.show()

    return fig


def plot_acf(x: np.ndarray, plot_lag_zero: bool = False) -> go.Figure:

    acf = np.correlate(x, x, mode="full")

    # get the normalized acf --> normalize to the lag 0 value
    acf = acf[acf.size // 2 :] / acf[acf.size // 2]

    upper_bound = 1.96 / np.sqrt(len(acf))
    lower_bound = upper_bound * (-1)

    lags = np.arange(len(acf))

    fig = go.Figure()

    # add the confidence bounds
    fig = fig.add_scatter(
        x=lags,
        y=[upper_bound] * len(lags),
        mode="lines",
        line_color="rgba(10,10,10,0.4)",
        line_width=1,
        line_dash="dash",
        showlegend=False,
        name="Confidence upper",
    )
    fig = fig.add_scatter(
        x=lags,
        y=[lower_bound] * len(lags),
        mode="lines",
        fill="tonexty",
        line_color="rgba(10,10,10,0.4)",
        line_width=1,
        line_dash="dash",
        fillcolor="rgba(10,10,10,0.1)",
        name="Confidence interval<br>standard normal",
    )

    ix_start = 0 if plot_lag_zero else 1
    fig = fig.add_scatter(
        x=lags,
        y=acf[ix_start:],
        mode="lines",
        name="acf",
        line=dict(color="blue"),
    )

    return fig


if __name__ == "__main__":
    y_true = np.random.randn(4000, 2)
    y_pred = y_true + 0.2 * np.random.randn(4000, 2)

    distgen = st.distributions.norm
    dist = fit_residual_dist(y_true, y_pred)
