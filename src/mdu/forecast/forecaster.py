from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa as tsa
from statsmodels.base.wrapper import ResultsWrapper
from statsmodels.tsa.api import ARDL
from statsmodels.tsa.ardl import ardl_select_order


class Forecaster:
    """Forecaster class to have a single entry point for forecasting
    based on different model instances. The primary model instance supported
    is statsmodels.tsa type models.

    Attributes
    ----------
    model : object
        the following instances are supported currently:
            sm.tsa (note, we are using the ResultsWrapper types, as we are
            expecting the fitted model).

    model_gen : Optional[callable]
        callable to generate the statsmodels

    """

    def __init__(self, model: object, model_gen: Optional[callable] = None):
        self.model = model
        self.forecaster_used = None  # bookkeeping for the last function used
        self.model_gen = model_gen

    def forecast(
        self,
        y0: (
            np.ndarray | pd.DataFrame
        ),  # we always expect starting conditions to be provided explicitly!
        n_fc: int = 100,
        n_step_pred: int = 1,
        exog: Optional[np.ndarray | pd.DataFrame] = None,
        callbacks: Optional[list[callable]] = None,
        ytrue: Optional[np.ndarray | pd.DataFrame] = None,
    ):
        """Unified entry point to call the model specific forecast function

        Parameters
        ----------
        y0 : np.ndarray | pd.DataFrame
            The starting conditions for the forecast. This is always required.
            For what statsmodels refers to as "in-sample" forecasts, just provide
            data from around the index you plan to start from -> `model.endog`.

        n_fc : int
            The number of steps to forecast

        n_step_pred : int
            The number of steps to predict ahead each time

        callbacks : Optional[list[callable]]
            A list of functions to call after each prediction of `n_step_pred`

        ytrue : Optional[np.ndarray | pd.DataFrame]
            The true values to use for the forecast after `n_step_pred` steps.
            This is used to realize a within sample forecast.

        Returns
        -------
            np.ndarray | pd.DataFrame
            The forecasted data of shape (n_fc, y0.shape[1])

        """

        model_fc = get_instance_specific_simulator(self.model)
        fc = model_fc(
            self,
            y0,
            n_fc=n_fc,
            n_step_pred=n_step_pred,
            exog=exog,
            callbacks=callbacks,
            ytrue=ytrue,
        )
        self.forecaster_used = model_fc

        return fc

    def trigger_callbacks(self, callbacks: list[callable], *args, **kwargs):
        for cb in callbacks:
            cb(*args, **kwargs)


def get_instance_specific_simulator(model: object):
    # try direct lookup
    sim_map = {}
    tp = type(model)

    if tp in sim_map.keys():
        forecaster = sim_map[tp]

    match model:
        case ResultsWrapper():
            if "statsmodels.tsa" in str(model.__class__):
                forecaster = forecast_statsmodels_tsa
            elif "statsmodels.regression" in str(model.__class__):
                forecaster = forecast_statsmodels_tsa

    if forecaster is not None:
        return forecaster
    else:
        raise NotImplementedError(
            f"Model type {type(model)} is not supported for simulation"
        )


def forecast_statsmodels_tsa(
    fc: Forecaster,
    y0: np.ndarray | pd.DataFrame,
    n_fc: int = 100,
    n_step_pred: int = 1,
    exog: Optional[np.ndarray | pd.DataFrame] = None,
    ytrue: Optional[np.ndarray | pd.DataFrame] = None,
    callbacks: Optional[list[callable]] = None,
    refit: bool = False,
) -> np.ndarray | pd.DataFrame:
    """
    Use the statsmodels specific syntax for insample and out of sample
    forecasts

    Return data type same as y0
    """
    if exog is not None:
        assert (
            exog.shape[0] == len(y0) + n_fc
        ), f"Exog {len(exog)=} must have same length as {len(y0) + n_fc=} (if provided)"

    fcm = deepcopy(fc.model)

    concat = pd.concat if isinstance(y0, pd.DataFrame) else np.vstack

    # Prepare the buffer for the forecasts, simple numpy or pandas
    fcs = np.full(n_fc, np.nan)
    if isinstance(y0, pd.DataFrame):
        fcs = pd.DataFrame(fcs, columns=y0.columns)
        # take index from ytrue if provided
        if ytrue is not None:
            fcs.index = ytrue.index
        elif exog is not None and isinstance(exog, pd.DataFrame):
            fcs.index = exog.index[len(y0) : len(y0) + n_fc]
        else:
            # extrapolate the index based on the last known step size
            dstep = y0.index[-1] - y0.index[-2]
            fcs.index = [y0.index[-1] + dstep * i for i in range(n_fc)]

    # replace data with the new initial conditions -> only for y0, then if ytrue
    # are provided, we extend the data
    #
    # replace only if endog != y0 or exog changed
    if (
        check_condition_compare(
            y0,
            fcm.model.data.orig_endog,
            exog[: len(y0)],
            fcm.model.data.orig_exog,
        )
        is False
    ):
        if hasattr(fcm.model, "_deterministics"):
            fcm.model.deterministics = fcm.model._deterministics
        fcm = fcm.apply(endog=y0, exog=exog[: len(y0)], refit=refit)
        assert (
            fcm.__class__ == fc.model.__class__
        ), f"Class changed by apply {fcm.__class__=} != {fc.model.__class__=}, this class cannot be supported beyond a single forecast"

    # continuously feed new data and refit if necessary
    i_fwd = 0

    while i_fwd < n_fc:
        # adjust the n_step_pred for the last segment, shortening it if necessary
        n_inc = n_step_pred if i_fwd + n_step_pred < n_fc else n_fc - i_fwd

        # the extended data is either bootstrapped from predictions or reflecting
        # the true observations, if provided
        if i_fwd > 0:
            last_slice = slice(i_fwd - n_step_pred, i_fwd)
            exog_known = exog[len(y0) :][last_slice] if exog is not None else None
            if ytrue is None:
                fcm = fcm.append(endog=fcs[last_slice], exog=exog_known, refit=refit)
            else:
                fcm = fcm.append(endog=ytrue[last_slice], exog=exog_known, refit=refit)

            assert (
                fcm.__class__ == fc.model.__class__
            ), f"Class changed by apply {fcm.__class__=} != {fc.model.__class__=}, this class cannot be supported beyond a single forecast"

        fc_res = fcm.forecast(
            steps=n_inc,
            exog=(
                exog[len(y0) + i_fwd : len(y0) + i_fwd + n_inc]
                if exog is not None
                else None
            ),
        )

        # ensure we have a dataframe as fcs will be a frame
        if isinstance(fc_res, pd.Series):
            fc_res = pd.DataFrame(fc_res)

        fcs[i_fwd : i_fwd + n_inc] = fc_res

        if callbacks is not None:
            fc.trigger_callbacks(callbacks)

        i_fwd += n_inc

    return fcs


def check_condition_compare(
    y0: np.ndarray | pd.DataFrame,
    endog_orig: np.ndarray | pd.DataFrame,
    exog: np.ndarray | pd.DataFrame | None = None,
    exog_orig: np.ndarray | pd.DataFrame | None = None,
) -> bool:
    if exog is not None:
        assert exog_orig is not None, f"If {exog=} is provided, {exog_orig=} is needed"

    # always work in numpy format
    y0v = y0.values if isinstance(y0, pd.DataFrame) else y0
    endogv = endog_orig.values if isinstance(y0, pd.DataFrame) else endog_orig

    if y0v.shape != endogv.shape:
        return False
    if not (y0v == endogv).all():
        return False
    if exog is not None:
        # use the recurse call to shorten the function
        return check_condition_compare(y0=exog, endog_orig=exog_orig)

    return True


# def prepate_statsmodels_data(
#     model: ResultsWrapper,
#     y0: np.ndarray,
#     exog: Optional[np.ndarray] = None,
#     ytrue: Optional[np.ndarray] = None,
# ) -> ResultsWrapper:
#     """Change the data in the results wrapper to contain the correct new
#     data for simulation by replacing the models fit data
#     """
#
#     if ytrue is not None:
#         y = np.vstack([y0, ytrue])
#     else:
#         y = y0
#
#     if exog is not None:
#         assert (
#             exog.shape[0] == y.shape[0]
#         ), f"Exog {len(exog)} must have same length as y0 + ytrue {len(y)} (if provided)"
#         model.data.exog = exog[: len(y)]
#
#     # only the y0 are to be considered within sample, others are out of sample
#     model.data.endog = y
#     model.model.endog = y
#
#     # the original data needs to be overwritten as well, in order to use forecast
#     model.data.orig_endog = y
#
#     # potentially y is longer than the old dates available -> if this is the case
#     # extrapolate the dates
#     start = model.data.dates[-len(y0)]
#     model.model._index = model.model._index[model.model._index >= start]
#
#     pred_dates = pd.date_range(
#         start, periods=len(y), freq=model.data.dates.freq
#     )
#
#     model.data.predict_start = start
#     model.data.predict_dates = pred_dates
#
#     # also overwrite cashed results to ensure that the update data is used
#     model.model._deterministics._cached_in_sample = (
#         model.model._deterministics._cached_in_sample[-len(y0) :]
#     )  # keep the cached results as they reflect fitted trends
#     # model.model._deterministics._index = model.model._index # <<<< this should not be replaced as otherwise the trending calculation
#     # no longer exactly reflects the one with the trained data
#
#     return model
#

# if __name__ == "__main__":
#     import numpy as np
#     import pandas as pd
#
#     df = sm.datasets.macrodata.load_pandas().data
#     df.index = pd.period_range("1959Q1", periods=len(df), freq="Q")
#
#     df["c"] = np.log(df.realcons)
#     df["g"] = np.log(df.realgdp)
#
#     sel_res = tsa.ardl.ardl_select_order(
#         df.c, 8, df[["g"]], 8, trend="c", seasonal=True, ic="aic"
#     )
#     ardl = sel_res.model
#     fit_res = ardl.fit(use_t=True)
#
#     fc = Forecaster(fit_res)
#
#     # # y0 = np.random.randn(15)
#     # y0 = fc.model.data.orig_endog.iloc[-10:]
#     # exog_replace = fc.model.data.orig_exog.iloc[-10:]
#     # exog_future = fc.model.data.orig_exog.iloc[:50]
#     #
#     # model: ResultsWrapper = deepcopy(fc.model)
#     # morig: ResultsWrapper = deepcopy(fc.model)
#     # morig.model._deterministics._cached_in_sample = None
#     # ytrue = None
#     #
#     # # write all data into the model, then treat simulation as within sample
#     # # forecasts
#     # model = prepate_statsmodels_data(model, y0, exog_replace, ytrue)
#     #
#     # print("----------------------")
#     # model.model._deterministics._cached_in_sample
#     # print("----------------------")
#     # morig.model._deterministics._cached_in_sample
#     # print("----------------------")
#
#     import matplotlib.pyplot as plt
#
#     print(f"++++++++++++:>MODEL")
#     # print([t for t in model.model._deterministics._deterministic_terms])
#     ax = replaced = model.forecast(steps=40, exog=exog_future).plot()
#
#     print(f"++++++++++++:>MORIG")
#     orig = morig.forecast(steps=40, exog=exog_future).plot(ax=ax)
#     plt.show()
#
#
