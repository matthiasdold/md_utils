from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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

    exog : np.ndarray | None
        The exogenous variables for the model to use during forecasting


    """

    def __init__(self, model: object):
        self.model = model
        self.exog = None
        # self.simulate_forward: callable = get_instance_specific_simulator(
        #     model
        # )

        self.sim_step = 0  # counter for simulation

    def simulate(
        self,
        y0: np.ndarray,
        n_sim: int = 100,
        n_step_pred: int = 1,
        exog: Optional[np.ndarray] = None,
        ytrue: Optional[np.ndarray] = None,
        callback: Optional[list[callable]] = None,
    ):
        pass


def get_instance_specific_simulator(model: object):
    # try direct lookup
    sim_map = {}
    tp = type(model)

    if tp in sim_map.keys():
        simulator = sim_map[tp]

    match model:
        case ResultsWrapper():
            if "statsmodels.tsa" in str(model.__class__):
                simulator = simulate_statsmodels_tsa
            elif "statsmodels.regression" in str(model.__class__):
                simulator = simulate_statsmodels_regression

    if simulator is not None:
        return simulator
    else:
        raise NotImplementedError(
            f"Model type {type(model)} is not supported for simulation"
        )


def simulate_statsmodels_tsa(
    fc: Forecaster,
    y0: np.ndarray,
    n_sim: int = 100,
    n_step_pred: int = 1,
    exog: Optional[np.ndarray] = None,
    ytrue: Optional[np.ndarray] = None,
    callback: Optional[list[callable]] = None,
):
    model: ResultsWrapper = deepcopy(fc.model)
    model.predict()

    data: statsmodels.base.data.PandasData = model.data

    while fc.sim_step < n_sim:
        # set the internal data to y0, then simulate forward
        pass

    return 0


def simulate_statsmodels_regression(
    fc: Forecaster,
    y0: np.ndarray,
    n_sim: int = 100,
    n_step_pred: int = 1,
    exog: Optional[np.ndarray] = None,
    ytrue: Optional[np.ndarray] = None,
    callback: Optional[list[callable]] = None,
):
    model: ResultsWrapper = deepcopy(fc.model)

    # write all data into the model, then treat simulation as within sample
    # forecasts
    model = prepate_statsmodels_data(model, y0, exog, ytrue)

    # use the forecast method
    model.forecast(steps=n_sim, exog=exog)

    return 0


def prepate_statsmodels_data(
    model: ResultsWrapper,
    y0: np.ndarray,
    exog: Optional[np.ndarray] = None,
    ytrue: Optional[np.ndarray] = None,
) -> ResultsWrapper:
    """Change the data in the results wrapper to contain the correct new
    data for simulation by replacing the models fit data
    """

    if ytrue is not None:
        y = np.vstack([y0, ytrue])
    else:
        y = y0

    if exog is not None:
        assert (
            exog.shape[0] == y.shape[0]
        ), f"Exog {len(exog)} must have same length as y0 + ytrue {len(y)} (if provided)"
        model.data.exog = exog[: len(y)]

    # only the y0 are to be considered within sample, others are out of sample
    model.data.endog = y
    model.model.endog = y

    # the original data needs to be overwritten as well, in order to use forecast
    model.data.orig_endog = y

    # potentially y is longer than the old dates available -> if this is the case
    # extrapolate the dates
    start = model.data.dates[-len(y0)]
    model.model._index = model.model._index[model.model._index >= start]

    pred_dates = pd.date_range(start, periods=len(y), freq=model.data.dates.freq)

    model.data.predict_start = start
    model.data.predict_dates = pred_dates

    # also overwrite cashed results to ensure that the update data is used
    model.model._deterministics._cached_in_sample = (
        model.model._deterministics._cached_in_sample[-len(y0) :]
    )  # keep the cached results as they reflect fitted trends
    # model.model._deterministics._index = model.model._index # <<<< this should not be replaced as otherwise the trending calculation
    # no longer exactly reflects the one with the trained data

    return model


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Use data from the statsmodels example here: https://www.statsmodels.org/dev/examples/notebooks/generated/autoregressive_distributed_lag.html
    df = pd.read_fwf(
        "https://raw.githubusercontent.com/statsmodels/smdatasets/main/data/autoregressive-distributed-lag/green/ardl_data.txt"
    )
    index = pd.to_datetime(
        df.Year.astype("int").astype("str") + "Q" + df.qtr.astype("int").astype("str")
    )
    df.index = index
    df.index.freq = df.index.inferred_freq
    df["c"] = np.log(df.realcons)
    df["g"] = np.log(df.realgdp)

    sel_res = tsa.ardl.ardl_select_order(
        df.c, 8, df[["g"]], 8, trend="c", seasonal=True, ic="aic"
    )
    ardl = sel_res.model
    fit_res = ardl.fit(use_t=True)

    fc = Forecaster(fit_res)

    # y0 = np.random.randn(15)
    y0 = fc.model.data.orig_endog.iloc[-10:]
    exog_replace = fc.model.data.orig_exog.iloc[-10:]
    exog_future = fc.model.data.orig_exog.iloc[:50]

    model: ResultsWrapper = deepcopy(fc.model)
    morig: ResultsWrapper = deepcopy(fc.model)
    morig.model._deterministics._cached_in_sample = None
    ytrue = None

    # write all data into the model, then treat simulation as within sample
    # forecasts
    model = prepate_statsmodels_data(model, y0, exog_replace, ytrue)

    # print("----------------------")
    # model.model._deterministics._cached_in_sample
    # print("----------------------")
    # morig.model._deterministics._cached_in_sample
    # print("----------------------")

    import matplotlib.pyplot as plt

    print(f"++++++++++++:>MODEL")
    # print([t for t in model.model._deterministics._deterministic_terms])
    ax = replaced = model.forecast(steps=40, exog=exog_future).plot()

    print(f"++++++++++++:>MORIG")
    orig = morig.forecast(steps=40, exog=exog_future).plot(ax=ax)
    plt.show()
