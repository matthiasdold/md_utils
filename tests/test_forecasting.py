from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.tsa.api import ARDL, AutoReg
from statsmodels.tsa.ardl import ardl_select_order

from mdu.forecast.forecaster import Forecaster

# from src.mdu.forecast.forecaster import Forecaster


def get_data() -> pd.DataFrame:
    data = sm.datasets.macrodata.load_pandas().data
    data.index = pd.period_range("1959Q1", periods=len(data), freq="Q")
    return data


# --- there seems to be an issue with copying over arld models, so testing
#     with new initial conditions and arld is not possible. After copy, the arld
#     will be an AutoReg model for the used data set, so the autoreg test will
#     be used instead
def arld_gen(endog, exog) -> ARDL:
    sel_res = ardl_select_order(
        endog=endog,
        maxorder=8,
        exog=exog,
        maxlag=10,
        trend="c",
        seasonal=True,
        ic="aic",
    )
    return sel_res.model


def auto_reg_gen(endog, exog) -> AutoReg:
    return AutoReg(endog=endog, lags=8, exog=exog, trend="c")


@pytest.fixture
def data_dict() -> dict:
    data = get_data()
    data_train = data.iloc[:-100]
    data_test = data.iloc[-100:]
    endog_vars = ["realgdp"]
    exog_vars = ["m1", "tbilrate"]

    data_dict = dict(
        data_train=data_train,
        data_test=data_test,
        endog_vars=endog_vars,
        exog_vars=exog_vars,
    )
    return data_dict


@pytest.mark.parametrize("n_step_pred", [1, 3, 10])
@pytest.mark.parametrize(
    "model_gen", [sm.tsa.ARIMA, auto_reg_gen, arld_gen]
)  # add models as we go
def test_statsmodels_forecaster_next_step_fc(
    data_dict,
    n_step_pred,
    model_gen,
    # get_data, n_step_pred: int = 1, model_gen: object = sm.tsa.ARIMA
):
    data_train = data_dict["data_train"]
    data_test = data_dict["data_test"]
    endog_vars = data_dict["endog_vars"]
    exog_vars = data_dict["exog_vars"]

    # model_gen = arld_gen
    model = model_gen(endog=data_train[endog_vars], exog=data_train[exog_vars])
    res = model.fit()
    fc = Forecaster(model=res)

    sm_fc = res.forecast(
        steps=n_step_pred,
        exog=data_test[exog_vars].iloc[:n_step_pred, :],
    )

    my_fc = fc.forecast(
        y0=data_train[endog_vars],
        n_fc=n_step_pred,
        n_step_pred=n_step_pred,
        exog=pd.concat(
            [
                data_train[exog_vars],
                data_test[exog_vars].iloc[:n_step_pred],
            ]
        ),
    )

    assert np.allclose(
        sm_fc.values.flatten(),
        my_fc.values.flatten(),
    )


@pytest.mark.parametrize(
    "model_gen", [sm.tsa.ARIMA, auto_reg_gen]
)  # add models as we go
def test_statsmodels_forecaster_within_sample_1step(
    data_dict,
    model_gen,
):
    data_train = data_dict["data_train"]
    endog_vars = data_dict["endog_vars"]
    exog_vars = data_dict["exog_vars"]
    model = model_gen(endog=data_train[endog_vars], exog=data_train[exog_vars])
    res = model.fit()
    fc = Forecaster(model=res)
    max_lag = 20  # to have sufficient data for the models

    # in sample, the forecast should be a prediction with the appropriate
    # end point
    # In sample, only 1 step ahead forecasts are made, eg.
    # _predict_dynamic@tsa.ar_model.py:607
    # For a correct comparison, we need to use forecast by applying the
    # old parameters to a new model
    n_fc = 10

    sm_fc = res.predict(
        start=data_train.index[max_lag],
        end=data_train.index[
            max_lag + n_fc - 1
        ],  # end is includede [start, end], -> -1
    )
    my_fc = fc.forecast(
        y0=data_train.iloc[:max_lag][endog_vars],
        n_fc=n_fc,
        n_step_pred=1,
        exog=data_train.iloc[: max_lag + n_fc][exog_vars],
        ytrue=data_train.iloc[max_lag : max_lag + n_fc][endog_vars],
    )

    assert np.allclose(
        sm_fc.values.flatten(),
        my_fc.values.flatten(),
    )


@pytest.mark.parametrize("n_step_pred", [1, 3, 10])
@pytest.mark.parametrize(
    "model_gen", [sm.tsa.ARIMA, auto_reg_gen]
)  # add models as we go
def test_statsmodels_forecaster_new_initial_values(
    data_dict,
    n_step_pred,
    model_gen,
    # get_data, n_step_pred: int = 1, model_gen: object = sm.tsa.ARIMA
):
    data_train = data_dict["data_train"]
    data_test = data_dict["data_test"]
    endog_vars = data_dict["endog_vars"]
    exog_vars = data_dict["exog_vars"]

    model = model_gen(endog=data_train[endog_vars], exog=data_train[exog_vars])
    res = model.fit()
    fc = Forecaster(model=res)

    max_lag = 20  # to have sufficient data for the models

    res_new = deepcopy(res)
    res_new = res_new.apply(
        endog=data_test[endog_vars].iloc[:max_lag, :],
        exog=data_test[exog_vars].iloc[:max_lag, :],
    )
    sm_fc = res_new.forecast(
        steps=n_step_pred,
        exog=data_test[exog_vars].iloc[max_lag : max_lag + n_step_pred, :],
    )

    my_fc = fc.forecast(
        y0=data_test.iloc[:max_lag][endog_vars],
        n_fc=20,
        n_step_pred=n_step_pred,
        exog=data_test.iloc[: max_lag + 20][exog_vars],
    )

    assert np.allclose(
        sm_fc.values.flatten(),
        my_fc.iloc[:n_step_pred].values.flatten(),
    )


if __name__ == "__main__":
    data = get_data()
    data_train = data.iloc[:-100]
    data_test = data.iloc[-100:]
    endog_vars = ["realgdp"]
    exog_vars = ["m1", "tbilrate"]

    model_gen = arld_gen
    # model_gen = auto_reg_gen
    model = model_gen(endog=data_train[endog_vars], exog=data_train[exog_vars])
    res = model.fit()
    max_lag = 20  # to have sufficient data for the models
    n_fc = 5

    fc = Forecaster(model=res, model_gen=model_gen)

    # # ---------   This basic forecast works
    print("My Forecaster")
    my_fc = fc.forecast(
        y0=data_train.iloc[-(max_lag):][endog_vars],
        n_fc=n_fc,
        n_step_pred=5,
        exog=pd.concat(
            [
                data_train.iloc[-(max_lag):][exog_vars],
                data_test[exog_vars].iloc[:n_fc],
            ]
        ),
    )

    print("Built in")
    sm_fc = res.forecast(steps=5, exog=data_test[exog_vars].iloc[:n_fc])

    # --------------- Within sample is tricky
    print("-" * 80)
    sm_fc

    print(".-" * 60)
    my_fc
