import numpy as np
import pandas as pd
import plotly.express as px
import pytest

from mdu.plotly.stats import add_box_significance_indicator


def get_test_data() -> pd.DataFrame:

    df = pd.DataFrame(np.random.randn(200, 2), columns=["a", "b"])  # type: ignore

    df["label"] = np.random.choice(["aa", "bb", "cc"], 200)
    df["color"] = np.random.choice(["xx", "yy", "zz"], 200)
    df["cat"] = np.random.choice(["rr", "ee"], 200)
    df.loc[df.color == "xx", "a"] += 20

    return df


def test_add_box_significance_indicator():
    df = get_test_data()

    fig = px.box(df, x="label", y="a", color="color", facet_col="cat")
    fig

    assert fig is not None
