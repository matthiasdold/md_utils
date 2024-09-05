import pytest
import numpy as np
from mdu.cross_val.chrono import ChronoGroupsSplit


@pytest.fixture
def get_test_data() -> dict:
    X = np.ones((20, 2)) * np.arange(1, 3)
    y = np.tile(np.asarray([0, 0, 1, 1]), (5))
    grps = np.arange(10).repeat(2)

    d = {"X": X, "y": y, "grps": grps}
    return d


def test_chrono_split(get_test_data):
    X = get_test_data["X"]
    y = get_test_data["y"]
    grps = get_test_data["grps"]

    cv = ChronoGroupsSplit()
    splits = cv.split(X, y, grps)

    assert len(splits) == 5
    assert splits[0][0].shape[0] == 16
    assert splits[0][1].shape[0] == 4
    assert np.allclose(splits[0][1], np.asarray([0, 1, 2, 3]))
    assert np.allclose(splits[1][1], np.asarray([4, 5, 6, 7]))
    assert np.allclose(splits[2][1], np.asarray([8, 9, 10, 11]))
    assert np.allclose(splits[3][1], np.asarray([12, 13, 14, 15]))
    assert np.allclose(splits[4][1], np.asarray([16, 17, 18, 19]))
