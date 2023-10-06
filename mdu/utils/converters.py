# All kind of formatting conversion which cannot be achieved as a one liner
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Callable

import numpy as np


@dataclass
class ToFloatConverter:
    """Conversion from numpy array of various data formats to float and back"""

    back_conversion: Callable | None = None

    def to_float(self, x: np.ndarray) -> np.ndarray:
        assert all([isinstance(e, type(x[0])) for e in x]), (
            "Input types are not all"
            f" {type(x[0])=}, please to a single type first."
        )

        if not isinstance(x[0], float | int):
            # Needs conversion
            match type(x[0]):
                case datetime:
                    x = np.array([_.timestamp() for _ in x])
                    self.back_conversion = partial(
                        self.from_timestamp_with_offset, offset=x[0]
                    )

                    # Offset to zero
                    x = x - x[0]

        return x

    def to_orig(self, x: np.ndarray) -> np.ndarray:
        """Convert to the type of the last input of `self.to_float`"""
        return np.asarray([self.back_conversion(e) for e in x])

    def from_timestamp_with_offset(
        self, ts: datetime.timestamp, offset: float
    ):
        return datetime.fromtimestamp(ts + offset)
