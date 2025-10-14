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
        """
        Convert numpy array to float, handling various data types.

        Parameters
        ----------
        x : np.ndarray
            Input array with elements of the same type (datetime, float, or int).

        Returns
        -------
        np.ndarray
            Array converted to float. For datetime objects, converts to timestamps
            offset from the first value.

        Raises
        ------
        AssertionError
            If input array contains mixed types.
        """
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
        """
        Convert back to original data type from float.

        Converts to the type of the last input of `self.to_float`.

        Parameters
        ----------
        x : np.ndarray
            Float array to convert back to original type.

        Returns
        -------
        np.ndarray
            Array converted to original data type.
        """
        return np.asarray([self.back_conversion(e) for e in x])

    def from_timestamp_with_offset(
        self, ts: datetime.timestamp, offset: float
    ):
        """
        Convert timestamp back to datetime with offset.

        Parameters
        ----------
        ts : float
            Timestamp value (seconds since epoch, with offset removed).
        offset : float
            Offset timestamp to add back (seconds since epoch).

        Returns
        -------
        datetime
            Datetime object reconstructed from timestamp and offset.
        """
        return datetime.fromtimestamp(ts + offset)
