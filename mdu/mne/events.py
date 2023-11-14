# utility to work with events
#
import re

import numpy as np


def inverse_map_events(ev: np.ndarray, evid: dict) -> np.ndarray:
    """Inverse map of events in evid -> use numbers in evid values if present

    Parameters
    ----------
    ev : np.ndarray
        Events to be mapped.
    evid : dict
        Mapping dictionary.

    Returns
    -------
    np.ndarray
        Mapped events with the evid values in the third column.
    """

    imap = {}
    for k, v in evid.items():
        try:
            imap[int(v)] = int(k)
        except ValueError:
            d = re.findall(r"(\d+)", k)
            if len(d) > 0:
                imap[v] = int(d[0])

    nev = ev.copy()
    nev[:, 2] = [imap.get(x, x) for x in ev[:, 2]]

    return nev
