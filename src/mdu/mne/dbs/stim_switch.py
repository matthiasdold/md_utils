import mne
import numpy as np


def find_stim_switch_on(
    epo: mne.BaseEpochs, sensitivity: float = 0.1
) -> list[dict]:
    """
    Find the epoch time where the stimulation setting changed following a
    heuristic looking for:
        - LFP channels (`pick_types(dbs=True)`)
        - identifying return channels
        - checking for the change point in the return channels

    Parameters
    ----------
    epo : mne.BaseEpochs
        epoch object to investigate, every epoch will be checked in which
        there is one channel which is within (1 + sensitivity) times the
        standard deviation of the channel with the least standard deviation.
        This works fine if there is at least one stim == "ON" epoch.

    sensitivity : float
        the sensitivity factor for the standard deviation of the return
        channel, the default is 0.1, which means that candidate return channels
        must have a standard deviation of at most 110% of the channel with
        the lowest standard deviation.


    Returns
    -------
    list[dict]
        epo: epoch nbr
        time: time in seconds
        ix: index in the epoch

    """

    lfp = epo.copy().pick_types(dbs=True)
    psd = lfp.compute_psd(n_jobs=-1, fmin=45, fmax=55)

    df = psd.to_data_frame()

    dm = df.melt(
        id_vars=["freq", "epoch", "condition"],
        value_vars=[c for c in df.columns if c in psd.ch_names],
    )
    dm["value"] = 10 * np.log10(dm["value"]) + 120
    dg = (
        dm.groupby(["epoch", "variable"])["value"]
        .std()
        .reset_index()
        .rename(columns={"value": "stdv", "variable": "channel"})
    )

    # the return channel will always have a very low std in the stim epochs
    dret = dg[dg.stdv < (1 + sensitivity) * dg.stdv.min()]

    assert (
        dret.channel.nunique() == 1
    ), f"Found {dret.channel.unique()} channels as candidates for return"
    #
    # create tuples of epoch nbrs and times for were the stim starts
    stim_starts = []
    ixc = list(lfp.ch_names).index(dret.channel.unique()[0])
    for rec in dret.iterrows():
        data = lfp[rec[1].epoch].get_data()[0, ixc, :]
        ix_max = np.argmax(data)

        assert data[ix_max] > np.mean(data) + 7 * np.std(
            data
        ), f"Peak at {ix_max=} is not 7 STD above mean for {rec=}"

        stim_starts.append(
            dict(epo=rec[1].epoch, time=lfp.times[ix_max], ix=ix_max)
        )

    return stim_starts
