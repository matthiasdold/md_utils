# MD Utils

This repo includes common functionality I often use during analysis of
electrophysiology data.

_Note_: Much of the data is processed as
[`xileh`](https://github.com/matthiasdold/xileh) containers.

## Time Frequency Analysis

I often just want to get a TFR plot of all data within a single file. The most
common files I use are `*.mat` (AlphaOmega), `*.vhdr` (BrainVision) and `*.fif`
(mne) files. Use this cli:

```bash
python -m mdu.tfr_plot --help
```

eg.

```bash
python -m mdu.tfr_plot --file=block_4_hand0001.mat --channels=1,2,3
```
