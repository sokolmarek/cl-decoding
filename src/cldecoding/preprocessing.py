"""Signal conditioning and segmentation (Section 4.1).

All recordings are resampled to 256 Hz. ECG is high-pass filtered (4th-order Butterworth, 0.5 Hz) and EDA is
low-pass filtered (4th-order Butterworth, 3 Hz). Signals are then cut into non-overlapping windows of 5 or 3 s
that lie entirely inside one labelled phase. One-second windows are centred on detected R peaks so that every
segment contains a heartbeat.
"""

from fractions import Fraction

import neurokit2 as nk
import numpy as np
from scipy import signal

FS = 256
WINDOWS = (5, 3, 1)


def resample(x, fs_in, fs_out=FS):
    """Polyphase resampling of a 1-D signal from ``fs_in`` to ``fs_out`` Hz."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=float)
    ratio = Fraction(fs_out, fs_in).limit_denominator(10_000)
    return signal.resample_poly(np.asarray(x, dtype=float), ratio.numerator, ratio.denominator)


def resample_labels(labels, fs_in, n_out, fs_out=FS):
    """Nearest-sample resampling of a label vector to ``n_out`` samples at ``fs_out`` Hz."""
    idx = np.floor(np.arange(n_out) * fs_in / fs_out).astype(int)
    return np.asarray(labels)[np.clip(idx, 0, len(labels) - 1)]


def filter_ecg(ecg, fs=FS, cutoff=0.5, order=4):
    """Remove baseline wander with a zero-phase Butterworth high-pass filter."""
    sos = signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, ecg)


def filter_eda(eda, fs=FS, cutoff=3.0, order=4):
    """Suppress high-frequency noise with a zero-phase Butterworth low-pass filter."""
    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, eda)


def window_starts(labels, fs, window):
    """Start indices of non-overlapping windows that do not cross a label change."""
    w = int(round(window * fs))
    labels = np.asarray(labels)
    change = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate(([0], change, [len(labels)]))

    starts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        starts.extend(range(a, b - w + 1, w))
    return np.asarray(starts, dtype=int)


def rpeak_window_starts(ecg, fs, window=1.0):
    """Start indices of windows centred on R peaks."""
    w = int(round(window * fs))
    _, info = nk.ecg_peaks(ecg, sampling_rate=fs)
    starts = np.asarray(info["ECG_R_Peaks"], dtype=int) - w // 2
    return starts[(starts >= 0) & (starts + w <= len(ecg))]


def segment(ecg, eda, labels, window, fs=FS):
    """Cut filtered ECG and EDA into windows of ``window`` seconds.

    Only windows whose samples all carry the same label ``>= 0`` are kept, so labels of ``-1`` can be used to
    mark phases that are not part of the classification task.

    Returns
    -------
    ecg_seg, eda_seg : ndarray, shape (n_segments, window * fs)
    seg_labels : ndarray, shape (n_segments,)
    """
    w = int(round(window * fs))
    labels = np.asarray(labels)
    if window == 1:
        starts = rpeak_window_starts(ecg, fs, window)
    else:
        starts = window_starts(labels, fs, window)

    if len(starts) == 0:
        return np.empty((0, w)), np.empty((0, w)), np.empty(0, dtype=int)

    idx = starts[:, None] + np.arange(w)[None, :]
    seg_labels = labels[idx]
    keep = (seg_labels == seg_labels[:, :1]).all(axis=1) & (seg_labels[:, 0] >= 0)
    idx = idx[keep]
    return ecg[idx], eda[idx], seg_labels[keep, 0].astype(int)


def preprocess_recording(ecg, eda, labels, fs_in, window):
    """Resample, filter and segment one recording whose labels are sampled at ``fs_in``."""
    ecg = filter_ecg(resample(ecg, fs_in))
    eda = filter_eda(resample(eda, fs_in))
    n = min(len(ecg), len(eda))
    ecg, eda = ecg[:n], eda[:n]
    labels = resample_labels(labels, fs_in, n)
    return segment(ecg, eda, labels, window)
