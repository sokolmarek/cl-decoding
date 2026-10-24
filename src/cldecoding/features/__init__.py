"""Image-like encodings of ECG and EDA segments."""

import numpy as np

from .gadf import gadf
from .gcn import fb_copula_granger, gcn
from .mdf import mdf, motif_difference_field

__all__ = ["gadf", "mdf", "gcn", "motif_difference_field", "fb_copula_granger", "extract"]


def extract(ecg, eda, n_jobs=1):
    """GADF, MDF and GCN features of ECG and EDA segments, stacked channel-wise (ECG first).

    Returns
    -------
    dict with ``gadf`` (n, 64, 64, 2), ``mdf`` (n, 32, 64, 2) and ``gcn`` (n, 32, 32, 2)
    """
    return {
        "gadf": np.stack([gadf(ecg), gadf(eda)], axis=-1),
        "mdf": np.stack([mdf(ecg), mdf(eda)], axis=-1),
        "gcn": np.stack([gcn(ecg, n_jobs=n_jobs), gcn(eda, n_jobs=n_jobs)], axis=-1),
    }
