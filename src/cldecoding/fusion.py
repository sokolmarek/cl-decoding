"""Fusion of the feature images into the network input (Section 4.7).

GADF (64 x 64), MDF (32 x 64, z-scored per sample and channel, zero-padded to 64 x 64) and GCN (32 x 32, resized
to 64 x 64 with bilinear interpolation) of ECG and EDA are stacked depth-wise into a 64 x 64 x 6 tensor.
"""

import numpy as np
from scipy.ndimage import zoom

MODALITIES = ("gadf", "mdf", "gcn")
IMAGE_SIZE = 64


def standardize(x, eps=1e-6):
    """Z-score every sample and channel of ``x`` (n, H, W, C) independently."""
    mean = np.nanmean(x, axis=(1, 2), keepdims=True)
    std = np.nanstd(x, axis=(1, 2), keepdims=True)
    return np.nan_to_num((x - mean) / (std + eps), nan=0.0)


def pad_to(x, size=IMAGE_SIZE):
    """Zero-pad images (n, H, W, C) at the bottom and right to ``size`` x ``size``."""
    out = np.zeros((x.shape[0], size, size, x.shape[-1]), dtype=np.float32)
    out[:, : x.shape[1], : x.shape[2]] = x
    return out


def resize_to(x, size=IMAGE_SIZE):
    """Bilinear resize of images (n, H, W, C) to ``size`` x ``size``."""
    return zoom(x, (1, size / x.shape[1], size / x.shape[2], 1), order=1)


def prepare(features):
    """Bring every modality to 64 x 64 x 2."""
    return {
        "gadf": features["gadf"].astype(np.float32),
        "mdf": pad_to(standardize(features["mdf"])),
        "gcn": resize_to(features["gcn"]).astype(np.float32),
    }


def fuse(features, modalities=MODALITIES):
    """Network input built from the selected ``modalities``, shape (n, 64, 64, 2 * len(modalities))."""
    prepared = prepare(features)
    return np.concatenate([prepared[m] for m in modalities], axis=-1)
