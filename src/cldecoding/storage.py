"""File layout of intermediate results."""

from pathlib import Path

import numpy as np


def segments_path(data_dir, dataset, window):
    return Path(data_dir) / "segments" / f"{dataset}_{window}s.npz"


def features_path(data_dir, dataset, window):
    return Path(data_dir) / "features" / f"{dataset}_{window}s.npz"


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}
