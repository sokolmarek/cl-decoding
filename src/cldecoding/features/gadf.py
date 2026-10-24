"""Gramian Angular Difference Field (Section 4.4).

Each segment is reduced to ``image_size`` points by Piecewise Aggregate Approximation, rescaled to [-1, 1],
mapped to polar coordinates (phi = arccos(x)) and encoded as ``sin(phi_i - phi_j)``.
"""

import numpy as np
from pyts.image import GramianAngularField


def gadf(segments, image_size=64):
    """GADF images of a batch of 1-D segments.

    Parameters
    ----------
    segments : array_like, shape (n_segments, n_samples) or (n_samples,)
    image_size : int

    Returns
    -------
    ndarray, shape (n_segments, image_size, image_size)
    """
    segments = np.atleast_2d(np.asarray(segments, dtype=float))
    transformer = GramianAngularField(image_size=image_size, method="difference", sample_range=(-1, 1))
    return transformer.fit_transform(segments)
