"""Motif Difference Field (Section 4.5), after Zhang and Chen (2023).

For every delay ``d = 1..d_max`` with ``d_max = (N - 1) // (n - 1)`` and every valid start ``s``, the motif
``(x_s, x_{s+d}, ..., x_{s+(n-1)d})`` is described by its first-order differences
``x_{s+id} - x_{s+(i-1)d}`` for ``i = 1..n-1`` (Eq. 12). The resulting ``d_max x (N - n + 1) x (n - 1)`` tensor is
zero-padded where a delay has fewer valid starts. The 2-D field combines the first difference component with
the reversed second component (Eq. 14) and is resized with bicubic interpolation.
"""

import numpy as np
from numba import njit
from scipy.ndimage import zoom

MOTIF_LENGTH = 3
FIELD_SHAPE = (32, 64)


@njit(cache=True)
def motif_differences(x, n=MOTIF_LENGTH):
    """Tensor of motif differences, shape ``(d_max, N - n + 1, n - 1)``."""
    N = x.shape[0]
    d_max = (N - 1) // (n - 1)
    out = np.zeros((d_max, N - n + 1, n - 1))
    for d in range(1, d_max + 1):
        for s in range(N - (n - 1) * d):
            for i in range(1, n):
                out[d - 1, s, i - 1] = x[s + i * d] - x[s + (i - 1) * d]
    return out


def motif_difference_field(x, n=MOTIF_LENGTH, shape=FIELD_SHAPE):
    """MDF image of one segment, resized to ``shape``."""
    if n < 3:
        raise ValueError("The 2-D field needs at least two difference components (n >= 3).")
    tensor = motif_differences(np.ascontiguousarray(x, dtype=np.float64), n)
    field = tensor[:, :, 0] + np.flipud(tensor[:, ::-1, 1])
    factors = (shape[0] / field.shape[0], shape[1] / field.shape[1])
    return zoom(field, factors, order=3, grid_mode=False)[: shape[0], : shape[1]]


def mdf(segments, n=MOTIF_LENGTH, shape=FIELD_SHAPE):
    """MDF images of a batch of segments, shape ``(n_segments, *shape)``."""
    segments = np.atleast_2d(segments)
    return np.stack([motif_difference_field(x, n, shape) for x in segments])
