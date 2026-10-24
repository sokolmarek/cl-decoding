import numpy as np
import pytest

from cldecoding.features import extract, fb_copula_granger, gadf, motif_difference_field
from cldecoding.features.mdf import motif_differences
from cldecoding.fusion import fuse
from cldecoding.preprocessing import FS, segment, window_starts


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("window", [5, 3, 1])
def test_feature_shapes(rng, window):
    x = rng.standard_normal((2, window * FS))
    feats = extract(x, x)
    assert feats["gadf"].shape == (2, 64, 64, 2)
    assert feats["mdf"].shape == (2, 32, 64, 2)
    assert feats["gcn"].shape == (2, 32, 32, 2)
    assert fuse(feats).shape == (2, 64, 64, 6)
    assert np.isfinite(fuse(feats)).all()


def test_gadf_diagonal_is_zero(rng):
    img = gadf(rng.standard_normal(320), image_size=64)[0]
    assert np.allclose(np.diag(img), 0.0)


def test_motif_differences_match_definition():
    x = np.arange(10, dtype=float) ** 2
    t = motif_differences(x, 3)
    d, s = 2, 1
    assert t[d - 1, s, 0] == x[s + d] - x[s]
    assert t[d - 1, s, 1] == x[s + 2 * d] - x[s + d]
    # starts beyond the last valid motif stay zero
    assert np.all(t[d - 1, len(x) - 2 * d:] == 0)


def test_mdf_shape_for_any_length(rng):
    for n in (256, 768, 1280):
        assert motif_difference_field(rng.standard_normal(n)).shape == (32, 64)


def test_gcn_detects_lagged_driver(rng):
    T, K = 40, 32
    z = rng.standard_normal((K, T))
    z[1, 1:] += 3.0 * z[0, :-1]  # series 0 drives series 1 with lag 1
    a = fb_copula_granger(z.ravel(), n_series=K)
    assert a.shape == (K, K)
    assert a[0, 1] > np.median(a)


def test_segmentation_respects_labels():
    n = 20 * FS
    labels = np.r_[np.zeros(7 * FS, int), -np.ones(3 * FS, int), np.ones(10 * FS, int)]
    starts = window_starts(labels, FS, 5)
    assert list(starts) == [0, 10 * FS, 15 * FS]
    ecg = np.zeros(n)
    _, _, y = segment(ecg, ecg, labels, 5)
    assert list(y) == [0, 1, 1]


def test_lasso_matches_sklearn(rng):
    from sklearn.linear_model import Lasso

    from cldecoding.features.gcn import lasso_path_cd

    X = rng.standard_normal((30, 12))
    X = (X - X.mean(0)) / X.std(0)
    y = X[:, 0] - 0.5 * X[:, 3] + 0.1 * rng.standard_normal(30)
    y = y - y.mean()
    lambdas = np.array([0.5, 0.05, 0.005])
    ours = lasso_path_cd(X, y, lambdas, 10_000, 1e-10)
    for k, lam in enumerate(lambdas):
        ref = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100_000).fit(X, y).coef_
        assert np.allclose(ours[:, k], ref, atol=1e-6)
