import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from cldecoding.model import build_model, capsnet  # noqa: E402


def test_parameter_count():
    assert capsnet((64, 64, 6), 2).count_params() == 323_104


def test_forward_pass():
    train_model, eval_model = build_model((64, 64, 6), 2)
    x = np.random.default_rng(0).standard_normal((3, 64, 64, 6)).astype("float32")
    y = np.eye(2, dtype="float32")[[0, 1, 0]]
    scores, rec = eval_model(x)
    assert scores.shape == (3, 2) and rec.shape == x.shape
    scores, rec = train_model([x, y])
    assert scores.shape == (3, 2) and rec.shape == x.shape
