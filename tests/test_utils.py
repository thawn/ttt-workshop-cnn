import math

import pytest

from ttt_workshop_cnn import ensure_seed, normalize_minmax, project_root


def test_ensure_seed_returns_seed():
    s = ensure_seed(123)
    assert s == 123


def test_normalize_list():
    out = normalize_minmax([0, 5, 10])
    assert out == [0.0, 0.5, 1.0]


def test_project_root_exists():
    root = project_root()
    assert root.exists()
