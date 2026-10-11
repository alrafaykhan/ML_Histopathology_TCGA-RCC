from pathlib import Path

import numpy as np

from src.utils import load_config, set_seed

ROOT = Path(__file__).resolve().parents[1]


def test_config_loads():
    cfg = load_config(ROOT / "configs" / "data.yaml")
    assert cfg["tiling"]["patch_size"] == 224
    assert not cfg["paths"]["raw_wsi"].startswith("~")


def test_seed_is_reproducible():
    set_seed(0)
    a = np.random.rand(3)
    set_seed(0)
    b = np.random.rand(3)
    assert np.allclose(a, b)
