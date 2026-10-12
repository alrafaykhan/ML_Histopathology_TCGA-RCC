import numpy as np

from src.preprocessing.export import slide_seed, verify_shard, write_shard
from src.preprocessing.qc import laplacian_variance, pen_mask


def _jpeg(colour: tuple[int, int, int]) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), colour).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_shard_roundtrip(tmp_path):
    shard = tmp_path / "S1.tar"
    write_shard(shard, "S1", [(0, 0, _jpeg((200, 100, 150))), (224, 0, _jpeg((150, 80, 160)))])
    assert verify_shard(shard, 2) == ""
    assert "expected 3" in verify_shard(shard, 3)
    assert verify_shard(tmp_path / "missing.tar", 1) == "shard missing"


def test_slide_seed_is_stable_and_distinct():
    assert slide_seed("TCGA-A", 42) == slide_seed("TCGA-A", 42)
    assert slide_seed("TCGA-A", 42) != slide_seed("TCGA-B", 42)


def test_pen_mask_separates_ink_from_stain():
    img = np.full((20, 40, 3), 240, dtype=np.uint8)
    img[:, :20] = (200, 90, 170)  # eosin/haematoxylin pink-purple
    img[:, 20:] = (40, 60, 200)  # blue marker
    mask = pen_mask(img)
    assert not mask[:, :18].any() and mask[:, 22:].all()


def test_laplacian_variance_detects_blur():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert laplacian_variance(sharp) > 100 * (laplacian_variance(flat) + 1)
