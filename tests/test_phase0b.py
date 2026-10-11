import numpy as np
import torch

from scripts.prepare_bench_patches import slide_type
from src.features.encoder_registry import _ConvStem, list_encoders
from src.preprocessing.tile import sample_coords, tissue_mask


def test_encoder_specs_valid():
    specs = list_encoders()
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    assert len({s.order for s in specs}) == len(specs)
    for s in specs:
        assert s.source in {"timm", "timm_ctranspath", "hf_transformers"}
        assert s.embedding in {"pooled", "cls", "cls_mean", "feature_map_mean"}
        assert (s.mean is None) == (s.std is None)


def test_slide_type_from_filename():
    assert slide_type("TCGA-A3-3329-01A-01-BS1.ebb876a2.svs") == "frozen"
    assert slide_type("TCGA-B0-5094-01Z-00-DX1.a1b2.svs") == "ffpe"
    assert slide_type("TCGA-B0-5094-11A-01-TS1.a1b2.svs") == "frozen"


def test_tissue_mask_finds_stained_region():
    img = np.full((200, 200, 3), 240, dtype=np.uint8)
    img[50:150, 50:150] = (180, 80, 160)
    mask = tissue_mask(img)
    assert mask[100, 100] and not mask[10, 10]
    assert 0.2 < mask.mean() < 0.3


def test_sample_coords_seeded_and_capped():
    coords = np.arange(5000 * 2).reshape(-1, 2)
    a = sample_coords(coords, 1000, seed=1)
    b = sample_coords(coords, 1000, seed=1)
    assert len(a) == 1000 and np.array_equal(a, b)
    assert len(sample_coords(coords[:10], 1000, seed=1)) == 10


def test_ctranspath_stem_shape():
    out = _ConvStem(embed_dim=96)(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 56, 56, 96)
