"""Patch encoder registry. Each encoder is described by ``configs/encoders/<name>.yaml``.

Loading recipes and embedding choices follow each model card; the non-timm recipes
(CTransPath, Phikon, kaiko, Midnight) match TRIDENT (mahmoodlab/TRIDENT @ e4c98b2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
ENCODER_DIR = ROOT / "configs" / "encoders"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EncoderSpec:
    name: str
    tier: str
    order: int
    source: str  # timm | timm_ctranspath | hf_transformers
    model_id: str
    embedding: str  # pooled | cls | cls_mean | feature_map_mean
    gated: bool = False
    licence: str = ""
    input_size: int = 224
    mean: tuple[float, ...] | None = None  # None: take from the timm pretrained config
    std: tuple[float, ...] | None = None
    kwargs: dict = field(default_factory=dict)


def load_spec(name: str, encoder_dir: Path = ENCODER_DIR) -> EncoderSpec:
    with open(Path(encoder_dir) / f"{name}.yaml") as f:
        return EncoderSpec(**yaml.safe_load(f))


def list_encoders(encoder_dir: Path = ENCODER_DIR, include_gated: bool = True) -> list[EncoderSpec]:
    """All encoder specs in run order (tier order, then ``order``)."""
    specs = [load_spec(p.stem, encoder_dir) for p in Path(encoder_dir).glob("*.yaml")]
    if not include_gated:
        specs = [s for s in specs if not s.gated]
    return sorted(specs, key=lambda s: s.order)


class _ConvStem(nn.Module):
    """CTransPath convolutional patch embedding, NHWC output for timm>=0.9 Swin."""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **_):
        super().__init__()
        assert patch_size == 4 and embed_dim % 8 == 0
        layers, c_in, c_out = [], in_chans, embed_dim // 8
        for _ in range(2):
            layers += [
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in, c_out = c_out, c_out * 2
        layers.append(nn.Conv2d(c_in, embed_dim, 1))
        self.proj = nn.Sequential(*layers)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


def _build_backbone(spec: EncoderSpec) -> tuple[nn.Module, tuple, tuple]:
    if spec.source in ("timm", "timm_ctranspath"):
        import timm

        kwargs = dict(spec.kwargs)
        if spec.source == "timm_ctranspath":
            kwargs["embed_layer"] = _ConvStem
        kwargs.setdefault("num_classes", 0)
        model = timm.create_model(spec.model_id, pretrained=True, **kwargs)
        cfg = model.pretrained_cfg
        mean = spec.mean or cfg.get("mean", IMAGENET_MEAN)
        std = spec.std or cfg.get("std", IMAGENET_STD)
        return model, tuple(mean), tuple(std)

    if spec.source == "hf_transformers":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(spec.model_id, **spec.kwargs)
        return model, tuple(spec.mean or IMAGENET_MEAN), tuple(spec.std or IMAGENET_STD)

    raise ValueError(f"unknown encoder source: {spec.source}")


class PatchEncoder(nn.Module):
    """Wraps a backbone: uint8 NCHW RGB patches in, one embedding per patch out."""

    def __init__(self, spec: EncoderSpec):
        super().__init__()
        self.spec = spec
        self.backbone, mean, std = _build_backbone(spec)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255)
        self.backbone.eval().requires_grad_(False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.shape[-1] != self.spec.input_size:
            x = F.interpolate(
                x, size=self.spec.input_size, mode="bilinear", antialias=True, align_corners=False
            )
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        if self.spec.source == "hf_transformers":
            tokens = self.backbone(pixel_values=x).last_hidden_state
            if self.spec.embedding == "cls":
                return tokens[:, 0]
            if self.spec.embedding == "cls_mean":
                return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)
            raise ValueError(f"unsupported embedding for transformers: {self.spec.embedding}")
        if self.spec.embedding in ("pooled", "cls"):
            return self.backbone(x)
        if self.spec.embedding == "feature_map_mean":
            fmap = self.backbone(x)[-1]
            return fmap.mean(dim=(2, 3))
        raise ValueError(f"unsupported embedding for timm: {self.spec.embedding}")


def build_encoder(name_or_spec: str | EncoderSpec) -> PatchEncoder:
    spec = load_spec(name_or_spec) if isinstance(name_or_spec, str) else name_or_spec
    return PatchEncoder(spec)
