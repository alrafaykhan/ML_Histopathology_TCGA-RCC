"""Per-slide segmentation, QC, coordinate storage and patch-shard export."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
from PIL import Image

from src.preprocessing.qc import laplacian_variance
from src.preprocessing.tile import read_patch, sample_coords, tissue_grid
from src.utils import wsi_open_path


def slide_seed(slide_id: str, seed: int) -> int:
    """Stable per-slide seed derived from the global seed and the slide id."""
    return (int(hashlib.md5(slide_id.encode()).hexdigest()[:8], 16) ^ seed) % 2**32


def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_coords(path: Path, grid: dict, sampled: np.ndarray, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("coords", data=grid["coords"], compression="gzip")
        f.create_dataset("coords_sampled", data=sampled, compression="gzip")
        f.create_dataset("tissue_mask", data=grid["tissue"], compression="gzip")
        for k, v in attrs.items():
            f.attrs[k] = v
    tmp.replace(path)


def save_thumbnail(path: Path, grid: dict, sampled: np.ndarray, max_side: int = 1024) -> None:
    """Thumbnail with tissue contours (green), pen marks (red) and sampled patches (blue)."""
    thumb = grid["thumb"].copy()
    for mask, colour in ((grid["tissue"], (0, 200, 0)), (grid["pen"], (220, 0, 0))):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(thumb, contours, -1, colour, 2)
    side = max(1, int(round(grid["footprint"] / grid["scale"])))
    for x, y in (sampled / grid["scale"]).astype(int):
        cv2.rectangle(thumb, (x, y), (x + side - 1, y + side - 1), (0, 0, 220), 1)
    img = Image.fromarray(thumb)
    img.thumbnail((max_side, max_side))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=85)


def write_shard(path: Path, slide_id: str, patches: list[tuple[int, int, bytes]]) -> None:
    """WebDataset-style tar: ``<slide_id>_<idx>.jpg`` plus ``.json`` with level-0 x, y."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tar.tmp")
    with tarfile.open(tmp, "w") as tar:
        for idx, (x, y, jpg) in enumerate(patches):
            key = f"{slide_id}_{idx:04d}"
            meta = json.dumps({"slide_id": slide_id, "idx": idx, "x": x, "y": y}).encode()
            for suffix, data in (("jpg", jpg), ("json", meta)):
                info = tarfile.TarInfo(f"{key}.{suffix}")
                info.size = len(data)
                info.mtime = 0
                tar.addfile(info, io.BytesIO(data))
    tmp.replace(path)


def verify_shard(path: Path, n_expected: int, sha256: str | None = None) -> str:
    """Return '' if the shard is intact (count, checksum, first/last JPEG decode), else why."""
    if not path.exists():
        return "shard missing"
    if sha256 is not None and sha256_of(path) != sha256:
        return "shard checksum mismatch"
    with tarfile.open(path) as tar:
        jpgs = [m for m in tar.getmembers() if m.name.endswith(".jpg")]
        if len(jpgs) != n_expected:
            return f"shard has {len(jpgs)} patches, expected {n_expected}"
        for m in (jpgs[0], jpgs[-1]):
            Image.open(io.BytesIO(tar.extractfile(m).read())).verify()
    return ""


def process_slide(
    wsi_path: str | Path,
    slide_id: str,
    out: dict[str, Path],
    tiling: dict,
    qc: dict,
    seed: int,
) -> dict:
    """Segment, QC, tile and export one slide. Returns a QC row (``status`` ok/excluded)."""
    t0 = time.time()
    row = {"slide_id": slide_id, "status": "excluded", "reason": ""}
    try:
        slide = openslide.OpenSlide(str(wsi_open_path(wsi_path)))
    except Exception as err:  # noqa: BLE001 - corrupt or unsupported file
        row["reason"] = f"unreadable: {err}"
        return row
    try:
        try:
            grid = tissue_grid(
                slide,
                tiling["target_mpp"],
                tiling["patch_size"],
                min_tissue=qc["min_tissue_frac_per_patch"],
                exclude_pen=True,
            )
        except ValueError as err:
            row["reason"] = str(err)
            return row
        tissue_px = int(grid["tissue"].sum())
        pen_px = int(grid["pen"].sum())
        marked_px = int((grid["tissue"] | grid["pen"]).sum())
        coords = grid["coords"]
        sampled = sample_coords(coords, tiling["max_patches_per_slide"], slide_seed(slide_id, seed))
        row.update(
            level0_mpp=round(tiling["patch_size"] * tiling["target_mpp"] / grid["footprint"], 4),
            width=slide.dimensions[0],
            height=slide.dimensions[1],
            footprint=round(grid["footprint"], 2),
            tissue_frac=round(tissue_px / grid["tissue"].size, 4),
            pen_frac=round(pen_px / max(1, marked_px), 4),
            n_tissue_patches=len(coords),
            n_sampled=len(sampled),
        )

        write_coords(
            out["coords"] / f"{slide_id}.h5",
            grid,
            sampled,
            {
                "slide_id": slide_id,
                "target_mpp": tiling["target_mpp"],
                "patch_size": tiling["patch_size"],
                "footprint": grid["footprint"],
                "seed": slide_seed(slide_id, seed),
            },
        )
        save_thumbnail(out["thumbnails"] / f"{slide_id}.jpg", grid, sampled)

        if row["pen_frac"] > qc["max_pen_frac"]:
            row["reason"] = f"mostly pen marks (pen fraction {row['pen_frac']:.2f})"
            return row
        if len(coords) < tiling["min_patches_per_slide"]:
            row["reason"] = f"too little tissue ({len(coords)} patches)"
            return row

        patches, sharpness = [], []
        for x, y in sampled:
            img = read_patch(slide, x, y, grid["footprint"], tiling["patch_size"])
            arr = np.asarray(img)
            sharpness.append(laplacian_variance(arr))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=tiling["jpeg_quality"])
            patches.append((int(x), int(y), buf.getvalue()))
        row["blur_median"] = round(float(np.median(sharpness)), 1)
        if row["blur_median"] < qc["min_blur_laplacian_var"]:
            row["reason"] = f"blurry (median Laplacian variance {row['blur_median']})"
            return row

        shard = out["shards"] / f"{slide_id}.tar"
        write_shard(shard, slide_id, patches)
        row["shard_sha256"] = sha256_of(shard)
        row["shard_bytes"] = shard.stat().st_size
        problem = verify_shard(shard, len(patches), row["shard_sha256"])
        if problem:
            row["reason"] = problem
            return row
        row["status"] = "ok"
        return row
    finally:
        slide.close()
        row["seconds"] = round(time.time() - t0, 1)
