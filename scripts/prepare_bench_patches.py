"""Download a few smoke-test slides and sample the patch set used by the Phase 0b benchmark.

Picks one frozen KIRC tumour, one FFPE KIRP tumour and one frozen KICH tumour slide (sizes
near a target so downloads stay small), tiles them at the configured magnification and
writes a seeded sample of JPEG-compressed patches to ``patch_export/bench/``.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.gdc import download_file, query_slide_files  # noqa: E402
from src.preprocessing.tile import read_patch, sample_coords, tissue_coords  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

PICKS = [
    ("TCGA-KIRC", "frozen", 250e6),
    ("TCGA-KIRP", "ffpe", 450e6),
    ("TCGA-KICH", "frozen", 250e6),
]


def slide_type(file_name: str) -> str:
    match = re.search(r"-(DX|TS|BS|MS)[0-9A-Z]*\.", file_name)
    if not match:
        return "unknown"
    return "ffpe" if match.group(1) == "DX" else "frozen"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--n-patches", type=int, default=2000)
    parser.add_argument("--max-tries", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = set_seed(cfg["seed"])
    tcfg = cfg["tiling"]
    raw_dir = Path(cfg["paths"]["raw_wsi"]) / "smoke"
    out_dir = Path(cfg["paths"]["patch_export"]).parent / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f"patches_{args.n_patches}.npy"
    if out_npy.exists():
        print(f"{out_npy} exists; nothing to do")
        return

    files = query_slide_files(cfg["gdc"]["projects"])
    files["slide_type"] = files.file_name.map(slide_type)
    files.to_csv(ROOT / cfg["paths"]["manifests"] / "gdc_open_svs_files.csv", index=False)
    print(f"GDC: {len(files)} open SVS files")

    per_slide = np.array_split(np.arange(args.n_patches), len(PICKS))
    patches, meta = [], []
    for (project, stype, target), idx in zip(PICKS, per_slide, strict=True):
        n_needed = len(idx)
        cand = files[
            (files.project == project)
            & (files.sample_type == "Primary Tumor")
            & (files.slide_type == stype)
        ].copy()
        cand["dist"] = (cand.file_size - target).abs()
        cand = cand.sort_values(["dist", "file_id"]).head(args.max_tries)

        for row in cand.itertuples():
            path = raw_dir / row.file_name
            print(f"{project} {stype}: {row.file_name} ({row.file_size / 1e6:.0f} MB)", flush=True)
            download_file(row.file_id, row.md5, path)
            slide = openslide.OpenSlide(str(path))
            try:
                coords, footprint = tissue_coords(slide, tcfg["target_mpp"], tcfg["patch_size"])
            except ValueError as err:
                print(f"  skipped: {err}")
                continue
            print(f"  {len(coords)} tissue patches at {tcfg['target_mpp']} um/px")
            if len(coords) < n_needed:
                print(f"  skipped: fewer than {n_needed} tissue patches")
                continue
            for x, y in sample_coords(coords, n_needed, seed):
                img = read_patch(slide, x, y, footprint, tcfg["patch_size"])
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=tcfg["jpeg_quality"])
                patches.append(np.asarray(Image.open(buf).convert("RGB")))
                meta.append({"file_name": row.file_name, "project": project, "x": x, "y": y})
            break
        else:
            raise RuntimeError(f"no usable slide found for {project} {stype}")

    np.save(out_npy, np.stack(patches))
    pd.DataFrame(meta).to_csv(out_dir / f"patches_{args.n_patches}.csv", index=False)
    print(f"wrote {len(patches)} patches to {out_npy}")


if __name__ == "__main__":
    main()
