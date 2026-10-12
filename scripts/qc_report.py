"""Write reports/qc_report.md and a contact sheet of random QC thumbnails."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402


def contact_sheet(paths: list[Path], labels: list[str], out: Path, cell: int = 256) -> None:
    cols = 6
    rows = int(np.ceil(len(paths) / cols))
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, text) in enumerate(zip(paths, labels, strict=True)):
        im = Image.open(p)
        im.thumbnail((cell, cell))
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        sheet.paste(im, (x + (cell - im.width) // 2, y + 16 + (cell - im.height) // 2))
        draw.text((x + 4, y + 2), text, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--n-thumbs", type=int, default=30)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]

    qc = pd.read_csv(ROOT / paths["slide_qc"])
    ok = qc[qc["status"] == "ok"]
    excluded = qc[qc["status"] != "ok"]

    thumbs = Path(paths["thumbnails"])
    have = ok[ok["slide_id"].map(lambda s: (thumbs / f"{s}.jpg").exists())]
    pick = have.sample(min(args.n_thumbs, len(have)), random_state=cfg["seed"])
    sheet = ROOT / paths["reports"] / "figures" / "qc_contact_sheet.jpg"
    contact_sheet(
        [thumbs / f"{s}.jpg" for s in pick["slide_id"]],
        [f"{s[:12]} {lbl}" for s, lbl in zip(pick["slide_id"], pick["label"], strict=True)],
        sheet,
    )

    summary = (
        ok.groupby(["track", "label"])
        .agg(
            slides=("slide_id", "size"),
            median_patches=("n_tissue_patches", "median"),
            min_patches=("n_tissue_patches", "min"),
            median_blur=("blur_median", "median"),
            shard_gb=("shard_bytes", lambda b: round(b.sum() / 1e9, 2)),
        )
        .reset_index()
    )
    lines = [
        "# Slide QC report",
        "",
        f"Processed {len(qc)} slides: {len(ok)} retained, {len(excluded)} excluded. "
        f"Tiles: {cfg['tiling']['patch_size']} px at {cfg['tiling']['target_mpp']} um/px, "
        f"up to {cfg['tiling']['max_patches_per_slide']} sampled per slide, "
        f"minimum {cfg['tiling']['min_patches_per_slide']} tissue patches.",
        "",
        "## Retained slides",
        "",
        summary.to_markdown(index=False) if len(summary) else "none",
        "",
        "Blur score: median variance of the Laplacian over a slide's sampled patches "
        f"(exclusion threshold {cfg['qc']['min_blur_laplacian_var']}). "
        f"Lowest retained: {ok['blur_median'].min() if len(ok) else 'n/a'}; "
        f"5th percentile: {ok['blur_median'].quantile(0.05) if len(ok) else 'n/a'}.",
        "",
        "## Excluded slides",
        "",
        (
            excluded[["slide_id", "label", "reason"]].to_markdown(index=False)
            if len(excluded)
            else "none"
        ),
        "",
        f"## Contact sheet ({len(pick)} random retained slides)",
        "",
        "Green: tissue contour; red: detected pen marks; blue: sampled patches.",
        "",
        "![contact sheet](figures/qc_contact_sheet.jpg)",
        "",
    ]
    report = ROOT / paths["reports"] / "qc_report.md"
    report.write_text("\n".join(lines))
    print(f"wrote {report} and {sheet}")


if __name__ == "__main__":
    main()
