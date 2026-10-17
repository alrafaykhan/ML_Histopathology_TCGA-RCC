"""Contact sheets of the CPTAC pilot: thumbnail plus four random 10x patches per slide.

Usage: python scripts/cptac_pilot_review.py qc/cptac_pilot
"""

import sys
from pathlib import Path

import openslide
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.tile import read_patch, sample_coords, tissue_grid  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(ROOT / "data/manifests/cptac_pilot.csv")
rows, stats = [], []
for r in m.itertuples():
    d = list((ROOT / "data/raw/cptac_pilot").glob(f"*/{r.series_uid}"))
    if not d:
        continue
    files = sorted(d[0].glob("*.dcm"))
    try:
        s = openslide.OpenSlide(str(files[0]))
        g = tissue_grid(s, exclude_pen=True)
        pick = sample_coords(g["coords"], 4, seed=0)
        patches = [read_patch(s, x, y, g["footprint"]).resize((300, 300)) for x, y in pick]
    except Exception as e:  # noqa: BLE001
        print(r.slide_id, "skip", type(e).__name__)
        continue
    th = Image.fromarray(g["thumb"])
    th.thumbnail((400, 300))
    row = Image.new("RGB", (400 + 4 * 300, 320), "white")
    row.paste(th, (0, 20))
    for i, p in enumerate(patches):
        row.paste(p, (400 + i * 300, 20))
    ImageDraw.Draw(row).text(
        (4, 2), f"{r.slide_id} {r.tissue} patches={len(g['coords'])}", fill="black"
    )
    rows.append(row)
    stats.append((r.slide_id, r.tissue, len(g["coords"]), int(g["pen"].sum())))
for k in range(0, len(rows), 4):
    page = Image.new("RGB", (1600, 320 * len(rows[k : k + 4])), "white")
    for i, row in enumerate(rows[k : k + 4]):
        page.paste(row, (0, i * 320))
    page.save(out / f"page{k // 4 + 1}.jpg", quality=80)
print(pd.DataFrame(stats, columns=["slide", "tissue", "patches_10x", "pen_px"]).to_string())
