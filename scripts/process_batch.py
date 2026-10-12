"""Phase 3: segment, QC, tile and export patch shards for one downloaded batch.

Slides already recorded in the QC table are skipped, so the script can be rerun after an
interruption. Raw WSIs of the batch are deleted only when every slide in it has a final
status (verified shard or logged exclusion).

Usage:
    python scripts/process_batch.py --batch 1
    python scripts/process_batch.py --wsi data/raw/smoke/*.svs --no-delete   # ad hoc test
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.export import process_slide, verify_shard  # noqa: E402
from src.utils import load_config  # noqa: E402

QC_COLUMNS = [
    "slide_id", "batch", "track", "label", "status", "reason", "level0_mpp", "width",
    "height", "footprint", "tissue_frac", "pen_frac", "n_tissue_patches", "n_sampled",
    "blur_median", "shard_bytes", "shard_sha256", "seconds",
]  # fmt: skip


def load_qc(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=QC_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=int)
    group.add_argument("--wsi", nargs="+", help="explicit slide files (not tied to a batch)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-delete", action="store_true", help="keep raw WSIs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]

    cohort = pd.read_csv(ROOT / paths["cohort"]).set_index("slide_id")
    raw_root = Path(paths["raw_wsi"])
    if args.batch is not None:
        rows = cohort[cohort["batch"] == args.batch]
        jobs = {sid: raw_root / f"track{r.track}" / r.file_name for sid, r in rows.iterrows()}
    else:
        jobs = {Path(p).name.split(".")[0]: Path(p) for p in args.wsi}

    qc_path = ROOT / paths["slide_qc"]
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc = load_qc(qc_path)
    done = set(qc["slide_id"])
    out = {
        "coords": Path(paths["coords"]),
        "thumbnails": Path(paths["thumbnails"]),
        "shards": Path(paths["patch_export"]),
    }

    todo = {sid: p for sid, p in jobs.items() if sid not in done and p.exists()}
    missing = [sid for sid, p in jobs.items() if sid not in done and not p.exists()]
    print(
        f"{len(jobs)} slides: {len(jobs) - len(todo) - len(missing)} already processed, "
        f"{len(todo)} to process, {len(missing)} not downloaded",
        flush=True,
    )

    workers = args.workers or cfg["streaming"]["workers"]
    with ProcessPoolExecutor(workers) as pool:
        futures = {
            pool.submit(process_slide, path, sid, out, cfg["tiling"], cfg["qc"], cfg["seed"]): sid
            for sid, path in todo.items()
        }
        for n, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                row = fut.result()
            except Exception as err:  # noqa: BLE001 - worker crash; record and continue
                row = {"slide_id": sid, "status": "excluded", "reason": f"error: {err!r}"}
            if sid in cohort.index:
                row.update(
                    batch=cohort.at[sid, "batch"],
                    track=cohort.at[sid, "track"],
                    label=cohort.at[sid, "label"],
                )
            qc = pd.concat([qc, pd.DataFrame([row])], ignore_index=True)
            qc.reindex(columns=QC_COLUMNS).to_csv(qc_path, index=False)
            print(
                f"[{n}/{len(todo)}] {sid} {row['status']} {row.get('reason', '')} "
                f"patches={row.get('n_sampled', '')} blur={row.get('blur_median', '')} "
                f"{row.get('seconds', '')}s",
                flush=True,
            )

    if args.no_delete or not cfg["streaming"]["delete_raw_after_export"]:
        return
    if missing:
        print(f"keeping raw WSIs: {len(missing)} slides of the batch not processed yet")
        return
    final = qc.set_index("slide_id").loc[list(jobs)]
    for sid, r in final[final["status"] == "ok"].iterrows():
        problem = verify_shard(out["shards"] / f"{sid}.tar", int(r.n_sampled), r.shard_sha256)
        if problem:
            sys.exit(f"not deleting raw WSIs: {sid}: {problem}")
    freed = 0
    for path in jobs.values():
        if path.exists():
            freed += path.stat().st_size
            path.unlink()
    print(f"deleted raw WSIs of the batch ({freed / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
