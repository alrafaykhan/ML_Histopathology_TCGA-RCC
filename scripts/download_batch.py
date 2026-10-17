"""Download one batch of selected slides with disk guards.

GDC slides are resumable and md5-verified; IDC (CPTAC DICOM-WSI) series are fetched with
s5cmd and verified against the per-instance sizes in the IDC index.

Usage: python scripts/download_batch.py --batch 1
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.gdc import download_file  # noqa: E402
from src.data.idc import download_series, series_instances, verify_series  # noqa: E402
from src.utils import load_config, path_bytes  # noqa: E402

GB = 1e9


def free_gb(path: str | Path) -> float:
    return shutil.disk_usage(path).free / GB


def dir_gb(path: Path) -> float:
    return (
        sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / GB if path.exists() else 0
    )


def check_disk(cfg: dict, raw_root: Path, pending_gb: float) -> None:
    """Exit with a message if this batch would break the disk limits in configs/data.yaml."""
    s = cfg["streaming"]
    raw_gb = dir_gb(raw_root)
    home_free = free_gb(Path.home())
    problems = []
    if raw_gb + pending_gb > s["max_raw_gb"]:
        problems.append(
            f"raw WSIs would reach {raw_gb + pending_gb:.0f} GB > max_raw_gb {s['max_raw_gb']}"
        )
    if home_free - pending_gb < s["stop_if_wsl_free_below_gb"]:
        problems.append(
            f"WSL free space would drop to {home_free - pending_gb:.0f} GB "
            f"< {s['stop_if_wsl_free_below_gb']} GB"
        )
    if home_free < s["max_raw_gb"] + s["min_free_margin_gb"]:
        problems.append(
            f"WSL free {home_free:.0f} GB < max_raw_gb + margin "
            f"({s['max_raw_gb'] + s['min_free_margin_gb']} GB)"
        )
    if os.path.isdir("/mnt/d") and free_gb("/mnt/d") < s["min_free_windows_d_gb"]:
        problems.append(f"D: free {free_gb('/mnt/d'):.0f} GB < {s['min_free_windows_d_gb']} GB")
    print(
        f"disk: raw {raw_gb:.1f} GB, batch pending {pending_gb:.1f} GB, WSL free {home_free:.0f} GB"
        + (f", D: free {free_gb('/mnt/d'):.0f} GB" if os.path.isdir("/mnt/d") else "")
    )
    if problems:
        sys.exit("stopping: " + "; ".join(problems))


def download_idc(batch: pd.DataFrame, dest: dict, batch_no: int, log_path: Path) -> None:
    """Fetch a batch of IDC series in one s5cmd transfer, then verify each series."""
    out_dir = next(iter(dest.values())).parent
    t0 = time.time()
    download_series(batch["file_id"].tolist(), out_dir)
    expected = series_instances(batch["file_id"].tolist()).groupby("SeriesInstanceUID")
    new_log = not log_path.exists()
    failures = 0
    with open(log_path, "a", newline="") as fh:
        log = csv.writer(fh)
        if new_log:
            log.writerow(["time", "batch", "slide_id", "file_id", "bytes", "seconds", "status"])
        for i, r in batch.iterrows():
            problem = verify_series(dest[i], expected.get_group(r.file_id))
            status = f"failed: {problem}" if problem else "ok"
            failures += bool(problem)
            log.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    batch_no,
                    r.slide_id,
                    r.file_id,
                    path_bytes(dest[i]),
                    round(time.time() - t0),
                    status,
                ]
            )
    print(f"batch {batch_no} done: {len(batch) - failures} ok, {failures} failed", flush=True)
    if failures:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    cohort = pd.read_csv(ROOT / cfg["paths"]["cohort"])
    batch = cohort[cohort["batch"] == args.batch]
    if batch.empty:
        sys.exit(f"batch {args.batch} has no slides")
    raw_root = Path(cfg["paths"]["raw_wsi"])
    smoke = raw_root / "smoke"
    dest = {i: raw_root / f"track{r.track}" / r.file_name for i, r in batch.iterrows()}

    for path in dest.values():
        cached = smoke / path.name
        if not path.exists() and cached.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(cached, path)

    pending = sum(max(0, r.file_size - path_bytes(dest[i])) for i, r in batch.iterrows())
    print(
        f"batch {args.batch}: {len(batch)} slides, {batch['file_size'].sum() / GB:.1f} GB, "
        f"classes {batch['label'].value_counts().to_dict()}"
    )
    check_disk(cfg, raw_root, pending / GB)

    if (batch["source"] == "idc").all():
        download_idc(batch, dest, args.batch, ROOT / cfg["paths"]["logs"] / "downloads.csv")
        return

    log_path = ROOT / cfg["paths"]["logs"] / "downloads.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    workers = args.workers or cfg["streaming"]["workers"]
    t0 = time.time()
    failures = 0
    with open(log_path, "a", newline="") as fh, ThreadPoolExecutor(workers) as pool:
        log = csv.writer(fh)
        if new_log:
            log.writerow(["time", "batch", "slide_id", "file_id", "bytes", "seconds", "status"])
        futures = {
            pool.submit(download_file, r.file_id, r.md5, dest[i]): (r, time.time())
            for i, r in batch.iterrows()
        }
        for n, fut in enumerate(as_completed(futures), 1):
            r, start = futures[fut]
            try:
                fut.result()
                status = "ok"
            except Exception as err:  # noqa: BLE001 - log and continue with the batch
                status = f"failed: {err}"
                failures += 1
            secs = round(time.time() - start)
            log.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    args.batch,
                    r.slide_id,
                    r.file_id,
                    r.file_size,
                    secs,
                    status,
                ]
            )
            fh.flush()
            elapsed = time.time() - t0
            print(
                f"[{n}/{len(batch)}] {r.slide_id} {r.file_size / 1e6:.0f} MB {status} "
                f"({elapsed / 3600:.1f} h elapsed)",
                flush=True,
            )
    print(f"batch {args.batch} done: {len(batch) - failures} ok, {failures} failed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
