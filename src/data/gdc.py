"""Minimal GDC API client: file queries and resumable, md5-verified downloads."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

GDC_API = "https://api.gdc.cancer.gov"

FILE_FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.sample_type",
    "cases.samples.submitter_id",
]


def query_slide_files(projects: list[str], page_size: int = 10000) -> pd.DataFrame:
    """Return one row per open-access SVS slide file in the given GDC projects."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": projects}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    payload = {
        "filters": filters,
        "fields": ",".join(FILE_FIELDS),
        "size": page_size,
        "format": "JSON",
    }
    resp = requests.post(f"{GDC_API}/files", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()["data"]
    if data["pagination"]["total"] > page_size:
        raise RuntimeError("GDC result exceeds page size; increase page_size")

    rows = []
    for hit in data["hits"]:
        case = hit["cases"][0]
        sample = case["samples"][0]
        rows.append(
            {
                "file_id": hit["file_id"],
                "file_name": hit["file_name"],
                "file_size": hit["file_size"],
                "md5": hit["md5sum"],
                "patient_id": case["submitter_id"],
                "project": case["project"]["project_id"],
                "sample_type": sample["sample_type"],
                "sample_id": sample["submitter_id"],
            }
        )
    return pd.DataFrame(rows)


def md5_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_file(
    file_id: str, md5: str, out_path: Path, retries: int = 20, chunk: int = 1 << 20
) -> Path:
    """Download a GDC file with HTTP range resume; verify md5. Returns the final path."""
    out_path = Path(out_path)
    if out_path.exists() and md5_of(out_path) == md5:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")

    failures = 0  # consecutive attempts that added no bytes
    while True:
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                f"{GDC_API}/data/{file_id}", headers=headers, stream=True, timeout=60
            ) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                mode = "ab" if offset and r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
            break
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.HTTPError,
            requests.exceptions.ChunkedEncodingError,
        ) as err:
            grew = part.exists() and part.stat().st_size > offset
            failures = 0 if grew else failures + 1
            if failures >= retries:
                raise RuntimeError(
                    f"download failed after {retries} attempts without progress: {file_id}"
                ) from err
            wait = min(60, 2**failures)
            print(f"download {file_id}: {type(err).__name__}; resuming in {wait}s", flush=True)
            time.sleep(wait)

    if md5_of(part) != md5:
        part.unlink()
        raise RuntimeError(f"md5 mismatch for {file_id}")
    part.rename(out_path)
    return out_path
