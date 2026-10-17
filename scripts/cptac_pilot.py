"""CPTAC-CCRCC pilot: download a seeded sample of normal slides to check fixation.

IDC metadata does not record whether CPTAC slides are FFPE or frozen, so a sample
is downloaded and inspected at 10x before the collection is added to the cohort.

Usage: python scripts/cptac_pilot.py --n 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.idc import download_series, query_sm_series  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--n", type=int, default=20, help="normal slides to sample")
    parser.add_argument("--n-tumor", type=int, default=4, help="tumour slides for comparison")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    seed = cfg["seed"]
    out_dir = Path(cfg["paths"]["raw_wsi"]).expanduser() / "cptac_pilot"

    series = query_sm_series("cptac_ccrcc")
    print(
        f"cptac_ccrcc: {len(series)} slides, {series.patient_id.nunique()} patients; "
        + ", ".join(f"{k}={v}" for k, v in series.tissue.value_counts().items())
    )

    def sample(tissue: str, n: int):
        sub = series[series.tissue == tissue]
        one_per_patient = sub.groupby("patient_id").sample(n=1, random_state=seed)
        return one_per_patient.sample(n=min(n, len(one_per_patient)), random_state=seed)

    picked = pd.concat([sample("Normal", args.n), sample("tumor", args.n_tumor)])
    manifest = ROOT / cfg["paths"]["manifests"] / "cptac_pilot.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    picked.to_csv(manifest, index=False)
    print(f"selected {len(picked)} slides, {picked.size_mb.sum() / 1024:.1f} GB -> {manifest}")

    download_series(picked.series_uid.tolist(), out_dir)
    print(f"downloaded to {out_dir}")


if __name__ == "__main__":
    main()
