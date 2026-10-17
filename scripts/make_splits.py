"""Phase 2: patient-level test hold-out, 5 CV folds and a site-held-out split.

Writes data/splits/{patients,main,site_holdout}.csv and reports/splits_summary.md.

Usage: python scripts/make_splits.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import (  # noqa: E402
    holdout_and_folds,
    patient_table,
    site_holdout,
    slide_splits,
)
from src.utils import load_config  # noqa: E402

CLASSES = ["ccRCC", "pRCC", "chRCC", "Normal"]


def counts_md(slides: pd.DataFrame, by: str) -> str:
    """Slides per class (columns) and split/fold (rows), separately for FFPE and frozen."""
    parts = []
    for name, sub in [("FFPE (Tracks B+C)", slides[slides["track"] != "A"]),
                      ("Frozen (Track A)", slides[slides["track"] == "A"])]:  # fmt: skip
        tab = pd.crosstab(sub[by], sub["label"], margins=True, margins_name="total")
        tab = tab.reindex(columns=[c for c in CLASSES if c in tab.columns] + ["total"])
        parts += [f"{name}:", "", tab.to_markdown(), ""]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    scfg = cfg["splits"]
    seed = cfg["seed"]

    cohort = pd.read_csv(ROOT / cfg["paths"]["cohort"])
    patients = patient_table(cohort)
    main_split = holdout_and_folds(patients, scfg["test_frac"], scfg["n_folds"], seed)
    site_split = site_holdout(patients, scfg["site_test_frac"], seed)

    out_dir = ROOT / cfg["paths"]["splits"]
    out_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(out_dir / "patients.csv", index=False)
    main_slides = slide_splits(cohort, main_split)
    site_slides = slide_splits(cohort, site_split)
    main_slides.to_csv(out_dir / "main.csv", index=False)
    site_slides.to_csv(out_dir / "site_holdout.csv", index=False)

    main_slides["part"] = main_slides["split"].where(
        main_slides["split"] == "test", "fold " + main_slides["fold"].astype(str)
    )
    held = site_slides[
        (site_slides["split"] == "test") & (site_slides["site"] != cfg["cptac"]["project"])
    ]
    test_sites = sorted(held["site"].unique())
    tcga = patients[patients["source"] == "tcga"].merge(site_split, on="patient_id")
    tcga_frac = tcga.groupby("ffpe_class")["split"].apply(lambda s: (s == "test").mean())
    report = [
        "# Splits",
        "",
        f"Seed {seed}. Patient-level: every selected slide of a patient (FFPE, paired frozen, "
        "CPTAC normal and tumour) is on the same side. Splits are made before slide QC; "
        "slides excluded in Phase 3 are dropped from whichever side they are on.",
        "",
        f"## Main split: {scfg['test_frac']:.0%} test hold-out + {scfg['n_folds']} folds",
        "",
        f"Patients: {len(patients)} ({(main_split.split == 'test').sum()} test). "
        "Stratified by FFPE class (TCGA tumour class or CPTAC) and presence of paired "
        "frozen slides.",
        "",
        counts_md(main_slides, "part"),
        "## Site-held-out split",
        "",
        f"Whole TCGA tissue source sites in test: {', '.join(test_sites)}. CPTAC has no "
        "site information and is split at patient level, so the Normal class is not "
        "site-held-out (caveat for the report).",
        "",
        "TCGA patients in test per FFPE class: "
        + ", ".join(f"{c} {f:.0%}" for c, f in tcga_frac.items())
        + ".",
        "",
        counts_md(site_slides, "split"),
    ]
    report_path = ROOT / cfg["paths"]["reports"] / "splits_summary.md"
    report_path.write_text("\n".join(report))
    print("\n".join(report[6:]))
    print(f"wrote {out_dir}/{{patients,main,site_holdout}}.csv and {report_path}")


if __name__ == "__main__":
    main()
