"""Phase 1: query GDC, build data/cohort.csv, manifests and reports/cohort_summary.md."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.cohort import (  # noqa: E402
    assign_batches,
    build_cohort,
    cptac_cohort,
    gdc_manifest,
    query_primary_diagnoses,
    select_frozen_subset,
    select_slides,
    track_of,
)
from src.data.gdc import query_slide_files  # noqa: E402
from src.data.idc import query_sm_series  # noqa: E402
from src.utils import load_config  # noqa: E402

CLASSES = ["ccRCC", "pRCC", "chRCC", "Normal"]


def gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:,.1f}"


def crosstab_md(df: pd.DataFrame, values: str | None = None) -> str:
    """Class x slide-type table of slide counts (or unique ``values``) with totals."""
    if values is None:
        tab = pd.crosstab(df["label"], df["slide_type"], margins=True, margins_name="total")
    else:
        tab = pd.crosstab(
            df["label"],
            df["slide_type"],
            values=df[values],
            aggfunc="nunique",
            margins=True,
            margins_name="total",
        )
    tab = tab.reindex(CLASSES + ["total"]).fillna(0).astype(int)
    return tab.to_markdown()


def summary_md(cohort: pd.DataFrame, excluded: pd.DataFrame) -> str:
    out = [
        "# Cohort summary",
        "",
        f"Generated {date.today().isoformat()} from the GDC API (open-access SVS files, "
        "TCGA-KIRC, TCGA-KIRP, TCGA-KICH; sample types 01 and 11).",
        "",
        f"Retained: {len(cohort):,} slides from {cohort['patient_id'].nunique():,} patients "
        f"and {cohort['site'].nunique()} tissue source sites, {gb(cohort['file_size'].sum())} GB. "
        f"Excluded: {len(excluded):,} files.",
        "",
        "## Slides per class and slide type",
        "",
        crosstab_md(cohort),
        "",
        "## Patients per class and slide type",
        "",
        crosstab_md(cohort, values="patient_id"),
        "",
        "## Tracks",
        "",
        "Track B: FFPE diagnostic slides (DX) for the three tumor classes, one per patient "
        "and class; the 12 FFPE normals are listed but not selected (FFPE normals come from "
        "CPTAC-CCRCC, added separately). Track A: frozen sections (TS/BS), restricted to the "
        "paired subset: frozen tumour slides of patients whose FFPE slide is selected, plus "
        "frozen normals sampled per project. Downloaded after Track B. Selected = `selected` "
        "column in `data/cohort.csv`.",
        "",
    ]
    rows = []
    for track, g in cohort.groupby("track"):
        for label in CLASSES:
            s = g[g["label"] == label]
            rows.append(
                {
                    "track": track,
                    "class": label,
                    "slides": len(s),
                    "patients": s["patient_id"].nunique(),
                    "sites": s["site"].nunique(),
                    "size_gb": round(s["file_size"].sum() / 1e9, 1),
                    "selected": int(s["selected"].sum()),
                    "selected_gb": round(s.loc[s["selected"], "file_size"].sum() / 1e9, 1),
                }
            )
        rows.append(
            {
                "track": track,
                "class": "total",
                "slides": len(g),
                "patients": g["patient_id"].nunique(),
                "sites": g["site"].nunique(),
                "size_gb": round(g["file_size"].sum() / 1e9, 1),
                "selected": int(g["selected"].sum()),
                "selected_gb": round(g.loc[g["selected"], "file_size"].sum() / 1e9, 1),
            }
        )
    out += [pd.DataFrame(rows).to_markdown(index=False), ""]

    sel = cohort[cohort["selected"]]
    batches = sel.groupby("batch").agg(
        track=("track", "first"),
        slides=("slide_id", "size"),
        size_gb=("file_size", lambda x: round(x.sum() / 1e9, 1)),
        **{c: ("label", lambda x, c=c: int((x == c).sum())) for c in CLASSES},
    )
    out += [
        f"Download batches: {len(batches)} (manifests in `data/manifests/batches/`).",
        "",
        batches.head(3).to_markdown(),
        "",
    ]

    per_patient = cohort.groupby(["track", "label", "patient_id"]).size()
    multi = per_patient[per_patient > 1].groupby(level=[0, 1]).size()
    out += [
        "Patients with more than one slide in the same track and class "
        "(kept; splits are patient-level):",
        "",
        multi.rename("patients").reset_index().to_markdown(index=False) if len(multi) else "none",
        "",
        "## Tissue source sites",
        "",
        "Slides per site and class (sites with at least 10 slides; remaining sites pooled).",
        "",
    ]
    site_tab = pd.crosstab(cohort["site"], cohort["label"]).reindex(columns=CLASSES, fill_value=0)
    site_tab["total"] = site_tab.sum(axis=1)
    site_tab = site_tab.sort_values("total", ascending=False)
    big = site_tab[site_tab["total"] >= 10]
    small = site_tab[site_tab["total"] < 10]
    if len(small):
        big.loc[f"other ({len(small)} sites)"] = small.sum()
    out += [big.to_markdown(), ""]

    classes_per_site = (site_tab[CLASSES] > 0).sum(axis=1)
    out += [
        f"Sites contributing slides of a single class: {(classes_per_site == 1).sum()} of "
        f"{len(site_tab)}. Site and class are strongly associated, which motivates the "
        "site-held-out evaluation.",
        "",
        "## Exclusions",
        "",
        (
            excluded["exclusion_reason"]
            .str.replace(r"'.*?'", "'...'", regex=True)
            .value_counts()
            .rename_axis("reason")
            .rename("files")
            .reset_index()
            .to_markdown(index=False)
            if len(excluded)
            else "none"
        ),
        "",
        "Full list: `data/cohort_excluded.csv`.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    projects = cfg["gdc"]["projects"]

    files = query_slide_files(projects)
    cases = query_primary_diagnoses(projects)
    print(f"GDC: {len(files)} open SVS files, {len(cases)} cases")
    cohort, excluded = build_cohort(files, cases, cfg["gdc"]["labels"])
    tracks_cfg = cfg["tracks"]
    cohort["track"] = track_of(cohort)
    cohort["selected"] = select_slides(
        cohort, tracks_cfg["classes"], tracks_cfg["slides_per_patient_class"]
    )
    subset = tracks_cfg["frozen_subset"]
    priority_path = ROOT / subset["priority"]
    priority = set(priority_path.read_text().split()) if priority_path.exists() else set()
    cohort["selected"] = select_frozen_subset(
        cohort,
        subset["tumor_per_class"],
        subset["normal_per_project"],
        seed=cfg["seed"],
        priority=priority,
    )
    cohort["source"] = "gdc"

    cp = cfg["cptac"]
    series = query_sm_series(cp["collection"])
    pilot_path = ROOT / cp["priority"]
    pilot = set(pd.read_csv(pilot_path)["series_uid"]) if pilot_path.exists() else set()
    cptac = cptac_cohort(series, cp["project"], cp["labels"], priority=pilot)
    print(f"IDC {cp['collection']}: {len(cptac)} slides, {cptac['patient_id'].nunique()} patients")
    cohort = pd.concat([cohort, cptac], ignore_index=True)
    cohort["batch"] = assign_batches(
        cohort, cfg["download"]["batch_gb"], tracks_cfg["order"], seed=cfg["seed"]
    )

    cohort_path = ROOT / cfg["paths"]["cohort"]
    cohort.to_csv(cohort_path, index=False)
    excluded.to_csv(cohort_path.with_name("cohort_excluded.csv"), index=False)

    manifest_dir = ROOT / cfg["paths"]["manifests"]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for track in tracks_cfg["order"]:
        if track == "C":
            continue
        gdc_manifest(cohort[cohort["selected"] & (cohort["track"] == track)]).to_csv(
            manifest_dir / f"gdc_manifest_track{track}.txt", sep="\t", index=False
        )
    batch_dir = manifest_dir / "batches"
    batch_dir.mkdir(exist_ok=True)
    for b, g in cohort[cohort["selected"]].groupby("batch"):
        if (g["source"] == "idc").all():
            g[["slide_id", "patient_id", "label", "file_id", "file_size"]].rename(
                columns={"file_id": "series_uid"}
            ).to_csv(batch_dir / f"batch_{b:03d}_idc.csv", index=False)
        else:
            gdc_manifest(g).to_csv(batch_dir / f"batch_{b:03d}.txt", sep="\t", index=False)

    report = ROOT / cfg["paths"]["reports"] / "cohort_summary.md"
    report.write_text(summary_md(cohort, excluded))
    print(f"retained {len(cohort)} slides, excluded {len(excluded)}")
    print(crosstab_md(cohort))
    print(f"wrote {cohort_path}, manifests and {report}")


if __name__ == "__main__":
    main()
