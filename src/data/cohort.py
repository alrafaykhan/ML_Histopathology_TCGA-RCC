"""Build the slide-level cohort table from GDC file and case metadata."""

from __future__ import annotations

import re

import pandas as pd
import requests

from src.data.gdc import GDC_API

EXPECTED_PRIMARY_DX = {
    "TCGA-KIRC": "Clear cell adenocarcinoma, NOS",
    "TCGA-KIRP": "Papillary adenocarcinoma, NOS",
    "TCGA-KICH": "Renal cell carcinoma, chromophobe type",
}
GDC_SAMPLE_TYPE = {"01": "Primary Tumor", "11": "Solid Tissue Normal"}

# TCGA-XX-XXXX-SSV-PP-TTN.uuid.svs, e.g. TCGA-A3-3329-01A-01-BS1.<uuid>.svs
_BARCODE = re.compile(
    r"^(?P<patient>TCGA-(?P<site>[A-Z0-9]{2})-[A-Z0-9]{4})-(?P<sample>\d{2})[A-Z]"
    r"-\d{2}-(?P<slide>(?P<kind>DX|TS|BS|MS)[0-9A-Z]*)\."
)


def parse_slide_name(file_name: str) -> dict | None:
    """Parse patient, site, sample code and slide section from a TCGA slide file name."""
    m = _BARCODE.match(file_name)
    if m is None:
        return None
    kind = m["kind"]
    return {
        "slide_id": file_name.split(".")[0],
        "patient_id": m["patient"],
        "site": m["site"],
        "sample_code": m["sample"],
        "slide_section": m["slide"],
        "slide_type": "ffpe" if kind == "DX" else "frozen",
    }


def query_primary_diagnoses(projects: list[str]) -> pd.DataFrame:
    """Return one row per case with its project and primary-disease diagnosis."""
    payload = {
        "filters": {"op": "in", "content": {"field": "project.project_id", "value": projects}},
        "fields": "submitter_id,project.project_id,diagnoses.primary_diagnosis,"
        "diagnoses.diagnosis_is_primary_disease",
        "size": 10000,
    }
    resp = requests.post(f"{GDC_API}/cases", json=payload, timeout=120)
    resp.raise_for_status()
    rows = []
    for hit in resp.json()["data"]["hits"]:
        primary = sorted(
            {
                d["primary_diagnosis"]
                for d in hit.get("diagnoses", [])
                if d.get("diagnosis_is_primary_disease") and d.get("primary_diagnosis")
            }
        )
        rows.append(
            {
                "patient_id": hit["submitter_id"],
                "case_project": hit["project"]["project_id"],
                "primary_diagnosis": "; ".join(primary),
            }
        )
    return pd.DataFrame(rows)


def build_cohort(
    files: pd.DataFrame, cases: pd.DataFrame, labels: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label slides and apply sanity checks.

    ``files`` comes from ``query_slide_files``; ``cases`` from ``query_primary_diagnoses``;
    ``labels`` maps project ids and ``"normal"`` to class names. Returns
    ``(cohort, excluded)``, where ``excluded`` carries an ``exclusion_reason`` column.
    """
    parsed = files["file_name"].map(parse_slide_name)
    df = files.rename(columns={"patient_id": "gdc_patient_id"})
    df["exclusion_reason"] = ""
    bad_name = parsed.isna()
    df.loc[bad_name, "exclusion_reason"] = "unparseable file name"
    df = df.join(pd.DataFrame(parsed[~bad_name].tolist(), index=parsed[~bad_name].index))
    df = df.merge(cases, on="patient_id", how="left")

    def reason(row) -> str:
        if row.exclusion_reason:
            return row.exclusion_reason
        if row.patient_id != row.gdc_patient_id:
            return f"barcode patient differs from GDC case {row.gdc_patient_id}"
        if row.sample_code not in GDC_SAMPLE_TYPE:
            return f"sample type {row.sample_code} not in scope"
        if row.sample_type != GDC_SAMPLE_TYPE[row.sample_code]:
            return f"barcode sample {row.sample_code} disagrees with GDC '{row.sample_type}'"
        if pd.isna(row.case_project):
            return "case missing from GDC clinical records"
        if row.case_project != row.project:
            return f"file project {row.project} differs from case project {row.case_project}"
        if row.sample_code == "01":
            expected = EXPECTED_PRIMARY_DX[row.project]
            if not row.primary_diagnosis:
                return "no primary diagnosis recorded"
            if row.primary_diagnosis != expected:
                return f"primary diagnosis '{row.primary_diagnosis}' not '{expected}'"
        return ""

    df["exclusion_reason"] = df.apply(reason, axis=1)

    dup = df["slide_id"].duplicated(keep=False) & df["slide_id"].notna()
    df.loc[dup & (df["exclusion_reason"] == ""), "exclusion_reason"] = "duplicate slide id"

    multi = df[df["exclusion_reason"] == ""].groupby("patient_id")["project"].nunique()
    multi = set(multi[multi > 1].index)
    df.loc[df["patient_id"].isin(multi), "exclusion_reason"] = "patient in multiple projects"

    df["label"] = [
        labels["normal"] if code == "11" else labels.get(proj, "")
        for code, proj in zip(df["sample_code"], df["project"], strict=True)
    ]
    columns = [
        "slide_id",
        "patient_id",
        "project",
        "sample_type",
        "label",
        "slide_type",
        "site",
        "file_size",
        "md5",
        "file_id",
        "file_name",
        "slide_section",
        "primary_diagnosis",
    ]
    keep = df["exclusion_reason"] == ""
    cohort = df.loc[keep, columns].sort_values(["label", "slide_type", "slide_id"])
    excluded = df.loc[~keep, ["file_id", "file_name", "project", "sample_type", "exclusion_reason"]]
    return cohort.reset_index(drop=True), excluded.reset_index(drop=True)


def track_of(cohort: pd.DataFrame) -> pd.Series:
    """Track A = frozen sections (TS/BS); Track B = FFPE diagnostic slides (DX)."""
    return cohort["slide_type"].map({"frozen": "A", "ffpe": "B"})


def select_slides(
    cohort: pd.DataFrame, track_classes: dict[str, list[str]], per_patient: int
) -> pd.Series:
    """Mark slides used for training: in-scope classes per track, ``per_patient`` slides per
    patient and class (lowest slide ids first, so the choice is deterministic)."""
    track = track_of(cohort)
    in_scope = pd.Series(
        [label in track_classes[t] for t, label in zip(track, cohort["label"], strict=True)],
        index=cohort.index,
    )
    rank = (
        cohort.assign(track=track)
        .sort_values("slide_id")
        .groupby(["track", "patient_id", "label"])
        .cumcount()
        .reindex(cohort.index)
    )
    return in_scope & (rank < per_patient)


def select_frozen_subset(
    cohort: pd.DataFrame,
    tumor_per_class: dict[str, int | str],
    normal_per_project: int,
    seed: int,
    priority: set[str] = frozenset(),
) -> pd.Series:
    """Restrict selected frozen (Track A) slides to the paired-robustness subset.

    Tumour slides are kept only for patients whose FFPE slide of the same class is
    selected, so every frozen tumour slide has an FFPE partner; ``tumor_per_class`` gives
    the number of patients per class (``"all"`` keeps every paired patient). Normal slides
    are sampled per source project, preferring patients with a selected FFPE tumour slide.
    Slide ids in ``priority`` (already downloaded) are taken first; the rest are drawn
    with ``seed``. Expects ``track`` and ``selected`` columns; returns the new ``selected``.
    """
    sel = cohort["selected"].copy()
    frozen = cohort["selected"] & (cohort["track"] == "A")
    ffpe_pairs = set(
        zip(
            cohort.loc[cohort["selected"] & (cohort["track"] == "B"), "patient_id"],
            cohort.loc[cohort["selected"] & (cohort["track"] == "B"), "label"],
            strict=True,
        )
    )
    ffpe_patients = {p for p, _ in ffpe_pairs}

    def take(candidates: pd.DataFrame, n: int | str, prefer: pd.Series) -> pd.Index:
        rng_order = candidates.sample(frac=1, random_state=seed)
        key = (~rng_order["slide_id"].isin(priority)).astype(int) * 2 + (
            ~prefer.reindex(rng_order.index)
        ).astype(int)
        ordered = rng_order.assign(_key=key).sort_values("_key", kind="stable")
        return ordered.index if n == "all" else ordered.index[: int(n)]

    keep = []
    for label, n in tumor_per_class.items():
        cand = cohort[frozen & (cohort["label"] == label)]
        paired = [(p, label) in ffpe_pairs for p in cand["patient_id"]]
        cand = cand[paired]
        keep += list(take(cand, n, pd.Series(True, index=cand.index)))
    normals = cohort[frozen & (cohort["label"] == "Normal")]
    for _, cand in normals.groupby("project"):
        keep += list(take(cand, normal_per_project, cand["patient_id"].isin(ffpe_patients)))

    sel[frozen] = False
    sel[keep] = True
    return sel


def assign_batches(
    cohort: pd.DataFrame, batch_gb: float, track_order: list[str], seed: int
) -> pd.Series:
    """Assign selected slides to download batches of about ``batch_gb``.

    Tracks are downloaded in ``track_order``. Within a track, slides are shuffled per class
    and interleaved round-robin across classes so every early batch contains all classes.
    Returns 1-based batch numbers (NA for unselected slides).
    """
    batch = pd.Series(pd.NA, index=cohort.index, dtype="Int64")
    track = track_of(cohort)
    limit = batch_gb * 1e9
    current, filled = 1, 0.0
    for t in track_order:
        sel = cohort[cohort["selected"] & (track == t)]
        queues = [
            list(g.sample(frac=1, random_state=seed).index)
            for _, g in sel.groupby("label", sort=True)
        ]
        order = [
            q[i] for i in range(max(map(len, queues), default=0)) for q in queues if i < len(q)
        ]
        for idx in order:
            size = cohort.at[idx, "file_size"]
            if filled and filled + size > limit:
                current, filled = current + 1, 0.0
            batch[idx] = current
            filled += size
        if filled:
            current, filled = current + 1, 0.0
    return batch


def gdc_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Format rows as a gdc-client manifest."""
    return pd.DataFrame(
        {
            "id": df["file_id"],
            "filename": df["file_name"],
            "md5": df["md5"],
            "size": df["file_size"],
            "state": "released",
        }
    )
