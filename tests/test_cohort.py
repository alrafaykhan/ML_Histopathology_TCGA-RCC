import pandas as pd

from src.data.cohort import (
    assign_batches,
    build_cohort,
    cptac_cohort,
    parse_slide_name,
    select_frozen_subset,
    select_slides,
)

LABELS = {"TCGA-KIRC": "ccRCC", "TCGA-KIRP": "pRCC", "TCGA-KICH": "chRCC", "normal": "Normal"}
UUID = "ebb876a2-ae65-4a58-90e7-9da5e5a71426"


def test_parse_frozen_and_ffpe():
    frozen = parse_slide_name(f"TCGA-A3-3329-01A-01-BS1.{UUID}.svs")
    assert frozen == {
        "slide_id": "TCGA-A3-3329-01A-01-BS1",
        "patient_id": "TCGA-A3-3329",
        "site": "A3",
        "sample_code": "01",
        "slide_section": "BS1",
        "slide_type": "frozen",
    }
    ffpe = parse_slide_name(f"TCGA-A4-7583-11A-01-DX1.{UUID}.svs")
    assert ffpe["slide_type"] == "ffpe" and ffpe["sample_code"] == "11"
    assert parse_slide_name("not_a_tcga_slide.svs") is None


def _file(name, project, sample_type):
    return {
        "file_id": name,
        "file_name": f"{name}.{UUID}.svs",
        "file_size": 1,
        "md5": "x",
        "project": project,
        "sample_type": sample_type,
        "sample_id": name[:16],
        "patient_id": name[:12],
    }


def test_labels_and_exclusions():
    files = pd.DataFrame(
        [
            _file("TCGA-AA-0001-01A-01-TS1", "TCGA-KIRC", "Primary Tumor"),
            _file("TCGA-AA-0001-11A-01-TS1", "TCGA-KIRC", "Solid Tissue Normal"),
            _file("TCGA-BB-0002-01A-01-DX1", "TCGA-KICH", "Primary Tumor"),
            _file("TCGA-CC-0003-01A-01-DX1", "TCGA-KIRC", "Primary Tumor"),
            _file("TCGA-CC-0003-11A-01-BS1", "TCGA-KIRC", "Solid Tissue Normal"),
            _file("TCGA-DD-0004-06A-01-TS1", "TCGA-KIRP", "Metastatic"),
            _file("TCGA-EE-0005-01A-01-TS1", "TCGA-KIRP", "Solid Tissue Normal"),
        ]
    )
    cases = pd.DataFrame(
        {
            "patient_id": ["TCGA-AA-0001", "TCGA-BB-0002", "TCGA-CC-0003", "TCGA-EE-0005"],
            "case_project": ["TCGA-KIRC", "TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP"],
            "primary_diagnosis": [
                "Clear cell adenocarcinoma, NOS",
                "Renal cell carcinoma, chromophobe type",
                "Renal cell carcinoma, NOS",
                "Papillary adenocarcinoma, NOS",
            ],
        }
    )
    cohort, excluded = build_cohort(files, cases, LABELS)
    labels = dict(zip(cohort["slide_id"], cohort["label"], strict=True))
    assert labels == {
        "TCGA-AA-0001-01A-01-TS1": "ccRCC",
        "TCGA-AA-0001-11A-01-TS1": "Normal",
        "TCGA-BB-0002-01A-01-DX1": "chRCC",
        "TCGA-CC-0003-11A-01-BS1": "Normal",
    }
    reasons = dict(zip(excluded["file_id"], excluded["exclusion_reason"], strict=True))
    assert "not 'Clear cell" in reasons["TCGA-CC-0003-01A-01-DX1"]
    assert "not in scope" in reasons["TCGA-DD-0004-06A-01-TS1"]
    assert "disagrees" in reasons["TCGA-EE-0005-01A-01-TS1"]


def test_select_and_batches():
    cohort = pd.DataFrame(
        {
            "slide_id": ["P1-TS1", "P1-BS1", "P1-N-TS1", "P2-DX1", "P3-DX1", "P4-TS1"],
            "patient_id": ["P1", "P1", "P1", "P2", "P3", "P4"],
            "label": ["ccRCC", "ccRCC", "Normal", "pRCC", "Normal", "chRCC"],
            "slide_type": ["frozen", "frozen", "frozen", "ffpe", "ffpe", "frozen"],
            "file_size": [10e9, 10e9, 5e9, 8e9, 1e9, 15e9],
        }
    )
    classes = {"A": ["ccRCC", "pRCC", "chRCC", "Normal"], "B": ["ccRCC", "pRCC", "chRCC"]}
    cohort["selected"] = select_slides(cohort, classes, per_patient=1)
    assert cohort["selected"].tolist() == [False, True, True, True, False, True]

    batch = assign_batches(cohort, batch_gb=20, track_order=["A", "B"], seed=0)
    assert batch[~cohort["selected"]].isna().all()
    frozen_batches = set(batch[cohort["selected"] & (cohort["slide_type"] == "frozen")])
    assert batch[3] > max(frozen_batches)
    sizes = cohort[cohort["selected"]].groupby(batch)["file_size"].sum()
    assert (sizes[sizes.index.isin(frozen_batches)] <= 20e9).all()


def test_frozen_subset_is_paired_and_prefers_downloaded():
    rows = []
    for i in range(6):
        p = f"P{i}"
        rows.append((f"{p}-DX1", p, "ccRCC", "B", "TCGA-KIRC"))
        rows.append((f"{p}-TS1", p, "ccRCC", "A", "TCGA-KIRC"))
    rows.append(("Q0-TS1", "Q0", "ccRCC", "A", "TCGA-KIRC"))  # frozen tumour without FFPE pair
    for i in range(4):
        rows.append((f"N{i}-11-TS1", f"N{i}", "Normal", "A", "TCGA-KIRP"))
    rows.append(("P5-11-TS1", "P5", "Normal", "A", "TCGA-KIRC"))
    cohort = pd.DataFrame(rows, columns=["slide_id", "patient_id", "label", "track", "project"])
    cohort["selected"] = True

    sel = select_frozen_subset(
        cohort, {"ccRCC": 3}, normal_per_project=2, seed=0, priority={"P4-TS1", "N3-11-TS1"}
    )
    chosen = set(cohort.loc[sel, "slide_id"])
    frozen_tumours = {s for s in chosen if s.endswith("-TS1") and "-11-" not in s}
    assert len(frozen_tumours) == 3 and "Q0-TS1" not in chosen and "P4-TS1" in chosen
    assert all(f"{s[:2]}-DX1" in chosen for s in frozen_tumours)
    assert sum(s.startswith("N") for s in chosen) == 2 and "N3-11-TS1" in chosen
    assert "P5-11-TS1" in chosen
    assert all(cohort.loc[cohort["track"] == "B", "slide_id"].isin(chosen))


def test_cptac_cohort_one_slide_per_patient_prefers_priority():
    series = pd.DataFrame(
        {
            "series_uid": ["u1", "u2", "u3", "u4"],
            "slide_id": ["C3L-1-21", "C3L-1-22", "C3L-1-26", "C3N-2-21"],
            "patient_id": ["C3L-1", "C3L-1", "C3L-1", "C3N-2"],
            "tissue": ["tumor", "tumor", "Normal", "tumor"],
            "size_mb": [100.0, 120.0, 50.0, 80.0],
        }
    )
    labels = {"tumor": "ccRCC", "Normal": "Normal"}
    df = cptac_cohort(series, "CPTAC-CCRCC", labels, priority={"u2"}).set_index("file_id")
    assert df.loc[["u2", "u3", "u4"], "selected"].all() and not df.at["u1", "selected"]
    assert (df["track"] == "C").all() and (df["slide_type"] == "ffpe").all()
    assert df.at["u3", "label"] == "Normal" and df.at["u1", "file_size"] == 100 * 2**20
