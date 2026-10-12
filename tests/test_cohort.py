import pandas as pd

from src.data.cohort import assign_batches, build_cohort, parse_slide_name, select_slides

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
