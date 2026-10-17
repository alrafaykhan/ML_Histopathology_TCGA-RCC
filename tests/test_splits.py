from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import holdout_and_folds, patient_table, site_holdout

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"


def _toy_cohort(n_per_class: int = 30) -> pd.DataFrame:
    rows = []
    for c, label in enumerate(["ccRCC", "pRCC", "chRCC"]):
        for i in range(n_per_class):
            pid, site = f"TCGA-S{c}{i % 4}-{c}{i:03d}", f"S{c}{i % 4}"
            rows.append((f"{pid}-DX1", pid, "B", label, "ffpe", site))
            if i % 2 == 0:
                rows.append((f"{pid}-TS1", pid, "A", label, "frozen", site))
            if i % 5 == 0:
                rows.append((f"{pid}-11-TS1", pid, "A", "Normal", "frozen", site))
    for i in range(40):
        pid = f"C3L-{i:05d}"
        rows.append((f"{pid}-26", pid, "C", "Normal", "ffpe", "CPTAC-CCRCC"))
        rows.append((f"{pid}-21", pid, "C", "ccRCC", "ffpe", "CPTAC-CCRCC"))
    df = pd.DataFrame(
        rows, columns=["slide_id", "patient_id", "track", "label", "slide_type", "site"]
    )
    return df.assign(selected=True)


def test_toy_splits_patient_and_site_disjoint():
    cohort = _toy_cohort()
    patients = patient_table(cohort)
    assert patients["patient_id"].is_unique
    main = holdout_and_folds(patients, test_frac=0.2, n_folds=5, seed=0)
    assert set(main.loc[main.split == "train", "fold"]) == set(range(5))
    assert (main.loc[main.split == "test", "fold"] == -1).all()

    site = site_holdout(patients, test_frac=0.2, seed=0).merge(patients, on="patient_id")
    tcga = site[site.source == "tcga"]
    train_sites = set(tcga.loc[tcga.split == "train", "site"])
    test_sites = set(tcga.loc[tcga.split == "test", "site"])
    assert test_sites and not train_sites & test_sites
    assert set(tcga.loc[tcga.split == "test", "ffpe_class"]) == {"ccRCC", "pRCC", "chRCC"}


@pytest.mark.skipif(not (SPLITS / "main.csv").exists(), reason="splits not generated")
def test_saved_splits():
    main = pd.read_csv(SPLITS / "main.csv")
    site = pd.read_csv(SPLITS / "site_holdout.csv")
    cohort = pd.read_csv(ROOT / "data" / "cohort.csv")
    assert set(main["slide_id"]) == set(cohort.loc[cohort["selected"], "slide_id"])
    assert set(site["slide_id"]) == set(main["slide_id"])

    # each patient on exactly one side / in exactly one fold
    assert (main.groupby("patient_id")[["split", "fold"]].nunique() == 1).all().all()
    assert (site.groupby("patient_id")["split"].nunique() == 1).all()
    test_patients = set(main.loc[main.split == "test", "patient_id"])
    assert not test_patients & set(main.loc[main.split == "train", "patient_id"])

    # every class present in test and in every fold, for FFPE and frozen
    for part in [main[main.track != "A"], main[main.track == "A"]]:
        assert part.groupby(["split", "fold"])["label"].nunique().eq(4).all()

    # whole TCGA sites held out; CPTAC split by patient
    tcga = site[site["site"] != "CPTAC-CCRCC"]
    assert not set(tcga.loc[tcga.split == "test", "site"]) & set(
        tcga.loc[tcga.split == "train", "site"]
    )
    test_ffpe = tcga[(tcga.split == "test") & (tcga.track == "B")]
    assert set(test_ffpe["label"]) == {"ccRCC", "pRCC", "chRCC"}
