"""Patient-level splits: held-out test set, cross-validation folds and a site-held-out split.

All splits are made over patients, so every slide of a patient (FFPE tumour, paired frozen
tumour, frozen normal, CPTAC normal and tumour) lands on the same side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def patient_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with selected slides: site, source and a stratification key.

    The key combines the patient's FFPE class (TCGA tumour class, or ``CPTAC`` with the
    CPTAC classes present) with whether paired frozen slides exist, so test and folds keep
    the frozen subset and the CPTAC normals in proportion.
    """
    sel = cohort[cohort["selected"]]
    rows = []
    for pid, g in sel.groupby("patient_id", sort=True):
        tracks = set(g["track"])
        if "C" in tracks:
            ffpe = "CPTAC:" + "+".join(sorted(set(g["label"])))
        else:
            ffpe = "+".join(sorted(set(g.loc[g["track"] == "B", "label"]))) or "none"
        frozen = g[g["track"] == "A"]
        suffix = ""
        if (frozen["label"] != "Normal").any():
            suffix += "|frozen"
        if (frozen["label"] == "Normal").any():
            suffix += "|frozen_normal"
        rows.append(
            {
                "patient_id": pid,
                "site": g["site"].iloc[0],
                "source": "cptac" if "C" in tracks else "tcga",
                "ffpe_class": ffpe,
                "strata": ffpe + suffix,
            }
        )
    return pd.DataFrame(rows)


def collapse_rare(strata: pd.Series, min_count: int) -> pd.Series:
    """Drop the ``|...`` suffixes of strata with fewer than ``min_count`` patients."""
    strata = strata.copy()
    for _ in range(3):
        counts = strata.map(strata.value_counts())
        rare = counts < min_count
        if not rare.any():
            break
        strata[rare] = strata[rare].str.rsplit("|", n=1).str[0]
    return strata


def holdout_and_folds(
    patients: pd.DataFrame, test_frac: float, n_folds: int, seed: int
) -> pd.DataFrame:
    """Stratified patient-level test hold-out, then stratified folds on the rest.

    Returns ``patient_id``, ``split`` (``train``/``test``) and ``fold`` (0..n_folds-1 for
    train patients, -1 for test).
    """
    strata = collapse_rare(patients["strata"], min_count=2 * n_folds)
    train_idx, test_idx = train_test_split(
        patients.index, test_size=test_frac, stratify=strata, random_state=seed
    )
    out = patients[["patient_id"]].assign(split="train", fold=-1)
    out.loc[test_idx, "split"] = "test"
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_strata = collapse_rare(strata.loc[train_idx], min_count=n_folds)
    for k, (_, val) in enumerate(skf.split(train_idx, train_strata)):
        out.loc[train_idx[val], "fold"] = k
    return out


def site_holdout(
    patients: pd.DataFrame, test_frac: float, seed: int, max_frac: float = 0.35
) -> pd.DataFrame:
    """Hold out whole TCGA tissue source sites, aiming for ``test_frac`` of each class.

    Sites are visited in a seeded random order; a site is added to the test side for its
    majority FFPE class while that class is below ``test_frac`` and adding the site keeps
    every class at or below ``max_frac``. If no site fits for a class, its smallest site
    is used. CPTAC records no sites, so CPTAC patients are split at patient level
    (stratified) and are not part of the site-held-out comparison.
    Returns ``patient_id`` and ``split``.
    """
    rng = np.random.default_rng(seed)
    tcga = patients[patients["source"] == "tcga"]
    by_site = pd.crosstab(tcga["site"], tcga["ffpe_class"])
    totals = by_site.sum()
    order = list(rng.permutation(by_site.index))
    test_sites: set[str] = set()

    def frac(sites: set[str]) -> pd.Series:
        return by_site.loc[sorted(sites)].sum() / totals if sites else totals * 0.0

    for cls in totals.sort_values().index:
        candidates = [s for s in order if by_site.at[s, cls] > 0 and s not in test_sites]
        for site in candidates:
            if frac(test_sites)[cls] >= test_frac:
                break
            if (frac(test_sites | {site}) <= max_frac).all():
                test_sites.add(site)
        if frac(test_sites)[cls] == 0 and candidates:
            test_sites.add(min(candidates, key=lambda s: by_site.loc[s].sum()))

    out = patients[["patient_id"]].assign(split="train")
    out.loc[patients["site"].isin(test_sites) & (patients["source"] == "tcga"), "split"] = "test"
    cptac = patients[patients["source"] == "cptac"]
    if len(cptac):
        _, test_idx = train_test_split(
            cptac.index, test_size=test_frac, stratify=cptac["strata"], random_state=seed
        )
        out.loc[test_idx, "split"] = "test"
    return out


def slide_splits(cohort: pd.DataFrame, patient_splits: pd.DataFrame) -> pd.DataFrame:
    """Expand patient-level splits to the selected slides."""
    sel = cohort.loc[
        cohort["selected"], ["slide_id", "patient_id", "track", "label", "slide_type", "site"]
    ]
    return sel.merge(patient_splits, on="patient_id", how="left", validate="many_to_one")
