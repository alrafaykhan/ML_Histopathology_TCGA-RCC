"""NCI Imaging Data Commons (IDC) access for CPTAC slide-microscopy (DICOM-WSI) series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from idc_index import IDCClient

TISSUE_LABELS = {"Normal": "Normal", "Neoplasm, Primary": "tumor"}


def _client() -> IDCClient:
    client = IDCClient()
    client.fetch_index("sm_index")
    return client


def query_sm_series(collection: str) -> pd.DataFrame:
    """Return one row per slide (DICOM series) in an IDC collection.

    Columns: series_uid, slide_id (container identifier), patient_id, tissue
    ("Normal" or "tumor"), size_mb, pixel_spacing_mm, embedding, fixative.
    Fixation is recorded only where the DICOM specimen metadata provides it.
    """
    client = _client()
    df = client.sql_query(f"""
        SELECT s.SeriesInstanceUID AS series_uid,
               s.ContainerIdentifier AS slide_id,
               i.PatientID AS patient_id,
               CAST(s.primaryAnatomicStructureModifier_CodeMeaning AS VARCHAR) AS tissue_raw,
               i.series_size_MB AS size_mb,
               s.min_PixelSpacing_2sf AS pixel_spacing_mm,
               CAST(s.embeddingMedium_CodeMeaning AS VARCHAR) AS embedding,
               CAST(s.tissueFixative_CodeMeaning AS VARCHAR) AS fixative
        FROM sm_index s JOIN index i USING (SeriesInstanceUID)
        WHERE i.collection_id = '{collection}'
        ORDER BY slide_id
        """)
    df["tissue"] = df.pop("tissue_raw").map(TISSUE_LABELS)
    return df


def download_series(series_uids: list[str], out_dir: str | Path) -> None:
    """Download DICOM series into ``out_dir/<PatientID>/.../<SeriesInstanceUID>/``.

    Uses the s5cmd transfer bundled with idc-index; completed files are skipped
    on re-run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _client().download_from_selection(
        seriesInstanceUID=list(series_uids),
        downloadDir=str(out_dir),
        dirTemplate="%PatientID/%SeriesInstanceUID",
        quiet=False,
    )
