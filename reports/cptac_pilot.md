# CPTAC-CCRCC fixation pilot

Date: 2026-10-17. Script: `scripts/cptac_pilot.py` (download), `scripts/cptac_pilot_review.py` (contact sheets, written to `qc/cptac_pilot/`). Manifest: `data/manifests/cptac_pilot.csv`.

## Why

TCGA has only 12 FFPE solid-tissue-normal slides (one KICH site), so the FFPE 4-class task (Track B) takes its Normal class from CPTAC-CCRCC, hosted on NCI Imaging Data Commons (IDC) as DICOM-WSI (CC BY 4.0). Neither the IDC index nor the DICOM specimen metadata records fixation, and at least one published study excluded some CPTAC slides as frozen, so a sample was checked visually.

## Sample

Seeded random sample (seed 42), one slide per patient: 20 normal and 4 tumour slides, 5.57 GB, 148 DICOM instances. All instances matched the IDC index sizes. Download from the IDC AWS bucket ran at about 1.0-1.5 MB/s (about 3x GDC on this connection).

OpenSlide 4.0.1 reads the series directly: level-0 0.494 um/px (20x), three pyramid levels, so the existing 10x tiler applies unchanged.

## Findings

| Group | Slides | Assessment |
|---|---:|---|
| Tumour | 4 | Typical ccRCC (clear cells, delicate vasculature); FFPE |
| Normal: renal cortex | 13 | Tubules and glomeruli; FFPE |
| Normal: non-neoplastic kidney with chronic changes (sclerosis, tubular atrophy, inflammation) | 4 | FFPE; usable as tumour-adjacent normal, as in TCGA |
| Normal: perirenal fat / vessels only | 2 | Not renal parenchyma; 0 and 36 tissue patches at 10x |
| Normal: very pale stain, no tissue detected | 1 | 0 tissue patches |

- No slide showed frozen-section artefacts (ice-crystal holes, tears, folds, poor nuclear detail). All 24 are consistent with FFPE.
- 3 of 20 normal slides (15%) are unusable; all three fall below the existing minimum of 50 tissue patches per slide, so slide QC removes them without a new rule.
- A teal-and-black bar along the lower slide edge (scanner artefact) occurs on some slides; the blue/green ink mask (`exclude_pen=True`) removes it from the tissue grid.
- Staining intensity varies (e.g. C3N-01261, C3N-01524 darker); one slide (C3N-01081) has out-of-focus regions. Colour augmentation and the blur QC cover these.

## Decision

Use CPTAC-CCRCC for Track B: all normal patients (one slide each) and one ccRCC tumour slide per patient. Expect about 85% of normal slides to pass QC. Report that CPTAC normals come from ccRCC nephrectomies only and include chronically damaged non-tumour kidney. Keep checking for frozen slides during QC in the full collection.
