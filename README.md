# RCC histopathology model benchmark (TCGA)

Benchmark of deep-learning models for classifying TCGA kidney whole-slide images into four classes:
clear cell RCC (TCGA-KIRC), papillary RCC (TCGA-KIRP), chromophobe RCC (TCGA-KICH) and solid tissue
normal. The main comparison uses frozen patch encoders (ImageNet and pathology foundation models)
with multiple-instance learning (MIL) aggregators. A second family fine-tunes CNN patch classifiers
end to end.

## Data

Slides are from the NCI Genomic Data Commons (projects TCGA-KIRC, TCGA-KIRP, TCGA-KICH). Labels are
derived from the GDC project and sample type (01 primary tumour, 11 solid tissue normal). All
splits are made at patient level. Frozen (TS/BS) and FFPE (DX) slides are analysed as separate
tracks to avoid confounding slide preparation with class.

## Setup

Tested on Ubuntu 24.04 (WSL2) with an NVIDIA GTX 1660 Ti (6 GB).

```bash
mamba env create -f environment.yml
conda activate rcc
python scripts/check_env.py
pytest
```

Gated Hugging Face models need an access token in the `HF_TOKEN` environment variable.

## Pipeline

| Stage | Description |
|---|---|
| 0b | Encoder throughput benchmark |
| 1 | GDC query, cohort table, manifests |
| 2 | Patient-level and site-held-out splits |
| 3 | Tissue segmentation, QC, tiling at 10x |
| 4 | Patch feature extraction per encoder |
| 5 | MIL and patch-classifier training (5-fold CV) |
| 6 | Cross-validation leaderboard |
| 7 | Held-out test evaluation with bootstrap CIs |
| 8 | Attention heatmaps and embedding analysis |

Commands for each stage are added as the stages are implemented.

## Results

Pending.

## Citation

If you use this code, please cite the repository and the original TCGA data and model publications.
