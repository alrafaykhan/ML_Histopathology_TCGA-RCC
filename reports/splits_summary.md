# Splits

Seed 42. Patient-level: every selected slide of a patient (FFPE, paired frozen, CPTAC normal and tumour) is on the same side. Splits are made before slide QC; slides excluded in Phase 3 are dropped from whichever side they are on.

## Main split: 20% test hold-out + 5 folds

Patients: 1106 (222 test). Stratified by FFPE class (TCGA tumour class or CPTAC) and presence of paired frozen slides.

FFPE (Tracks B+C):

| part   |   ccRCC |   pRCC |   chRCC |   Normal |   total |
|:-------|--------:|-------:|--------:|---------:|--------:|
| fold 0 |     117 |     43 |      17 |       33 |     210 |
| fold 1 |     116 |     44 |      17 |       33 |     210 |
| fold 2 |     115 |     45 |      17 |       33 |     210 |
| fold 3 |     114 |     45 |      18 |       32 |     209 |
| fold 4 |     116 |     42 |      18 |       33 |     209 |
| test   |     144 |     56 |      22 |       41 |     263 |
| total  |     722 |    275 |     109 |      205 |    1311 |

Frozen (Track A):

| part   |   ccRCC |   pRCC |   chRCC |   Normal |   total |
|:-------|--------:|-------:|--------:|---------:|--------:|
| fold 0 |      24 |     23 |      13 |       23 |      83 |
| fold 1 |      25 |     23 |      14 |       24 |      86 |
| fold 2 |      24 |     25 |      14 |       25 |      88 |
| fold 3 |      24 |     25 |      14 |       25 |      88 |
| fold 4 |      23 |     23 |      14 |       23 |      83 |
| test   |      30 |     31 |      17 |       30 |     108 |
| total  |     150 |    150 |      86 |      150 |     536 |

## Site-held-out split

Whole TCGA tissue source sites in test: 6D, A4, B8, CJ, DV, EU, F9, KN, MH, MM, NP, T7, V9, Y8. CPTAC has no site information and is split at patient level, so the Normal class is not site-held-out (caveat for the report).

TCGA patients in test per FFPE class: ccRCC 23%, chRCC 24%, pRCC 20%.

FFPE (Tracks B+C):

| split   |   ccRCC |   pRCC |   chRCC |   Normal |   total |
|:--------|--------:|-------:|--------:|---------:|--------:|
| test    |     160 |     55 |      26 |       42 |     283 |
| train   |     562 |    220 |      83 |      163 |    1028 |
| total   |     722 |    275 |     109 |      205 |    1311 |

Frozen (Track A):

| split   |   ccRCC |   pRCC |   chRCC |   Normal |   total |
|:--------|--------:|-------:|--------:|---------:|--------:|
| test    |      31 |     24 |      26 |       35 |     116 |
| train   |     119 |    126 |      60 |      115 |     420 |
| total   |     150 |    150 |      86 |      150 |     536 |
