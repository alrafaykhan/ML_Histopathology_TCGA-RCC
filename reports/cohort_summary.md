# Cohort summary

Generated 2026-10-17 from the GDC API (open-access SVS files, TCGA-KIRC, TCGA-KIRP, TCGA-KICH; sample types 01 and 11).

Retained: 3,240 slides from 929 patients and 58 tissue source sites, 1,506.3 GB. Excluded: 32 files.

## Slides per class and slide type

| label   |   ffpe |   frozen |   total |
|:--------|-------:|---------:|--------:|
| ccRCC   |    505 |     1075 |    1580 |
| pRCC    |    299 |      378 |     677 |
| chRCC   |    109 |      146 |     255 |
| Normal  |     12 |      716 |     728 |
| total   |    925 |     2315 |    3240 |

## Patients per class and slide type

| label   |   ffpe |   frozen |   total |
|:--------|-------:|---------:|--------:|
| ccRCC   |    500 |      522 |     523 |
| pRCC    |    275 |      284 |     290 |
| chRCC   |    109 |       90 |     113 |
| Normal  |     12 |      589 |     601 |
| total   |    884 |      902 |     929 |

## Tracks

Track B: FFPE diagnostic slides (DX) for the three tumor classes, one per patient and class; the 12 FFPE normals are listed but not selected (FFPE normals come from CPTAC-CCRCC, added separately). Track A: frozen sections (TS/BS), restricted to the paired subset: frozen tumour slides of patients whose FFPE slide is selected, plus frozen normals sampled per project. Downloaded after Track B. Selected = `selected` column in `data/cohort.csv`.

| track   | class   |   slides |   patients |   sites |   size_gb |   selected |   selected_gb |
|:--------|:--------|---------:|-----------:|--------:|----------:|-----------:|--------------:|
| A       | ccRCC   |     1075 |        522 |      18 |     228.6 |        150 |          35   |
| A       | pRCC    |      378 |        284 |      33 |      86.7 |        150 |          37.7 |
| A       | chRCC   |      146 |         90 |       6 |     111   |         86 |          54.3 |
| A       | Normal  |      716 |        589 |      25 |     157.6 |        150 |          54   |
| A       | total   |     2315 |        902 |      58 |     583.9 |        536 |         181.1 |
| B       | ccRCC   |      505 |        500 |      17 |     460.8 |        500 |         455.1 |
| B       | pRCC    |      299 |        275 |      31 |     307.8 |        275 |         279.2 |
| B       | chRCC   |      109 |        109 |       6 |     140.9 |        109 |         140.9 |
| B       | Normal  |       12 |         12 |       1 |      12.9 |          0 |           0   |
| B       | total   |      925 |        884 |      54 |     922.4 |        884 |         875.2 |

Download batches: 56 (manifests in `data/manifests/batches/`).

|   batch | track   |   slides |   size_gb |   ccRCC |   pRCC |   chRCC |   Normal |
|--------:|:--------|---------:|----------:|--------:|-------:|--------:|---------:|
|       1 | B       |       18 |      18.4 |       6 |      6 |       6 |        0 |
|       2 | B       |       21 |      19.7 |       7 |      7 |       7 |        0 |
|       3 | B       |       16 |      19.3 |       6 |      5 |       5 |        0 |

Patients with more than one slide in the same track and class (kept; splits are patient-level):

| track   | label   |   patients |
|:--------|:--------|-----------:|
| A       | Normal  |        127 |
| A       | ccRCC   |        496 |
| A       | chRCC   |         56 |
| A       | pRCC    |         84 |
| B       | ccRCC   |          4 |
| B       | pRCC    |         11 |

## Tissue source sites

Slides per site and class (sites with at least 10 slides; remaining sites pooled).

| site             |   ccRCC |   pRCC |   chRCC |   Normal |   total |
|:-----------------|--------:|-------:|--------:|---------:|--------:|
| BP               |     428 |      0 |       0 |      183 |     611 |
| B0               |     317 |      0 |       0 |      147 |     464 |
| CJ               |     221 |      0 |       0 |       82 |     303 |
| A3               |     158 |      0 |       0 |       48 |     206 |
| CZ               |     119 |      0 |       0 |       53 |     172 |
| BQ               |       0 |     89 |       0 |       35 |     124 |
| KL               |       0 |      0 |      71 |       24 |      95 |
| B8               |      76 |      0 |       0 |       16 |      92 |
| AK               |      85 |      0 |       0 |        0 |      85 |
| A4               |       0 |     66 |       0 |       11 |      77 |
| KN               |       0 |      0 |      56 |       19 |      75 |
| B2               |      46 |      0 |       0 |       18 |      64 |
| KO               |       0 |      0 |      42 |       14 |      56 |
| UW               |       0 |      0 |      38 |       12 |      50 |
| CW               |      30 |      0 |       0 |       16 |      46 |
| B9               |       0 |     43 |       0 |        3 |      46 |
| 2Z               |       0 |     46 |       0 |        0 |      46 |
| G7               |       0 |     36 |       0 |        0 |      36 |
| KM               |       0 |      0 |      34 |        0 |      34 |
| DV               |      33 |      0 |       0 |        0 |      33 |
| 5P               |       0 |     32 |       0 |        0 |      32 |
| UZ               |       0 |     32 |       0 |        0 |      32 |
| DW               |       0 |     31 |       0 |        0 |      31 |
| AL               |       0 |     31 |       0 |        0 |      31 |
| P4               |       0 |     20 |       0 |       10 |      30 |
| B3               |       0 |     21 |       0 |        9 |      30 |
| GL               |       0 |     21 |       0 |        7 |      28 |
| SX               |       0 |     27 |       0 |        0 |      27 |
| B4               |      27 |      0 |       0 |        0 |      27 |
| KV               |       0 |     20 |       0 |        2 |      22 |
| Y8               |       0 |     18 |       0 |        4 |      22 |
| HE               |       0 |     21 |       0 |        0 |      21 |
| MH               |       0 |     19 |       0 |        0 |      19 |
| IA               |       0 |     14 |       0 |        4 |      18 |
| B1               |       0 |     15 |       0 |        1 |      16 |
| NP               |       0 |      0 |      14 |        2 |      16 |
| DZ               |       0 |     10 |       0 |        5 |      15 |
| EU               |      12 |      0 |       0 |        0 |      12 |
| IZ               |       0 |      8 |       0 |        2 |      10 |
| J7               |       0 |     10 |       0 |        0 |      10 |
| F9               |       0 |     10 |       0 |        0 |      10 |
| other (17 sites) |      28 |     37 |       0 |        1 |      66 |

Sites contributing slides of a single class: 33 of 58. Site and class are strongly associated, which motivates the site-held-out evaluation.

## Exclusions

| reason                            |   files |
|:----------------------------------|--------:|
| primary diagnosis '...' not '...' |      27 |
| sample type 05 not in scope       |       3 |
| duplicate slide id                |       2 |

Full list: `data/cohort_excluded.csv`.
