# Compute plan

## Hardware and environment

- OS: "Ubuntu 24.04.5 LTS" (WSL2), kernel 6.18.40.1-microsoft-standard-WSL2
- Python 3.11.17, PyTorch 2.7.1+cu126 (CUDA 12.6, cuDNN 90501), timm 1.0.30, OpenSlide 4.0.1
- CUDA available: True; device: NVIDIA GeForce GTX 1660 Ti; compute capability: (7, 5)
- Free space: WSL `~` 297 GB, Windows D: 369 GB
- `max_raw_gb` = min(100, 297 - 170) = 100 GB

### lscpu
```
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           39 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  10
On-line CPU(s) list:                     0-9
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
CPU family:                              6
Model:                                   158
Thread(s) per core:                      2
Core(s) per socket:                      5
Socket(s):                               1
Stepping:                                10
BogoMIPS:                                5184.05
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm arch_perfmon rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pdcm pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch pti ssbd ibrs ibpb stibp tpr_shadow ept vpid ept_ad fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves vnmi md_clear flush_l1d arch_capabilities
Virtualization:                          VT-x
Hypervisor vendor:                       Microsoft
Virtualization type:                     full
L1d cache:                               160 KiB (5 instances)
L1i cache:                               160 KiB (5 instances)
L2 cache:                                1.3 MiB (5 instances)
L3 cache:                                12 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0-9
```

### free -h
```
total        used        free      shared  buff/cache   available
Mem:            11Gi       1.4Gi       2.8Gi       4.0Mi       7.8Gi        10Gi
Swap:           16Gi       440Ki        15Gi
```

### df -h
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd        305G   13G  277G   5% /
D:\             477G  134G  344G  28% /mnt/d
```

### nvidia-smi
```
Sun Oct 11 14:24:50 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.72                 Driver Version: 566.14         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1660 Ti     On  |   00000000:01:00.0 Off |                  N/A |
| N/A   52C    P8              6W /   80W |       0MiB /   6144MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## Encoder throughput

Benchmark: 2000 patches (224x224, 10x, JPEG q90) from 3 smoke-test slides (frozen KIRC, FFPE KIRP, frozen KICH), `scripts/benchmark_encoders.py`, Rafay. Throughput includes host-to-device copy and normalisation; JPEG decoding is measured separately below.

Selected slides (`data/cohort.csv`): 1847; FFPE 4-class task (Tracks B + C) 1311, paired frozen subset (Track A) 536. Counts are before QC, so estimates are upper bounds. Estimate = slides x 1000 patches / throughput.

DataLoader ceiling: JPEG decode 1907 patches/s per core, about 7627 patches/s with 4 workers. Encoders faster than this are input-bound; their estimate uses the ceiling.

Precision: float16 autocast and float32 are both timed; the faster one is used, and float16 only if its embeddings agree with float32 (mean cosine >= 0.999 on 200 patches, no NaN/Inf). The GTX 1660 Ti (TU116) has no tensor cores, so float16 is not automatically faster.

| # | Encoder | Tier | Params (M) | Dim | fp16 cos | fp32/s | fp16/s | Used | Batch | Peak VRAM (GB) | Max temp (C) | FFPE B+C (h) | All selected (h) | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | resnet18 | 1a | 11.2 | 512 | 0.999999 | 1163 | 252 | fp32 | 256 | 1.85 | 67 | 0.3 | 0.4 | ok |
| 2 | efficientnet_b0 | 1a | 4.0 | 1280 | 0.999981 | 927 | 529 | fp32 | 256 | 3.07 | 72 | 0.4 | 0.6 | ok |
| 3 | convnext_tiny | 1a | 27.8 | 768 | 0.999997 | 219 | 62 | fp32 | 256 | 3.45 | 79 | 1.7 | 2.3 | ok |
| 4 | convnext_base | 1a | 87.6 | 1024 | nan | 91 |  | fp32 | 256 | 4.72 | 87 | 4.0 | 5.7 | ok |
| 5 | swin_t | 1a | 27.5 | 768 | 0.999999 | 96 | 75 | fp32 | 256 | 3.89 | 88 | 3.8 | 5.3 | ok |
| 6 | vit_b16 | 1a | 85.8 | 768 | 0.999999 | 64 | 35 | fp32 | 256 | 2.24 | 89 | 5.7 | 8.1 | ok |
| 7 | kaiko_vits16 | 1b | 21.7 | 384 | 0.999999 | 254 | 64 | fp32 | 256 | 1.14 | 81 | 1.4 | 2.0 | ok |
| 8 | distillpath_ks16_virchow2 | 1b | 21.7 | 384 | 0.999999 | 251 | 131 | fp32 | 256 | 1.14 | 82 | 1.5 | 2.0 | ok |
| 9 | ctranspath | 1b | 27.5 | 768 | 1.0 | 39 | 78 | fp16 | 256 | 3.25 | 86 | 4.7 | 6.6 | ok |
| 10 | phikon | 1b | 85.8 | 768 | 0.999999 | 68 | 27 | fp32 | 256 | 2.24 | 89 | 5.3 | 7.5 | ok |
| 11 | conch | 1b | 395.2 | 512 | 0.999999 | 66 | 37 | fp32 | 256 | 3.4 | 87 | 5.6 | 7.8 | ok |
| 12 | h0_mini | 1b | | | | | | | | | | | | failed: RuntimeError: weights could not be downloaded |
| 13 | resnet50 | 1b | 8.5 | 1024 | 1.0 | 416 | 211 | fp32 | 256 | 2.71 | 89 | 0.9 | 1.2 | ok |
| 14 | uni | 2 | 303.4 | 1024 | 0.999999 | 38 | 11 | fp32 | 256 | 3.63 | 90 | 9.6 | 13.5 | ok |
| 15 | swin_b | 2 | 86.7 | 1024 | 0.999997 | 50 | 20 | fp32 | 256 | 5.31 | 88 | 7.2 | 10.2 | ok |
| 16 | virchow2 | 3 | 631.2 | 2560 | 0.999995 | 9 |  | fp32 | 32 | 2.99 | 87 | 41.4 | 58.3 | ok |
| 17 | uni2_h | 3 | 681.4 | 1536 | 1.0 | 8 |  | fp32 | 64 | 4.04 | 88 | 43.9 | 61.8 | ok |
| 18 | h_optimus_1 | 3 | | | | | | | | | | | | failed: RuntimeError: weights could not be downloaded |
| 19 | prov_gigapath | 3 | 1135.0 | 1536 | 0.999998 | 7 |  | fp32 | 8 | 4.41 | 86 | 51.3 | 72.3 | ok |
| 20 | midnight12k | 3 | 1136.5 | 3072 | 0.999996 | 5 |  | fp32 | 16 | 4.54 | 86 | 75.9 | 106.9 | ok |

Total GPU time, benchmarked encoders, all slides: 373 h (15.5 days).

Not benchmarked (see Status): h0_mini, h_optimus_1.

### Run order

Tier order as specified (1a, 1b, 2, 3; within a tier by `order` in `configs/encoders/*.yaml`); FFPE slides (Tracks B, C) before the paired frozen subset (Track A), following the download order.

### Encoders over 3.0 days

- prov_gigapath: 3.0 days at 1000 patches/slide; a cap of 996 patches/slide fits in 3.0 days.
- midnight12k: 4.5 days at 1000 patches/slide; a cap of 673 patches/slide fits in 3.0 days.
