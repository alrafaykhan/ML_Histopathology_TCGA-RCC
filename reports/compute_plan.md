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

Filled in Phase 0b.
