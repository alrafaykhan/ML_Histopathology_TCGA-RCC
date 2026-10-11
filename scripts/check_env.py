"""Verify the CUDA/OpenSlide environment and record hardware specs in reports/compute_plan.md."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "reports" / "compute_plan.md"))
    args = parser.parse_args()

    import openslide
    import timm
    import torch

    cuda = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "n/a"
    cap = torch.cuda.get_device_capability(0) if cuda else None
    print(cuda, name, cap)

    wsl_free = free_gb(str(Path.home()))
    d_free = free_gb("/mnt/d") if Path("/mnt/d").exists() else float("nan")
    max_raw_gb = max(30, min(100, int(wsl_free - 170)))
    feasible = wsl_free - 170 >= 30

    is_wsl = "microsoft" in platform.release().lower()
    lines = [
        "# Compute plan",
        "",
        "## Hardware and environment",
        "",
        f"- OS: {run('grep PRETTY_NAME /etc/os-release | cut -d= -f2')} "
        f"({'WSL2' if is_wsl else 'native Linux'}), kernel {platform.release()}",
        f"- Python {platform.python_version()}, PyTorch {torch.__version__} "
        f"(CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}), "
        f"timm {timm.__version__}, OpenSlide {openslide.__library_version__}",
        f"- CUDA available: {cuda}; device: {name}; compute capability: {cap}",
        f"- Free space: WSL `~` {wsl_free:.0f} GB, Windows D: {d_free:.0f} GB",
        f"- `max_raw_gb` = min(100, {wsl_free:.0f} - 170) = {max_raw_gb} GB"
        + ("" if feasible else " (WARNING: below the 30 GB floor; owner action needed)"),
        "",
        "### lscpu",
        "```",
        run("lscpu | head -25"),
        "```",
        "",
        "### free -h",
        "```",
        run("free -h"),
        "```",
        "",
        "### df -h",
        "```",
        run("df -h ~ /mnt/d"),
        "```",
        "",
        "### nvidia-smi",
        "```",
        run("nvidia-smi"),
        "```",
        "",
        "## Encoder throughput",
        "",
        "Filled in Phase 0b.",
        "",
    ]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
