"""Phase 0b: GPU throughput benchmark of every patch encoder on a fixed set of patches.

For each encoder: float16-autocast vs float32 agreement check, batch-size auto-tuning,
throughput (patches/s, including host-to-device copy and normalisation), peak VRAM and
GPU temperature. Results are appended to ``results/benchmark_0b.csv`` (resumable) and the
throughput section of ``reports/compute_plan.md`` is regenerated from that file.
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

BATCH_CANDIDATES = [256, 128, 64, 32, 16, 8, 4, 2, 1]
PLAN_START = "## Encoder throughput"
SECONDS_PER_DAY = 86400


def is_oom(err: BaseException) -> bool:
    import torch

    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err)


def free_cuda() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def embed(model, x, batch_size: int, fp16: bool, device: str):
    """Embed a uint8 NCHW CPU tensor in batches; returns float32 CPU embeddings."""
    import torch

    outs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        for i in range(0, len(x), batch_size):
            batch = x[i : i + batch_size].to(device, non_blocking=True)
            outs.append(model(batch).float().cpu())
    return torch.cat(outs)


def precision_check(model, x, device: str, threshold: float) -> dict:
    """Compare float16-autocast and float32 embeddings on the same patches."""
    import torch
    import torch.nn.functional as F

    for bs in (32, 16, 8, 4, 2, 1):
        try:
            ref = embed(model, x, bs, fp16=False, device=device)
            half = embed(model, x, bs, fp16=True, device=device)
            break
        except RuntimeError as err:
            if not is_oom(err):
                raise
            free_cuda()
    else:
        return {"cosine_fp16": float("nan"), "fp16_ok": True, "precision_note": "fp32 OOM"}

    finite = bool(torch.isfinite(half).all())
    cos = F.cosine_similarity(ref, half, dim=1).mean().item() if finite else float("nan")
    ok = finite and cos >= threshold
    return {
        "cosine_fp16": round(cos, 6),
        "fp16_ok": ok,
        "precision_note": "" if ok else ("non-finite" if not finite else "cosine below threshold"),
    }


def tune_batch(model, x, fp16: bool, device: str) -> int:
    """Largest candidate batch size that runs two forward passes without OOM."""
    for bs in BATCH_CANDIDATES:
        if bs > len(x):
            continue
        try:
            embed(model, x[: 2 * bs], bs, fp16, device)
            return bs
        except RuntimeError as err:
            if not is_oom(err):
                raise
            free_cuda()
    raise RuntimeError("OOM at batch size 1")


def time_throughput(model, x, bs: int, fp16: bool, device: str, monitor) -> tuple[float, int]:
    """Patches/s over all of ``x`` (thermal pauses excluded); halves batch on OOM."""
    import torch

    while True:
        try:
            embed(model, x[: 2 * bs], bs, fp16, device)  # warm-up
            torch.cuda.synchronize()
            paused, start = 0.0, time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                for i in range(0, len(x), bs):
                    paused += monitor.wait_if_hot()
                    model(x[i : i + bs].to(device, non_blocking=True))
            torch.cuda.synchronize()
            return len(x) / (time.perf_counter() - start - paused), bs
        except RuntimeError as err:
            if not is_oom(err) or bs == 1:
                raise
            free_cuda()
            bs //= 2
            print(f"  OOM during timing; retrying with batch {bs}", flush=True)


def jpeg_decode_rate(patches: np.ndarray, quality: int, n: int = 300) -> float:
    """Single-core JPEG decode rate (patches/s) for 224x224 patches."""
    from PIL import Image

    blobs = []
    for arr in patches[:n]:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
        blobs.append(buf.getvalue())
    start = time.perf_counter()
    for blob in blobs:
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    return len(blobs) / (time.perf_counter() - start)


def run_benchmark(args, cfg) -> None:
    import torch

    from src.features.encoder_registry import build_encoder, list_encoders
    from src.gpu import GpuMonitor

    assert torch.cuda.is_available(), "Phase 0b requires the NVIDIA GPU"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    out_csv = Path(args.out)
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(prev.loc[prev.status == "ok", "encoder"])

    patches = np.load(args.patches)
    x = torch.from_numpy(patches).permute(0, 3, 1, 2).contiguous().pin_memory()
    print(f"{len(x)} patches loaded from {args.patches}", flush=True)

    specs = list_encoders(include_gated=False)
    if args.encoders:
        specs = [s for s in specs if s.name in args.encoders]

    gcfg = cfg["gpu"]
    monitor = GpuMonitor(
        ROOT / cfg["paths"]["logs"] / "gpu_benchmark_0b.csv",
        gcfg["monitor_interval_s"],
        gcfg["temp_pause_c"],
        gcfg["temp_resume_c"],
    )
    with monitor:
        for spec in specs:
            if spec.name in done:
                print(f"{spec.name}: already benchmarked, skipping")
                continue
            print(f"{spec.name} (tier {spec.tier}): loading", flush=True)
            row = {"encoder": spec.name, "tier": spec.tier, "order": spec.order, "gpu": gpu_name}
            model = None
            try:
                t0 = time.perf_counter()
                for attempt in range(5):
                    try:
                        model = build_encoder(spec)
                        break
                    except OSError as err:  # interrupted weight download; hub resumes it
                        print(f"  load attempt {attempt + 1} failed: {err}", flush=True)
                        time.sleep(30)
                else:
                    raise RuntimeError("weights could not be downloaded")
                row["load_s"] = round(time.perf_counter() - t0, 1)
                row["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
                model = model.to(device)
                row["embed_dim"] = embed(model, x[:2], 2, True, device).shape[1]

                row.update(precision_check(model, x[: args.check_n], device, args.threshold))
                # Turing TU116 has no tensor cores, so fp16 is not always faster: time both
                # precisions and keep the faster one that passes the agreement check.
                timings = {}
                for precision in ["fp32", "fp16"] if row["fp16_ok"] else ["fp32"]:
                    fp16 = precision == "fp16"
                    try:
                        bs = tune_batch(model, x, fp16, device)
                    except RuntimeError as err:
                        if not is_oom(err):
                            raise
                        print(f"  {precision}: OOM at batch 1", flush=True)
                        continue
                    monitor.max_temp_c = 0
                    torch.cuda.reset_peak_memory_stats()
                    rate, bs = time_throughput(model, x, bs, fp16, device, monitor)
                    timings[precision] = {
                        "batch_size": bs,
                        "patches_per_s": round(rate, 1),
                        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                        "max_temp_c": monitor.max_temp_c,
                    }
                    row[f"patches_per_s_{precision}"] = round(rate, 1)
                    print(f"  {precision}: bs={bs} {rate:.0f} patches/s", flush=True)
                if not timings:
                    raise RuntimeError("OOM at batch size 1 in every precision")
                best = max(timings, key=lambda p: timings[p]["patches_per_s"])
                row.update(precision=best, **timings[best], status="ok", reason="")
                print(
                    f"  -> {best} cos={row['cosine_fp16']} bs={row['batch_size']} "
                    f"{row['patches_per_s']:.0f} patches/s peak {row['peak_vram_gb']} GB "
                    f"max {row['max_temp_c']} C",
                    flush=True,
                )
            except Exception as err:  # log and continue with the next encoder
                traceback.print_exc()
                row.update(status="failed", reason=f"{type(err).__name__}: {err}"[:300])
            finally:
                del model
                free_cuda()

            pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)


def fmt(value) -> str:
    return "" if value is None or pd.isna(value) else f"{value:.0f}"


def write_plan(args, cfg) -> None:
    """Regenerate the throughput section of reports/compute_plan.md from the benchmark CSV."""
    bench = pd.read_csv(args.out).drop_duplicates("encoder", keep="last").sort_values("order")
    files = pd.read_csv(ROOT / cfg["paths"]["manifests"] / "gdc_open_svs_files.csv")
    files = files[files.sample_type.isin(["Primary Tumor", "Solid Tissue Normal"])]
    n_all = len(files)
    n_frozen = int((files.slide_type == "frozen").sum())
    n_ffpe = int((files.slide_type == "ffpe").sum())
    cap = cfg["tiling"]["max_patches_per_slide"]
    decode = jpeg_decode_rate(np.load(args.patches), cfg["tiling"]["jpeg_quality"])
    loader_ceiling = decode * cfg["loader"]["num_workers"]
    limit_days = args.max_days

    lines = [
        PLAN_START,
        "",
        f"Benchmark: {args.n_patches} patches (224x224, 10x, "
        f"JPEG q{cfg['tiling']['jpeg_quality']}) "
        "from 3 smoke-test slides (frozen KIRC, FFPE KIRP, frozen KICH), "
        f"`scripts/benchmark_encoders.py`, {platform.node()}. Throughput includes host-to-device "
        "copy and normalisation; JPEG decoding is measured separately below.",
        "",
        f"Slides in scope (open SVS, primary tumour + solid tissue normal): {n_all} "
        f"(frozen {n_frozen}, FFPE {n_ffpe}); before QC, so estimates are upper bounds. "
        f"Estimate = slides x {cap} patches / throughput.",
        "",
        f"DataLoader ceiling: JPEG decode {decode:.0f} patches/s per core, about "
        f"{loader_ceiling:.0f} patches/s with {cfg['loader']['num_workers']} workers. "
        "Encoders faster than this are input-bound; their estimate uses the ceiling.",
        "",
        "Precision: float16 autocast and float32 are both timed; the faster one is used, and "
        "float16 only if its embeddings agree with float32 (mean cosine >= "
        f"{args.threshold} on {args.check_n} patches, no NaN/Inf). The GTX 1660 Ti (TU116) has "
        "no tensor cores, so float16 is not automatically faster.",
        "",
        "| # | Encoder | Tier | Params (M) | Dim | fp16 cos | fp32/s | fp16/s | Used | Batch "
        "| Peak VRAM (GB) | Max temp (C) | Track A (h) | All slides (h) | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    total_h = 0.0
    slow = []
    for r in bench.itertuples():
        if r.status != "ok":
            lines.append(
                f"| {r.order} | {r.encoder} | {r.tier} |" + " |" * 11 + f" failed: {r.reason} |"
            )
            continue
        eff = min(r.patches_per_s, loader_ceiling)
        h_a = n_frozen * cap / eff / 3600
        h_all = n_all * cap / eff / 3600
        total_h += h_all
        if h_all * 3600 > limit_days * SECONDS_PER_DAY:
            slow.append((r.encoder, h_all, int(limit_days * SECONDS_PER_DAY * eff / n_all)))
        fp32 = fmt(getattr(r, "patches_per_s_fp32", None))
        fp16 = fmt(getattr(r, "patches_per_s_fp16", None))
        lines.append(
            f"| {r.order} | {r.encoder} | {r.tier} | {r.params_m} | {int(r.embed_dim)} "
            f"| {r.cosine_fp16} | {fp32} | {fp16} "
            f"| {r.precision} | {int(r.batch_size)} "
            f"| {r.peak_vram_gb} | {int(r.max_temp_c)} | {h_a:.1f} | {h_all:.1f} | ok |"
        )

    lines += [
        "",
        f"Total GPU time, benchmarked encoders, all slides: {total_h:.0f} h "
        f"({total_h / 24:.1f} days).",
        "",
        "Not benchmarked (gated on Hugging Face, no access token yet): CONCH, H0-mini, UNI, "
        "Virchow2, UNI2-h, H-optimus-1, Prov-GigaPath.",
        "",
        "### Run order",
        "",
        "Tier order as specified (1a, 1b, 2, 3; within a tier by `order` in "
        "`configs/encoders/*.yaml`), Track A (frozen) slides before Track B (FFPE).",
    ]
    if slow:
        lines += ["", f"### Encoders over {limit_days} days", ""]
        for name, hours, new_cap in slow:
            lines.append(
                f"- {name}: {hours / 24:.1f} days at {cap} patches/slide; a cap of "
                f"{new_cap} patches/slide fits in {limit_days} days."
            )

    plan_path = ROOT / "reports" / "compute_plan.md"
    text = plan_path.read_text()
    head = text.split(PLAN_START)[0]
    plan_path.write_text(head + "\n".join(lines) + "\n")
    print(f"updated {plan_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--n-patches", type=int, default=2000)
    parser.add_argument("--patches", default=None)
    parser.add_argument("--out", default=str(ROOT / "results" / "benchmark_0b.csv"))
    parser.add_argument("--encoders", nargs="*", default=None)
    parser.add_argument("--check-n", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--max-days", type=float, default=3.0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["paths"]["hf_home"])
    if args.patches is None:
        bench_dir = Path(cfg["paths"]["patch_export"]).parent / "bench"
        args.patches = str(bench_dir / f"patches_{args.n_patches}.npy")

    if not args.plan_only:
        run_benchmark(args, cfg)
    write_plan(args, cfg)


if __name__ == "__main__":
    main()
