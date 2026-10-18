"""GPU telemetry logging and thermal guard (NVML)."""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import pynvml


class GpuMonitor:
    """Logs temperature, clocks, utilisation and memory every ``interval_s`` seconds to CSV.

    ``wait_if_hot()`` blocks while the GPU is above ``pause_c`` until it cools below
    ``resume_c``; call it between batches.
    """

    FIELDS = [
        "time",
        "temp_c",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "util_pct",
        "mem_used_mb",
        "power_w",
    ]

    def __init__(
        self,
        log_path: str | Path,
        interval_s: int = 60,
        pause_c: int = 85,
        resume_c: int = 75,
        index: int = 0,
    ):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.log_path = Path(log_path)
        self.interval_s = interval_s
        self.pause_c = pause_c
        self.resume_c = resume_c
        self.max_temp_c = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def read(self) -> dict:
        h = self.handle
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
        except pynvml.NVMLError:
            power = float("nan")
        sample = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "util_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
            "mem_used_mb": pynvml.nvmlDeviceGetMemoryInfo(h).used // 2**20,
            "power_w": power,
        }
        self.max_temp_c = max(self.max_temp_c, sample["temp_c"])
        return sample

    def temperature(self) -> int:
        temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        self.max_temp_c = max(self.max_temp_c, temp)
        return temp

    def wait_if_hot(self) -> float:
        """Block while the GPU is too hot. Returns seconds spent paused."""
        if self.temperature() <= self.pause_c:
            return 0.0
        import torch

        # Finish queued kernels first so callers that subtract the pause from a timing
        # do not also subtract compute.
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        start = time.time()
        print(f"GPU at {self.temperature()} C > {self.pause_c} C; pausing", flush=True)
        while self.temperature() >= self.resume_c:
            time.sleep(10)
        paused = time.time() - start
        print(f"GPU cooled to {self.temperature()} C after {paused:.0f}s; resuming", flush=True)
        return paused

    def _run(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                writer.writeheader()
            while not self._stop.is_set():
                writer.writerow(self.read())
                f.flush()
                self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuMonitor:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
