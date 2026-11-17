"""The Windows tile/mask math is shared verbatim; device selection is Mac-specific."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from _windowmask_pipeline import (
    InferenceRuntime as TiledRuntime,
    fill_batch,
    quality_difference,
)
from windowmask_engines import OnnxEngine, cpu_threads
from windowmask_worker import WorkerEngine


class InferenceRuntime(TiledRuntime):
    def __init__(self, active, cpu, profile=None):
        super().__init__(active, cpu, profile)
        # CoreML's dynamic batch support depends on the compiled graph. Batch 1
        # preserves the validated path and still overlaps CPU preparation.
        self.batch_limit = 1


def select_engines(model_bytes, profile_path=None):
    target = os.environ.get("WINDOWMASK_DEVICE", "auto").lower()
    if target not in ("cpu", "auto", "gpu"):
        raise ValueError("WINDOWMASK_DEVICE must be auto/cpu/gpu")
    threads = cpu_threads()
    cpu = OnnxEngine(model_bytes, threads)
    identity = {
        "model": hashlib.sha256(model_bytes).hexdigest(),
        "threads": threads,
        "os": platform.platform(),
        "arch": platform.machine(),
        "target": target,
        "ort": importlib.metadata.version("onnxruntime"),
        "policy": 1,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    profile = {"fingerprint": fingerprint, "winner": "cpu"}
    if profile_path:
        try:
            cached = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            if (
                cached.get("fingerprint") == fingerprint
                and cached.get("winner") == "cpu"
            ):
                return cpu, cpu, cached
        except (OSError, ValueError):
            pass
    active = cpu
    gpu = None
    if target != "cpu":
        try:
            image = np.random.default_rng(13500).integers(
                0, 256, (768, 768, 3), dtype=np.uint8
            )
            batch = fill_batch(image, [(0, 0)], np.empty((1, 768, 768, 3), np.float32))
            expected = cpu.run(batch)

            def measure(engine):
                actual = engine.run(batch)
                quality = quality_difference(expected, actual)
                if not quality["ok"]:
                    raise ValueError(f"CoreML mask quality rejected: {quality}")
                durations = []
                for _ in range(2):
                    started = time.perf_counter()
                    engine.run(batch)
                    durations.append(time.perf_counter() - started)
                return statistics.median(durations)

            before = measure(cpu)
            gpu = WorkerEngine(model_bytes, threads, "coreml", label="Apple")
            after = measure(gpu)
            profile.update(cpu_seconds=before, coreml_seconds=after)
            if after < before * 0.9:
                active, gpu = gpu, None
                profile["winner"] = "coreml"
        except Exception as error:
            profile["coreml_rejected"] = str(error)[:1000]
        finally:
            if gpu is not None:
                gpu.close()
    if profile_path:
        path = Path(profile_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass  # A read-only profile directory must not prevent inference.
    return active, cpu, profile
