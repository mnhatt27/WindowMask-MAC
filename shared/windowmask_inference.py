"""Full-resolution tiled masks, automatic FP32 engine selection and CPU fallback."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from windowmask_devices import gpu_adapters
from windowmask_engines import OnnxEngine, cpu_threads
from windowmask_worker import WorkerEngine

TILE_SIZE, STRIDE = 768, 576
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    result = list(range(0, length - tile + 1, stride))
    if result[-1] != length - tile:
        result.append(length - tile)
    return result


@lru_cache(maxsize=2)
def blend_weights(tile: int):
    axis = np.hanning(tile).astype(np.float32) * 0.95 + 0.05
    weight = np.outer(axis, axis).astype(np.float32)
    axis.setflags(write=False)
    weight.setflags(write=False)
    return axis, weight


def axis_weights(length, positions, axis):
    result = np.zeros(length, np.float32)
    for offset in positions:
        result[offset : offset + len(axis)] += axis
    return result


def fill_batch(image, positions, buffer, tile=TILE_SIZE):
    for index, (y, x) in enumerate(positions):
        target = buffer[index]
        np.copyto(target, image[y : y + tile, x : x + tile], casting="unsafe")
        target *= 0.00392156862745098
        target -= MEAN
        target /= STD
    return buffer[: len(positions)].transpose(0, 3, 1, 2)


def prepared_batches(image, positions, count, prefetch, tile=TILE_SIZE):
    buffers = [
        np.empty((count, tile, tile, 3), np.float32)
        for _ in range(2 if prefetch else 1)
    ]
    if not prefetch:
        for start in range(0, len(positions), count):
            chunk = positions[start : start + count]
            yield chunk, fill_batch(image, chunk, buffers[0], tile)
        return
    # Only one batch may be prepared ahead. The consumer finishes reading a
    # buffer before this generator reuses it, including OOM retries.
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="WindowMaskPrepare"
    ) as executor:
        chunk = positions[:count]
        pending = executor.submit(fill_batch, image, chunk, buffers[0], tile)
        for start in range(0, len(positions), count):
            batch = pending.result()
            next_start = start + count
            next_chunk = positions[next_start : next_start + count]
            if next_chunk:
                pending = executor.submit(
                    fill_batch,
                    image,
                    next_chunk,
                    buffers[(next_start // count) % 2],
                    tile,
                )
            yield chunk, batch
            chunk = next_chunk


def normalise_mask(sums, weight_y, weight_x, height, width):
    result = np.empty((height, width), np.uint8)
    x_weights = np.maximum(weight_x[:width], 1e-08)
    scratch = np.empty((min(64, height), width), np.float32)
    for start in range(0, height, len(scratch)):
        stop = min(height, start + len(scratch))
        block = scratch[: stop - start]
        np.multiply(weight_y[start:stop, None], x_weights[None, :], out=block)
        np.divide(sums[start:stop, :width], block, out=block)
        np.multiply(block, 255.0, out=block)
        np.clip(block, 0.0, 255.0, out=block)
        result[start:stop] = block
    return result


def quality_difference(expected, actual):
    if (
        actual.shape != expected.shape
        or actual.dtype != np.float32
        or not np.isfinite(actual).all()
    ):
        return {"ok": False, "reason": "invalid output shape, precision or values"}
    delta = np.abs(actual - expected)
    a = np.clip(actual * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(expected * 255.0, 0, 255).astype(np.uint8)
    pixels = np.abs(a.astype(np.int16) - b.astype(np.int16))
    changed = int(np.count_nonzero(pixels))
    flips = int(np.count_nonzero((actual >= 0.5) != (expected >= 0.5)))
    maximum = float(delta.max())
    return {
        "ok": maximum <= 1e-4
        and int(pixels.max()) <= 1
        and changed / pixels.size <= 0.001
        and flips / pixels.size <= 0.00001,
        "max_probability_error": maximum,
        "different_pixels": changed,
        "max_pixel_difference": int(pixels.max()),
        "threshold_flips": flips,
    }


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def select_engines(model_bytes: bytes, profile_path: Path | None = None):
    threads = cpu_threads()
    baseline = OnnxEngine(model_bytes, threads)
    cpu = baseline
    requested = os.environ.get("WINDOWMASK_ENGINE", "auto").lower()
    target = os.environ.get("WINDOWMASK_DEVICE", "auto").lower()
    if requested not in ("auto", "onnx") or target not in (
        "auto",
        "cpu",
        "gpu",
    ):
        raise ValueError(
            "WINDOWMASK_ENGINE must be auto/onnx; WINDOWMASK_DEVICE must be auto/cpu/gpu"
        )
    adapters = gpu_adapters() if target != "cpu" else []
    identity = {
        "model": hashlib.sha256(model_bytes).hexdigest(),
        "threads": threads,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "adapters": adapters,
        "ort": _version("onnxruntime-directml"),
        "engine": requested,
        "target": target,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    cached = None
    if profile_path:
        try:
            cached = json.loads(profile_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") != fingerprint:
                cached = None
        except (OSError, ValueError):
            pass
    rng = np.random.default_rng(13500)
    probe = rng.integers(0, 256, (TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    buffer = np.empty((1, TILE_SIZE, TILE_SIZE, 3), np.float32)
    batch = fill_batch(probe, [(0, 0)], buffer)
    expected = baseline.run(batch)

    def measure(engine):
        actual = engine.run(batch)
        quality = quality_difference(expected, actual)
        if not quality["ok"]:
            raise ValueError(f"FP32 quality gate rejected {engine.name}: {quality}")
        times = []
        for _ in range(2):
            started = time.perf_counter()
            engine.run(batch)
            times.append(time.perf_counter() - started)
        return statistics.median(times), quality

    base_time, _ = measure(baseline)
    best, best_time = baseline, base_time
    choices = [("onnx-cpu", baseline.name, base_time)]
    factories = []
    if target != "cpu":
        for adapter in adapters:
            factories.append(
                (
                    f"directml-{adapter['index']}",
                    lambda a=adapter: WorkerEngine(
                        model_bytes, threads, "directml", a["index"], a["name"]
                    ),
                )
            )
    # A saved winner avoids recompiling every rejected GPU on each application
    # launch. It is still validated against the CPU on every load.
    if cached:
        wanted = {cached.get("cpu"), cached.get("selected")}
        factories = [(key, factory) for key, factory in factories if key in wanted]
    cpu_key = selected_key = "onnx-cpu"
    for key, factory in factories:
        engine = None
        try:
            print(f"WindowMask: kiểm tra {key} FP32...", flush=True)
            engine = factory()
            elapsed, quality = measure(engine)
            choices.append((key, engine.name, elapsed))
            print(
                f"WindowMask: {engine.name} {elapsed:.3f}s/tile; quality={quality}",
                flush=True,
            )
            if not engine.gpu and elapsed < best_time:
                cpu, cpu_key = engine, key
            # Prefer CPU for marginal GPU gains, to leave Photoshop GPU headroom.
            if elapsed < best_time * (0.9 if engine.gpu else 1.0):
                previous = best
                best, best_time, selected_key = engine, elapsed, key
                if previous is not cpu and hasattr(previous, "close"):
                    previous.close()
            elif engine is not cpu and hasattr(engine, "close"):
                engine.close()
        except Exception as error:
            if engine is not None and hasattr(engine, "close"):
                engine.close()
            print(f"WindowMask: bỏ qua {key}: {error}", flush=True)
    profile = {
        "fingerprint": fingerprint,
        "selected": selected_key,
        "cpu": cpu_key,
        "selected_name": best.name,
        "seconds_per_tile": best_time,
        "choices": choices,
        "precision": "FP32",
    }
    if profile_path:
        try:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = profile_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            temporary.replace(profile_path)
        except OSError as error:
            print(f"WindowMask: không lưu được cấu hình tốc độ: {error}", flush=True)
    return best, cpu, profile


class InferenceRuntime:
    def __init__(self, active, cpu, profile=None):
        self.active, self.cpu = active, cpu
        self.profile = profile or {}
        try:
            self.batch_limit = max(
                1, min(16, int(os.environ.get("WINDOWMASK_BATCH", "4")))
            )
        except ValueError:
            self.batch_limit = 4

    @property
    def name(self):
        return self.active.name

    def _try_batch(self, batch):
        try:
            return self.active.run(batch), None
        except Exception as error:
            if not self.active.gpu:
                raise
            message = str(error)
        return None, message

    def _batches(self, batch):
        offset = 0
        while offset < len(batch):
            count = min(self.batch_limit if self.active.gpu else 1, len(batch) - offset)
            probability, error = self._try_batch(batch[offset : offset + count])
            if error is not None:
                oom = any(
                    marker in error.lower()
                    for marker in (
                        "out of memory",
                        "e_outofmemory",
                        "0x8007000e",
                        "failed to allocate",
                    )
                )
                if oom and count > 1:
                    self.batch_limit = max(1, count // 2)
                    print(
                        f"WindowMask: GPU thiếu bộ nhớ; thử lại batch={self.batch_limit}.",
                        flush=True,
                    )
                    continue
                failed = self.active
                self.active = self.cpu
                if hasattr(failed, "close"):
                    failed.close()
                print(
                    f"WindowMask: chuyển CPU để hoàn thành mask: {error[:300]}",
                    flush=True,
                )
                continue
            expected_shape = (count, batch.shape[2], batch.shape[3])
            if (
                probability.shape != expected_shape
                or not np.isfinite(probability).all()
            ):
                raise RuntimeError("Inference returned an invalid mask")
            yield offset, probability
            offset += count

    def predict(self, rgb, tile=TILE_SIZE, stride=STRIDE):
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Ảnh đầu vào phải có 3 kênh RGB.")
        h, w = rgb.shape[:2]
        if min(h, w) < 1:
            raise ValueError("Ảnh đầu vào có kích thước rỗng.")
        pad_h, pad_w = max(0, tile - h), max(0, tile - w)
        padded = (
            np.pad(
                rgb,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect" if min(h, w) > 1 else "edge",
            )
            if pad_h or pad_w
            else rgb
        )
        height, width = padded.shape[:2]
        ys, xs = (
            tile_positions(height, tile, stride),
            tile_positions(width, tile, stride),
        )
        axis, weight = blend_weights(tile)
        sums = np.zeros((height, width), np.float32)
        weight_y, weight_x = (
            axis_weights(height, ys, axis),
            axis_weights(width, xs, axis),
        )
        positions = [(y, x) for y in ys for x in xs]
        count = min(self.batch_limit if self.active.gpu else 1, len(positions))
        prepared = prepared_batches(padded, positions, count, self.active.gpu, tile)
        start = time.perf_counter()
        try:
            for chunk, batch in prepared:
                for offset, probabilities in self._batches(batch):
                    for (y, x), probability in zip(
                        chunk[offset : offset + len(probabilities)], probabilities
                    ):
                        np.multiply(probability, weight, out=probability)
                        sums[y : y + tile, x : x + tile] += probability
        finally:
            prepared.close()
        result = normalise_mask(sums, weight_y, weight_x, h, w)
        print(
            f"WindowMask inference: {w}x{h} | tiles={len(positions)} ({tile}/{stride}) | "
            f"{self.name} FP32 | {time.perf_counter() - start:.3f}s",
            flush=True,
        )
        return result

    def close(self):
        if hasattr(self.active, "close"):
            self.active.close()
        if self.cpu is not self.active and hasattr(self.cpu, "close"):
            self.cpu.close()
