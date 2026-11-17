"""Private GPU process: bounded binary pipe protocol, no HTTP or license bypass."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import weakref

import numpy as np


def _read_exact(stream, count):
    parts = bytearray()
    while len(parts) < count:
        chunk = stream.read(count - len(parts))
        if not chunk:
            raise RuntimeError("GPU worker closed its output")
        parts.extend(chunk)
    return bytes(parts)


def _stop(process):
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    for stream in (process.stdin, process.stdout):
        if stream:
            try:
                stream.close()
            except OSError:
                pass


class WorkerEngine:
    gpu = True

    def __init__(
        self, model_bytes: bytes, threads: int, kind: str, device_id=0, label="GPU"
    ):
        if getattr(sys, "frozen", False):
            command = [
                sys.executable,
                "--engine-worker",
                kind,
                "--worker-threads",
                str(threads),
                "--device-id",
                str(device_id),
            ]
        else:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                kind,
                str(threads),
                str(device_id),
            ]
        provider = "CoreML" if kind == "coreml" else "DirectML"
        self.name = f"{label} / {provider}"
        self.warmed_up = False
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._cleanup = weakref.finalize(self, _stop, self.process)
        try:
            with self._deadline(60):
                self.process.stdin.write(
                    (json.dumps({"model_bytes": len(model_bytes)}) + "\n").encode()
                )
                self.process.stdin.write(model_bytes)
                self.process.stdin.flush()
                response = self._response()
                if not response.get("ready"):
                    raise RuntimeError(
                        response.get("error", "GPU worker did not initialise")
                    )
        except Exception:
            self.close()
            raise

    def _deadline(self, seconds):
        process = self.process

        class Deadline:
            def __enter__(self):
                self.expired = threading.Event()

                def expire():
                    self.expired.set()
                    if process.poll() is None:
                        process.kill()

                self.timer = threading.Timer(seconds, expire)
                self.timer.daemon = True
                self.timer.start()

            def __exit__(self, *_):
                self.timer.cancel()
                if self.expired.is_set():
                    raise RuntimeError("GPU response timed out; CPU fallback required")

        return Deadline()

    def _response(self):
        line = self.process.stdout.readline(65536)
        if not line:
            raise RuntimeError("GPU worker exited")
        return json.loads(line)

    def run(self, batch):
        contiguous = np.ascontiguousarray(batch)
        # DirectML compiles the first full convolution graph on this device.
        # Later calls retain a shorter watchdog so a failed driver cannot hang UI.
        with self._deadline(90 if not self.warmed_up else 30):
            self.process.stdin.write(
                (
                    json.dumps({"shape": list(batch.shape), "bytes": contiguous.nbytes})
                    + "\n"
                ).encode()
            )
            self.process.stdin.write(contiguous.tobytes())
            self.process.stdin.flush()
            response = self._response()
            if "error" in response:
                raise RuntimeError(response["error"])
            shape = (len(batch), batch.shape[2], batch.shape[3])
            count = int(np.prod(shape)) * 4
            if response.get("bytes") != count:
                raise RuntimeError("GPU worker returned an invalid mask size")
            result = (
                np.frombuffer(_read_exact(self.process.stdout, count), np.float32)
                .reshape(shape)
                .copy()
            )
            self.warmed_up = True
            return result

    def close(self):
        self._cleanup()


def serve_worker(kind: str, threads: int, device_id: int):
    # Source execution with embedded Python does not automatically add the script directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    source, output = sys.stdin.buffer, sys.stdout.buffer

    def respond(value):
        output.write((json.dumps(value) + "\n").encode())
        output.flush()

    try:
        initial = json.loads(source.readline(65536))
        count = initial["model_bytes"]
        if not isinstance(count, int) or not 0 < count <= 256 * 1024**2:
            raise ValueError("Invalid worker model size")
        model_bytes = _read_exact(source, count)
        from windowmask_engines import OnnxEngine

        engine = OnnxEngine(model_bytes, threads, device_id)
        del model_bytes
        respond({"ready": True})
        for line in source:
            try:
                request = json.loads(line)
                shape = request["shape"]
                if (
                    len(shape) != 4
                    or not 1 <= shape[0] <= 16
                    or shape[1:] != [3, 768, 768]
                ):
                    raise ValueError("Invalid worker input shape")
                size = int(np.prod(shape)) * 4
                if request["bytes"] != size:
                    raise ValueError("Invalid worker input size")
                batch = np.frombuffer(_read_exact(source, size), np.float32).reshape(
                    shape
                )
                probability = np.ascontiguousarray(engine.run(batch), dtype=np.float32)
                respond({"bytes": probability.nbytes})
                output.write(probability.tobytes())
                output.flush()
            except Exception as error:
                respond({"error": str(error)[:2000]})
    except Exception as error:
        respond({"error": str(error)[:2000]})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(serve_worker(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
