"""FP32 inference adapters. Training/export libraries are deliberately absent."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class OnnxEngine:
    def __init__(self, model_bytes: bytes, threads: int, device_id: int | None = None):
        self.gpu = device_id is not None
        self.name = "ONNX DirectML" if self.gpu else "ONNX CPU"
        # Session creation and execution share a stable native thread. WSGI
        # creates a different caller thread per request; FP32 denormal settings
        # are thread-local and must not disappear between HTTP requests.
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="WindowMaskInfer"
        )
        try:
            self.session = self.executor.submit(
                self._create_session, model_bytes, threads, device_id
            ).result()
        except Exception:
            self.executor.shutdown(wait=False, cancel_futures=True)
            raise
        self.input_name = self.session.get_inputs()[0].name

    def _create_session(self, model_bytes, threads, device_id):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        # Avoid slow arithmetic on extremely small activations and idle CPU spin.
        options.add_session_config_entry("session.set_denormal_as_zero", "1")
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if self.gpu:
            options.enable_mem_pattern = False
            providers = [
                ("DmlExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=providers
        )
        if self.gpu and "DmlExecutionProvider" not in session.get_providers():
            raise RuntimeError("DirectML did not initialise on the requested GPU")
        return session

    def run(self, batch: np.ndarray) -> np.ndarray:
        return self.executor.submit(self._run, np.ascontiguousarray(batch)).result()

    def _run(self, batch):
        output = self.session.run(None, {self.input_name: batch})[0]
        return np.array(output[:, 0], dtype=np.float32, copy=True)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.session = None


def cpu_threads() -> int:
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    try:
        requested = int(os.environ.get("WINDOWMASK_CPU_THREADS", min(16, available)))
    except ValueError:
        requested = min(16, available)
    return max(1, min(available, requested))
