"""Mac CPU adapter and optional CoreML GPU adapter, sharing the CPU executor."""

import numpy as np

from _windowmask_cpu import OnnxEngine as CpuEngine, cpu_threads  # noqa: F401


class OnnxEngine(CpuEngine):
    def __init__(self, model_bytes, threads, device_id=None):
        super().__init__(model_bytes, threads, device_id)
        self.name = "CoreML CPU/GPU" if self.gpu else "ONNX CPU"

    def _create_session(self, model_bytes, threads, device_id):
        if device_id is None:
            return super()._create_session(model_bytes, threads, device_id)
        import onnxruntime as ort

        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CoreML is not available in this runtime.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = [
            (
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "NeuralNetwork",
                    "MLComputeUnits": "CPUAndGPU",
                    "AllowLowPrecisionAccumulationOnGPU": "0",
                    "RequireStaticInputShapes": "0",
                    "EnableOnSubgraphs": "0",
                },
            ),
            "CPUExecutionProvider",
        ]
        session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=providers
        )
        if "CoreMLExecutionProvider" not in session.get_providers():
            raise RuntimeError("CoreML provider did not initialise.")
        return session

    def _run(self, batch):
        output = self.session.run(None, {self.input_name: batch})[0]
        return np.array(output[:, 0], dtype=np.float32, copy=True)
