"""Hidden Mac backend; no Windows mutexes, DPAPI, services or shell launchers."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import sys
import threading

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

VERSION = "1.1.3"


@contextmanager
def instance_lock(root):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / "backend.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def self_test(report=None):
    """Never touches production Keychain items or decrypts production weights."""
    from unittest.mock import patch
    import numpy as np
    import windowmask_server as backend
    import windowmask_license_client as licensing
    from windowmask_engines import OnnxEngine, cpu_threads

    with patch.object(licensing, "is_entitled", return_value=False):
        response = backend.app.test_client().post(
            "/predict?width=1&height=1", data=b"\0\0\0"
        )
    assert response.status_code == 403 and backend.model is None
    engine = OnnxEngine(
        (Path(__file__).parent / "packaging/runtime_smoke.onnx").read_bytes(),
        cpu_threads(),
    )
    try:
        mask = engine.run(np.zeros((1, 3, 768, 768), np.float32))
        assert mask.shape == (1, 768, 768) and np.all(mask == np.float32(0.5))
    finally:
        engine.close()
    result = {
        "ok": True,
        "version": VERSION,
        "architecture": __import__("platform").machine(),
        "license_gated": True,
        "inference_tested": True,
        "production_weights_tested": False,
    }
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--engine-worker", choices=("coreml",))
    parser.add_argument("--worker-threads", type=int, default=8)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    if args.engine_worker:
        from windowmask_worker import serve_worker

        return serve_worker(args.engine_worker, args.worker_threads, args.device_id)
    if args.self_test:
        return self_test(args.report)
    root = (
        args.state_dir or Path.home() / "Library/Application Support/WindowMaskMystic"
    )
    os.environ["WINDOWMASK_RUNTIME_STATE"] = str(root)
    with instance_lock(root) as acquired:
        if not acquired:
            return 0
        logs = root / "Logs"
        logs.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            logs / "backend.log", maxBytes=2_000_000, backupCount=2
        )
        logging.basicConfig(
            level=logging.INFO,
            handlers=[handler],
            format="%(asctime)s %(levelname)s %(message)s",
        )
        import windowmask_server as backend
        from werkzeug.serving import make_server

        server = make_server("127.0.0.1", args.port, backend.app, threaded=True)

        def stop(*_):
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            logging.info(
                "WindowMask Mac %s listening on 127.0.0.1:%s", VERSION, args.port
            )
            server.serve_forever()
        finally:
            server.server_close()
            if backend.model is not None:
                backend.model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
