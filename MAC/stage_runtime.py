"""Prepare an explicit Mac source set; never copy developer credentials or checkpoints."""

from pathlib import Path
import shutil

MAC = Path(__file__).resolve().parent
ROOT = MAC.parent
BUILD = ROOT / ".build"
SHARED = ROOT / "shared" if (ROOT / "shared").is_dir() else ROOT
FILES = {
    "windowmask_server.py": "windowmask_server.py",
    "windowmask_license_client.py": "windowmask_license_client.py",
    "windowmask_worker.py": "windowmask_worker.py",
    "windowmask_devices.py": "windowmask_devices.py",
    "windowmask_engines.py": "_windowmask_cpu.py",
    "windowmask_inference.py": "_windowmask_pipeline.py",
    "models/windowmask_onnx.model": "models/windowmask_onnx.model",
    "models/windowmask_onnx.json": "models/windowmask_onnx.json",
    "config/windowmask_client.json": "config/windowmask_client.json",
    "packaging/runtime_smoke.onnx": "packaging/runtime_smoke.onnx",
}


def stage():
    destination = BUILD / "mac-stage"
    BUILD.mkdir(exist_ok=True)
    if destination.resolve().parent != BUILD.resolve() or destination.is_symlink():
        raise ValueError("Unsafe Mac staging directory.")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    for source, target in FILES.items():
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SHARED / source, output)
    for source in (MAC / "runtime").glob("*.py"):
        shutil.copy2(source, destination / source.name)
    return destination


if __name__ == "__main__":
    print(stage())
