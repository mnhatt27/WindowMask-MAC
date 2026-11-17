"""Build one native slice on its matching macOS runner."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_runtime import MAC, ROOT, stage


def main():
    if sys.platform != "darwin" or platform.machine() not in ("arm64", "x86_64"):
        raise SystemExit(
            "Run this build on a native Apple Silicon or Intel macOS runner."
        )
    architecture = platform.machine()
    stage()
    output = ROOT / ".build" / f"mac-{architecture}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            str(output),
            "--workpath",
            str(ROOT / ".build/mac-pyinstaller"),
            str(MAC / "backend.spec"),
        ],
        check=True,
    )
    runtime = output / "WindowMaskRuntime"
    with zipfile.ZipFile(runtime / "_internal/base_library.zip") as library:
        assert {"encodings/__init__.pyc", "encodings/utf_8.pyc"}.issubset(
            library.namelist()
        )
        assert library.testzip() is None
    forbidden = {
        "torch",
        "torchvision",
        "timm",
        "segmentation_models_pytorch",
        "openvino",
        "onnx",
        "pandas",
        "scipy",
    }
    assert not any(
        path.name in forbidden or path.suffix == ".pth" for path in runtime.rglob("*")
    )
    report = output / "self-test.json"
    subprocess.run(
        [str(runtime / "WindowMaskRuntime"), "--self-test", "--report", str(report)],
        check=True,
        timeout=90,
    )
    result = json.loads(report.read_text())
    assert result["ok"] and result["architecture"] == architecture
    subprocess.run(
        [
            "codesign",
            "--verify",
            "--deep",
            "--strict",
            str(runtime / "WindowMaskRuntime"),
        ],
        check=True,
    )
    artifact = ROOT / f"runtime-{architecture}.tar.gz"
    subprocess.run(
        [
            "tar",
            "-czf",
            str(artifact),
            "-C",
            str(output),
            "WindowMaskRuntime",
            "self-test.json",
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
