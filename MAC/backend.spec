from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata

mac = Path(SPECPATH)
root = mac.parent / ".build/mac-stage"
data = []
for item in ("models/windowmask_onnx.model", "models/windowmask_onnx.json",
             "config/windowmask_client.json", "packaging/runtime_smoke.onnx"):
    data.append((str(root / item), str(Path(item).parent)))
for package in ("onnxruntime", "numpy", "Pillow", "Flask", "cryptography"):
    data += copy_metadata(package)
analysis = Analysis(
    [str(root / "mac_entry.py")], pathex=[str(root)], binaries=[], datas=data,
    hiddenimports=["windowmask_license_client", "windowmask_worker"],
    hookspath=[], runtime_hooks=[],
    excludes=["torch", "torchvision", "timm", "segmentation_models_pytorch", "onnx",
              "onnxscript", "openvino", "scipy", "pandas", "matplotlib", "tkinter",
              "pytest", "sympy", "onnxruntime.tools", "onnxruntime.training", "onnxruntime.quantization"],
    noarchive=False,
)
archive = PYZ(analysis.pure)
executable = EXE(archive, analysis.scripts, [], exclude_binaries=True,
                 name="WindowMaskRuntime", console=True, strip=False, upx=False)
bundle = COLLECT(executable, analysis.binaries, analysis.datas,
                 strip=False, upx=False, name="WindowMaskRuntime")
