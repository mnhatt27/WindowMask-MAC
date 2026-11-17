"""Install the final DMG/pkg and test the bundled native dispatch on either Mac."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".build/mac-installed-test"
APP = Path("/Applications/WindowMask Backend.app")


def run(*command, **kwargs):
    subprocess.run([str(value) for value in command], check=True, **kwargs)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    image = next((ROOT / "dist").glob("*.dmg"))
    mount = WORK / "volume"
    mount.mkdir(exist_ok=True)
    run("hdiutil", "attach", image, "-readonly", "-nobrowse", "-mountpoint", mount)
    try:
        run(
            "sudo",
            "installer",
            "-pkg",
            mount / "WindowMask Backend.pkg",
            "-target",
            "/",
        )
    finally:
        run("hdiutil", "detach", mount)
    run("codesign", "--verify", "--deep", "--strict", APP)
    for architecture in ("arm64", "x86_64"):
        model = (
            APP
            / f"Contents/Resources/runtimes/{architecture}/WindowMaskRuntime/_internal/models/windowmask_onnx.model"
        )
        common = APP / "Contents/Resources/models/windowmask_onnx.model"
        assert model.resolve() == common.resolve()
        assert model.read_bytes()[:4] == b"WMO1"
    launcher = APP / "Contents/MacOS/WindowMaskLauncher"
    report_path = WORK / "runtime-self-test.json"
    run(launcher, "--self-test", "--report", report_path, timeout=90)
    report = json.loads(report_path.read_text())
    assert report["ok"] and report["architecture"] == platform.machine()
    process = subprocess.Popen(
        [str(launcher), "--port", "18779", "--state-dir", str(WORK / "state")]
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Installed backend exited during startup.")
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:18779/health", timeout=1
                ) as reply:
                    assert json.load(reply)["status"] == "ok"
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise TimeoutError("Installed backend did not become ready.")
        run(launcher, "--stop", timeout=30)
        process.wait(timeout=15)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    report.update(
        installed_pkg_tested=True,
        native_launcher_tested=True,
        native_stop_tested=True,
        shared_model_links_valid=True,
        dmg_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
    )
    destination = ROOT / "dist" / f"installed-{platform.machine()}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
