"""Install the final DMG/pkg and test the bundled native dispatch on either Mac."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".build/mac-installed-test"
APP = Path("/Applications/WindowMask Backend.app")
sys.path.insert(0, str(ROOT / "MAC/runtime"))
from mac_entry import VERSION as RUNTIME_VERSION


def run(*command, **kwargs):
    subprocess.run([str(value) for value in command], check=True, **kwargs)


def wait_for_backend(port, state, process=None):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            break
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1
            ) as reply:
                assert json.load(reply)["status"] == "ok"
                return
        except OSError:
            time.sleep(0.2)
    log = state / "Logs/backend.log"
    if log.exists():
        print(log.read_text(errors="replace")[-8000:], flush=True)
    raise RuntimeError(f"Installed backend did not start on port {port}.")


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    image = next((ROOT / "dist").glob("*.dmg"))
    mount = WORK / "volume"
    mount.mkdir(exist_ok=True)
    run("hdiutil", "attach", image, "-readonly", "-nobrowse", "-mountpoint", mount)
    try:
        source_manifest = json.loads((ROOT / "MAC/uxp/manifest.json").read_text())
        with zipfile.ZipFile(mount / "com.mystic.windowmask.mac_PS.ccx") as panel:
            panel_manifest = json.loads(panel.read("manifest.json"))
            assert panel_manifest["version"] == source_manifest["version"]
            assert panel_manifest["requiredPermissions"]["network"]["domains"] == "all"
            assert "backend.js" in panel.namelist()
            assert panel.testzip() is None
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
    assert report["version"] == RUNTIME_VERSION
    process = subprocess.Popen(
        [str(launcher), "--port", "18779", "--state-dir", str(WORK / "state")]
    )
    try:
        wait_for_backend(18779, WORK / "state", process)
        run(launcher, "--stop", timeout=30)
        process.wait(timeout=15)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    # Exercise LaunchServices as Finder/UXP do, including the normal app startup
    # environment. Directly executing CFBundleExecutable does not cover this.
    launch_state = WORK / "launch-services-state"
    try:
        run(
            "/usr/bin/open", "-n", APP, "--args", "--port", "18780",
            "--state-dir", launch_state, timeout=15,
        )
        wait_for_backend(18780, launch_state)
    finally:
        run(launcher, "--stop", timeout=30)
    report.update(
        installed_pkg_tested=True,
        native_launcher_tested=True,
        native_stop_tested=True,
        launch_services_tested=True,
        shared_model_links_valid=True,
        panel_version=panel_manifest["version"],
        bundled_panel_manifest_tested=True,
        dmg_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
    )
    destination = ROOT / "dist" / f"installed-{platform.machine()}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
