"""Assemble both native runtimes into one app and DMG on a macOS runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tarfile
import zipfile

MAC = Path(__file__).resolve().parent
ROOT = MAC.parent
VERSION = "1.1.2"


def run(*args, **kwargs):
    subprocess.run([str(value) for value in args], check=True, **kwargs)


def icons():
    from PIL import Image, ImageDraw

    folder = MAC / "uxp/icons"
    folder.mkdir(exist_ok=True)
    for size, name in ((48, "plugin"), (23, "panel")):
        image = Image.new("RGBA", (size, size), "#2563df")
        draw = ImageDraw.Draw(image)
        inset, center = size // 5, size // 2
        draw.rectangle(
            (inset, inset, size - inset - 1, size - inset - 1),
            outline="white",
            width=max(1, size // 16),
        )
        draw.line(
            (center, inset, center, size - inset - 1),
            fill="white",
            width=max(1, size // 16),
        )
        draw.line(
            (inset, center, size - inset - 1, center),
            fill="white",
            width=max(1, size // 16),
        )
        image.save(folder / f"{name}.png")


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # Locally built PyInstaller archives contain symlinks; reject traversal and
    # escaping links while retaining their layout (tarfile's data filter).
    with tarfile.open(archive, "r:gz") as package:
        package.extractall(destination, filter="data")
    report = json.loads((destination / "self-test.json").read_text())
    assert report["ok"] and report["inference_tested"]
    return report


def main():
    if sys.platform != "darwin":
        raise SystemExit(
            "Packaging requires macOS (codesign, clang, pkgbuild, hdiutil)."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    work = ROOT / ".build/mac-package"
    if work.resolve().parent != (ROOT / ".build").resolve() or work.is_symlink():
        raise ValueError("Unsafe package staging directory")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    payload = work / "payload"
    app = payload / "Applications/WindowMask Backend.app"
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True)
    reports = {}
    for architecture in ("arm64", "x86_64"):
        reports[architecture] = unpack(
            args.artifacts / f"runtime-{architecture}.tar.gz",
            contents / f"Resources/runtimes/{architecture}",
        )
        assert reports[architecture]["architecture"] == architecture
    # Both slices use identical encrypted weights. Keep one copy in the bundle
    # and preserve the original runtime-relative model paths through local links.
    shared_models = contents / "Resources/models"
    shared_models.mkdir()
    for name in ("windowmask_onnx.model", "windowmask_onnx.json"):
        shared_model = shared_models / name
        digest = None
        for architecture in ("arm64", "x86_64"):
            model = (
                contents
                / f"Resources/runtimes/{architecture}/WindowMaskRuntime/_internal/models/{name}"
            )
            with model.open("rb") as stream:
                current = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest is None:
                digest = current
                model.replace(shared_model)
            else:
                if current != digest:
                    raise ValueError("Native slices contain different model weights.")
                model.unlink()
            model.symlink_to(os.path.relpath(shared_model, model.parent))
            assert model.resolve() == shared_model.resolve()
    launcher = contents / "MacOS/WindowMaskLauncher"
    run(
        "clang",
        "-Os",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-mmacosx-version-min=13.0",
        "-arch",
        "arm64",
        "-arch",
        "x86_64",
        MAC / "installer/launcher.c",
        "-o",
        launcher,
    )
    run("lipo", launcher, "-verify_arch", "arm64", "x86_64")
    info = {
        "CFBundleIdentifier": "com.mystic.windowmask.backend",
        "CFBundleName": "WindowMask Backend",
        "CFBundleExecutable": launcher.name,
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": VERSION,
        "CFBundleShortVersionString": VERSION,
        "LSMinimumSystemVersion": "13.0",
        "LSBackgroundOnly": True,
    }
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    identity = os.environ.get("MAC_CODESIGN_IDENTITY", "").strip()
    # Without a Developer ID, produce explicitly marked test artifacts only.
    if identity:
        for file in contents.rglob("*"):
            if file.is_file() and not file.is_symlink():
                magic = file.open("rb").read(4)
                if magic in (
                    b"\xcf\xfa\xed\xfe",
                    b"\xfe\xed\xfa\xcf",
                    b"\xca\xfe\xba\xbe",
                ):
                    run(
                        "codesign",
                        "--force",
                        "--timestamp",
                        "--options",
                        "runtime",
                        "--sign",
                        identity,
                        file,
                    )
    sign = ["codesign", "--force", "--sign", identity or "-"]
    if identity:
        sign += ["--timestamp", "--options", "runtime"]
    run(*sign, app)
    run("codesign", "--verify", "--deep", "--strict", app)
    # Validate the assembled launcher's architecture dispatch, not just each slice.
    run(
        launcher,
        "--self-test",
        "--report",
        work / "combined-self-test.json",
        timeout=90,
    )
    scripts = work / "scripts"
    scripts.mkdir()
    shutil.copy2(launcher, scripts / "WindowMaskStop")
    shutil.copy2(MAC / "installer/preinstall", scripts / "preinstall")
    (scripts / "preinstall").chmod(0o755)
    image_root = work / "image"
    image_root.mkdir()
    pkg = image_root / "WindowMask Backend.pkg"
    installer_identity = os.environ.get("MAC_INSTALLER_IDENTITY", "").strip()
    component_plist = work / "components.plist"
    run("pkgbuild", "--analyze", "--root", payload, component_plist)
    components = plistlib.loads(component_plist.read_bytes())
    for component in components:
        component["BundleIsRelocatable"] = False
    component_plist.write_bytes(plistlib.dumps(components))
    command = [
        "pkgbuild",
        "--root",
        payload,
        "--scripts",
        scripts,
        "--component-plist",
        component_plist,
        "--identifier",
        "com.mystic.windowmask.backend.pkg",
        "--version",
        VERSION,
        "--install-location",
        "/",
    ]
    if installer_identity:
        command += ["--sign", installer_identity]
    run(*command, pkg)
    icons()
    run("node", MAC / "package_plugin.cjs", image_root)
    ccx = image_root / "com.mystic.windowmask.mac_PS.ccx"
    with zipfile.ZipFile(ccx) as package:
        assert package.testzip() is None
        manifest = json.loads(package.read("manifest.json"))
        assert manifest["version"] == VERSION and manifest["host"]["app"] == "PS"
    shutil.copy2(MAC / "INSTALL.txt", image_root / "INSTALL.txt")
    signed = bool(identity and installer_identity)
    notarized = False
    if signed and os.environ.get("MAC_NOTARY_PROFILE"):
        run(
            "xcrun",
            "notarytool",
            "submit",
            pkg,
            "--keychain-profile",
            os.environ["MAC_NOTARY_PROFILE"],
            "--wait",
        )
        run("xcrun", "stapler", "staple", pkg)
        notarized = True
    tag = "" if notarized else "-UNNOTARIZED-TEST"
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    dmg = output / f"WindowMask-Mac-{VERSION}{tag}.dmg"
    if dmg.exists():
        dmg.unlink()
    run(
        "hdiutil",
        "create",
        "-volname",
        "WindowMask",
        "-srcfolder",
        image_root,
        "-format",
        "UDZO",
        dmg,
    )
    if notarized:
        run("codesign", "--timestamp", "--sign", identity, dmg)
        run(
            "xcrun",
            "notarytool",
            "submit",
            dmg,
            "--keychain-profile",
            os.environ["MAC_NOTARY_PROFILE"],
            "--wait",
        )
        run("xcrun", "stapler", "staple", dmg)
    checksum = hashlib.sha256(dmg.read_bytes()).hexdigest()
    (output / "SHA256SUMS-MAC.txt").write_text(
        f"{checksum}  {dmg.name}\n", encoding="ascii"
    )
    (output / "mac-build-report.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "architectures": reports,
                "signed": signed,
                "notarized": notarized,
                "photoshop_integration_tested": False,
                "minimum_macos": "13.0",
                "minimum_photoshop": "24.4.0",
                "artifact": dmg.name,
                "sha256": checksum,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(dmg)


if __name__ == "__main__":
    main()
