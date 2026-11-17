"""Export a small, reviewable GitHub tree using an explicit file allowlist."""

from pathlib import Path
import hashlib
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_runtime import BUILD, FILES, MAC, SHARED


def main():
    destination = BUILD / "mac-github"
    destination.mkdir(parents=True, exist_ok=True)
    if destination.resolve().parent != BUILD.resolve() or destination.is_symlink():
        raise ValueError("Unsafe repository staging directory")
    exported = []

    def copy(source, relative):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() in {".png", ".onnx", ".model"}:
            shutil.copy2(source, target)
        else:
            target.write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
            )
        exported.append(target)

    for name in FILES:
        copy(SHARED / name, Path("shared") / name)
    # Implementation only: no generated build trees, node_modules or secrets.
    for source in MAC.rglob("*"):
        if not source.is_file() or any(
            part in {"node_modules", "__pycache__", ".build", "dist"}
            for part in source.relative_to(MAC).parts
        ):
            continue
        if (
            source.suffix
            in {
                ".py",
                ".c",
                ".cjs",
                ".json",
                ".txt",
                ".md",
                ".spec",
                ".png",
                ".js",
                ".html",
                ".css",
                ".yml",
            }
            or source.name == "preinstall"
        ):
            copy(source, Path("MAC") / source.relative_to(MAC))
    copy(MAC / "ci/build-macos.yml", Path(".github/workflows/build-macos.yml"))
    (destination / ".gitattributes").write_text(
        "* text=auto eol=lf\n*.model binary\n*.onnx binary\n*.png binary\n",
        encoding="utf-8",
    )
    (destination / ".gitignore").write_text(
        ".build/\ndist/\nnode_modules/\n__pycache__/\n*.pyc\n*.p12\n*.pfx\n.DS_Store\n",
        encoding="utf-8",
    )
    (destination / "README.md").write_text(
        "# WindowMask for macOS\n\nSee [MAC/README.md](MAC/README.md) for build, compatibility and validation status.\n",
        encoding="utf-8",
    )
    report = {
        str(path.relative_to(destination)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in exported
    }
    (destination / "SOURCE_MANIFEST.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    forbidden = {".pth", ".p12", ".pfx", ".dpapi", ".exe"}
    for path in destination.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if (
            path.suffix.lower() in forbidden
            or "decrypted" in path.name.lower()
            or "password" in path.name.lower()
        ):
            raise ValueError(
                f"Unexpected private/build file: {path.relative_to(destination)}"
            )
        if (
            path.suffix in {".json", ".txt", ".pem"}
            and b"-----BEGIN PRIVATE KEY-----" in path.read_bytes()
        ):
            raise ValueError("Private key in export")
    print(f"Prepared {len(exported)} explicitly selected files at {destination}")


if __name__ == "__main__":
    main()
