"""Optional CI-only Developer ID import. No credentials are written to the repo."""

import base64
import os
from pathlib import Path
import secrets
import subprocess


def run(*args):
    subprocess.run(args, check=True)


def main():
    encoded = os.environ.get("MAC_CERTIFICATE_P12_BASE64", "")
    if not encoded:
        if os.environ.get("MAC_CODESIGN_IDENTITY") or os.environ.get(
            "MAC_INSTALLER_IDENTITY"
        ):
            raise SystemExit(
                "Apple identities are configured but the CI certificate is missing."
            )
        print("No Developer ID configured: building UNNOTARIZED-TEST artifacts.")
        return
    root = Path(os.environ["RUNNER_TEMP"])
    certificate = root / "windowmask-ci.p12"
    keychain = root / "windowmask-ci.keychain-db"
    password = secrets.token_urlsafe(32)
    certificate.write_bytes(base64.b64decode(encoded, validate=True))
    certificate.chmod(0o600)
    try:
        run("security", "create-keychain", "-p", password, str(keychain))
        run("security", "unlock-keychain", "-p", password, str(keychain))
        run("security", "set-keychain-settings", "-lut", "21600", str(keychain))
        run(
            "security",
            "import",
            str(certificate),
            "-k",
            str(keychain),
            "-P",
            os.environ.get("MAC_CERTIFICATE_PASSWORD", ""),
            "-T",
            "/usr/bin/codesign",
            "-T",
            "/usr/bin/productsign",
            "-T",
            "/usr/bin/pkgbuild",
        )
        run(
            "security",
            "set-key-partition-list",
            "-S",
            "apple-tool:,apple:",
            "-s",
            "-k",
            password,
            str(keychain),
        )
        run("security", "list-keychains", "-d", "user", "-s", str(keychain))
        run("security", "default-keychain", "-d", "user", "-s", str(keychain))
        notary = [
            os.environ.get(name)
            for name in (
                "MAC_NOTARY_APPLE_ID",
                "MAC_NOTARY_TEAM_ID",
                "MAC_NOTARY_PASSWORD",
            )
        ]
        if any(notary) and not all(notary):
            raise SystemExit("All three Apple notarization secrets are required.")
        if all(notary):
            run(
                "xcrun",
                "notarytool",
                "store-credentials",
                "windowmask-notary",
                "--apple-id",
                notary[0],
                "--team-id",
                notary[1],
                "--password",
                notary[2],
                "--keychain",
                str(keychain),
            )
            with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as output:
                output.write("MAC_NOTARY_PROFILE=windowmask-notary\n")
    finally:
        certificate.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
