# WindowMask Mac 1.1.3

The complete 1.1.3 DMG includes both native backend slices and panel 1.1.3.
Final installation tests inspect the CCX manifest inside the DMG as well as
installing, starting and stopping the backend on each architecture.

Panel hotfix **1.1.3** fixes UXP's localhost permission matching. Existing 1.1.2
backend installations can install the new CCX directly. Save documents and quit
Photoshop before updating; a manifest change requires a fresh plugin load.
The panel shows `Mac · v1.1.3` so the installed revision can be checked.

Affected UXP hosts reject the localhost/port domain list with "Manifest entry not
found". The manifest now uses the documented `network.domains: "all"` form;
`uxp/backend.js` centralizes requests to the fixed `127.0.0.1:8766` origin and
known API routes, requesting no redirects. No user-provided URL is accepted.
This is a host-compatibility workaround, not a runtime sandbox guarantee beyond
UXP's manifest permissions. Real Photoshop confirmation remains necessary.

The separate **Build WindowMask Mac Panel** workflow tests and packages the CCX
without rebuilding unchanged native runtimes. Local command:
`node MAC/build_panel.cjs MAC/dist` (after `npm ci --prefix MAC/ci --ignore-scripts`).

Native backend for Apple Silicon (`arm64`, including M-series Macs) and Intel
(`x86_64`), with a UXP Photoshop panel. Minimum target: **macOS 13**, **Photoshop
24.4**, Creative Cloud Desktop. This does not promise support for old macOS or
Photoshop releases. CEP/Rosetta is not used.

## Build

The GitHub Actions workflow builds each architecture on its matching macOS
runner, then packages both into one `WindowMask Backend.app`. A small universal
native launcher selects the physical machine's architecture. The DMG contains
the backend `.pkg`, Adobe-packaged Photoshop `.ccx`, and installation instructions.

On Windows, prepare the exact GitHub source tree with:

```powershell
.\.build\python314\python.exe MAC\prepare_repository.py
```

Push **only `.build/mac-github`** to the Mac repository. The workflow is copied
to `.github/workflows/build-macos.yml` and runs on pushes to `main` or manually.
`SOURCE_MANIFEST.json` lists the exported files and hashes. No private signing
keys, DPAPI credentials, plaintext model, Photoshop photos, Windows binaries,
training dependencies or local build directories are exported.

For a local Mac build, use Python 3.12 and the pinned dependencies, run
`python MAC/build_runtime.py` on each architecture, collect both runtime tarballs
in `artifacts`, install the build-only Node dependencies using
`npm ci --prefix MAC/ci --ignore-scripts`, and run
`python MAC/package_macos.py --artifacts artifacts`.

## Behavior

- Shared model weights, full resolution, 768/576 tiling, normalization and mask
  blending match the Windows implementation. Images stay on the local machine.
- CPU is always available. CoreML CPU/GPU is optional, guarded by the existing
  numerical mask gate, a speed check and an isolated worker with timeouts.
  Rejected/failed GPU execution falls back to CPU. CoreML batches stay at one.
  Physical GPU speed/quality still needs testing on customer Mac hardware.
- The activation API and encrypted model-key envelope are unchanged. Credentials
  live in the user's login Keychain. Mac hardware identity uses IOPlatformUUID;
  changing hostname or switching Rosetta/native does not change that identity.
- UXP reads RGB pixels directly without PNG/JPEG encoding, restores transparent
  margins on a white full-size canvas, and applies the returned soft grayscale
  pixels with Photoshop's Imaging API. Existing masks are protected. Document,
  layer, dimensions and history must still match before the mask is applied.
- Backend starts when the panel connects, without a login item or Windows-style
  service. Updating stops the old backend with a native helper. It does not
  delete photos or Keychain credentials.

## Validation and distribution

Unit tests cover the envelope protocol, license gate, raw mask contract, native
Keychain and hardware ID (on macOS), CPU fallback, pixel placement, busy state,
existing-mask protection and changed-target refusal. Each frozen architecture
and the final combined launcher run an untrained native inference smoke test.
These checks **do not test encrypted production inference or real Photoshop
integration**. A Photoshop Mac QA pass is required before sending to customers.

Adobe's pinned `uxp-devtools-core` headless packager validates and creates the
CCX. Host-side manifest/API validation requires a real Photoshop installation.
The Node dependencies are build-only and are not shipped in the app.

Without Apple Developer ID secrets the workflow produces a clearly named
`UNNOTARIZED-TEST.dmg`, not a signed customer release. Do not disable Gatekeeper.
For signed/notarized builds, configure these as **GitHub Actions secrets**:

- `MAC_CERTIFICATE_P12_BASE64`, `MAC_CERTIFICATE_PASSWORD`: P12 containing the
  Developer ID Application and Developer ID Installer identities.
- `MAC_CODESIGN_IDENTITY`, `MAC_INSTALLER_IDENTITY`: their exact identity names.
- `MAC_NOTARY_APPLE_ID`, `MAC_NOTARY_TEAM_ID`, `MAC_NOTARY_PASSWORD`: notarization
  credentials; the password must be an Apple app-specific password.

The CI imports these only into a temporary runner Keychain. No credential is
committed. `mac-build-report.json` records signing/notarization and test scope.

Developer references: [Adobe Imaging API](https://developer.adobe.com/photoshop/uxp/2022/ps-reference/media/imaging),
[Adobe packaging](https://developer.adobe.com/uxp/guides/how-to/distribution/package/),
[ONNX CoreML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).
