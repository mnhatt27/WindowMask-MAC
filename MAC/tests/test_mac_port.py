"""Mac protocol, key storage, device fallback and shared full-size mask checks."""

import hashlib
from pathlib import Path
import plistlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

MAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAC))
from stage_runtime import stage

STAGE = stage()
sys.path.insert(0, str(STAGE))
import model_protection as credentials
import windowmask_license_client as licensing
import windowmask_server as backend
import windowmask_inference as inference
from mac_keychain import Keychain


class MemoryStore:
    def __init__(self):
        self.items = {}

    def get(self, name):
        return self.items.get(name)

    def set(self, name, value):
        self.items[name] = value

    def delete(self, name):
        self.items.pop(name, None)


class MacTests(unittest.TestCase):
    def test_existing_envelope_protocol_and_no_plaintext_files(self):
        store = MemoryStore()
        with patch.object(credentials, "_store", return_value=store):
            public = serialization.load_pem_public_key(
                credentials.ensure_device_keypair().encode()
            )
            key = bytes(range(32))
            encrypted = public.encrypt(
                key,
                padding.OAEP(
                    mgf=padding.MGF1(hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            import base64

            credentials.store_model_key_envelope(
                base64.urlsafe_b64encode(encrypted).decode()
            )
            self.assertEqual(credentials.load_model_key(), key)
            state = {"key": "test-only", "user": "Người thử", "status": "active"}
            credentials.save_license_state(state)
            self.assertEqual(credentials.load_license_state(), state)
            self.assertTrue(credentials.has_model_key())
            credentials.clear_device_model_credentials()
            self.assertEqual(store.items, {})

    def test_mac_hwid_is_independent_of_rosetta_and_host_name(self):
        data = plistlib.dumps([{"IOPlatformUUID": "test-machine-uuid"}])
        with (
            patch.object(licensing.platform, "system", return_value="Darwin"),
            patch.object(licensing.subprocess, "check_output", return_value=data),
        ):
            with (
                patch.object(licensing.platform, "machine", return_value="arm64"),
                patch.object(licensing.platform, "node", return_value="old-name"),
            ):
                native = licensing.get_hwid()
            with (
                patch.object(licensing.platform, "machine", return_value="x86_64"),
                patch.object(licensing.platform, "node", return_value="renamed-mac"),
            ):
                self.assertEqual(licensing.get_hwid(), native)
        self.assertEqual(
            native, hashlib.sha256(b"WindowMask|Darwin|test-machine-uuid").hexdigest()
        )

    def test_mac_hwid_fails_instead_of_using_an_unstable_fallback(self):
        with (
            patch.object(licensing.platform, "system", return_value="Darwin"),
            patch.object(
                licensing.subprocess, "check_output", return_value=plistlib.dumps([{}])
            ),
        ):
            with self.assertRaises(licensing.LicenseError):
                licensing.get_hwid()

    def test_coreml_failure_returns_cpu(self):
        cpu = SimpleNamespace(
            name="CPU", gpu=False, run=lambda batch: np.zeros((1, 768, 768), np.float32)
        )
        with (
            patch.object(inference, "OnnxEngine", return_value=cpu),
            patch.object(inference.importlib.metadata, "version", return_value="test"),
            patch.object(
                inference,
                "WorkerEngine",
                side_effect=RuntimeError("CoreML unavailable"),
            ),
        ):
            active, fallback, profile = inference.select_engines(b"test-model")
            self.assertIs(active, cpu)
            self.assertIs(fallback, cpu)
            self.assertIn("CoreML unavailable", profile["coreml_rejected"])

    def test_raw_mask_preserves_dimensions_and_soft_values(self):
        rgb = np.zeros((13, 17, 3), np.uint8)
        mask = np.arange(13 * 17, dtype=np.uint8).reshape(13, 17)
        model = SimpleNamespace(
            name="test",
            predict=lambda image: mask if np.array_equal(image, rgb) else None,
        )
        with (
            patch.object(licensing, "is_entitled", return_value=True),
            patch.object(backend, "model", model),
        ):
            reply = backend.app.test_client().post(
                "/predict?width=17&height=13", data=rgb.tobytes()
            )
        self.assertEqual(reply.status_code, 200)
        self.assertEqual(reply.data, mask.tobytes())
        self.assertEqual(reply.headers["X-Mask-Format"], "grayscale-u8-soft")

    def test_raw_mask_license_gate(self):
        with patch.object(licensing, "is_entitled", return_value=False):
            self.assertEqual(
                backend.app.test_client().post("/predict", data=b"test").status_code,
                403,
            )

    def test_source_payload_contains_only_encrypted_production_model(self):
        model = STAGE / "models/windowmask_onnx.model"
        with model.open("rb") as stream:
            self.assertEqual(stream.read(4), b"WMO1")
        self.assertFalse(list(STAGE.rglob("*.pth")))
        self.assertFalse(list(STAGE.rglob("*.dpapi")))

    @unittest.skipUnless(
        sys.platform == "darwin", "Native Keychain test requires macOS"
    )
    def test_native_keychain_binary_roundtrip(self):
        store = Keychain("com.mystic.windowmask.ci." + uuid.uuid4().hex)
        try:
            self.assertIsNone(store.get("test"))
            store.set("test", b"\0binary\xff\0")
            self.assertEqual(store.get("test"), b"\0binary\xff\0")
            store.set("test", b"updated")
            self.assertEqual(store.get("test"), b"updated")
            store.delete("test")
            self.assertIsNone(store.get("test"))
        finally:
            store.delete("test")

    @unittest.skipUnless(
        sys.platform == "darwin", "Native hardware ID test requires macOS"
    )
    def test_native_mac_hardware_identifier(self):
        identity = licensing.get_hwid()
        self.assertEqual(len(identity), 64)
        self.assertEqual(licensing.get_hwid(), identity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
