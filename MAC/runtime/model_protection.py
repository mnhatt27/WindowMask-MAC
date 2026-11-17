"""Mac credentials use the login Keychain; model-key envelope protocol is unchanged."""

from __future__ import annotations

import base64
from functools import lru_cache
import json

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from mac_keychain import Keychain


@lru_cache(maxsize=1)
def _store():
    return Keychain()


def _private_key():
    payload = _store().get("model-private-key")
    if payload is None:
        raise FileNotFoundError("Thiết bị chưa có khóa model. Hãy kích hoạt license.")
    key = serialization.load_pem_private_key(payload, password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("Khóa thiết bị không phải RSA.")
    return key


def ensure_device_keypair():
    if _store().get("model-private-key") is None:
        key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        payload = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        _store().set("model-private-key", payload)
    return (
        _private_key()
        .public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        .decode("ascii")
    )


def store_model_key_envelope(envelope):
    if not envelope or len(envelope) > 8192:
        raise ValueError("Model key envelope không hợp lệ.")
    encrypted = base64.urlsafe_b64decode(envelope + "=" * (-len(envelope) % 4))
    key = _private_key().decrypt(
        encrypted,
        padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None
        ),
    )
    if len(key) != 32:
        raise ValueError("Khóa model không phải AES-256.")
    _store().set("model-key", key)


def load_model_key():
    key = _store().get("model-key")
    if key is None:
        raise FileNotFoundError("Chưa nhận khóa model. Hãy kích hoạt license.")
    if len(key) != 32:
        raise ValueError("Khóa model lưu trên máy không hợp lệ.")
    return key


def has_model_key():
    return _store().get("model-key") is not None


def save_license_state(state):
    _store().set("license-state", json.dumps(state, ensure_ascii=False).encode("utf-8"))


def load_license_state():
    raw = _store().get("license-state")
    if raw is None:
        return None
    state = json.loads(raw.decode("utf-8"))
    return state if isinstance(state, dict) else None


def clear_license_state():
    _store().delete("license-state")


def clear_device_model_credentials():
    for name in ("license-state", "model-key", "model-private-key"):
        _store().delete(name)
