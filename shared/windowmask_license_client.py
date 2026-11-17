"""WindowMask Commercial — license client for the shared AutoHDR/WindowMask VPS.

Replaces the old `agent/` package (api.areamask.com, Ed25519 lease). The flow now
mirrors AutoHDR's key+HWID model, plus an RSA-OAEP model-key envelope so the AI
model can stay encrypted on disk and is only usable after activation:

  activate(key):
      HWID + device RSA public key  --POST /api/windowmask/activate-->  VPS
      VPS verifies key/HWID, returns model_key_envelope (+ expiry)
      -> store envelope (DPAPI), save activation state (DPAPI)

  check()/status(): best-effort online re-verify, offline fallback via stored expiry.
  deactivate(): clear local credentials (admin resets HWID server-side if needed).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import plistlib
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from model_protection import (
    clear_device_model_credentials,
    ensure_device_keypair,
    has_model_key,
    load_license_state,
    save_license_state,
    store_model_key_envelope,
)

_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "windowmask_client.json"
_DEFAULTS = {
    "server_url": "http://103.74.103.68:8080",
    "product": "windowmask",
    "request_timeout_seconds": 8,
}


class LicenseError(Exception):
    def __init__(
        self, message: str, *, code: str = "LICENSE_ERROR", status_code: int = 400
    ):
        super().__init__(message)
        self.detail = message
        self.code = code
        self.status_code = status_code


def load_config() -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    try:
        if _CONFIG_PATH.is_file():
            cfg.update(json.loads(_CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return cfg


def get_hwid() -> str:
    """Stable machine fingerprint; Mac identity survives native/Rosetta switches."""
    if platform.system() == "Darwin":
        output = subprocess.check_output(
            ["/usr/sbin/ioreg", "-a", "-r", "-d", "1", "-c", "IOPlatformExpertDevice"],
            timeout=15,
        )
        devices = plistlib.loads(output)
        identity = next(
            (
                item.get("IOPlatformUUID")
                for item in devices
                if item.get("IOPlatformUUID")
            ),
            None,
        )
        if not identity:
            raise LicenseError(
                "Không đọc được định danh máy Mac.", code="HWID_UNAVAILABLE"
            )
        return hashlib.sha256(
            ("WindowMask|Darwin|" + str(identity)).encode("utf-8")
        ).hexdigest()
    elements = []
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography"
            ) as key:
                guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                if guid:
                    elements.append(str(guid).strip())
        except Exception:
            pass
        if not elements:
            try:
                out = subprocess.check_output(
                    'powershell -NoProfile -Command "(Get-CimInstance -Class Win32_ComputerSystemProduct).UUID"',
                    shell=True,
                    text=True,
                    timeout=15,
                ).strip()
                if out:
                    elements.append(out)
            except Exception:
                pass
    elements.extend(
        [
            platform.system(),
            platform.machine(),
            platform.node(),
            os.environ.get("PROCESSOR_IDENTIFIER", ""),
        ]
    )
    raw = "WindowMask|" + "|".join(e for e in elements if e)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_config()
    url = str(cfg["server_url"]).rstrip("/") + path
    timeout = int(cfg.get("request_timeout_seconds", 8))
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(1048576)
            result = json.loads(body.decode("utf-8"))
            if not isinstance(result, dict):
                raise LicenseError(
                    "License Server trả phản hồi không hợp lệ.",
                    code="BAD_RESPONSE",
                    status_code=502,
                )
            return result
    except urllib.error.HTTPError as exc:
        try:
            detail_obj = json.loads(exc.read(65536).decode("utf-8", "replace"))
            detail = str(
                detail_obj.get("error")
                or detail_obj.get("detail")
                or f"HTTP {exc.code}"
            )
        except Exception:
            detail = f"HTTP {exc.code}"
        # 4xx (not 429) are authoritative rejections (bad key, expired, wrong HWID).
        if 400 <= exc.code < 500 and exc.code != 429:
            raise LicenseError(
                detail, code="SERVER_REJECTED", status_code=exc.code
            ) from exc
        raise LicenseError(
            f"License Server tạm thời không khả dụng ({exc.code}).",
            code="SERVER_UNAVAILABLE",
            status_code=503,
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise LicenseError(
            "Không kết nối được License Server.",
            code="SERVER_UNAVAILABLE",
            status_code=503,
        ) from exc
    except json.JSONDecodeError as exc:
        raise LicenseError(
            "License Server trả dữ liệu không đọc được.",
            code="BAD_RESPONSE",
            status_code=502,
        ) from exc


def _is_expired(expires_at: Optional[str]) -> bool:
    text = str(expires_at or "").strip().lower()
    if not text or text in ("lifetime", "none", "null", "vĩnh viễn"):
        return False
    try:
        exp = datetime.datetime.fromisoformat(str(expires_at))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=datetime.timezone.utc)
        return datetime.datetime.now(datetime.timezone.utc) > exp
    except Exception:
        return False


def _public_status(state: Dict[str, Any], mode: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "status": state.get("status", "active"),
        "mode": mode,
        "license_id": state.get("key", ""),
        "user": state.get("user", ""),
        "expires_at": state.get("expires_at") or "Lifetime",
        "product": "windowmask",
    }


class LicenseAgent:
    """Local entitlement agent for WindowMask Commercial (shared VPS)."""

    def activate(self, license_key: str) -> Dict[str, Any]:
        license_key = (license_key or "").strip()
        if not license_key:
            raise LicenseError("License Key trống.", code="INVALID_LICENSE_KEY")
        if load_license_state() is not None:
            raise LicenseError(
                "Máy này đã kích hoạt license. Hãy hủy kích hoạt trước khi đổi key.",
                code="ALREADY_ACTIVATED",
            )
        hwid = get_hwid()
        response = _post(
            "/api/windowmask/activate",
            {
                "key": license_key,
                "hwid": hwid,
                "model_public_key": ensure_device_keypair(),
            },
        )
        if not response.get("success"):
            raise LicenseError(
                str(response.get("error") or "Kích hoạt thất bại."),
                code="SERVER_REJECTED",
                status_code=403,
            )
        envelope = str(response.get("model_key_envelope", ""))
        if not envelope:
            raise LicenseError(
                "License Server chưa cấp khóa model. Liên hệ admin.",
                code="NO_MODEL_KEY",
            )
        store_model_key_envelope(envelope)
        state = {
            "key": license_key,
            "hwid": hwid,
            "user": response.get("user", ""),
            "status": response.get("status", "active"),
            "expires_at": response.get("expires_at") or "Lifetime",
            "activated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        save_license_state(state)
        return _public_status(state, "activated-online")

    def check(self) -> Dict[str, Any]:
        state = load_license_state()
        if not state:
            raise LicenseError(
                "WindowMask chưa được kích hoạt trên máy này.",
                code="NOT_ACTIVATED",
                status_code=403,
            )
        try:
            response = _post(
                "/api/windowmask/verify", {"key": state["key"], "hwid": state["hwid"]}
            )
            if not response.get("success"):
                raise LicenseError(
                    str(response.get("error") or "License bị từ chối."),
                    code="SERVER_REJECTED",
                    status_code=403,
                )
            state["status"] = response.get("status", "active")
            state["expires_at"] = (
                response.get("expires_at") or state.get("expires_at") or "Lifetime"
            )
            state["user"] = response.get("user", state.get("user", ""))
            save_license_state(state)
            return _public_status(state, "online-refreshed")
        except LicenseError as exc:
            if exc.code == "SERVER_UNAVAILABLE":
                # Offline fallback: trust the last stored expiry.
                if _is_expired(state.get("expires_at")):
                    raise LicenseError(
                        "License đã hết hạn. Kết nối Internet để gia hạn.",
                        code="EXPIRED",
                        status_code=403,
                    )
                return _public_status(state, "offline-valid")
            raise

    def deactivate(self) -> Dict[str, Any]:
        state = load_license_state()
        if not state:
            raise LicenseError(
                "Máy này chưa có license cục bộ.", code="NOT_ACTIVATED", status_code=400
            )
        clear_device_model_credentials()
        return {
            "ok": True,
            "message": "Đã hủy kích hoạt máy này. Nếu cần dùng máy khác, liên hệ admin để reset thiết bị.",
        }


def is_entitled() -> bool:
    """Fast local gate for inference: activated, not expired, and model key present."""
    state = load_license_state()
    if not state:
        return False
    if _is_expired(state.get("expires_at")):
        return False
    if str(state.get("status", "active")).lower() != "active":
        return False
    return has_model_key()
