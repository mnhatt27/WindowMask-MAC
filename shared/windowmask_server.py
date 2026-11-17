from __future__ import annotations
import sys as _wm_stdio_sys

for _wm_stdio_stream in (_wm_stdio_sys.stdout, _wm_stdio_sys.stderr):
    if _wm_stdio_stream is not None:
        try:
            _wm_stdio_stream.reconfigure(errors="replace")
        except Exception:
            pass

import io
import os
import tempfile
import threading
import time
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image, ImageOps

from model_protection import load_model_key
from windowmask_engines import cpu_threads
from windowmask_inference import InferenceRuntime, select_engines

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "windowmask_onnx.model"
MODEL_MAGIC = b"WMO1"
MODEL_AAD = b"AreaMask WindowMask ONNX v1"
TILE_SIZE = 768
STRIDE = 576
MAX_UPLOAD_BYTES = 262144000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

LOCAL_PANEL_TOKEN_PATH = BASE_DIR / "license" / "cep_local_api_token.txt"
model = None
device_label = "CPU"
model_lock = threading.Lock()


def configure_cpu_threads() -> int:
    return cpu_threads()


@app.before_request
def enforce_commercial_license_for_inference():
    if request.method == "OPTIONS":
        return None
    if request.method == "POST" and request.path.startswith("/predict"):
        from windowmask_license_client import is_entitled

        if not is_entitled():
            return (
                jsonify(
                    {
                        "ok": False,
                        "code": "LICENSE_REQUIRED",
                        "error": "WindowMask chưa kích hoạt hoặc license đã hết hạn. Mở panel và nhập License Key.",
                    }
                ),
                403,
            )
        if model is None:
            with model_lock:
                if model is None:
                    load_model()
        return None
    return None


@app.route("/license/status", methods=["GET"])
def license_status():
    from windowmask_license_client import LicenseAgent, LicenseError

    try:
        return jsonify(LicenseAgent().check())
    except LicenseError as exc:
        return jsonify(
            {
                "ok": False,
                "status": "inactive",
                "mode": "not-activated",
                "code": exc.code,
                "error": exc.detail,
            }
        )


def _authorize_local_panel_request():
    return None


@app.route("/license/activate", methods=["POST", "OPTIONS"])
def activate_local_license():
    if request.method == "OPTIONS":
        return "", 204
    denied = _authorize_local_panel_request()
    if denied is not None:
        return denied
    payload = request.get_json(silent=True) or {}
    license_key = str(payload.get("license_key", "")).strip()
    if not license_key or len(license_key) > 128:
        return (
            jsonify(
                {
                    "ok": False,
                    "code": "INVALID_LICENSE_KEY",
                    "error": "License Key trống hoặc không hợp lệ.",
                }
            ),
            400,
        )
    try:
        from windowmask_license_client import LicenseAgent

        result = LicenseAgent().activate(license_key)
        return jsonify(result), 200
    except Exception as exc:
        status_code = int(getattr(exc, "status_code", 400))
        if status_code < 400 or status_code > 599:
            status_code = 400
        detail = str(getattr(exc, "detail", "") or str(exc))
        code = getattr(exc, "code", type(exc).__name__).upper()
        return jsonify({"ok": False, "code": code, "error": detail}), status_code


@app.route("/license/deactivate", methods=["POST", "OPTIONS"])
def deactivate_local_license():
    if request.method == "OPTIONS":
        return "", 204
    denied = _authorize_local_panel_request()
    if denied is not None:
        return denied
    try:
        from windowmask_license_client import LicenseAgent

        result = LicenseAgent().deactivate()
        return jsonify({"ok": True, "message": result.get("message", "Đã hủy kích hoạt.")}), 200
    except Exception as exc:
        status_code = int(getattr(exc, "status_code", 400))
        if status_code < 400 or status_code > 599:
            status_code = 400
        detail = str(getattr(exc, "detail", "") or str(exc))
        code = getattr(exc, "code", type(exc).__name__).upper()
        return jsonify({"ok": False, "code": code, "error": detail}), status_code


def load_model() -> None:
    global model, device_label
    encrypted = MODEL_PATH.read_bytes()
    if len(encrypted) < 32 or encrypted[:4] != MODEL_MAGIC:
        raise ValueError("Model ONNX mã hóa không hợp lệ.")
    key = bytearray(load_model_key())
    try:
        model_bytes = AESGCM(bytes(key)).decrypt(encrypted[4:16], encrypted[16:], MODEL_AAD)
    finally:
        for index in range(len(key)):
            key[index] = 0
    state = Path(os.environ.get("WINDOWMASK_RUNTIME_STATE", str(BASE_DIR / ".build/runtime-state")))
    active, cpu, profile = select_engines(model_bytes, state / "engine-profile.json")
    model = InferenceRuntime(active, cpu, profile)
    device_label = model.name
    print(f"WindowMask đã nạp model: {device_label} FP32 | CPU threads={cpu_threads()}", flush=True)


def _predict_soft_mask(rgb: np.ndarray) -> np.ndarray:
    global device_label
    if model is None:
        load_model()
    try:
        return model.predict(rgb)
    finally:
        device_label = model.name


@app.after_request
def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-WindowMask-Local-Token"
    response.headers["Access-Control-Expose-Headers"] = (
        "X-Image-Width, X-Image-Height, X-Mask-Format, X-Image-Resolution, X-Model-Device"
    )
    response.headers["Access-Control-Max-Age"] = "600"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "model": MODEL_PATH.name,
            "device": device_label,
            "tile_size": TILE_SIZE,
            "stride": STRIDE,
            "precision": "FP32",
            "engine": model.name if model is not None else None,
            "selected_layer_output": "grayscale-soft-mask-fullcanvas",
            "memory_optimized": True,
            "license_required": True,
            "license_status_endpoint": "/license/status",
        }
    )


@app.route("/predict-image", methods=["POST", "OPTIONS"])
def predict_image():
    if request.method == "OPTIONS":
        return "", 204
    try:
        raw = request.get_data(cache=False)
        if not raw:
            return jsonify({"error": "File ảnh đầu vào đang rỗng."}), 400
        try:
            source = Image.open(io.BytesIO(raw))
            rgb_image = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(rgb_image, dtype=np.uint8)
        except Exception as image_error:
            return jsonify({"error": f"Không đọc được JPG/PNG/TIFF: {image_error}"}), 400

        height, width = rgb.shape[:2]
        with model_lock:
            soft_mask = _predict_soft_mask(rgb)

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = soft_mask

        output = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", compress_level=1)
        png_bytes = output.getvalue()

        response = Response(png_bytes, mimetype="image/png")
        response.headers["X-Image-Width"] = str(width)
        response.headers["X-Image-Height"] = str(height)
        response.headers["X-Mask-Format"] = "rgba-white-alpha-soft"
        response.headers["X-Model-Device"] = str(device_label)
        return response
    except Exception as exc:
        app.logger.exception("WindowMask file inference failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict-selected-layer-gray", methods=["POST", "OPTIONS"])
def predict_selected_layer_gray():
    if request.method == "OPTIONS":
        return "", 204
    try:
        raw = request.get_data(cache=False)
        if not raw:
            return jsonify({"error": "Dữ liệu pixel của layer đang chọn bị rỗng."}), 400
        try:
            source = Image.open(io.BytesIO(raw))
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba_source = image.convert("RGBA")
                white = Image.new("RGBA", rgba_source.size, (255, 255, 255, 255))
                image = Image.alpha_composite(white, rgba_source)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as image_error:
            return jsonify({"error": f"Không đọc được ảnh xuất từ layer đang chọn: {image_error}"}), 400

        height, width = rgb.shape[:2]
        try:
            resolution = float(request.args.get("resolution", "72"))
        except (TypeError, ValueError):
            resolution = 72.0
        if not np.isfinite(resolution) or resolution <= 0 or resolution > 9600:
            resolution = 72.0

        with model_lock:
            soft_mask = _predict_soft_mask(rgb)

        output = io.BytesIO()
        Image.fromarray(soft_mask, mode="L").save(output, format="PNG", dpi=(resolution, resolution), compress_level=1)
        png_bytes = output.getvalue()

        response = Response(png_bytes, mimetype="image/png")
        response.headers["X-Image-Width"] = str(width)
        response.headers["X-Image-Height"] = str(height)
        response.headers["X-Image-Resolution"] = str(resolution)
        response.headers["X-Mask-Format"] = "grayscale-soft-mask-fullcanvas"
        response.headers["X-Model-Device"] = str(device_label)
        response.headers["X-Mask-Bytes"] = str(len(png_bytes))
        return response
    except Exception as exc:
        app.logger.exception("WindowMask grayscale selected-layer inference failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict-selected-layer-gray-file", methods=["POST", "OPTIONS"])
def predict_selected_layer_gray_file():
    if request.method == "OPTIONS":
        return "", 204
    try:
        payload = request.get_json(silent=True) or {}
        source_path_text = payload.get("source_path", "")
        if not isinstance(source_path_text, str) or not source_path_text.strip():
            return jsonify({"error": "Thiếu đường dẫn PNG tạm do Photoshop xuất."}), 400

        supplied_path = Path(source_path_text.strip())
        if not supplied_path.is_absolute():
            return jsonify({"error": "Đường dẫn PNG tạm phải là đường dẫn tuyệt đối."}), 400
        if supplied_path.suffix.lower() != ".png" or not supplied_path.name.lower().startswith("selected_layer_"):
            return jsonify({"error": "Chỉ chấp nhận PNG selected_layer_ do WindowMask tạo."}), 400

        allowed_dir = (Path(tempfile.gettempdir()).resolve() / "WindowMask_CEP_Stage2").resolve()
        try:
            source_path = supplied_path.resolve(strict=True)
        except FileNotFoundError:
            return jsonify({"error": f"Không tìm thấy PNG tạm: {supplied_path}"}), 404

        if source_path.parent != allowed_dir:
            return jsonify({"error": "PNG không nằm trong thư mục tạm WindowMask_CEP_Stage2 được phép."}), 403
        if not source_path.is_file():
            return jsonify({"error": "Đường dẫn PNG tạm không phải file."}), 400

        file_size = source_path.stat().st_size
        if file_size < 100:
            return jsonify({"error": f"PNG tạm quá nhỏ hoặc rỗng ({file_size} bytes)."}), 400
        if file_size > MAX_UPLOAD_BYTES:
            return jsonify({"error": f"PNG tạm vượt giới hạn {MAX_UPLOAD_BYTES // 1048576} MB."}), 413

        try:
            with Image.open(source_path) as source:
                source.load()
                image = ImageOps.exif_transpose(source)
                if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                    rgba_source = image.convert("RGBA")
                    white = Image.new("RGBA", rgba_source.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(white, rgba_source)
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as image_error:
            return jsonify({"error": f"Không đọc được PNG tạm từ ổ đĩa: {image_error}"}), 400

        height, width = rgb.shape[:2]
        try:
            expected_width = int(payload.get("width"))
            expected_height = int(payload.get("height"))
        except (TypeError, ValueError):
            return jsonify({"error": "Width/height do CEP gửi không hợp lệ."}), 400

        if width != expected_width or height != expected_height:
            return (
                jsonify(
                    {
                        "error": f"Kích thước PNG thực tế {width} × {height} không khớp tài liệu Photoshop {expected_width} × {expected_height}."
                    }
                ),
                400,
            )

        try:
            resolution = float(payload.get("resolution", 72.0))
        except (TypeError, ValueError):
            resolution = 72.0
        if not np.isfinite(resolution) or resolution <= 0 or resolution > 9600:
            resolution = 72.0

        print(f"[WindowMask] File-path inference started: {width}x{height} | {file_size} bytes | {device_label}")
        started = time.perf_counter()

        with model_lock:
            soft_mask = _predict_soft_mask(rgb)

        output = io.BytesIO()
        Image.fromarray(soft_mask, mode="L").save(output, format="PNG", dpi=(resolution, resolution), compress_level=1)
        png_bytes = output.getvalue()
        elapsed = time.perf_counter() - started
        print(f"[WindowMask] File-path inference completed: {width}x{height} | mask {len(png_bytes)} bytes | {elapsed:.1f}s | {device_label}")

        response = Response(png_bytes, mimetype="image/png")
        response.headers["X-Image-Width"] = str(width)
        response.headers["X-Image-Height"] = str(height)
        response.headers["X-Image-Resolution"] = str(resolution)
        response.headers["X-Mask-Format"] = "grayscale-soft-mask-fullcanvas"
        response.headers["X-Model-Device"] = str(device_label)
        response.headers["X-Mask-Bytes"] = str(len(png_bytes))
        return response
    except Exception as exc:
        app.logger.exception("WindowMask selected-layer file-path inference failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict-selected-layer", methods=["POST", "OPTIONS"])
def predict_selected_layer():
    if request.method == "OPTIONS":
        return "", 204
    try:
        raw = request.get_data(cache=False)
        if not raw:
            return jsonify({"error": "Dữ liệu pixel của layer đang chọn bị rỗng."}), 400
        try:
            source = Image.open(io.BytesIO(raw))
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba_source = image.convert("RGBA")
                white = Image.new("RGBA", rgba_source.size, (255, 255, 255, 255))
                image = Image.alpha_composite(white, rgba_source)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as image_error:
            return jsonify({"error": f"Không đọc được ảnh xuất từ layer đang chọn: {image_error}"}), 400

        height, width = rgb.shape[:2]
        try:
            resolution = float(request.args.get("resolution", "72"))
        except (TypeError, ValueError):
            resolution = 72.0
        if not np.isfinite(resolution) or resolution <= 0 or resolution > 9600:
            resolution = 72.0

        with model_lock:
            soft_mask = _predict_soft_mask(rgb)

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = soft_mask

        output = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", dpi=(resolution, resolution), compress_level=1)
        response = Response(output.getvalue(), mimetype="image/png")
        response.headers["X-Image-Width"] = str(width)
        response.headers["X-Image-Height"] = str(height)
        response.headers["X-Image-Resolution"] = str(resolution)
        response.headers["X-Mask-Format"] = "rgba-white-alpha-soft-fullcanvas"
        response.headers["X-Model-Device"] = str(device_label)
        return response
    except Exception as exc:
        app.logger.exception("WindowMask selected-layer inference failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict-path", methods=["POST", "OPTIONS"])
def predict_path():
    if request.method == "OPTIONS":
        return "", 204
    try:
        payload = request.get_json(silent=True) or {}
        source_path_text = payload.get("source_path", "")
        if not isinstance(source_path_text, str) or not source_path_text.strip():
            return jsonify({"error": "Thiếu đường dẫn ảnh đang mở trong Photoshop."}), 400

        raw_path = source_path_text.strip()
        if raw_path.lower().startswith("file:///"):
            raw_path = raw_path[8:]
            if os.name == "nt" and len(raw_path) >= 3 and raw_path[1:3] in (":/", ":\\"):
                pass
            elif raw_path.startswith("/"):
                raw_path = raw_path[1:]

        source_path = Path(raw_path.replace("/", "\\")).expanduser()
        allowed_extensions = frozenset({".jpg", ".tiff", ".jpeg", ".tif", ".png"})
        if source_path.suffix.lower() not in allowed_extensions:
            return (
                jsonify(
                    {
                        "error": "Tài liệu phải là JPG/PNG/TIFF đã lưu. Với PSD/PSB, hãy lưu một bản JPG hoặc TIFF đã gộp hiển thị rồi mở bản đó."
                    }
                ),
                400,
            )
        if not source_path.is_absolute():
            return jsonify({"error": f"Đường dẫn ảnh không phải đường dẫn tuyệt đối: {source_path}"}), 400
        if not source_path.is_file():
            return jsonify({"error": f"Không tìm thấy file ảnh đã lưu: {source_path}. Hãy lưu tài liệu Photoshop rồi thử lại."}), 404

        try:
            source = Image.open(source_path)
            rgb_image = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(rgb_image, dtype=np.uint8)
        except Exception as image_error:
            return jsonify({"error": f"Không đọc được ảnh hiện tại: {image_error}"}), 400

        height, width = rgb.shape[:2]
        try:
            resolution = float(payload.get("resolution", 72.0))
        except (TypeError, ValueError):
            resolution = 72.0
        if not np.isfinite(resolution) or resolution <= 0 or resolution > 9600:
            resolution = 72.0

        with model_lock:
            soft_mask = _predict_soft_mask(rgb)

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = soft_mask

        output = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(output, format="PNG", dpi=(resolution, resolution), compress_level=1)
        response = Response(output.getvalue(), mimetype="image/png")
        response.headers["X-Image-Width"] = str(width)
        response.headers["X-Image-Height"] = str(height)
        response.headers["X-Image-Resolution"] = str(resolution)
        response.headers["X-Mask-Format"] = "rgba-white-alpha-soft-png"
        response.headers["X-Model-Device"] = str(device_label)
        return response
    except Exception as exc:
        app.logger.exception("WindowMask active-document path inference failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204
    try:
        width = int(request.args.get("width", "0"))
        height = int(request.args.get("height", "0"))
        components = int(request.args.get("components", "3"))
        if width < 1 or height < 1:
            return jsonify({"error": "Thiếu hoặc sai width/height."}), 400
        if components != 3:
            return jsonify({"error": f"Cần 3 kênh RGB, nhận được {components}."}), 400

        raw = request.get_data(cache=False)
        expected = width * height * 3
        if len(raw) != expected:
            return jsonify({"error": f"Kích thước pixel không khớp: nhận {len(raw)} bytes, cần {expected}."}), 400

        rgb = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
        with model_lock:
            soft_mask = _predict_soft_mask(rgb)

        response = Response(soft_mask.tobytes(), mimetype="application/octet-stream")
        response.headers["X-Image-Width"] = str(width)
        response.headers["X-Image-Height"] = str(height)
        response.headers["X-Mask-Format"] = "grayscale-u8-soft"
        return response
    except Exception as exc:
        app.logger.exception("WindowMask inference failed")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    commercial_port = int(os.environ.get("WINDOWMASK_PORT", "8766"))
    try:
        load_model()
    except Exception as exc:
        print(f"Cảnh báo nạp model trước: {exc}")
    app.run(host="127.0.0.1", port=commercial_port, debug=False, threaded=True)
