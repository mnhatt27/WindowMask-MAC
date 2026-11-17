"use strict";

// UXP's localhost domain-list matching is unreliable in affected hosts. The
// manifest uses its supported "all" form; every request here stays on this
// fixed loopback origin and a known API route. No user-provided URLs are used.
const ORIGIN = "http://127.0.0.1:8766";
const ROUTES = new Set(["/health", "/license/status", "/license/activate", "/license/deactivate", "/predict"]);

function createBackend(transport) {
  async function request(path, options = {}) {
    const route = path.split("?")[0];
    if (!ROUTES.has(route) || /[\\#\r\n]/.test(path)) {
      throw new Error("Đường dẫn backend không hợp lệ.");
    }
    const response = await transport(ORIGIN + path, {...options, redirect: "error"});
    if (response.redirected || (response.url && !response.url.startsWith(ORIGIN + "/"))) {
      throw new Error("Backend trả về địa chỉ chuyển hướng không hợp lệ.");
    }
    return response;
  }

  async function json(path, payload) {
    if (path === "/predict") throw new Error("Dùng dữ liệu RGB để tạo mask.");
    const options = payload === undefined ? {} : {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
    };
    const response = await request(path, options);
    const result = await response.json();
    if (!response.ok || result.error || result.ok === false) {
      throw new Error(result.error || "Backend từ chối yêu cầu.");
    }
    return result;
  }

  function predict(rgb, width, height) {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 ||
        rgb.byteLength !== width * height * 3) {
      throw new Error("Kích thước RGB không hợp lệ.");
    }
    const body = rgb.byteOffset === 0 && rgb.byteLength === rgb.buffer.byteLength ? rgb.buffer :
      rgb.buffer.slice(rgb.byteOffset, rgb.byteOffset + rgb.byteLength);
    return request(`/predict?width=${width}&height=${height}&components=3`, {
      method: "POST", headers: {"Content-Type": "application/octet-stream"}, body
    });
  }

  return {json, predict};
}

module.exports = {createBackend};
