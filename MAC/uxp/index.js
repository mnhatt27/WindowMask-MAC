"use strict";

const { app, core, imaging, action } = require("photoshop");
const { shell } = require("uxp");
const { fullCanvasRGB, sameTarget } = require("./pixels.js");
const { createBackend } = require("./backend.js");
const backend = createBackend((url, options) => fetch(url, options));
const jsonRequest = backend.json;
const ui = Object.fromEntries(["key", "activate", "deactivate", "connect", "create", "retry", "status"]
  .map(id => [id, document.getElementById(id)]));
let busy = false, entitled = false, ready = false, savedMask = null;

function status(message) { ui.status.textContent = message; }
function update() {
  ui.create.disabled = busy || !ready || !entitled;
  ui.retry.hidden = !savedMask;
  ui.retry.disabled = busy || !ready || !entitled;
  ui.activate.disabled = ui.deactivate.disabled = ui.connect.disabled = busy;
}
async function exclusive(work) {
  if (busy) return;
  busy = true; update();
  try { await work(); } catch (error) { status(error.message || String(error)); }
  finally { busy = false; update(); }
}
async function refreshLicense() {
  entitled = false;
  const result = await jsonRequest("/license/status");
  entitled = result.status === "active";
  return result;
}
async function connect() {
  ready = false; entitled = false;
  status("Đang kết nối backend…");
  try { await jsonRequest("/health"); }
  catch (_) {
    const error = await shell.openPath("/Applications/WindowMask Backend.app");
    if (error) throw new Error("Hãy cài WindowMask Backend.pkg trước. " + error);
  }
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const health = await jsonRequest("/health");
      if (health.status === "ok") { ready = true; break; }
    } catch (_) { /* The app may still be starting. */ }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  if (!ready) throw new Error("Backend chưa sẵn sàng. Kiểm tra WindowMaskMystic/Logs trong Library/Application Support.");
  try {
    await refreshLicense();
    status(entitled ? "Đã kết nối. Chọn layer ảnh để bắt đầu." : "Nhập License Key để kích hoạt.");
  } catch (error) { status(error.message); }
}
function currentTarget() {
  if (!app.documents.length) return null;
  const doc = app.activeDocument;
  if (doc.activeLayers.length !== 1) return null;
  return {documentID: doc.id, layerID: doc.activeLayers[0].id,
    width: Number(doc.width), height: Number(doc.height), historyID: doc.activeHistoryState.id};
}
async function checkMask(target) {
  const [descriptor] = await action.batchPlay([{
    _obj: "get", _target: [{_ref: "layer", _id: target.layerID}, {_ref: "document", _id: target.documentID}],
    _options: {dialogOptions: "dontDisplay"}
  }], {});
  if (descriptor._obj === "error" || typeof descriptor.hasUserMask !== "boolean") {
    throw new Error("Không xác minh được mask hiện tại. WindowMask chưa thay đổi layer.");
  }
  if (descriptor.hasUserMask) throw new Error("Layer đã có mask. Hãy chọn một layer chưa có mask.");
  if (descriptor.background) throw new Error("Hãy chuyển Background thành layer thường trước khi gắn mask.");
}
async function exportLayer() {
  return core.executeAsModal(async () => {
    const target = currentTarget();
    if (!target) throw new Error("Mở tài liệu và chọn đúng một layer ảnh.");
    await checkMask(target);
    const pixels = await imaging.getPixels({documentID: target.documentID, layerID: target.layerID,
      sourceBounds: {left: 0, top: 0, right: target.width, bottom: target.height},
      colorSpace: "RGB", colorProfile: "sRGB IEC61966-2.1", componentSize: 8, applyAlpha: true});
    try {
      if (pixels.level !== 0) throw new Error("Photoshop trả ảnh đã thu nhỏ. WindowMask dừng để giữ chất lượng.");
      const raw = await pixels.imageData.getData({chunky: true});
      return {target, rgb: fullCanvasRGB(raw, pixels.imageData, pixels.sourceBounds, target.width, target.height)};
    } finally { pixels.imageData.dispose(); }
  }, {commandName: "Đọc layer cho WindowMask"});
}
async function applyMask() {
  if (!savedMask) throw new Error("Chưa có mask để gắn.");
  const {target, data} = savedMask;
  await core.executeAsModal(async context => {
    if (!sameTarget(target, currentTarget())) throw new Error("Tài liệu, layer hoặc nội dung đã thay đổi. Hãy chạy AI lại.");
    await checkMask(target);
    const history = await context.hostControl.suspendHistory({documentID: target.documentID, name: "WindowMask Soft Mask"});
    let committed = false;
    try {
      const mask = await imaging.createImageDataFromBuffer(data, {width: target.width, height: target.height,
        components: 1, colorSpace: "Grayscale", colorProfile: "Gray Gamma 2.2"});
      try {
        await imaging.putLayerMask({documentID: target.documentID, layerID: target.layerID,
          imageData: mask, replace: true, targetBounds: {left: 0, top: 0}});
      } finally { mask.dispose(); }
      committed = true;
    } finally { await context.hostControl.resumeHistory(history, committed); }
  }, {commandName: "WindowMask Soft Mask"});
  savedMask = null;
  status("Đã tạo và gắn soft mask. Có thể Undo bằng Cmd+Z.");
}
ui.connect.addEventListener("click", () => exclusive(connect));
ui.activate.addEventListener("click", () => exclusive(async () => {
  const key = ui.key.value.trim();
  if (!key) throw new Error("Nhập License Key.");
  status("Đang kích hoạt…");
  await jsonRequest("/license/activate", {license_key: key});
  ui.key.value = ""; await refreshLicense(); status("Kích hoạt thành công.");
}));
ui.deactivate.addEventListener("click", () => exclusive(async () => {
  await jsonRequest("/license/deactivate", {});
  entitled = false; savedMask = null; status("Đã huỷ kích hoạt trên máy này.");
}));
ui.retry.addEventListener("click", () => exclusive(applyMask));
ui.create.addEventListener("click", () => exclusive(async () => {
  await refreshLicense();
  if (!entitled) throw new Error("Cần kích hoạt license.");
  savedMask = null;
  status("Đang đọc layer…");
  const {target, rgb} = await exportLayer();
  status("Đang nhận diện… Lần đầu có thể cần thêm thời gian chọn thiết bị.");
  const response = await backend.predict(rgb, target.width, target.height);
  if (!response.ok) {
    const error = await response.json();
    if (response.status === 403) entitled = false;
    throw new Error(error.error || "Không tạo được mask.");
  }
  const data = new Uint8Array(await response.arrayBuffer());
  if (data.length !== target.width * target.height) throw new Error("Kích thước mask trả về không khớp.");
  savedMask = {target, data};
  status("Đang gắn mask…");
  await applyMask();
}));
exclusive(connect);
