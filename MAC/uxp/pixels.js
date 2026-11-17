"use strict";

// Photoshop trims empty margins. Restore the full canvas without resizing.
function fullCanvasRGB(data, image, bounds, width, height) {
  if (image.components !== 3 || image.componentSize !== 8 ||
      !Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 ||
      width * height * 3 > 262144000 ||
      data.length !== image.width * image.height * 3 ||
      !Number.isInteger(bounds.left) || !Number.isInteger(bounds.top)) {
    throw new Error("Dữ liệu RGB không hợp lệ hoặc ảnh quá lớn.");
  }
  const result = new Uint8Array(width * height * 3);
  result.fill(255);
  const left = Math.max(0, bounds.left), top = Math.max(0, bounds.top);
  const right = Math.min(width, bounds.left + image.width);
  const bottom = Math.min(height, bounds.top + image.height);
  for (let y = top; y < bottom && right > left; y++) {
    const offset = ((y - bounds.top) * image.width + left - bounds.left) * 3;
    result.set(data.subarray(offset, offset + (right - left) * 3), (y * width + left) * 3);
  }
  return result;
}

function sameTarget(target, current) {
  return !!current && ["documentID", "layerID", "width", "height", "historyID"]
    .every(key => target[key] === current[key]);
}

module.exports = { fullCanvasRGB, sameTarget };
