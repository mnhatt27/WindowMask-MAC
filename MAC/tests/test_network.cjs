"use strict";
const assert = require("node:assert/strict");
const manifest = require("../uxp/manifest.json");
const {createBackend} = require("../uxp/backend.js");

(async () => {
  assert.equal(manifest.requiredPermissions.network.domains, "all");
  assert.deepEqual(manifest.requiredPermissions.launchProcess.extensions, [".app"]);
  const calls = [];
  const backend = createBackend(async (url, options) => {
    calls.push({url, options});
    return {ok: true, url, redirected: false, json: async () => ({ok: true, status: "active"})};
  });
  await backend.json("/health");
  await backend.json("/license/status");
  await backend.json("/license/activate", {license_key: "test-only"});
  await backend.json("/license/deactivate", {});
  await backend.predict(new Uint8Array(18), 3, 2);
  assert.equal(calls.length, 5);
  for (const {url, options} of calls) {
    assert.equal(new URL(url).origin, "http://127.0.0.1:8766");
    assert.equal(options.redirect, "error");
  }
  assert.equal(calls[2].options.method, "POST");
  assert.equal(calls[4].options.body.byteLength, 18);
  for (const path of ["https://example.com", "//example.com", "/health#test", "/health\\test", "/license/activate/../../test"]) {
    await assert.rejects(backend.json(path));
  }
  assert.equal(calls.length, 5);
  const redirected = createBackend(async () => ({url: "https://example.com", redirected: true}));
  await assert.rejects(redirected.json("/health"), /chuyển hướng/);
  const offline = createBackend(async () => { throw new TypeError("Network request failed"); });
  await assert.rejects(offline.json("/health"), error =>
    error.code === "BACKEND_UNAVAILABLE" && error.message.includes("Applications"));
  console.log("Passed: manifest compatibility, all local API routes, binary upload, URL rejection and redirect refusal.");
})().catch(error => {console.error(error); process.exitCode = 1;});
