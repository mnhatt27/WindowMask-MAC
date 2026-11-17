"use strict";
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const code = fs.readFileSync(require("node:path").join(__dirname, "../uxp/index.js"), "utf8");
const tick = () => new Promise(resolve => setImmediate(resolve));

async function panel({existingMask = false, changeTarget = false} = {}) {
  const elements = {};
  for (const id of ["key", "activate", "deactivate", "connect", "create", "retry", "status"]) {
    elements[id] = {value: "", hidden: false, disabled: false, textContent: "", listeners: {},
      addEventListener(event, listener) { this.listeners[event] = listener; }};
  }
  const doc = {id: 1, width: 3, height: 2, activeLayers: [{id: 5}], activeHistoryState: {id: 10}};
  const calls = {export: 0, apply: 0, inference: 0, commits: []};
  const ps = {
    app: {documents: [doc], activeDocument: doc},
    core: {executeAsModal: async work => work({hostControl: {
      suspendHistory: async () => "history", resumeHistory: async (_, commit) => calls.commits.push(commit)
    }})},
    action: {batchPlay: async () => [{hasUserMask: existingMask}]},
    imaging: {
      getPixels: async () => { calls.export++; return {level: 0, sourceBounds: {left: 0, top: 0},
        imageData: {width: 3, height: 2, components: 3, componentSize: 8,
          getData: async () => new Uint8Array(18), dispose() {}}}; },
      createImageDataFromBuffer: async data => ({data, dispose() {}}),
      putLayerMask: async () => { calls.apply++; }
    }
  };
  let finish;
  const inference = new Promise(resolve => { finish = resolve; });
  const response = value => ({ok: true, status: 200, json: async () => value});
  const context = {
    document: {getElementById: id => elements[id]}, console, Uint8Array, setTimeout,
    require(name) {
      if (name === "photoshop") return ps;
      if (name === "uxp") return {shell: {openPath: async () => ""}};
      if (name === "./backend.js") return require("../uxp/backend.js");
      return require("../uxp/pixels.js");
    },
    fetch: async url => {
      if (url.endsWith("/health")) return response({status: "ok"});
      if (url.endsWith("/license/status")) return response({status: "active"});
      calls.inference++;
      await inference;
      if (changeTarget) doc.activeLayers = [{id: 6}];
      return {ok: true, arrayBuffer: async () => Uint8Array.from([0,1,64,128,200,255]).buffer};
    }
  };
  vm.runInNewContext(code, context);
  await tick();
  return {elements, calls, finish};
}

(async () => {
  const normal = await panel();
  const job = normal.elements.create.listeners.click();
  await tick();
  assert.equal(normal.elements.create.disabled, true);
  assert.equal(normal.elements.activate.disabled, true);
  await normal.elements.create.listeners.click();
  assert.equal(normal.calls.inference, 1);
  normal.finish(); await job;
  assert.equal(normal.calls.apply, 1);
  assert.deepEqual(normal.calls.commits, [true]);
  assert.equal(normal.elements.retry.hidden, true);
  assert.equal(normal.elements.create.disabled, false);

  const changed = await panel({changeTarget: true});
  const pending = changed.elements.create.listeners.click();
  await tick(); changed.finish(); await pending;
  assert.equal(changed.calls.apply, 0);
  assert.equal(changed.elements.retry.hidden, false);
  assert.match(changed.elements.status.textContent, /đã thay đổi/);

  const masked = await panel({existingMask: true});
  await masked.elements.create.listeners.click();
  assert.equal(masked.calls.export, 0);
  assert.equal(masked.calls.inference, 0);
  assert.match(masked.elements.status.textContent, /đã có mask/);
  console.log("Passed: busy lock, one inference per click, apply, target-change refusal and existing-mask protection.");
})().catch(error => { console.error(error); process.exitCode = 1; });
