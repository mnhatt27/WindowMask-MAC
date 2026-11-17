"use strict";
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const {spawnSync} = require("node:child_process");
const manifest = require("./uxp/manifest.json");
const output = path.resolve(process.argv[2] || path.join(__dirname, "dist"));
fs.mkdirSync(output, {recursive: true});
const result = spawnSync(process.execPath, [path.join(__dirname, "package_plugin.cjs"), output], {stdio: "inherit"});
if (result.status !== 0) process.exit(result.status || 1);
const source = path.join(output, manifest.id + "_PS.ccx");
const filename = `WindowMask-Mac-Panel-${manifest.version}.ccx`;
const destination = path.join(output, filename);
fs.renameSync(source, destination);
const sha256 = crypto.createHash("sha256").update(fs.readFileSync(destination)).digest("hex");
fs.writeFileSync(path.join(output, "SHA256SUMS-PANEL.txt"), `${sha256}  ${filename}\n`);
fs.writeFileSync(path.join(output, "panel-build-report.json"), JSON.stringify({version: manifest.version,
  artifact: filename, sha256, backend_compatible: "1.1.2", photoshop_integration_tested: false}, null, 2));
console.log(destination);
