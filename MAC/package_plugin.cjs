"use strict";
// Use Adobe's pinned headless packager, including its packaging validators.
// Full host-side manifest/API validation still requires Photoshop in Mac QA.
const path = require("node:path");
const Packager = require("./ci/node_modules/@adobe/uxp-devtools-core/src/core/client/plugin/actions/PluginPackageCommand");
const output = path.resolve(process.argv[2]);
new Packager(null, {manifest: path.join(__dirname, "uxp/manifest.json"), packageDir: output, apps: ["PS"]})
  .package().then(results => {
    if (results.length !== 1 || !results[0].success) throw new Error(results[0]?.error || "CCX packaging failed");
    console.log("Adobe headless packager created the Photoshop CCX.");
  }).catch(error => { console.error(error); process.exitCode = 1; });
