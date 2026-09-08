#!/usr/bin/env node
const { spawn } = require("child_process");
const electron = require("electron");
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
const child = spawn(electron, ["."], {
  cwd: __dirname,
  env,
  stdio: "inherit",
});
child.on("close", (code, signal) => {
  if (code === null) {
    process.exit(1);
  }
  process.exit(code);
});
