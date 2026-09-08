const {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  shell,
} = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const OUTPUT_ROOT = path.join(PROJECT_ROOT, "output");
const ASK_MARKER = "この内容を採用しますか？ [y/n] ";
const OUTPUT_FILES = [
  "世界観.txt",
  "プロット.txt",
  "本編.txt",
  "あとがき.txt",
  "統計.txt",
  "時事.txt",
  "ベース.txt",
];

let mainWindow = null;
let child = null;
let pendingAsk = "";
let outputWatcher = null;
let outputPoll = null;

function send(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 960,
    minHeight: 700,
    title: "Storywriter",
    backgroundColor: "#f3eee6",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

function pythonCommand() {
  return process.env.STORYWRITER_PYTHON || "python3";
}

function workDirFor(inputPath) {
  return path.join(OUTPUT_ROOT, path.parse(inputPath).name);
}

function readOutputs(inputPath) {
  const dir = workDirFor(inputPath);
  const files = {};
  let exists = false;
  if (fs.existsSync(dir) && fs.statSync(dir).isDirectory()) {
    exists = true;
    for (const name of OUTPUT_FILES) {
      const filePath = path.join(dir, name);
      if (fs.existsSync(filePath)) {
        files[name] = fs.readFileSync(filePath, "utf8");
      }
    }
  }
  return { dir, exists, files };
}

function watchOutputs(inputPath) {
  stopWatch();
  const emit = () => send("outputs", readOutputs(inputPath));
  emit();
  outputPoll = setInterval(emit, 2000);
  const dir = workDirFor(inputPath);
  if (!fs.existsSync(dir)) {
    return;
  }
  try {
    outputWatcher = fs.watch(dir, { persistent: false }, emit);
  } catch (_err) {
    outputWatcher = null;
  }
}

function stopWatch() {
  if (outputPoll) {
    clearInterval(outputPoll);
    outputPoll = null;
  }
  if (outputWatcher) {
    outputWatcher.close();
    outputWatcher = null;
  }
}

function stripAnsi(text) {
  return text.replace(/\x1b\[[0-9;]*m/g, "");
}

function parseAsk(before) {
  const text = before.replace(/\r/g, "").replace(/\s+$/, "");
  const match = text.match(/===== (.+?) =====\n([\s\S]*)\n===== \1 =====$/);
  if (!match) {
    return { kind: "確認", body: text.trim() || "（内容を取得できませんでした）" };
  }
  return { kind: match[1], body: match[2].trim() || "（空）" };
}

function feedAskBuffer(chunk) {
  pendingAsk += chunk;
  while (true) {
    const index = pendingAsk.indexOf(ASK_MARKER);
    if (index === -1) {
      if (pendingAsk.length > 4000) {
        pendingAsk = pendingAsk.slice(-200);
      }
      return;
    }
    const before = pendingAsk.slice(0, index);
    pendingAsk = pendingAsk.slice(index + ASK_MARKER.length);
    send("interactive", parseAsk(before));
    send("status", { state: "ask", message: "採用する内容を確認してください" });
  }
}

function stopChild(signal = "SIGTERM") {
  if (!child) {
    return false;
  }
  const proc = child;
  child = null;
  pendingAsk = "";
  try {
    proc.kill(signal);
  } catch (_err) {
    /* already gone */
  }
  return true;
}

function spawnScript(scriptName, extraArgs, inputPath) {
  if (child) {
    return { ok: false, error: "すでに処理が実行中です。" };
  }
  if (!inputPath) {
    return { ok: false, error: "入力ファイルを選んでください。" };
  }
  const scriptPath = path.join(PROJECT_ROOT, scriptName);
  if (!fs.existsSync(scriptPath)) {
    return { ok: false, error: `${scriptName} が見つかりません。` };
  }
  pendingAsk = "";
  watchOutputs(inputPath);
  const args = ["-u", scriptPath, inputPath, ...extraArgs];
  const env = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    NO_COLOR: "1",
  };
  delete env.ELECTRON_RUN_AS_NODE;
  const proc = spawn(pythonCommand(), args, {
    cwd: PROJECT_ROOT,
    env,
    stdio: ["pipe", "pipe", "pipe"],
  });
  child = proc;
  send("status", { state: "running", message: `${scriptName} を実行しています` });

  const onData = (buf) => {
    const text = stripAnsi(buf.toString("utf8"));
    if (text) {
      send("log", text);
      feedAskBuffer(text);
    }
  };
  proc.stdout.on("data", onData);
  proc.stderr.on("data", onData);
  proc.on("error", (err) => {
    if (child === proc) {
      child = null;
    }
    send("log", `\n起動に失敗しました: ${err.message}\n`);
    send("status", { state: "error", message: err.message });
  });
  proc.on("close", (code, signalName) => {
    if (child === proc) {
      child = null;
    }
    pendingAsk = "";
    stopWatch();
    send("outputs", readOutputs(inputPath));
    if (signalName) {
      send("status", { state: "idle", message: "処理を停止しました" });
      return;
    }
    if (code === 0) {
      send("status", { state: "done", message: "完了しました" });
    } else {
      send("status", {
        state: "error",
        message: `終了コード ${code}`,
      });
    }
  });
  return { ok: true };
}

function writerArgs(options) {
  const args = [
    "-n",
    String(Math.max(1, Number(options.scenes) || 3)),
    "--url",
    options.url || "http://localhost:11434",
    "--model",
    options.model ||
      "hf.co/bartowski/Ateron_Gemma-4-Novelist-Eclipse-31B-GGUF:Q4_K_M",
    "--timeout",
    String(Math.max(1, Number(options.timeout) || 1800)),
  ];
  if (options.fresh) {
    args.push("--fresh");
  }
  if (options.news) {
    args.push("--news");
  }
  if (options.plotOnly) {
    args.push("--plot-only");
  }
  if (options.quiet) {
    args.push("-q");
  }
  if (options.noShift) {
    args.push("--no-shift");
  }
  if (options.ending) {
    args.push("--ending");
  }
  if (options.interactive) {
    args.push("-i");
  }
  return args;
}

function afterwordArgs(options) {
  const args = [
    "--url",
    options.url || "http://localhost:11434",
    "--model",
    options.model ||
      "hf.co/bartowski/Ateron_Gemma-4-Novelist-Eclipse-31B-GGUF:Q4_K_M",
    "--timeout",
    String(Math.max(1, Number(options.timeout) || 1800)),
  ];
  if (options.quiet) {
    args.push("-q");
  }
  return args;
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  stopChild("SIGTERM");
  stopWatch();
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  stopChild("SIGTERM");
  stopWatch();
});

ipcMain.handle("select-input", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "小説テキストを選択",
    defaultPath: PROJECT_ROOT,
    properties: ["openFile"],
    filters: [
      { name: "テキスト", extensions: ["txt"] },
      { name: "すべてのファイル", extensions: ["*"] },
    ],
  });
  if (result.canceled || !result.filePaths[0]) {
    return null;
  }
  return result.filePaths[0];
});

ipcMain.handle("run-writer", (_event, options) => {
  return spawnScript(
    "storywriter.py",
    writerArgs(options || {}),
    options && options.input,
  );
});

ipcMain.handle("run-afterword", (_event, options) => {
  return spawnScript(
    "afterword.py",
    afterwordArgs(options || {}),
    options && options.input,
  );
});

ipcMain.handle("stop", () => {
  const stopped = stopChild("SIGTERM");
  if (stopped) {
    send("log", "\n処理を停止しました。\n");
    send("status", { state: "idle", message: "処理を停止しました" });
  }
  return { ok: stopped };
});

ipcMain.handle("answer", (_event, yes) => {
  if (!child || !child.stdin.writable) {
    return { ok: false, error: "確認待ちの処理がありません。" };
  }
  child.stdin.write(yes ? "y\n" : "n\n");
  send("status", {
    state: "running",
    message: yes ? "採用して続行します" : "不採用のため再生成します",
  });
  return { ok: true };
});

ipcMain.handle("read-outputs", (_event, inputPath) => {
  if (!inputPath) {
    return { dir: "", exists: false, files: {} };
  }
  return readOutputs(inputPath);
});

ipcMain.handle("open-output", async (_event, inputPath) => {
  if (!inputPath) {
    return { ok: false, error: "入力ファイルを選んでください。" };
  }
  const dir = workDirFor(inputPath);
  if (!fs.existsSync(dir)) {
    return { ok: false, error: "作業フォルダがまだありません。" };
  }
  const result = await shell.openPath(dir);
  if (result) {
    return { ok: false, error: result };
  }
  return { ok: true };
});
