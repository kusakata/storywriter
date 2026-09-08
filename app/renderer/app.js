const DEFAULT_MODEL =
  "hf.co/bartowski/Ateron_Gemma-4-Novelist-Eclipse-31B-GGUF:Q4_K_M";

const els = {
  pickFile: document.getElementById("pick-file"),
  fileName: document.getElementById("file-name"),
  scenes: document.getElementById("scenes"),
  scenesLabel: document.getElementById("scenes-label"),
  url: document.getElementById("url"),
  model: document.getElementById("model"),
  timeout: document.getElementById("timeout"),
  fresh: document.getElementById("fresh"),
  news: document.getElementById("news"),
  plotOnly: document.getElementById("plot-only"),
  quiet: document.getElementById("quiet"),
  noShift: document.getElementById("no-shift"),
  ending: document.getElementById("ending"),
  interactive: document.getElementById("interactive"),
  runWriter: document.getElementById("run-writer"),
  runAfterword: document.getElementById("run-afterword"),
  stop: document.getElementById("stop"),
  openOutput: document.getElementById("open-output"),
  status: document.getElementById("status"),
  log: document.getElementById("log"),
  tabs: document.getElementById("tabs"),
  viewLog: document.getElementById("view-log"),
  viewFile: document.getElementById("view-file"),
  fileView: document.getElementById("file-view"),
  askPanel: document.getElementById("ask-panel"),
  askKind: document.getElementById("ask-kind"),
  askBody: document.getElementById("ask-body"),
  askYes: document.getElementById("ask-yes"),
  askNo: document.getElementById("ask-no"),
};

const state = {
  input: "",
  running: false,
  waitingAsk: false,
  activeTab: "log",
  files: {},
};

function loadSettings() {
  try {
    const raw = localStorage.getItem("storywriter-gui");
    if (!raw) {
      return;
    }
    const data = JSON.parse(raw);
    if (data.input) {
      state.input = data.input;
      els.fileName.textContent = data.input;
    }
    if (data.scenes) {
      els.scenes.value = data.scenes;
    }
    if (data.url) {
      els.url.value = data.url;
    }
    if (data.model) {
      els.model.value = data.model;
    }
    if (data.timeout) {
      els.timeout.value = data.timeout;
    }
    els.fresh.checked = Boolean(data.fresh);
    els.news.checked = Boolean(data.news);
    els.plotOnly.checked = Boolean(data.plotOnly);
    els.quiet.checked = Boolean(data.quiet);
    els.noShift.checked = Boolean(data.noShift);
    els.ending.checked = Boolean(data.ending);
    els.interactive.checked =
      data.interactive === undefined ? true : Boolean(data.interactive);
  } catch (_err) {
    /* ignore broken local settings */
  }
}

function saveSettings() {
  localStorage.setItem(
    "storywriter-gui",
    JSON.stringify({
      input: state.input,
      scenes: els.scenes.value,
      url: els.url.value,
      model: els.model.value,
      timeout: els.timeout.value,
      fresh: els.fresh.checked,
      news: els.news.checked,
      plotOnly: els.plotOnly.checked,
      quiet: els.quiet.checked,
      noShift: els.noShift.checked,
      ending: els.ending.checked,
      interactive: els.interactive.checked,
    }),
  );
}

function options() {
  return {
    input: state.input,
    scenes: Number(els.scenes.value),
    url: els.url.value.trim(),
    model: els.model.value.trim() || DEFAULT_MODEL,
    timeout: Number(els.timeout.value),
    fresh: els.fresh.checked,
    news: els.news.checked,
    plotOnly: els.plotOnly.checked,
    quiet: els.quiet.checked,
    noShift: els.noShift.checked,
    ending: els.ending.checked,
    interactive: els.interactive.checked,
  };
}

function setStatus(message) {
  els.status.textContent = message;
}

function setRunning(running) {
  state.running = running;
  els.runWriter.disabled = running;
  els.runAfterword.disabled = running;
  els.stop.disabled = !running;
  els.pickFile.disabled = running;
}

function hideAsk() {
  state.waitingAsk = false;
  els.askPanel.classList.add("hidden");
}

function showAsk(kind, body) {
  state.waitingAsk = true;
  els.askKind.textContent = kind;
  els.askBody.textContent = body;
  els.askPanel.classList.remove("hidden");
}

function appendLog(chunk) {
  els.log.textContent += chunk;
  if (els.log.textContent.length > 200000) {
    els.log.textContent = els.log.textContent.slice(-160000);
  }
  els.log.scrollTop = els.log.scrollHeight;
}

function renderFile() {
  if (state.activeTab === "log") {
    els.viewLog.classList.remove("hidden");
    els.viewFile.classList.add("hidden");
    return;
  }
  els.viewLog.classList.add("hidden");
  els.viewFile.classList.remove("hidden");
  const text = state.files[state.activeTab];
  if (text && text.trim()) {
    els.fileView.classList.remove("empty");
    els.fileView.textContent = text;
  } else {
    els.fileView.classList.add("empty");
    els.fileView.textContent = "まだファイルがありません";
  }
}

function setTab(name) {
  state.activeTab = name;
  for (const button of els.tabs.querySelectorAll(".tab")) {
    button.classList.toggle("active", button.dataset.tab === name);
  }
  renderFile();
}

async function refreshOutputs() {
  if (!state.input) {
    return;
  }
  const result = await window.storywriter.readOutputs(state.input);
  state.files = result.files || {};
  renderFile();
}

function syncSceneLabel() {
  els.scenesLabel.textContent = els.plotOnly.checked
    ? "プロットへ追加するシーン数"
    : "シーン数";
}

async function startJob(kind) {
  if (!state.input) {
    setStatus("入力ファイルを選んでください");
    return;
  }
  saveSettings();
  hideAsk();
  if (kind === "writer" && els.fresh.checked) {
    const ok = window.confirm(
      "既存の世界観・プロット・本編を削除して最初から生成します。よろしいですか？",
    );
    if (!ok) {
      return;
    }
  }
  els.log.textContent = "";
  setTab("log");
  setRunning(true);
  const result =
    kind === "writer"
      ? await window.storywriter.runWriter(options())
      : await window.storywriter.runAfterword(options());
  if (!result.ok) {
    setRunning(false);
    setStatus(result.error || "起動に失敗しました");
    appendLog(`${result.error || "起動に失敗しました"}\n`);
  }
}

els.pickFile.addEventListener("click", async () => {
  const selected = await window.storywriter.selectInput();
  if (!selected) {
    return;
  }
  state.input = selected;
  els.fileName.textContent = selected;
  els.fileName.title = selected;
  saveSettings();
  setStatus("作業フォルダを読み込みました");
  await refreshOutputs();
});

els.plotOnly.addEventListener("change", () => {
  syncSceneLabel();
  saveSettings();
});

for (const id of [
  "scenes",
  "url",
  "model",
  "timeout",
  "fresh",
  "news",
  "quiet",
  "no-shift",
  "ending",
  "interactive",
]) {
  document.getElementById(id).addEventListener("change", saveSettings);
}

els.runWriter.addEventListener("click", () => startJob("writer"));
els.runAfterword.addEventListener("click", () => startJob("afterword"));

els.stop.addEventListener("click", async () => {
  await window.storywriter.stop();
  hideAsk();
  setRunning(false);
});

els.openOutput.addEventListener("click", async () => {
  const result = await window.storywriter.openOutput(state.input);
  if (!result.ok) {
    setStatus(result.error || "フォルダを開けませんでした");
  }
});

els.tabs.addEventListener("click", (event) => {
  const button = event.target.closest(".tab");
  if (!button) {
    return;
  }
  setTab(button.dataset.tab);
});

els.askYes.addEventListener("click", async () => {
  hideAsk();
  await window.storywriter.answer(true);
});

els.askNo.addEventListener("click", async () => {
  hideAsk();
  await window.storywriter.answer(false);
});

window.storywriter.onLog((chunk) => appendLog(chunk));
window.storywriter.onInteractive((payload) => {
  showAsk(payload.kind, payload.body);
});
window.storywriter.onStatus((payload) => {
  setStatus(payload.message || "");
  if (payload.state === "running") {
    setRunning(true);
  }
  if (
    payload.state === "idle" ||
    payload.state === "done" ||
    payload.state === "error"
  ) {
    hideAsk();
    setRunning(false);
    refreshOutputs();
  }
});
window.storywriter.onOutputs((payload) => {
  state.files = payload.files || {};
  renderFile();
});

loadSettings();
syncSceneLabel();
if (state.input) {
  refreshOutputs();
}
