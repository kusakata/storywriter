const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("storywriter", {
  selectInput: () => ipcRenderer.invoke("select-input"),
  runWriter: (options) => ipcRenderer.invoke("run-writer", options),
  runAfterword: (options) => ipcRenderer.invoke("run-afterword", options),
  stop: () => ipcRenderer.invoke("stop"),
  answer: (yes) => ipcRenderer.invoke("answer", yes),
  readOutputs: (inputPath) => ipcRenderer.invoke("read-outputs", inputPath),
  openOutput: (inputPath) => ipcRenderer.invoke("open-output", inputPath),
  onLog: (handler) => {
    const listener = (_event, chunk) => handler(chunk);
    ipcRenderer.on("log", listener);
    return () => ipcRenderer.removeListener("log", listener);
  },
  onInteractive: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("interactive", listener);
    return () => ipcRenderer.removeListener("interactive", listener);
  },
  onStatus: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("status", listener);
    return () => ipcRenderer.removeListener("status", listener);
  },
  onOutputs: (handler) => {
    const listener = (_event, payload) => handler(payload);
    ipcRenderer.on("outputs", listener);
    return () => ipcRenderer.removeListener("outputs", listener);
  },
});
