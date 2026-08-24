/* Browser transport for a live, model-scoped LingBot World 2 session.
 * The API key remains on the FastAPI server; this client receives only a
 * short-lived token and connects directly to Reactor for WebRTC media.
 */

const $ = (selector) => document.querySelector(selector);
const message = (text, isError = false) => {
  const target = $("#reactor-message");
  target.textContent = text;
  target.style.color = isError ? "var(--red)" : "var(--green)";
};
const state = { model: null, connected: false, started: false, pair: null, recorder: null, recordingChunks: [] };
const REACTOR_READY_TIMEOUT_MS = 120000;
const REACTOR_CONDITIONS_TIMEOUT_MS = 60000;

function setEnabled() {
  $("#start-reactor").disabled = !state.connected;
  $("#pause-reactor").disabled = !state.started;
  $("#reset-reactor").disabled = !state.connected;
  $("#finish-pair").disabled = !state.recorder || state.recorder.state === "inactive";
}

function recordingMimeType() {
  return ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"].find((type) => MediaRecorder.isTypeSupported(type));
}

function beginPairedRecording(stream) {
  if (!state.pair || state.recorder) return;
  try {
    const mimeType = recordingMimeType();
    state.recordingChunks = [];
    state.recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    state.recorder.ondataavailable = (event) => { if (event.data.size) state.recordingChunks.push(event.data); };
    state.recorder.onstop = savePairedRecording;
    state.recorder.start(1000);
    message("Paired Reactor recording started.");
    setEnabled();
  } catch (error) {
    message(`Unable to record the Reactor video: ${error.message || error}`, true);
  }
}

async function savePairedRecording() {
  const pair = state.pair;
  const recorder = state.recorder;
  state.recorder = null;
  setEnabled();
  if (!pair || !recorder || !state.recordingChunks.length) return;
  try {
    message("Saving the paired Reactor recording…");
    const blob = new Blob(state.recordingChunks, { type: recorder.mimeType || "video/webm" });
    const response = await fetch(`/api/pair-captures/${encodeURIComponent(pair.pair_id)}/recording`, {
      method: "POST", headers: { "Content-Type": blob.type }, body: blob,
    });
    const responseBody = await response.text();
    let result = {};
    try { result = responseBody ? JSON.parse(responseBody) : {}; } catch { result = { detail: responseBody }; }
    if (!response.ok) throw new Error(result.detail || `Unable to save paired recording (HTTP ${response.status}).`);
    message("Paired recording saved. Opening comparison…");
    state.pair = null;
    location.assign(result.comparison_url);
  } catch (error) {
    message(error.message || "Unable to save paired recording.", true);
  }
}

async function sdk() {
  // The official SDK is loaded as an ES module only when a live session starts.
  // Keeping it out of the dashboard bundle leaves the evidence-review page offline-safe.
  return import("https://esm.sh/@reactor-models/lingbot-world-2");
}

async function token() {
  const response = await fetch("/api/reactor/token", { method: "POST" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Unable to mint a Reactor session token.");
  return payload.jwt;
}

function event(name, callback) {
  const method = `on${name}`;
  if (typeof state.model?.[method] === "function") return state.model[method](callback);
  return undefined;
}

function waitForEvent(name, timeoutMs) {
  return new Promise((resolve, reject) => {
    let unsubscribe;
    const timeout = window.setTimeout(() => {
      if (typeof unsubscribe === "function") unsubscribe();
      reject(new Error(`Timed out waiting for Reactor ${name}.`));
    }, timeoutMs);
    unsubscribe = event(name, (payload) => {
      window.clearTimeout(timeout);
      if (typeof unsubscribe === "function") unsubscribe();
      resolve(payload);
    });
  });
}

async function waitForReactorReady() {
  const deadline = Date.now() + REACTOR_READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const status = state.model?.getStatus?.();
    if (status === "ready") return;
    if (status === "disconnected") throw new Error("Reactor disconnected before becoming ready.");
    $("#reactor-status").textContent = `Connected — ${status || "initializing"}`;
    message(`Waiting for Reactor capacity (${status || "initializing"})…`);
    await new Promise((resolve) => window.setTimeout(resolve, 1000));
  }
  throw new Error("Timed out waiting for Reactor session capacity.");
}

function installListeners() {
  event("MainVideo", (_track, stream) => {
    const video = $("#reactor-video");
    video.srcObject = stream;
    video.classList.add("active");
    $("#video-placeholder").classList.add("hidden");
    beginPairedRecording(stream);
  });
  event("ChunkComplete", (chunk) => {
    $("#chunk-status").textContent = `Chunk ${chunk?.chunk_index ?? "complete"}`;
  });
  event("GenerationStarted", () => {
    state.started = true;
    $("#reactor-status").textContent = "Generating live";
    message("World generation started.");
    setEnabled();
  });
  event("GenerationPaused", () => {
    state.started = false;
    $("#reactor-status").textContent = "Paused";
    $("#pause-reactor").textContent = "Resume";
    setEnabled();
  });
  event("GenerationResumed", () => {
    state.started = true;
    $("#reactor-status").textContent = "Generating live";
    $("#pause-reactor").textContent = "Pause";
    setEnabled();
  });
  event("GenerationReset", () => {
    state.started = false;
    $("#reactor-status").textContent = "Connected — ready";
    $("#pause-reactor").textContent = "Pause";
    $("#chunk-status").textContent = "No session";
    setEnabled();
  });
  event("CommandError", (error) => message(`Reactor rejected ${error?.command ?? "a command"}: ${error?.reason ?? "unknown reason"}`, true));
}

async function connect() {
  try {
    $("#connect-reactor").disabled = true;
    message("Minting a short-lived session token…");
    const { LingbotWorld2Model } = await sdk();
    state.model = new LingbotWorld2Model();
    installListeners();
    await state.model.connect(await token());
    await waitForReactorReady();
    state.connected = true;
    $("#reactor-status").textContent = "Connected — ready";
    message("Connected. Add a prompt and seed image, then start the world.");
  } catch (error) {
    message(error.message || "Unable to connect to Reactor.", true);
    $("#connect-reactor").disabled = false;
  }
  setEnabled();
}

async function start() {
  const image = state.pair?.image || $("#reactor-image").files[0];
  const prompt = state.pair?.prompt || $("#reactor-prompt").value.trim();
  const seed = state.pair?.seed ?? Number($("#reactor-seed").value);
  if (!image || !prompt || !Number.isInteger(seed) || seed < 0) {
    message("A prompt, an image, and a non-negative integer seed are required.", true);
    return;
  }
  try {
    $("#start-reactor").disabled = true;
    message("Uploading image and configuring the world…");
    await waitForReactorReady();
    const conditionsReady = waitForEvent("ConditionsReady", REACTOR_CONDITIONS_TIMEOUT_MS);
    await state.model.setSeed({ seed });
    const imageRef = await state.model.uploadFile(image);
    await state.model.setImage({ image: imageRef });
    await state.model.setPrompt({ prompt });
    await state.model.setRotationSpeedDeg({ rotation_speed_deg: Number($("#reactor-speed").value) });
    await conditionsReady;
    message("Seed and prompt accepted. Starting world generation…");
    await state.model.start();
  } catch (error) {
    message(error.message || "Unable to start the Reactor world.", true);
    setEnabled();
  }
}

async function loadIsaacRecordings() {
  const select = $("#paired-isaac-run");
  try {
    const response = await fetch("/api/isaac-recordings");
    if (!response.ok) throw new Error();
    const recordings = await response.json();
    select.replaceChildren();
    if (!recordings.length) {
      select.append(new Option("No indexed Isaac recordings", ""));
      select.disabled = true;
      return;
    }
    for (const recording of recordings) {
      select.append(new Option(`${recording.task} · ${recording.run_id}`, recording.run_id));
    }
  } catch {
    select.replaceChildren(new Option("Isaac recording list unavailable", ""));
    select.disabled = true;
  }
}

async function preparePair() {
  if (!state.connected) { message("Connect to Reactor before preparing a paired run.", true); return; }
  const isaacRunId = $("#paired-isaac-run").value;
  if (!isaacRunId) { message("Choose an indexed Isaac recording first.", true); return; }
  try {
    $("#prepare-pair").disabled = true;
    message("Generating a world-model prompt from the Isaac initial frame…");
    const response = await fetch("/api/pair-captures", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ isaac_run_id: isaacRunId, objective: $("#paired-objective").value.trim() }),
    });
    const pair = await response.json();
    if (!response.ok) throw new Error(pair.detail || "Unable to prepare paired run.");
    const imageResponse = await fetch(pair.seed_image_url);
    if (!imageResponse.ok) throw new Error("Unable to load the Isaac seed frame.");
    const blob = await imageResponse.blob();
    state.pair = { ...pair, image: new File([blob], "isaac_initial_frame.png", { type: blob.type || "image/png" }), seed: Number(pair.seed) || 42 };
    $("#reactor-prompt").value = pair.prompt;
    message("Starting Reactor with the Isaac seed frame and generated prompt…");
    await start();
  } catch (error) {
    state.pair = null;
    message(error.message || "Unable to prepare paired run.", true);
  } finally {
    $("#prepare-pair").disabled = false;
  }
}

async function loadPreparedPair() {
  const pairId = new URLSearchParams(location.search).get("pair_id");
  if (!pairId) return;
  try {
    message("Loading the agent-prepared Reactor comparison…");
    const response = await fetch(`/api/pair-captures/${encodeURIComponent(pairId)}`);
    const pair = await response.json();
    if (!response.ok) throw new Error(pair.detail || "Unable to load the prepared pair.");
    const imageResponse = await fetch(pair.seed_image_url);
    if (!imageResponse.ok) throw new Error("Unable to load the prepared Isaac seed frame.");
    const blob = await imageResponse.blob();
    state.pair = {
      ...pair,
      image: new File([blob], "isaac_initial_frame.png", { type: blob.type || "image/png" }),
      seed: Number(pair.seed) || 42,
    };
    $("#paired-isaac-run").value = pair.isaac_run_id;
    $("#paired-objective").value = pair.objective || "";
    $("#reactor-prompt").value = pair.prompt;
    message("Agent-prepared pair loaded. Connect, then start the world.");
  } catch (error) {
    state.pair = null;
    message(error.message || "Unable to load the prepared pair.", true);
  }
}

async function controls(changes) {
  if (!state.started) return;
  try {
    const calls = [];
    if (changes.move_longitudinal) calls.push(state.model.setMoveLongitudinal({ move_longitudinal: changes.move_longitudinal }));
    if (changes.move_lateral) calls.push(state.model.setMoveLateral({ move_lateral: changes.move_lateral }));
    if (changes.look_horizontal) calls.push(state.model.setLookHorizontal({ look_horizontal: changes.look_horizontal }));
    if (changes.look_vertical) calls.push(state.model.setLookVertical({ look_vertical: changes.look_vertical }));
    if (changes.rotation_speed_deg !== undefined) calls.push(state.model.setRotationSpeedDeg({ rotation_speed_deg: changes.rotation_speed_deg }));
    await Promise.all(calls);
  } catch (error) {
    message(error.message || "Unable to send controls.", true);
  }
}

const keyMap = {
  KeyW: ["move_longitudinal", "forward"], KeyS: ["move_longitudinal", "back"],
  KeyA: ["move_lateral", "strafe_left"], KeyD: ["move_lateral", "strafe_right"],
  ArrowLeft: ["look_horizontal", "left"], ArrowRight: ["look_horizontal", "right"],
  ArrowUp: ["look_vertical", "up"], ArrowDown: ["look_vertical", "down"],
};
const pressed = new Map();
function applyKey(code, down) {
  const entry = keyMap[code];
  if (!entry || pressed.get(code) === down) return;
  pressed.set(code, down);
  controls({ [entry[0]]: down ? entry[1] : "idle" });
}

$("#connect-reactor").onclick = connect;
$("#start-reactor").onclick = start;
$("#prepare-pair").onclick = preparePair;
$("#finish-pair").onclick = () => { if (state.recorder?.state === "recording") state.recorder.stop(); };
$("#pause-reactor").onclick = async () => {
  if (state.started) await state.model.pause(); else await state.model.resume();
};
$("#reset-reactor").onclick = () => state.model.reset();
$("#stop-reactor").onclick = () => controls({ move_longitudinal: "idle", move_lateral: "idle", look_horizontal: "idle", look_vertical: "idle" });
$("#reactor-speed").oninput = (event) => {
  $("#speed-value").textContent = `${event.target.value}°/frame`;
  controls({ rotation_speed_deg: Number(event.target.value) });
};
document.querySelectorAll("[data-control]").forEach((button) => {
  const change = { [button.dataset.control]: button.dataset.value };
  button.onpointerdown = () => { button.classList.add("active"); controls(change); };
  for (const eventName of ["pointerup", "pointerleave", "pointercancel"]) button.addEventListener(eventName, () => { button.classList.remove("active"); controls({ [button.dataset.control]: "idle" }); });
});
function isTextEntry(target) {
  return target instanceof HTMLElement && target.matches("input, textarea, select");
}
window.addEventListener("keydown", (event) => {
  if (keyMap[event.code] && !isTextEntry(event.target)) {
    event.preventDefault();
    applyKey(event.code, true);
  }
});
window.addEventListener("keyup", (event) => {
  if (!isTextEntry(event.target)) applyKey(event.code, false);
});
window.addEventListener("blur", () => { pressed.forEach((_value, code) => applyKey(code, false)); });

fetch("/api/reactor/live-config").then((response) => response.json()).then((config) => {
  if (config.enabled) { $("#reactor-status").textContent = "Ready to connect"; message("Reactor is configured on this dashboard server."); }
  else { $("#reactor-status").textContent = "REACTOR_API_KEY not configured"; message("Set REACTOR_API_KEY on the dashboard server to enable live sessions.", true); $("#connect-reactor").disabled = true; }
}).catch(() => { $("#reactor-status").textContent = "Configuration unavailable"; });

loadIsaacRecordings().then(loadPreparedPair);
