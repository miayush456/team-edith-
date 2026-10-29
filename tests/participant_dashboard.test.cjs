"use strict";

// Execute the actual embedded dashboard script with native-control/DOM doubles.
// No browser, npm package, renderer, or simulator implementation is needed here.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../tools/participant_ui.py"), "utf8");
const page = source.split("PAGE = r'''")[1].split("</html>'''")[0];
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
const flush = () => new Promise(resolve => setImmediate(resolve));

class Element {
  constructor(tag = "div", attrs = {}) {
    this.tagName = tag; this.attrs = attrs; this.value = attrs.value || "";
    this.disabled = "disabled" in attrs; this.hidden = "hidden" in attrs;
    this.children = []; this.dataset = {}; this.style = {}; this.listeners = {};
    this._text = ""; this.focused = false;
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  get options() { return this.tagName === "select" ? this.children : undefined; }
  setAttribute(key, value) { this.attrs[key] = String(value); }
  removeAttribute(key) { delete this.attrs[key]; }
  getAttribute(key) { return this.attrs[key] ?? null; }
  appendChild(child) {
    this.children.push(child);
    if (this.tagName === "select" && this.children.length === 1) this.value = child.value;
    return child;
  }
  replaceChildren(...children) { this._text = ""; this.children = children; if (this.tagName === "select") this.value = ""; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  async emit(name) { await Promise.all((this.listeners[name] || []).map(callback => callback({ target: this }))); }
  focus() { this.focused = true; }
}

async function dashboard() {
  const elements = {}, calls = [], intervals = [], deferred = [];
  for (const match of page.matchAll(/<([\w-]+)\b([^>]*\bid="[^"]+"[^>]*)>/g)) {
    const attrs = {};
    for (const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) attrs[attr[1]] = attr[2] ?? "";
    elements[attrs.id] = new Element(match[1], attrs);
  }
  elements["watch-source"].value = "reference";
  elements["watch-renderer"].value = "pygame";
  let status = { state: "idle", running: false, lines: [], renderer: { state: "idle", running: false } };
  let delayStatus = false;
  const context = vm.createContext({
    document: {
      getElementById: id => { assert.ok(elements[id], `Unknown element ${id}`); return elements[id]; },
      createElement: tag => new Element(tag),
      createElementNS: (_, tag) => new Element(tag),
      createTextNode: text => { const node = new Element("#text"); node.textContent = text; return node; }
    },
    fetch: (url, options) => {
      calls.push({ url, options });
       if (url === "/api/maps") return Promise.resolve(response([{ id: "monaco", display_name: "Monaco", path: "tracks/train/monaco.json", centerline: [[0, 0], [10, 0], [10, 10]] }]));
      if (url === "/api/status" && !delayStatus) return Promise.resolve(response(status));
      return new Promise(resolve => deferred.push({ url, resolve }));
    },
    setInterval: fn => intervals.push(fn),
    alert: () => assert.fail("Dashboard errors must be inline"),
  });
  vm.runInContext(script, context, { filename: "participant-dashboard.js" });
  await flush();
  return {
    el: id => elements[id], calls,
    renderStatus() { context.render(status); },
    async input(id, value) { elements[id].value = value; await elements[id].emit("input"); },
    click: id => elements[id].emit("click"),
    setStatus: data => { status = { ...status, ...data }; },
    delayStatus: value => { delayStatus = value; },
    async poll() { intervals[0](); await flush(); },
    resolve(url, body, ok = true) { const index = deferred.findIndex(item => item.url === url); assert.ok(index >= 0, `No pending ${url}`); deferred.splice(index, 1)[0].resolve(response(body, ok)); },
  };
}

function response(body, ok = true) { return { ok, json: async () => body }; }
const runFields = ["trainer", "track-dir", "output-dir", "steps", "n-envs", "rollout-length", "seed", "ray-count", "ray-angles", "reward-progress", "reward-steer_smoothness"];
const raceFields = ["participant-a", "participant-b", "artifact-a", "artifact-b", "race-track", "race-config", "race-output", "race-port", "race-budget-ms", "race-max-steps", "race-late-policy", "race-isolate"];
function assertFrozen(h, expected) { for (const id of runFields) assert.equal(h.el(id).disabled, expected, id); }
function assertRaceFrozen(h, expected) { for (const id of raceFields) assert.equal(h.el(id).disabled, expected, id); }

test("initial budget, explicit renderer/source controls and ordered live ray preview", async () => {
  const h = await dashboard();
  assert.equal(h.el("remaining-label").textContent, "1,000,000 remaining");
  assert.equal(h.el("training-car").attrs.points, "11,0 -8,-7 -5,0 -8,7");
  assert.equal(h.el("start").disabled, false);
  assert.equal(h.el("watch").disabled, false);
  assert.equal(h.el("artifact-label").hidden, true);
  assert.equal(h.el("artifact-dir").disabled, true);
  await h.input("ray-count", "3");
  await h.input("ray-angles", "90,-90,0");
  const rays = h.el("ray-lines").children.filter(node => node.tagName === "line");
  assert.deepEqual(rays.map(ray => [ray.attrs.x2, ray.attrs.y2]), [["126.00", "270.00"], ["514.00", "270.00"], ["320.00", "76.00"]]);
  assert.match(h.el("ray-summary").children[0].textContent, /ray_0 · \+90°/);
  assert.equal(h.el("start").disabled, false);
});

test("training status compares public-map lap times and shows live speed and episode time", async () => {
  const h = await dashboard();
  h.setStatus({ metrics: { training_position: { kind: "training_position", update: 12, cars: [
    { lane: 0, track_id: "monaco", x: 2, y: 3, heading: 0.2, progress_fraction: 0.4,
      zones_reached: 6, speed_mps: 42.3, episode_time_s: 12, off_track: false, end_reason: null },
  ] }, public_validation: {
    latest_update: 12, latest_improved: false,
    latest: { finish_count: 0, track_count: 2, off_track_seconds: 4.2, tracks: [
      { track_id: "monaco", status: "dnf_crash", lap_time_s: null, mean_speed_mps: 18.25, off_track_seconds: 2.1 },
      { track_id: "spa", status: "dnf_timeout", lap_time_s: null, mean_speed_mps: 16, off_track_seconds: 2.1 },
    ] },
    best_update: 8,
    best: { finish_count: 1, track_count: 2, off_track_seconds: 1.2, tracks: [
      { track_id: "monaco", status: "finished", lap_time_s: 95.432, mean_speed_mps: 35.5, off_track_seconds: 0.4 },
      { track_id: "spa", status: "dnf_crash", lap_time_s: null, mean_speed_mps: 16, off_track_seconds: 0.8 },
    ] },
  } } });
  h.renderStatus();
  assert.match(h.el("validation-summary").textContent, /Best checkpoint · update 8: 1\/2 laps finished/);
  assert.match(h.el("validation-summary").textContent, /monaco: latest dnf_crash · no finish.*best finished · 95\.43 s lap/);
  assert.match(h.el("training-map-status").textContent, /42\.3 m\/s · 12\.0 s in episode/);
});

test("invalid rays, numeric input and static weights disable submission with accessible inline errors", async () => {
  const h = await dashboard();
  for (const [id, value] of [["ray-count", "2"], ["steps", "1.5"], ["reward-progress", "101"], ["reward-speed", ""], ["trainer", " "]]) {
    const before = h.el(id).value;
    await h.input(id, value);
    assert.equal(h.el("start").disabled, true, id);
    assert.notEqual(h.el("run-error").textContent, "");
    assert.equal(h.el("run-error").attrs.role, "alert");
    assert.match(h.el(id).attrs["aria-describedby"], /run-error/);
    await h.click("start");
    assert.equal(h.calls.filter(call => call.url === "/api/start").length, 0);
    await h.input(id, before);
    assert.equal(h.el("start").disabled, false);
  }
  await h.input("ray-count", "3");
  await h.input("ray-angles", "91,0,-90");
  assert.equal(h.el("ray-lines").children.filter(node => node.tagName === "line").length, 2);
  assert.equal(h.el("ray-angles").attrs["aria-invalid"], "true");
});

test("start is immediately single-flight and freezes all run inputs through stopping", async () => {
  const h = await dashboard();
  const first = h.click("start");
  assertFrozen(h, true);
  assert.equal(h.el("watch").disabled, true);
  await h.click("start");
  assert.equal(h.calls.filter(call => call.url === "/api/start").length, 1);
  h.setStatus({ state: "running", running: true });
  h.resolve("/api/start", { state: "starting", running: true });
  await first; await flush();
  assertFrozen(h, true);
  assert.equal(h.el("stop").disabled, false);
  h.setStatus({ state: "stopping", running: true });
  await h.poll();
  assertFrozen(h, true);
  assert.equal(h.el("stop").disabled, true);
  h.setStatus({ state: "stopped", running: false });
  await h.poll();
  assertFrozen(h, false);
  assert.equal(h.el("start").disabled, false);
});

test("failed start is inline, focused and leaves editable controls", async () => {
  const h = await dashboard();
  const action = h.click("start");
  h.resolve("/api/start", { error: "Existing run uses different rays" }, false);
  await action; await flush();
  assert.match(h.el("run-error").textContent, /different rays/);
  assert.equal(h.el("run-error").focused, true);
  assertFrozen(h, false);
});

test("watch sends selected generic source and browser, is single-flight, and exposes real links", async () => {
  const h = await dashboard();
  await h.input("watch-source", "artifact");
  await h.input("watch-renderer", "browser");
  await h.input("artifact-dir", "my-run/artifact");
  assert.equal(h.el("artifact-label").hidden, false);
  const first = h.click("watch");
  assert.equal(h.el("watch").attrs["aria-busy"], "true");
  assertFrozen(h, true);
  await h.click("watch");
  const calls = h.calls.filter(call => call.url === "/api/watch");
  assert.equal(calls.length, 1);
  const payload = JSON.parse(calls[0].options.body);
  assert.equal(payload.source, "artifact");
  assert.equal(payload.renderer, "browser");
  assert.equal(payload.artifact_dir, "my-run/artifact");
  h.resolve("/api/watch", { source_label: "Safe NumPy participant artifact", display_name: "Oval", status: "dnf_timeout", renderer: "browser", renderer_state: "ready", browser_url: "/replays/123", download_url: "/replays/123/download" });
  await first; await flush();
  const links = h.el("watch-result").children.filter(node => node.tagName === "a");
  assert.deepEqual(links.map(link => link.href), ["/replays/123", "/replays/123/download"]);
  assert.equal(links[0].target, "_blank");
  assert.equal(links[1].download, "replay.json");
  assert.match(h.el("renderer-status").textContent, /Browser fallback ready/);
  assert.equal(h.el("watch").attrs["aria-busy"], "false");
  assertFrozen(h, false);
});

test("headless Pygame is labeled a check; renderer failure retains fallback links", async () => {
  const h = await dashboard();
  const first = h.click("watch");
  h.resolve("/api/watch", { source_label: "Latest saved reference artifact", display_name: "Oval", status: "dnf_timeout", renderer: "pygame", renderer_state: "unavailable", renderer_error: "SDL display unavailable", headless: true, browser_url: "/replays/x", download_url: "/replays/x/download" });
  await first; await flush();
  assert.match(h.el("renderer-status").textContent, /Headless Pygame · one-frame check only/);
  assert.match(h.el("watch-error").textContent, /SDL display unavailable/);
  assert.equal(h.el("watch-result").children.filter(node => node.tagName === "a").length, 2);
});

test("stale status polls cannot unfreeze an in-flight watch or start", async () => {
  const h = await dashboard();
  h.delayStatus(true);
  await h.poll();
  const action = h.click("watch");
  h.resolve("/api/status", { state: "idle", running: false });
  await flush();
  assert.equal(h.el("watch").disabled, true);
  assertFrozen(h, true);
  h.delayStatus(false);
  h.resolve("/api/watch", { error: "Artifact failed validation" }, false);
  await action; await flush();
  assert.equal(h.el("watch-error").focused, true);
  assert.match(h.el("watch-error").textContent, /failed validation/);
  assert.equal(h.el("watch").disabled, false);
});

test("live heat starts as a single-flight race action and renders telemetry/resource status", async () => {
  const h = await dashboard();
  const first = h.click("race-start");
  assertRaceFrozen(h, true);
  const calls = h.calls.filter(call => call.url === "/api/race/start");
  assert.equal(calls.length, 1);
  const payload = JSON.parse(calls[0].options.body);
  assert.equal(payload.participant_a, "racer-alpha");
  assert.equal(payload.participant_b, "racer-beta");
  await h.click("race-start");
  assert.equal(h.calls.filter(call => call.url === "/api/race/start").length, 1);
  h.setStatus({ race: { state: "starting", running: true, lines: [], telemetry: {}, resources: {} } });
  h.resolve("/api/race/start", { state: "starting", running: true, lines: ["Live WebSocket endpoint: ws://127.0.0.1:8777/live/v1"] });
  await first; await flush();
  assertRaceFrozen(h, true);
  assert.equal(h.el("race-state-chip").textContent, "starting");
  h.setStatus({ race: { state: "running", running: true, lines: ["LIVE_TELEMETRY tick=4"], resources: { rss_mb: 42.5, peak_rss_mb: 48.1, pids: 3, threads: 5 }, telemetry: { entrants: [
    { label: "Alpha", status: "running", latency_ms: 3.25, model_mb: 1.5, deadline_misses: 1, inference_failures: 0 },
    { label: "Beta", status: "dnf_inference", latency_ms: 22.0, model_mb: 2.5, deadline_misses: 2, inference_failures: 1 },
  ] } } });
  await h.poll();
  assert.equal(h.el("race-latency-a").textContent, "3.25 ms");
  assert.equal(h.el("race-model-a").textContent, "1.500 MB");
  assert.equal(h.el("race-deadlines-b").textContent, "2");
  assert.match(h.el("race-resource").textContent, /42\.5 MB/);
  assert.match(h.el("race-terminal").textContent, /LIVE_TELEMETRY/);
  assert.equal(h.el("race-stop").disabled, false);
  h.setStatus({ race: { state: "stopped", running: false, lines: [], telemetry: {}, resources: {} } });
  await h.poll();
  assertRaceFrozen(h, false);
  assert.equal(h.el("race-start").disabled, false);
});

test("ONNX package exports only the model submission and leaves live inputs to the heat", async () => {
  const h = await dashboard();
  await h.input("package-output", "artifacts/live-packages/ui-test");
  const first = h.click("package");
  assert.equal(h.el("package").attrs["aria-busy"], "true");
  const calls = h.calls.filter(call => call.url === "/api/package");
  assert.equal(calls.length, 1);
  const payload = JSON.parse(calls[0].options.body);
  assert.equal(payload.package_track, undefined);
  h.resolve("/api/package", {
    package_dir: "/workspace/artifacts/live-packages/ui-test",
    submission_dir: "/workspace/artifacts/live-packages/ui-test/submission",
    manifest_path: "/workspace/artifacts/live-packages/ui-test/package_manifest.json",
    config_path: "/workspace/artifacts/live-packages/ui-test/config/simulation.json",
    participant_id: "submission-alpha", source_kind: "numpy_artifact",
  });
  await first; await flush();
  assert.match(h.el("package-result").textContent, /ONNX submission package ready/);
  assert.equal(h.el("artifact-a").value, "artifacts/onnx-imitation");
  assert.equal(h.el("race-track").value, "tracks/train/monaco.json");
  assert.equal(h.el("race-config").value, "config/simulation.json");
  assert.equal(h.el("package").attrs["aria-busy"], "false");
});
