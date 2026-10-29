"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const replay = require("../racing_event/assets/replay.js");
const html = fs.readFileSync(path.join(__dirname, "../racing_event/assets/replay.html"), "utf8");

function fixture() {
  const hash = "a".repeat(64);
  return {
    schema_version: "1.1", kind: "racing_event_replay",
    engine: { code_hash: hash, config_hash: hash },
    source: { mode: "evaluation", participant_id: "driver-01" },
    tick_hz: 30, car: { length_m: 4.8, width_m: 2 },
    track: {
      version: "1.0",
      centerline: [[0, 0], [40, 0], [40, 30], [0, 30]],
      left_boundary: [[3, 3], [37, 3], [37, 27], [3, 27]],
      right_boundary: [[-3, -3], [43, -3], [43, 33], [-3, 33]],
      barrier_left: [[5, 5], [35, 5], [35, 25], [5, 25]],
      barrier_right: [[-5, -5], [45, -5], [45, 35], [-5, 35]],
      s: [0, 40, 70, 110], heading: [0, Math.PI / 2, Math.PI, -Math.PI / 2],
      curvature: [0, 0, 0, 0], half_width_left: [3, 3, 3, 3], half_width_right: [3, 3, 3, 3],
      L: 140, runoff_width: 2, start_line_s: 0,
      meta: { id: "test-loop", display_name: "Test Loop", content_hash: hash }
    },
    frames: Array.from({ length: 4 }, function (_, tick) {
      return {
        tick, x: 5 + tick * 3, y: tick,
        heading: [179 * Math.PI / 180, -179 * Math.PI / 180, -2.9, 7.5][tick],
        speed: 10 + tick * 2, lateral_velocity: tick * 0.2, yaw_rate: tick * 0.1,
        steer: tick * 0.05, steering_command: tick * 0.1, throttle_brake: tick === 3 ? -0.5 : 0.75,
        off_track: tick > 0, events: tick === 1 ? ["off_track_start"] : tick === 3 ? ["barrier_contact"] : [],
        terminated: tick === 3, truncated: false
      };
    }),
    result: { status: "dnf_crash", steps: 3, lap_time_s: null, elapsed_s: 0.1, trajectory_hash: hash },
    payload_hash: hash
  };
}

function near(actual, expected, tolerance = 1e-10) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} should be close to ${expected}`);
}

function freezeDeep(value) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freezeDeep);
    Object.freeze(value);
  }
  return value;
}

test("template retains exactly the parent's three bundling markers in order", function () {
  for (const marker of ["/*__REPLAY_CSS__*/", "/*__REPLAY_JS__*/", "<!--__REPLAY_DATA__-->"]) {
    assert.equal(html.split(marker).length - 1, 1);
  }
  assert.match(html, /<style>\/\*__REPLAY_CSS__\*\/<\/style>/);
  assert.match(html, /<script>\/\*__REPLAY_JS__\*\/<\/script>/);
  assert.ok(html.indexOf("<!--__REPLAY_DATA__-->") < html.indexOf("/*__REPLAY_JS__*/"));
  assert.doesNotMatch(html, /\b(?:src|href)\s*=/i);
});

test("valid replay is accepted without copying or mutating data", function () {
  const data = freezeDeep(fixture());
  const before = JSON.stringify(data);
  assert.equal(replay.validateReplay(data), data);
  for (const time of [-10, 0, 1 / 60, 1 / 30, 0.09, 0.1, 100]) replay.sampleAtTime(data, time);
  replay.replayBounds(data);
  assert.equal(JSON.stringify(data), before);
});

test("event index preserves recorded order and exact tick times", function () {
  const data = fixture();
  data.frames[0].events = ["off_track_start", "off_track_end"];
  const events = replay.buildEventIndex(data);
  assert.deepEqual(events, [
    { tick: 0, time_s: 0, events: ["off_track_start", "off_track_end"], label: "Off track · Back on track" },
    { tick: 1, time_s: 1 / 30, events: ["off_track_start"], label: "Off track" },
    { tick: 3, time_s: 0.1, events: ["barrier_contact"], label: "Barrier contact" }
  ]);
});

const invalidCases = [
  ["missing version", d => { delete d.schema_version; }],
  ["old version", d => { d.schema_version = "1.0"; }],
  ["future version", d => { d.schema_version = "1.2"; }],
  ["wrong kind", d => { d.kind = "other"; }],
  ["zero tick rate", d => { d.tick_hz = 0; }],
  ["nonfinite tick rate", d => { d.tick_hz = Infinity; }],
  ["string tick rate", d => { d.tick_hz = "30"; }],
  ["overflowing duration", d => { d.tick_hz = Number.MIN_VALUE; }],
  ["missing engine", d => { delete d.engine; }],
  ["invalid mode", d => { d.source.mode = "live"; }],
  ["invalid participant", d => { d.source.participant_id = {}; }],
  ["invalid car size", d => { d.car.width_m = -2; }],
  ["wrong track version", d => { d.track.version = "2.0"; }],
  ["too few track points", d => { d.track.centerline = [[0, 0]]; }],
  ["mismatched boundary", d => { d.track.left_boundary.pop(); }],
  ["bad coordinate shape", d => { d.track.barrier_left[0] = [1]; }],
  ["nonfinite track coordinate", d => { d.track.right_boundary[0][1] = NaN; }],
  ["missing track scalar", d => { delete d.track.s; }],
  ["nonfinite track scalar", d => { d.track.heading[0] = Infinity; }],
  ["invalid track size", d => { d.track.L = -1; }],
  ["invalid track metadata", d => { d.track.meta.display_name = []; }],
  ["empty frames", d => { d.frames = []; }],
  ["nonzero initial tick", d => { d.frames[0].tick = 1; }],
  ["skipped tick", d => { d.frames[1].tick = 2; }],
  ["nonfinite frame coordinate", d => { d.frames[1].x = Infinity; }],
  ["missing recorded speed", d => { delete d.frames[1].speed; }],
  ["string steering", d => { d.frames[1].steer = "0.1"; }],
  ["invalid state flag", d => { d.frames[0].off_track = 0; }],
  ["invalid event", d => { d.frames[0].events = [{}]; }],
  ["unrecognized result", d => { d.result.status = "won"; }],
  ["prototype result name", d => { d.result.status = "toString"; }],
  ["mismatched final tick", d => { d.result.steps = 4; }],
  ["nonfinite reported lap", d => { d.result.lap_time_s = Infinity; }],
  ["missing reported lap", d => { delete d.result.lap_time_s; }],
  ["negative recorded elapsed", d => { d.result.elapsed_s = -1; }],
  ["missing payload metadata", d => { delete d.payload_hash; }],
  ["overflowing coordinate span", d => { d.frames[0].x = -Number.MAX_VALUE; d.frames[1].x = Number.MAX_VALUE; }]
];
for (const [name, change] of invalidCases) {
  test("rejects malformed replay: " + name, function () {
    const data = fixture();
    change(data);
    assert.throws(() => replay.validateReplay(data), Error);
  });
}

test("null, arrays, and missing recordings are rejected", function () {
  for (const value of [undefined, null, [], "replay"]) assert.throws(() => replay.validateReplay(value));
});

test("browser shape checks do not authenticate hashes or infer a result", function () {
  const data = fixture();
  data.payload_hash = "Python already validates this";
  data.result.trajectory_hash = "not checked in the browser";
  data.result.status = "finished";
  data.result.lap_time_s = 7.125;
  data.result.elapsed_s = 9.75;
  assert.equal(replay.validateReplay(data), data);
  assert.equal(data.frames.at(-1).events[0], "barrier_contact");
  assert.equal(data.result.status, "finished");
});

test("interpolates only pose and follows shortest heading across the wrap", function () {
  const data = fixture();
  const sample = replay.sampleAtTime(data, 1 / 60);
  near(sample.x, 6.5);
  near(sample.y, 0.5);
  near(sample.heading, Math.PI);
  assert.equal(sample.tick, 0);
  near(sample.alpha, 0.5);
  assert.equal(sample.frame, data.frames[0]);
  assert.equal(sample.frame.speed, 10);
  assert.equal(sample.frame.off_track, false);
  assert.deepEqual(sample.frame.events, []);
  assert.equal(sample.frame.terminated, false);
  const completed = replay.sampleAtTime(data, 1 / 30);
  assert.equal(completed.frame, data.frames[1]);
  assert.equal(completed.frame.off_track, true);
  assert.deepEqual(completed.frame.events, ["off_track_start"]);
});

test("reduced motion snaps only the visual pose and keeps completed telemetry exact", function () {
  const data = fixture();
  const normal = replay.sampleAtTime(data, 1 / 60);
  const reduced = replay.sampleAtTime(data, 1 / 60, { reduced_motion: true });
  assert.equal(reduced.tick, normal.tick);
  assert.equal(reduced.frame, normal.frame);
  assert.equal(reduced.alpha, 0);
  assert.equal(reduced.x, data.frames[0].x);
  assert.equal(reduced.y, data.frames[0].y);
  assert.equal(reduced.heading, data.frames[0].heading);
});

test("clamps and preserves all exact stored endpoints, including unwrapped headings", function () {
  const data = fixture();
  for (const time of [-100, 0]) {
    const sample = replay.sampleAtTime(data, time);
    assert.equal(sample.frame, data.frames[0]);
    assert.equal(sample.x, data.frames[0].x);
    assert.equal(sample.y, data.frames[0].y);
    assert.equal(sample.heading, data.frames[0].heading);
  }
  for (let tick = 0; tick < data.frames.length; tick += 1) {
    const sample = replay.sampleAtTime(data, tick / data.tick_hz);
    assert.equal(sample.tick, tick);
    assert.equal(sample.alpha, 0);
    assert.equal(sample.heading, data.frames[tick].heading);
  }
  for (const time of [0.1, 0.100001, 1000]) {
    const sample = replay.sampleAtTime(data, time);
    assert.equal(sample.frame, data.frames[3]);
    assert.equal(sample.x, 14);
    assert.equal(sample.y, 3);
    assert.equal(sample.heading, 7.5);
    assert.equal(sample.time_s, 0.1);
    assert.equal(sample.alpha, 0);
  }
});

test("heading interpolation handles both wrap directions and exact endpoint clamps", function () {
  const a = 179 * Math.PI / 180;
  const b = -a;
  near(replay.interpolateHeading(a, b, 0.5), Math.PI);
  near(replay.interpolateHeading(b, a, 0.5), -Math.PI);
  near(replay.interpolateHeading(0, Math.PI / 2, 0.5), Math.PI / 4);
  assert.equal(replay.interpolateHeading(7.5, -7.5, 0), 7.5);
  assert.equal(replay.interpolateHeading(7.5, -7.5, 1), -7.5);
  assert.equal(replay.interpolateHeading(7.5, -7.5, -1), 7.5);
  assert.equal(replay.interpolateHeading(7.5, -7.5, 2), -7.5);
});

test("tick timing survives round trips and does not advance discrete data early", function () {
  for (const hz of [30, 60, 29.97]) {
    for (let tick = 0; tick <= 1000; tick += 1) {
      const timing = replay.timingAtTime(tick / hz, hz, 1000);
      assert.equal(timing.tick, tick);
      assert.equal(timing.alpha, 0);
    }
  }
  const before = replay.timingAtTime(1 / 30 - 1e-8, 30, 3);
  assert.equal(before.tick, 0);
  assert.ok(before.alpha > 0.99);
  assert.equal(replay.timingAtTime(-1, 30, 3).tick, 0);
  assert.equal(replay.timingAtTime(1, 30, 3).tick, 3);
  for (const time of [NaN, Infinity, "0"]) assert.throws(() => replay.timingAtTime(time, 30, 3));
});

test("a recording with only its initial frame has a stable exact pose", function () {
  const data = fixture();
  data.frames = [data.frames[0]];
  data.result.steps = 0;
  replay.validateReplay(data);
  for (const time of [-1, 0, 1]) assert.equal(replay.sampleAtTime(data, time).frame, data.frames[0]);
  const state = replay.transitionPlayback(replay.createPlayback(30, 0), { type: "play" }, 0);
  assert.equal(state.playing, false);
  assert.equal(state.time_s, 0);
});

test("playback clock rebases on seek, speed change, pause, and resume", function () {
  const initial = freezeDeep(replay.createPlayback(30, 300));
  let state = replay.transitionPlayback(initial, { type: "play" }, 1000);
  assert.equal(initial.playing, false);
  near(replay.playbackTime(state, 2000), 1);
  state = replay.transitionPlayback(state, { type: "seek", time_s: 4 }, 2000);
  assert.equal(state.playing, true);
  near(replay.playbackTime(state, 2500), 4.5);
  state = replay.transitionPlayback(state, { type: "speed", speed: 2 }, 2500);
  near(state.time_s, 4.5);
  near(replay.playbackTime(state, 3000), 5.5);
  state = replay.transitionPlayback(state, { type: "pause" }, 3000);
  near(replay.playbackTime(state, 5000), 5.5);
  state = replay.transitionPlayback(state, { type: "toggle" }, 5000);
  near(replay.playbackTime(state, 5500), 6.5);
  state = replay.transitionPlayback(state, { type: "toggle" }, 5500);
  assert.equal(state.playing, false);
  near(state.time_s, 6.5);
});

test("playback stops exactly at end without looping and Play explicitly replays", function () {
  let state = replay.transitionPlayback(replay.createPlayback(30, 300), { type: "play" }, 0);
  state = replay.transitionPlayback(state, { type: "sync" }, 15000);
  assert.equal(state.time_s, 10);
  assert.equal(state.playing, false);
  state = replay.transitionPlayback(state, { type: "sync" }, 20000);
  assert.equal(state.time_s, 10);
  state = replay.transitionPlayback(state, { type: "play" }, 20000);
  assert.equal(state.time_s, 0);
  assert.equal(state.playing, true);
  state = replay.transitionPlayback(state, { type: "seek", time_s: 100 }, 20000);
  assert.equal(state.time_s, 10);
  assert.equal(state.playing, false);
});

test("Pause at the end before the next RAF stays on the final frame", function () {
  const playing = replay.transitionPlayback(replay.createPlayback(30, 3), { type: "play" }, 0);
  const paused = replay.transitionPlayback(playing, { type: "toggle" }, 101);
  assert.equal(paused.time_s, 0.1);
  assert.equal(paused.playing, false);
});

test("animation sync preserves the wall-clock anchor until an explicit control change", function () {
  const playing = replay.transitionPlayback(replay.createPlayback(30, 300), { type: "play" }, 1000);
  const synced = replay.transitionPlayback(playing, { type: "sync" }, 2000);
  assert.equal(synced.time_s, 1);
  assert.equal(synced.anchor_ms, 1000);
  assert.equal(synced.anchor_time_s, 0);
  near(replay.playbackTime(synced, 2500), 1.5);
  const sought = replay.transitionPlayback(synced, { type: "seek", time_s: 4 }, 2500);
  assert.equal(sought.anchor_ms, 2500);
  assert.equal(sought.anchor_time_s, 4);
});

test("next tick pauses at the next complete tick and resets from the end", function () {
  let state = replay.transitionPlayback(replay.createPlayback(30, 3), { type: "play" }, 0);
  state = replay.transitionPlayback(state, { type: "next" }, 15);
  near(state.time_s, 1 / 30);
  assert.equal(state.playing, false);
  state = replay.transitionPlayback(state, { type: "next" }, 30);
  near(state.time_s, 2 / 30);
  state = replay.transitionPlayback(state, { type: "next" }, 40);
  assert.equal(state.time_s, 0.1);
  state = replay.transitionPlayback(state, { type: "next" }, 50);
  assert.equal(state.time_s, 0);
  state = replay.transitionPlayback(state, { type: "play" }, 50);
  state = replay.transitionPlayback(state, { type: "restart" }, 75);
  assert.equal(state.time_s, 0);
  assert.equal(state.playing, false);
});

test("playback controls reject invalid inputs and clamp seeking to the recording", function () {
  const state = freezeDeep(replay.createPlayback(30, 300));
  assert.equal(replay.transitionPlayback(state, { type: "seek", time_s: -1 }, 0).time_s, 0);
  assert.equal(replay.transitionPlayback(state, { type: "seek", time_s: 100 }, 0).time_s, 10);
  for (const speed of [0, -1, Infinity, NaN, "2"]) assert.throws(() => replay.transitionPlayback(state, { type: "speed", speed }, 0));
  assert.throws(() => replay.transitionPlayback(state, { type: "seek", time_s: NaN }, 0));
  assert.throws(() => replay.transitionPlayback(state, { type: "unknown" }, 0));
  assert.throws(() => replay.createPlayback(0, 3));
  assert.throws(() => replay.createPlayback(30, -1));
  assert.equal(state.time_s, 0);
});

test("exact tick playback action pauses at the requested event frame", function () {
  let state = replay.transitionPlayback(replay.createPlayback(30, 3), { type: "play" }, 0);
  state = replay.transitionPlayback(state, { type: "tick", tick: 2 }, 100);
  assert.equal(state.time_s, 2 / 30);
  assert.equal(state.playing, false);
  assert.throws(() => replay.transitionPlayback(state, { type: "tick", tick: 4 }, 100));
});

test("track fit preserves aspect ratio, fits every sample, and inverts screen Y", function () {
  const data = fixture();
  data.frames[3].x = 80;
  const bounds = replay.replayBounds(data);
  assert.ok(bounds.maxX > 80, "fit includes the car footprint outside the track");
  for (const [width, height] of [[900, 500], [320, 500], [500, 160]]) {
    const fit = replay.fitViewport(bounds, width, height, 20);
    const origin = replay.projectPoint(fit, 0, 0);
    const east = replay.projectPoint(fit, 1, 0);
    const north = replay.projectPoint(fit, 0, 1);
    near(east.x - origin.x, origin.y - north.y);
    assert.ok(north.y < origin.y);
    for (const name of ["centerline", "left_boundary", "right_boundary", "barrier_left", "barrier_right"]) {
      for (const [x, y] of data.track[name]) {
        const point = replay.projectPoint(fit, x, y);
        assert.ok(point.x >= 20 && point.x <= width - 20);
        assert.ok(point.y >= 20 && point.y <= height - 20);
      }
    }
    const car = replay.projectPoint(fit, 80, 3);
    assert.ok(car.x <= width - 20);
  }
});

test("simulation-clock formatting carries across seconds and minutes", function () {
  assert.equal(replay.formatTime(0), "00:00.000");
  assert.equal(replay.formatTime(1 / 30), "00:00.033");
  assert.equal(replay.formatTime(59.9996), "01:00.000");
  assert.equal(replay.formatTime(125.125), "02:05.125");
});

// Small DOM/canvas boundary double: tests run real validation, drawing and
// control handlers with a controlled RAF clock. No browser or packages needed.
function harness(embedded, reducedMotion = false) {
  const elements = new Map();
  const backgrounds = [];
  const controlIds = ["show-centerline", "play-pause", "restart", "next-tick", "scrubber", "playback-speed", "previous-event", "next-event"];
  function makeContext() {
    const context = { calls: [] };
    for (const method of ["setTransform", "clearRect", "beginPath", "moveTo", "lineTo", "closePath", "setLineDash", "stroke", "fill", "save", "translate", "arc", "rotate", "fillRect", "strokeRect", "restore", "drawImage"]) {
      context[method] = (...args) => context.calls.push([method, ...args]);
    }
    return context;
  }
  function makeElement(id, tagName = "div") {
    const handlers = new Map();
    return {
      id, tagName: tagName.toUpperCase(), textContent: "", dataset: {}, style: {}, attributes: {}, value: "", disabled: false, hidden: false, checked: true, children: [],
      className: "", title: "", type: "", isContentEditable: false,
      set innerHTML(_) { throw new Error("Source strings must never be inserted as HTML"); },
      appendChild(child) { this.children.push(child); return child; },
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener(type, handler) {
        if (!handlers.has(type)) handlers.set(type, []);
        handlers.get(type).push(handler);
      },
      emit(type, extra = {}) {
        const event = Object.assign({ target: this, preventDefault() {} }, extra);
        for (const handler of handlers.get(type) || []) handler(event);
      },
      getBoundingClientRect() { return { width: 960, height: 500 }; }
    };
  }
  for (const match of html.matchAll(/\bid="([^"]+)"/g)) elements.set(match[1], makeElement(match[1]));
  for (const id of controlIds) elements.get(id).disabled = true;
  elements.get("replay-error").hidden = true;
  if (embedded !== undefined) {
    const dataNode = makeElement("replay-data");
    dataNode.textContent = typeof embedded === "string" ? embedded : JSON.stringify(embedded);
    elements.set("replay-data", dataNode);
  }
  const context = makeContext();
  elements.get("track-canvas").getContext = () => context;
  const documentHandlers = new Map();
  const document = {
    getElementById: id => elements.get(id) || null,
    querySelectorAll: () => controlIds.map(id => elements.get(id)),
    createElement(tag) {
      if (tag === "canvas") {
        const canvas = makeElement("", "canvas");
        const ctx = makeContext();
        canvas.getContext = () => ctx;
        backgrounds.push(ctx);
        return canvas;
      }
      return makeElement("", tag);
    },
    addEventListener(type, handler) {
      if (!documentHandlers.has(type)) documentHandlers.set(type, []);
      documentHandlers.get(type).push(handler);
    }
  };
  let now = 0;
  let nextId = 0;
  const callbacks = new Map();
  const window = {
    devicePixelRatio: 2,
    performance: { now: () => now },
    addEventListener() {},
    requestAnimationFrame(callback) { callbacks.set(++nextId, callback); return nextId; },
    cancelAnimationFrame(id) { callbacks.delete(id); },
    matchMedia() { return { matches: reducedMotion, addEventListener() {}, removeEventListener() {} }; }
  };
  replay.bootReplay(document, window);
  return {
    get: id => elements.get(id), context, backgrounds, controls: controlIds.map(id => elements.get(id)),
    pending: () => callbacks.size,
    frameAt(time) {
      now = time;
      const current = Array.from(callbacks.values());
      callbacks.clear();
      current.forEach(callback => callback(now));
    },
    setNow(time) { now = time; },
    emitKey(key, extra = {}) {
      const event = Object.assign({ key, target: { tagName: "body", isContentEditable: false }, shiftKey: false, preventDefault() {} }, extra);
      for (const handler of documentHandlers.get("keydown") || []) handler(event);
    }
  };
}

test("DOM boot displays source strings as text and keeps the Python result authoritative", function () {
  const data = fixture();
  const source = '<img src=x onerror="throw 1"> & <script>bad()</script>';
  data.source.participant_id = source;
  data.track.meta.display_name = source;
  data.frames[0].events = [source];
  // Deliberately unlike the frame duration: the viewer must not substitute it.
  data.result.status = "finished";
  data.result.lap_time_s = 7.125;
  data.result.elapsed_s = 9.75;
  const app = harness(data);
  assert.equal(app.get("replay-error").hidden, true);
  assert.equal(app.get("track-name").textContent, source);
  assert.equal(app.get("participant-id").textContent, source);
  assert.equal(app.get("frame-events").textContent, source);
  assert.equal(app.get("lap-time").textContent, "7.125 s");
  assert.equal(app.get("result-elapsed").textContent, "9.750 s");
  assert.equal(app.get("result-status").textContent, "FINISHED");
  assert.equal(app.get("simulation-time").textContent, "00:00.000");
  assert.equal(app.get("tick-hz").textContent, "30");
  assert.equal(app.get("speed-ms").textContent, "10.00");
  assert.equal(app.get("speed-kmh").textContent, "36.0");
  assert.equal(app.get("playback-status").textContent, "Paused");
  assert.equal(app.pending(), 0, "does not autoplay");
  assert.ok(app.controls.filter(control => control.id !== "previous-event").every(control => !control.disabled));
  assert.equal(app.get("previous-event").disabled, true);
  app.get("play-pause").emit("click");
  app.frameAt(200);
  assert.equal(app.get("lap-time").textContent, "7.125 s");
  assert.equal(app.get("result-code").textContent, "finished");
});

test("DOM playback interpolates the car, holds tick telemetry, and stops on the exact DNF frame", function () {
  const data = fixture();
  const app = harness(data);
  const backgroundCalls = app.backgrounds[0].calls.length;
  app.get("play-pause").emit("click");
  assert.equal(app.get("play-label").textContent, "Pause");
  assert.equal(app.get("play-pause").attributes["aria-pressed"], "true");
  assert.equal(app.pending(), 1);
  app.frameAt(1000 / 60);
  assert.equal(app.get("current-tick").textContent, "0");
  assert.equal(app.get("off-track").textContent, "ON TRACK");
  assert.equal(app.get("speed-ms").textContent, "10.00");
  assert.equal(app.get("frame-events").textContent, "None recorded");
  const fit = replay.fitViewport(replay.replayBounds(data), 960, 500);
  const expected = replay.projectPoint(fit, 6.5, 0.5);
  const translated = app.context.calls.filter(call => call[0] === "translate").at(-1);
  near(translated[1], expected.x);
  near(translated[2], expected.y);
  near(app.context.calls.filter(call => call[0] === "rotate").at(-1)[1], -Math.PI);
  app.frameAt(1000 / 30);
  assert.equal(app.get("current-tick").textContent, "1");
  assert.equal(app.get("off-track").textContent, "OFF TRACK");
  assert.equal(app.get("frame-events").textContent, "off_track_start");
  app.frameAt(200);
  assert.equal(app.get("current-tick").textContent, "3");
  assert.equal(app.get("simulation-time").textContent, "00:00.100");
  assert.equal(app.get("terminated-flag").textContent, "Terminated: yes");
  assert.equal(app.get("frame-events").textContent, "barrier_contact");
  assert.equal(app.get("result-code").textContent, "dnf_crash");
  assert.equal(app.get("lap-time").textContent, "—");
  assert.equal(app.get("playback-status").textContent, "End of recording");
  assert.equal(app.get("play-pause").attributes["aria-pressed"], "false");
  assert.equal(app.pending(), 0);
  const last = replay.projectPoint(fit, 14, 3);
  near(app.context.calls.filter(call => call[0] === "translate").at(-1)[1], last.x);
  near(app.context.calls.filter(call => call[0] === "rotate").at(-1)[1], -(7.5 % (2 * Math.PI)));
  assert.equal(app.backgrounds[0].calls.length, backgroundCalls, "track/path are not redrawn on each animation frame");
  app.get("next-tick").emit("click");
  assert.equal(app.get("current-tick").textContent, "0");
  assert.equal(app.get("playback-status").textContent, "Paused");
});

test("DOM controls keep seeking anchored while playing and support speed, pause, restart and keyboard", function () {
  const app = harness(fixture());
  app.get("play-pause").emit("click");
  app.frameAt(10);
  app.get("scrubber").value = "0.04";
  app.get("scrubber").emit("input");
  assert.equal(app.get("play-label").textContent, "Pause");
  app.frameAt(20);
  near(Number(app.get("scrubber").value), 0.05);
  app.get("playback-speed").value = "2";
  app.get("playback-speed").emit("change");
  app.frameAt(30);
  near(Number(app.get("scrubber").value), 0.07);
  app.get("play-pause").emit("click");
  assert.equal(app.pending(), 0);
  app.get("restart").emit("click");
  assert.equal(app.get("current-tick").textContent, "0");
  let prevented = false;
  app.get("scrubber").emit("keydown", { key: "ArrowRight", preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(app.get("current-tick").textContent, "1");
  app.get("scrubber").emit("keydown", { key: "End" });
  assert.equal(app.get("current-tick").textContent, "3");
  app.get("play-pause").emit("click");
  assert.equal(app.get("current-tick").textContent, "0");
  assert.equal(app.pending(), 1);
  const oldBackgroundCalls = app.backgrounds[0].calls.length;
  app.get("show-centerline").checked = false;
  app.get("show-centerline").emit("change");
  const redrawn = app.backgrounds[0].calls.slice(oldBackgroundCalls);
  assert.ok(redrawn.length > 0);
  assert.ok(!redrawn.some(call => call[0] === "setLineDash" && String(call[1]) === "3,7"));
});

test("event markers jump to exact ticks and navigate without changing the recording", function () {
  const data = fixture();
  const before = JSON.stringify(data);
  const app = harness(data);
  const markers = app.get("event-markers").children;
  assert.equal(markers.length, 2);
  assert.equal(markers[0].style.left, (1 / 3 * 100) + "%");
  assert.equal(markers[1].style.left, "100%");
  assert.equal(app.get("next-event").disabled, false);
  markers[0].emit("click");
  assert.equal(app.get("current-tick").textContent, "1");
  assert.equal(app.get("event-status").textContent, "Off track · exact tick 1");
  assert.equal(app.get("playback-status").textContent, "Paused");
  assert.equal(app.get("previous-event").disabled, true);
  app.get("next-event").emit("click");
  assert.equal(app.get("current-tick").textContent, "3");
  assert.equal(app.get("next-event").disabled, true);
  app.get("previous-event").emit("click");
  assert.equal(app.get("current-tick").textContent, "1");
  assert.equal(JSON.stringify(data), before);
});

test("global keyboard shortcuts are available without stealing control focus", function () {
  const app = harness(fixture());
  let prevented = false;
  app.emitKey("ArrowRight", { preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(app.get("current-tick").textContent, "1");
  app.emitKey("ArrowLeft", { shiftKey: true });
  assert.equal(app.get("current-tick").textContent, "0");
  app.emitKey("[", { target: { tagName: "body", isContentEditable: false } });
  assert.equal(app.get("playback-speed").value, "0.5");
  const before = app.get("current-tick").textContent;
  app.emitKey("ArrowRight", { target: { tagName: "input", isContentEditable: false } });
  assert.equal(app.get("current-tick").textContent, before);
  app.get("event-markers").children[0].emit("click");
  assert.equal(app.get("current-tick").textContent, "1");
});

test("reduced motion snaps the car while retaining exact tick telemetry", function () {
  const data = fixture();
  const app = harness(data, true);
  app.get("play-pause").emit("click");
  app.frameAt(1000 / 60);
  const fit = replay.fitViewport(replay.replayBounds(data), 960, 500);
  const expected = replay.projectPoint(fit, data.frames[0].x, data.frames[0].y);
  const translated = app.context.calls.filter(call => call[0] === "translate").at(-1);
  near(translated[1], expected.x);
  near(translated[2], expected.y);
  assert.equal(app.get("current-tick").textContent, "0");
  assert.equal(app.get("speed-ms").textContent, "10.00");
});

test("DOM malformed/missing data shows a visible error and disables controls before drawing", function () {
  const wrongVersion = fixture();
  wrongVersion.schema_version = "1.0";
  for (const input of [undefined, "", "{broken", "null", wrongVersion]) {
    const app = harness(input);
    assert.equal(app.get("replay-error").hidden, false);
    assert.match(app.get("error-message").textContent, /Unable to display this recording/);
    assert.equal(app.get("playback-status").textContent, "Replay unavailable");
    assert.ok(app.controls.every(control => control.disabled));
    assert.equal(app.context.calls.length, 0);
    assert.equal(app.pending(), 0);
  }
});

test("DOM singleton recordings display their state with idle transport controls", function () {
  const data = fixture();
  data.frames = [data.frames[0]];
  data.result.steps = 0;
  const app = harness(data);
  assert.equal(app.get("replay-error").hidden, true);
  assert.equal(app.get("current-tick").textContent, "0");
  assert.equal(app.get("total-ticks").textContent, "0");
  assert.equal(app.get("play-pause").disabled, true);
  assert.equal(app.get("scrubber").disabled, true);
  assert.equal(app.get("show-centerline").disabled, false);
  assert.equal(app.get("event-markers").children[0].textContent, "No recorded events");
  assert.equal(app.get("event-count").textContent, "No recorded events");
  assert.equal(app.pending(), 0);
});
