(function () {
  "use strict";

  const TAU = 2 * Math.PI;
  const RESULT_LABELS = Object.freeze({
    finished: "FINISHED",
    dnf_crash: "DNF · CRASH",
    dnf_stuck: "DNF · STUCK",
    dnf_backward: "DNF · BACKWARD",
    dnf_timeout: "DNF · TIMEOUT"
  });
  const POLYLINES = ["centerline", "left_boundary", "right_boundary", "barrier_left", "barrier_right"];
  const TRACK_SCALARS = ["s", "heading", "curvature", "half_width_left", "half_width_right"];
  const FRAME_SCALARS = ["x", "y", "heading", "speed", "lateral_velocity", "yaw_rate", "steer", "steering_command", "throttle_brake"];
  const EVENT_LABELS = Object.freeze({
    off_track_start: "Off track",
    off_track_end: "Back on track",
    barrier_contact: "Barrier contact",
    lap_complete: "Lap complete",
    checkpoint: "Progress zone reached"
  });

  function requireShape(condition, message) {
    if (!condition) throw new Error(message);
  }

  function isObject(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function finite(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  // Shape/finite checks protect rendering only. Python validates the recording
  // and its hashes; this viewer neither authenticates nor re-evaluates a run.
  function validateReplay(data) {
    requireShape(isObject(data), "Replay data must be a JSON object.");
    requireShape(data.schema_version === "1.1", "Unsupported replay schema; expected version 1.1.");
    requireShape(data.kind === "racing_event_replay", "Data is not a racing event replay.");
    requireShape(finite(data.tick_hz) && data.tick_hz > 0, "Replay tick_hz must be positive and finite.");
    requireShape(isObject(data.engine) && typeof data.engine.code_hash === "string" && typeof data.engine.config_hash === "string", "Missing engine metadata.");
    requireShape(isObject(data.source) && ["evaluation", "scripted"].includes(data.source.mode) && typeof data.source.participant_id === "string", "Invalid replay source.");
    requireShape(isObject(data.car) && finite(data.car.length_m) && data.car.length_m > 0 && finite(data.car.width_m) && data.car.width_m > 0, "Car dimensions must be positive and finite.");
    const track = data.track;
    requireShape(isObject(track) && track.version === "1.0", "Unsupported or missing track version; expected 1.0.");
    requireShape(Array.isArray(track.centerline) && track.centerline.length >= 3, "Track needs at least three samples.");
    const count = track.centerline.length;
    for (const name of POLYLINES) {
      const points = track[name];
      requireShape(Array.isArray(points) && points.length === count, "Track " + name + " must match the centreline sample count.");
      for (let i = 0; i < count; i += 1) {
        const point = points[i];
        requireShape(Array.isArray(point) && point.length === 2 && finite(point[0]) && finite(point[1]), "Track " + name + " contains an invalid coordinate.");
      }
    }
    for (const name of TRACK_SCALARS) {
      const values = track[name];
      requireShape(Array.isArray(values) && values.length === count, "Track " + name + " must match the centreline sample count.");
      for (let i = 0; i < count; i += 1) {
        requireShape(finite(values[i]), "Track " + name + " must contain finite numbers.");
      }
    }
    requireShape(finite(track.L) && track.L > 0 && finite(track.runoff_width) && track.runoff_width >= 0 && finite(track.start_line_s), "Invalid track dimensions.");
    requireShape(isObject(track.meta) && ["id", "display_name", "content_hash"].every(function (key) { return typeof track.meta[key] === "string"; }), "Missing track metadata.");
    requireShape(Array.isArray(data.frames) && data.frames.length > 0, "Replay must contain at least the initial frame.");
    for (let i = 0; i < data.frames.length; i += 1) {
      const frame = data.frames[i];
      requireShape(isObject(frame) && frame.tick === i, "Frames must have consecutive ticks starting at 0.");
      for (const name of FRAME_SCALARS) {
        requireShape(finite(frame[name]), "Frame " + i + " has an invalid " + name + ".");
      }
      requireShape(typeof frame.off_track === "boolean" && typeof frame.terminated === "boolean" && typeof frame.truncated === "boolean", "Frame " + i + " has invalid state flags.");
      requireShape(Array.isArray(frame.events) && frame.events.every(function (event) { return typeof event === "string"; }), "Frame " + i + " has invalid events.");
    }
    const duration = (data.frames.length - 1) / data.tick_hz;
    requireShape(finite(duration) && (data.frames.length === 1 || duration > 0), "Replay duration is outside the supported numeric range.");
    const result = data.result;
    requireShape(isObject(result) && Object.prototype.hasOwnProperty.call(RESULT_LABELS, result.status), "Invalid recorded result status.");
    requireShape(Number.isSafeInteger(result.steps) && result.steps === data.frames.length - 1, "Recorded steps must match the final frame tick.");
    requireShape(finite(result.elapsed_s) && result.elapsed_s >= 0 && (result.lap_time_s === null || (finite(result.lap_time_s) && result.lap_time_s >= 0)), "Invalid recorded result timing.");
    requireShape(typeof result.trajectory_hash === "string" && typeof data.payload_hash === "string", "Missing recording metadata.");
    // Also reject finite coordinates whose combined span would overflow a fit.
    replayBounds(data);
    return data;
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  function timingAtTime(time_s, tick_hz, last_tick) {
    requireShape(finite(time_s) && finite(tick_hz) && tick_hz > 0 && Number.isSafeInteger(last_tick) && last_tick >= 0, "Invalid playback timing.");
    const duration_s = last_tick / tick_hz;
    requireShape(finite(duration_s), "Invalid playback duration.");
    const time = clamp(time_s, 0, duration_s);
    let position = time === duration_s ? last_tick : time * tick_hz;
    // A tick/hz round trip can land a few floating-point ULPs below a tick.
    const nearest = Math.round(position);
    if (Math.abs(position - nearest) <= 8 * Number.EPSILON * Math.max(1, Math.abs(position))) position = nearest;
    const tick = clamp(Math.floor(position), 0, last_tick);
    return { time_s: time, tick: tick, next_tick: Math.min(tick + 1, last_tick), alpha: tick === last_tick ? 0 : position - tick };
  }

  function interpolateHeading(from, to, alpha) {
    if (alpha <= 0) return from;
    if (alpha >= 1) return to;
    const difference = (to % TAU) - (from % TAU);
    const shortest = Math.atan2(Math.sin(difference), Math.cos(difference));
    return from + shortest * alpha;
  }

  function sampleAtTime(data, time_s, options) {
    const timing = timingAtTime(time_s, data.tick_hz, data.frames.length - 1);
    const frame = data.frames[timing.tick];
    const next = data.frames[timing.next_tick];
    const reducedMotion = Boolean(options && options.reduced_motion);
    const alpha = reducedMotion ? 0 : timing.alpha;
    // Preserve every stored value at exact ticks (especially the final pose).
    // Telemetry/events are taken from frame, never blended or inferred.
    return {
      time_s: timing.time_s, tick: timing.tick, alpha: alpha, frame: frame,
      x: alpha === 0 ? frame.x : (1 - alpha) * frame.x + alpha * next.x,
      y: alpha === 0 ? frame.y : (1 - alpha) * frame.y + alpha * next.y,
      heading: alpha === 0 ? frame.heading : interpolateHeading(frame.heading, next.heading, alpha)
    };
  }

  function buildEventIndex(data) {
    requireShape(isObject(data) && Array.isArray(data.frames) && finite(data.tick_hz) && data.tick_hz > 0, "Invalid replay event source.");
    const events = [];
    for (const frame of data.frames) {
      if (!frame.events || frame.events.length === 0) continue;
      const names = frame.events.slice();
      events.push({
        tick: frame.tick,
        time_s: frame.tick / data.tick_hz,
        events: names,
        label: names.map(function (name) { return EVENT_LABELS[name] || name; }).join(" · ")
      });
    }
    return events;
  }

  function createPlayback(tick_hz, last_tick) {
    timingAtTime(0, tick_hz, last_tick);
    return { tick_hz: tick_hz, last_tick: last_tick, duration_s: last_tick / tick_hz, time_s: 0, playing: false, speed: 1, anchor_time_s: 0, anchor_ms: 0 };
  }

  function playbackTime(state, now_ms) {
    requireShape(finite(now_ms), "Invalid playback clock.");
    return state.playing ? clamp(state.anchor_time_s + Math.max(0, now_ms - state.anchor_ms) * state.speed / 1000, 0, state.duration_s) : state.time_s;
  }

  // These transitions change only the visual clock. Nothing writes to frames.
  function transitionPlayback(state, action, now_ms) {
    let time = playbackTime(state, now_ms);
    let playing = state.playing && time < state.duration_s;
    let speed = state.speed;
    switch (action.type) {
      case "toggle":
      case "play":
        // A Pause click just after the final tick must not become a replay
        // merely because the next animation callback has not run yet.
        if (action.type === "toggle" && state.playing) playing = false;
        else {
          if (time >= state.duration_s) time = 0;
          playing = state.duration_s > 0;
        }
        break;
      case "pause": playing = false; break;
      case "restart": time = 0; playing = false; break;
      case "seek":
        requireShape(finite(action.time_s), "Invalid seek time.");
        time = clamp(action.time_s, 0, state.duration_s);
        playing = playing && time < state.duration_s;
        break;
      case "next": {
        const tick = timingAtTime(time, state.tick_hz, state.last_tick).tick;
        time = tick === state.last_tick ? 0 : (tick + 1) / state.tick_hz;
        playing = false;
        break;
      }
      case "tick":
        requireShape(Number.isSafeInteger(action.tick) && action.tick >= 0 && action.tick <= state.last_tick, "Invalid playback tick.");
        time = action.tick / state.tick_hz;
        playing = false;
        break;
      case "speed":
        requireShape(finite(action.speed) && action.speed > 0, "Invalid playback speed.");
        speed = action.speed;
        break;
      case "sync":
        if (playing) return Object.assign({}, state, { time_s: time });
        break;
      default: throw new Error("Unknown playback action.");
    }
    return Object.assign({}, state, { time_s: time, playing: playing, speed: speed, anchor_time_s: time, anchor_ms: now_ms });
  }

  function replayBounds(data) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    function include(x, y) {
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
    for (const name of POLYLINES) {
      for (const point of data.track[name]) include(point[0], point[1]);
    }
    // Include the recorded path and car footprint so a recorded off-track pose
    // remains visible. This is a view fit, not a track-state calculation.
    for (const frame of data.frames) include(frame.x, frame.y);
    const margin = Math.hypot(data.car.length_m / 2, data.car.width_m / 2);
    minX -= margin; minY -= margin; maxX += margin; maxY += margin;
    requireShape([minX, minY, maxX, maxY, maxX - minX, maxY - minY].every(finite), "Replay coordinates exceed the supported drawing range.");
    return { minX: minX, minY: minY, maxX: maxX, maxY: maxY };
  }

  function fitViewport(bounds, width, height, padding) {
    requireShape(finite(width) && width > 0 && finite(height) && height > 0, "Invalid canvas size.");
    const inset = Math.min(padding === undefined ? 36 : Math.max(0, padding), width / 4, height / 4);
    return {
      width: width, height: height,
      centerX: bounds.minX / 2 + bounds.maxX / 2,
      centerY: bounds.minY / 2 + bounds.maxY / 2,
      scale: Math.min((width - 2 * inset) / Math.max(1, bounds.maxX - bounds.minX), (height - 2 * inset) / Math.max(1, bounds.maxY - bounds.minY))
    };
  }

  function projectPoint(fit, x, y) {
    return { x: fit.width / 2 + (x - fit.centerX) * fit.scale, y: fit.height / 2 - (y - fit.centerY) * fit.scale };
  }

  function formatTime(seconds) {
    // Simulation-clock formatting only; reported lap times use result data.
    const milliseconds = Math.round(seconds * 1000);
    const minutes = Math.floor(milliseconds / 60000);
    return String(minutes).padStart(2, "0") + ":" + String(Math.floor(milliseconds / 1000) % 60).padStart(2, "0") + "." + String(milliseconds % 1000).padStart(3, "0");
  }

  function bootReplay(doc, win) {
    const get = function (id) { return doc.getElementById(id); };
    const controls = doc.querySelectorAll("[data-control]");
    let raf = null;
    let observer = null;
    let motionQuery = null;
    let reducedMotion = false;
    let failed = false;
    function text(id, value) {
      const node = get(id);
      const content = String(value);
      if (node && node.textContent !== content) node.textContent = content;
    }
    function fail(error) {
      failed = true;
      if (raf !== null) win.cancelAnimationFrame(raf);
      raf = null;
      if (observer) observer.disconnect();
      if (motionQuery) {
        if (typeof motionQuery.removeEventListener === "function") motionQuery.removeEventListener("change", onMotionChange);
        else if (typeof motionQuery.removeListener === "function") motionQuery.removeListener(onMotionChange);
      }
      for (const control of controls) control.disabled = true;
      const panel = get("replay-error");
      if (panel) panel.hidden = false;
      text("error-message", "Unable to display this recording. " + (error instanceof Error ? error.message : "Invalid replay data."));
      text("playback-status", "Replay unavailable");
      text("play-label", "Play");
      const button = get("play-pause");
      if (button) button.setAttribute("aria-pressed", "false");
    }

    try {
      const dataNode = get("replay-data");
      requireShape(dataNode && dataNode.textContent.trim(), "The embedded replay-data JSON is missing.");
      const data = validateReplay(JSON.parse(dataNode.textContent));
      const canvas = get("track-canvas");
      const context = canvas.getContext("2d");
      requireShape(context, "This browser does not support a 2D canvas.");
      // One cached background: no copied trajectory arrays or per-frame track
      // meshes. The full recorded path is drawn only on resize/toggle.
      const background = doc.createElement("canvas");
      const backgroundContext = background.getContext("2d");
      requireShape(backgroundContext, "Unable to create the track drawing surface.");
      const bounds = replayBounds(data);
      const scrubber = get("scrubber");
      const centerline = get("show-centerline");
      const eventIndex = buildEventIndex(data);
      const eventMarkers = get("event-markers");
      const previousEvent = get("previous-event");
      const nextEvent = get("next-event");
      let state = createPlayback(data.tick_hz, data.frames.length - 1);
      let fit = null;
      let pixelRatio = 1;
      let lastTelemetryTick = -1;

      if (typeof win.matchMedia === "function") {
        motionQuery = win.matchMedia("(prefers-reduced-motion: reduce)");
        reducedMotion = Boolean(motionQuery.matches);
      }

      text("track-name", data.track.meta.display_name);
      text("participant-id", data.source.participant_id);
      text("track-id", data.track.meta.id);
      text("source-mode", data.source.mode === "evaluation" ? "EVALUATION" : "SCRIPTED");
      text("result-status", RESULT_LABELS[data.result.status]);
      get("result-status").dataset.tone = data.result.status === "finished" ? "good" : "warning";
      text("result-code", data.result.status);
      text("lap-time", data.result.lap_time_s === null ? "—" : data.result.lap_time_s.toFixed(3) + " s");
      text("result-elapsed", data.result.elapsed_s.toFixed(3) + " s");
      text("result-steps", data.result.steps);
      text("code-hash", data.engine.code_hash);
      text("config-hash", data.engine.config_hash);
      text("track-hash", data.track.meta.content_hash);
      text("trajectory-hash", data.result.trajectory_hash);
      text("payload-hash", data.payload_hash);
      text("duration", formatTime(state.duration_s));
      text("total-ticks", state.last_tick);
      text("tick-hz", data.tick_hz);
      scrubber.max = String(state.duration_s);
      text("event-count", eventIndex.length === 0 ? "No recorded events" : eventIndex.length === 1 ? "1 recorded event" : eventIndex.length + " recorded events");
      text("event-status", eventIndex.length ? "Select a marker to inspect its exact tick." : "No recorded events.");
      if (eventMarkers) {
        eventMarkers.textContent = "";
        if (eventIndex.length === 0) {
          const empty = doc.createElement("span");
          empty.className = "event-rail-empty";
          empty.textContent = "No recorded events";
          eventMarkers.appendChild(empty);
        } else {
          for (const entry of eventIndex) {
            const marker = doc.createElement("button");
            marker.type = "button";
            marker.className = "event-marker";
            marker.style.left = (state.last_tick === 0 ? 0 : entry.tick / state.last_tick * 100) + "%";
            marker.textContent = "●";
            marker.title = entry.label + " · tick " + entry.tick;
            marker.setAttribute("aria-label", entry.label + ", tick " + entry.tick);
            marker.setAttribute("aria-controls", "scrubber");
            marker.addEventListener("click", function () { act({ type: "tick", tick: entry.tick }); });
            eventMarkers.appendChild(marker);
          }
        }
      }
      for (const control of controls) control.disabled = false;
      if (state.last_tick === 0) {
        for (const id of ["play-pause", "restart", "next-tick", "scrubber", "playback-speed", "previous-event", "next-event"]) get(id).disabled = true;
      }

      function plot(ctx, method, x, y) {
        ctx[method](fit.width / 2 + (x - fit.centerX) * fit.scale, fit.height / 2 - (y - fit.centerY) * fit.scale);
      }

      function line(ctx, points, color, width, dash) {
        ctx.beginPath();
        for (let i = 0; i < points.length; i += 1) plot(ctx, i === 0 ? "moveTo" : "lineTo", points[i][0], points[i][1]);
        ctx.closePath();
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.setLineDash(dash);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      function strip(ctx, left, right, color) {
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = 0.6;
        for (let i = 0; i < left.length; i += 1) {
          const next = (i + 1) % left.length;
          ctx.beginPath();
          plot(ctx, "moveTo", left[i][0], left[i][1]);
          plot(ctx, "lineTo", left[next][0], left[next][1]);
          plot(ctx, "lineTo", right[next][0], right[next][1]);
          plot(ctx, "lineTo", right[i][0], right[i][1]);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
      }

      function buildBackground() {
        const ctx = backgroundContext;
        ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        ctx.clearRect(0, 0, fit.width, fit.height);
        const track = data.track;
        strip(ctx, track.barrier_left, track.left_boundary, "#1c2c30");
        strip(ctx, track.right_boundary, track.barrier_right, "#1c2c30");
        strip(ctx, track.left_boundary, track.right_boundary, "#2a3748");
        line(ctx, track.left_boundary, "#7c8b9c", 1, []);
        line(ctx, track.right_boundary, "#7c8b9c", 1, []);
        line(ctx, track.barrier_left, "#abb6c6", 1.2, [5, 5]);
        line(ctx, track.barrier_right, "#abb6c6", 1.2, [5, 5]);
        if (centerline.checked) line(ctx, track.centerline, "#8092aa", 0.9, [3, 7]);
        ctx.beginPath();
        for (let i = 0; i < data.frames.length; i += 1) plot(ctx, i === 0 ? "moveTo" : "lineTo", data.frames[i].x, data.frames[i].y);
        ctx.strokeStyle = "#62ead1aa";
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }

      function drawCar(sample) {
        const position = projectPoint(fit, sample.x, sample.y);
        const length = data.car.length_m * fit.scale;
        const width = data.car.width_m * fit.scale;
        const color = sample.frame.off_track ? "#ffd28a" : "#62ead1";
        context.save();
        context.translate(position.x, position.y);
        context.beginPath();
        context.arc(0, 0, Math.max(8, width), 0, TAU);
        context.strokeStyle = sample.frame.off_track ? "#ffd28a55" : "#62ead155";
        context.lineWidth = 1;
        context.stroke();
        // World heading is CCW. The screen's inverted Y makes rotation negative.
        context.rotate(-(sample.heading % TAU));
        context.shadowColor = color;
        context.shadowBlur = 10;
        context.fillStyle = color;
        context.fillRect(-length / 2, -width / 2, length, width);
        context.shadowBlur = 0;
        context.strokeStyle = "#e5fff8";
        context.lineWidth = 1;
        context.strokeRect(-length / 2, -width / 2, length, width);
        context.fillStyle = "#13343c";
        context.fillRect(length * 0.03, -width * 0.35, length * 0.18, width * 0.7);
        context.fillStyle = "#f3fffc";
        context.fillRect(length * 0.35, -width * 0.4, length * 0.08, width * 0.8);
        context.restore();
      }

      function updateTelemetry(sample) {
        if (sample.tick === lastTelemetryTick) return;
        lastTelemetryTick = sample.tick;
        const frame = sample.frame;
        text("current-tick", frame.tick);
        text("speed-kmh", (frame.speed * 3.6).toFixed(1));
        text("speed-ms", frame.speed.toFixed(2));
        text("steer-angle", frame.steer.toFixed(3));
        text("steering-command", frame.steering_command.toFixed(3));
        text("lateral-velocity", frame.lateral_velocity.toFixed(2));
        text("yaw-rate", frame.yaw_rate.toFixed(3));
        text("throttle-value", (frame.throttle_brake > 0 ? "+" : "") + frame.throttle_brake.toFixed(3));
        const fill = get("throttle-fill");
        const input = clamp(frame.throttle_brake, -1, 1);
        fill.style.left = (input < 0 ? 50 + input * 50 : 50) + "%";
        fill.style.width = Math.abs(input) * 50 + "%";
        fill.dataset.negative = String(input < 0);
        text("off-track", frame.off_track ? "OFF TRACK" : "ON TRACK");
        get("off-track").dataset.tone = frame.off_track ? "warning" : "good";
        text("terminated-flag", "Terminated: " + (frame.terminated ? "yes" : "no"));
        text("truncated-flag", "Truncated: " + (frame.truncated ? "yes" : "no"));
        text("frame-events", frame.events.length ? frame.events.join(" · ") : "None recorded");
        const atTick = eventIndex.find(function (entry) { return entry.tick === frame.tick; });
        text("event-status", atTick ? atTick.label + " · exact tick " + atTick.tick : (eventIndex.length ? "Select a marker to inspect its exact tick." : "No recorded events."));
        text("canvas-description", "Full-track plan with equal axis scale and +Y up. Recorded tick " + frame.tick + ": x " + frame.x.toFixed(2) + " m, y " + frame.y.toFixed(2) + " m, heading " + frame.heading.toFixed(3) + " rad. The highlighted car nose points forward. Only position and heading are visually interpolated; telemetry uses completed ticks.");
      }

      function updateEventNavigation(tick) {
        if (!previousEvent || !nextEvent) return;
        const previous = eventIndex.findLast ? eventIndex.findLast(function (entry) { return entry.tick < tick; }) : eventIndex.slice().reverse().find(function (entry) { return entry.tick < tick; });
        const next = eventIndex.find(function (entry) { return entry.tick > tick; });
        previousEvent.disabled = !previous;
        nextEvent.disabled = !next;
        previousEvent.setAttribute("aria-label", previous ? "Previous event, tick " + previous.tick : "No previous event");
        nextEvent.setAttribute("aria-label", next ? "Next event, tick " + next.tick : "No next event");
      }

      function render() {
        if (failed || !fit) return;
        const sample = sampleAtTime(data, state.time_s, { reduced_motion: reducedMotion });
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        context.clearRect(0, 0, fit.width, fit.height);
        context.drawImage(background, 0, 0, fit.width, fit.height);
        drawCar(sample);
        updateTelemetry(sample);
        updateEventNavigation(sample.tick);
        text("simulation-time", formatTime(sample.time_s));
        scrubber.value = String(sample.time_s);
        scrubber.setAttribute("aria-valuetext", formatTime(sample.time_s) + ", tick " + sample.tick + " of " + state.last_tick);
        text("play-label", state.playing ? "Pause" : "Play");
        text("play-icon", state.playing ? "Ⅱ" : "▶");
        get("play-pause").setAttribute("aria-pressed", String(state.playing));
        text("playback-status", state.playing ? "Playing" : state.time_s >= state.duration_s ? "End of recording" : "Paused");
      }

      function schedule() {
        if (!failed && state.playing && raf === null) raf = win.requestAnimationFrame(onFrame);
      }

      function onFrame(now) {
        raf = null;
        if (failed) return;
        try {
          state = transitionPlayback(state, { type: "sync" }, now);
          render();
          schedule();
        } catch (error) { fail(error); }
      }

      function act(action) {
        if (failed) return;
        try {
          state = transitionPlayback(state, action, win.performance.now());
          if (raf !== null) win.cancelAnimationFrame(raf);
          raf = null;
          render();
          schedule();
        } catch (error) { fail(error); }
      }

      function onMotionChange(event) {
        reducedMotion = Boolean(event.matches);
        render();
      }

      function isEditableTarget(target) {
        if (!target) return false;
        const tag = typeof target.tagName === "string" ? target.tagName.toLowerCase() : "";
        return tag === "input" || tag === "select" || tag === "textarea" || tag === "button" || Boolean(target.isContentEditable);
      }

      function handleGlobalKeydown(event) {
        if (isEditableTarget(event.target)) return;
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
          event.preventDefault();
          const direction = event.key === "ArrowRight" ? 1 : -1;
          if (event.shiftKey) {
            act({ type: "seek", time_s: playbackTime(state, win.performance.now()) + direction });
          } else {
            const tick = timingAtTime(playbackTime(state, win.performance.now()), data.tick_hz, state.last_tick).tick;
            act({ type: "tick", tick: clamp(tick + direction, 0, state.last_tick) });
          }
        } else if (event.key === "PageDown" || event.key === "PageUp") {
          event.preventDefault();
          act({ type: "seek", time_s: playbackTime(state, win.performance.now()) + (event.key === "PageUp" ? 10 : -10) });
        } else if (event.key === " ") {
          event.preventDefault();
          act({ type: "toggle" });
        } else if (event.key === "Home" || event.key === "End") {
          event.preventDefault();
          act({ type: "seek", time_s: event.key === "Home" ? 0 : state.duration_s });
        } else if (event.key === "r" || event.key === "R") {
          event.preventDefault();
          act({ type: "restart" });
        } else if (event.key === "[" || event.key === "]") {
          event.preventDefault();
          const speeds = [0.25, 0.5, 1, 2, 4];
          const index = speeds.indexOf(state.speed);
          const nextIndex = clamp(index + (event.key === "]" ? 1 : -1), 0, speeds.length - 1);
          const speedControl = get("playback-speed");
          speedControl.value = String(speeds[nextIndex]);
          act({ type: "speed", speed: speeds[nextIndex] });
        }
      }

      function resize() {
        if (failed) return;
        try {
          const rect = get("canvas-viewport").getBoundingClientRect();
          const width = Math.max(1, Math.round(rect.width || 960));
          const height = Math.max(1, Math.round(rect.height || 600));
          const ratio = clamp(win.devicePixelRatio || 1, 1, 2);
          if (fit && fit.width === width && fit.height === height && pixelRatio === ratio) return;
          pixelRatio = ratio;
          fit = fitViewport(bounds, width, height);
          canvas.width = background.width = Math.round(width * pixelRatio);
          canvas.height = background.height = Math.round(height * pixelRatio);
          buildBackground();
          render();
        } catch (error) { fail(error); }
      }

      get("play-pause").addEventListener("click", function () { act({ type: "toggle" }); });
      get("restart").addEventListener("click", function () { act({ type: "restart" }); });
      get("next-tick").addEventListener("click", function () { act({ type: "next" }); });
      if (previousEvent) previousEvent.addEventListener("click", function () {
        const tick = timingAtTime(playbackTime(state, win.performance.now()), data.tick_hz, state.last_tick).tick;
        const entry = eventIndex.slice().reverse().find(function (candidate) { return candidate.tick < tick; });
        if (entry) act({ type: "tick", tick: entry.tick });
      });
      if (nextEvent) nextEvent.addEventListener("click", function () {
        const tick = timingAtTime(playbackTime(state, win.performance.now()), data.tick_hz, state.last_tick).tick;
        const entry = eventIndex.find(function (candidate) { return candidate.tick > tick; });
        if (entry) act({ type: "tick", tick: entry.tick });
      });
      get("playback-speed").addEventListener("change", function (event) { act({ type: "speed", speed: Number(event.target.value) }); });
      scrubber.addEventListener("input", function () { act({ type: "seek", time_s: Number(scrubber.value) }); });
      scrubber.addEventListener("keydown", function (event) {
        const increments = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10 };
        const now = playbackTime(state, win.performance.now());
        if (Object.prototype.hasOwnProperty.call(increments, event.key)) {
          event.preventDefault();
          act({ type: "seek", time_s: now + increments[event.key] / data.tick_hz });
        } else if (event.key === "Home" || event.key === "End") {
          event.preventDefault();
          act({ type: "seek", time_s: event.key === "Home" ? 0 : state.duration_s });
        }
      });
      centerline.addEventListener("change", function () {
        if (failed) return;
        try { buildBackground(); render(); } catch (error) { fail(error); }
      });
      if (typeof win.ResizeObserver === "function") {
        observer = new win.ResizeObserver(resize);
        observer.observe(get("canvas-viewport"));
      }
      if (motionQuery) {
        if (typeof motionQuery.addEventListener === "function") motionQuery.addEventListener("change", onMotionChange);
        else if (typeof motionQuery.addListener === "function") motionQuery.addListener(onMotionChange);
      }
      if (typeof doc.addEventListener === "function") doc.addEventListener("keydown", handleGlobalKeydown);
      win.addEventListener("resize", resize);
      resize();
    } catch (error) { fail(error); }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { validateReplay, timingAtTime, interpolateHeading, sampleAtTime, buildEventIndex, createPlayback, playbackTime, transitionPlayback, replayBounds, fitViewport, projectPoint, formatTime, bootReplay };
  }
  if (typeof document !== "undefined" && typeof window !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", function () { bootReplay(document, window); }, { once: true });
    else bootReplay(document, window);
  }
}());
