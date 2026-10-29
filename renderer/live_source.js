/* Read-only live state adapter. It validates packets but never simulates. */

export const LIVE_SCHEMA_VERSION = 1;
const MAX_MESSAGE_BYTES = 256 * 1024;

function finite(value, name) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Live ${name} must be finite`);
  }
  return value;
}

function requiredString(value, name) {
  if (typeof value !== 'string' || value.length < 1 || value.length > 256) {
    throw new Error(`Live ${name} is invalid`);
  }
  return value;
}

function validateCar(car) {
  if (!car || typeof car !== 'object') throw new Error('Live car is invalid');
  for (const name of ['x', 'y', 'heading', 'speed', 'lateral_velocity', 'yaw_rate', 'steer']) {
    finite(car[name], `car.${name}`);
  }
  for (const name of ['steering_command', 'throttle_brake']) {
    if (car[name] !== null) finite(car[name], `car.${name}`);
  }
  if (!Array.isArray(car.events) || car.events.some((event) => typeof event !== 'string')) {
    throw new Error('Live car events are invalid');
  }
  requiredString(car.participant_id, 'participant_id');
  requiredString(car.status, 'status');
  return car;
}

export function parseLiveMessage(raw, previous = {}) {
  if (typeof raw !== 'string' || new TextEncoder().encode(raw).length > MAX_MESSAGE_BYTES) {
    throw new Error('Live message exceeds the byte limit');
  }
  let message;
  try {
    message = JSON.parse(raw);
  } catch (error) {
    throw new Error(`Invalid live JSON: ${error.message}`);
  }
  if (!message || typeof message !== 'object' || message.schema_version !== LIVE_SCHEMA_VERSION) {
    throw new Error('Unsupported live message schema');
  }
  requiredString(message.run_id, 'run_id');
  if (previous.runId && previous.runId !== message.run_id) {
    throw new Error('Live run identity changed without a new connection');
  }
  const kind = message.kind;
  if (!['synapse.live.hello', 'synapse.live.state', 'synapse.live.end'].includes(kind)) {
    throw new Error('Unsupported live message kind');
  }
  if (kind !== 'synapse.live.hello') {
    if (!Number.isInteger(message.seq) || message.seq < 0 || message.seq <= (previous.seq ?? -1)) {
      throw new Error('Live sequence is stale or invalid');
    }
    if (!Number.isInteger(message.tick) || message.tick < 0) throw new Error('Live tick is invalid');
    finite(message.sim_time_s, 'sim_time_s');
  }
  if (kind === 'synapse.live.hello') {
    if (!Number.isInteger(message.physics_hz) || message.physics_hz < 1) throw new Error('Live physics_hz is invalid');
    if (!Number.isInteger(message.control_hz) || message.control_hz < 1) throw new Error('Live control_hz is invalid');
    if (!message.track || !message.track.meta || !Array.isArray(message.track.centerline)) {
      throw new Error('Live hello track is invalid');
    }
    if (!Array.isArray(message.entrants) || message.entrants.length < 1 || message.entrants.length > 4) {
      throw new Error('Live hello entrants are invalid');
    }
    return { kind, message, runId: message.run_id, seq: previous.seq ?? -1 };
  }
  if (!Array.isArray(message.cars) || message.cars.length < 1 || message.cars.length > 4) {
    throw new Error('Live state cars are invalid');
  }
  message.cars.forEach(validateCar);
  if (kind === 'synapse.live.end' && !Array.isArray(message.results)) {
    throw new Error('Live end results are invalid');
  }
  return { kind, message, runId: message.run_id, seq: message.seq };
}

export function connectLive(url, callbacks = {}) {
  const parsed = new URL(url, window.location.href);
  if (!['ws:', 'wss:'].includes(parsed.protocol)) throw new Error('Live URL must use ws:// or wss://');
  const socket = new WebSocket(parsed.href);
  socket.addEventListener('open', () => callbacks.onOpen?.());
  socket.addEventListener('message', (event) => callbacks.onMessage?.(event.data));
  socket.addEventListener('error', (event) => callbacks.onError?.(event));
  socket.addEventListener('close', (event) => callbacks.onClose?.(event));
  return socket;
}
