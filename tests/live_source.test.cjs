const test = require('node:test');
const assert = require('node:assert/strict');

let parseLiveMessage;
test.before(async () => {
  ({ parseLiveMessage } = await import('../renderer/live_source.js'));
});

function hello() {
  return {
    kind: 'synapse.live.hello', schema_version: 1, run_id: 'run-1',
    physics_hz: 30, control_hz: 30,
    track: { meta: {}, centerline: [[0, 0]] }, entrants: [{ participant_id: 'a' }],
  };
}

function state(seq = 1) {
  return {
    kind: 'synapse.live.state', schema_version: 1, run_id: 'run-1', seq, tick: seq,
    sim_time_s: seq / 30,
    cars: [{
      participant_id: 'a', status: 'running', x: 0, y: 0, heading: 0, speed: 0,
      lateral_velocity: 0, yaw_rate: 0, steer: 0, steering_command: 0,
      throttle_brake: 0, events: [],
    }],
  };
}

test('live source accepts hello and ordered state, then rejects stale state', () => {
  const parsedHello = parseLiveMessage(JSON.stringify(hello()));
  assert.equal(parsedHello.kind, 'synapse.live.hello');
  const parsedState = parseLiveMessage(JSON.stringify(state()), { runId: 'run-1', seq: -1 });
  assert.equal(parsedState.message.tick, 1);
  assert.throws(() => parseLiveMessage(JSON.stringify(state()), { runId: 'run-1', seq: 1 }), /stale/);
});

test('live source rejects nonfinite and unsupported packets', () => {
  const invalid = state();
  invalid.cars[0].speed = NaN;
  assert.throws(() => parseLiveMessage(JSON.stringify(invalid)), /finite/);
  assert.throws(() => parseLiveMessage(JSON.stringify({ ...hello(), kind: 'action' })), /Unsupported live message kind/);
});
