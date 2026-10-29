from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest

from racing_event.live_protocol import LiveStatePublisher, parse_live_message
from racing_event.live_race import LiveRaceOptions, run_live_onnx_race
from racing_event.race import RaceEntrant
from tests.test_model_artifact import CONFIG, TRACK, write_artifact


HAS_ONNX = bool(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"))


@unittest.skipUnless(HAS_ONNX, "live ONNX checks require requirements-live.lock")
class LiveRaceTests(unittest.TestCase):
    def setUp(self) -> None:
        from tools.export_onnx import export

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        source_a, source_b = root / "source-a", root / "source-b"
        write_artifact(source_a)
        write_artifact(source_b)
        self.artifact_a, self.artifact_b = root / "team-a", root / "team-b"
        export(source_a, self.artifact_a)
        export(source_b, self.artifact_b)

    def test_fixed_rate_live_run_emits_fresh_state_and_report_without_replays(self) -> None:
        output = Path(self.temp.name) / "live"
        publisher = LiveStatePublisher(port=0)
        report = run_live_onnx_race(
            [RaceEntrant("a", self.artifact_a), RaceEntrant("b", self.artifact_b)],
            TRACK,
            CONFIG,
            publisher=publisher,
            options=LiveRaceOptions(max_steps=4, policy_budget_ms=100, pace=False),
            output_dir=output,
        )
        try:
            data = report.to_dict()
            self.assertEqual(data["kind"], "synapse_live_onnx_race")
            self.assertEqual(data["tick_rate"]["physics_hz"], CONFIG.tick_hz)
            self.assertEqual(data["tick_rate"]["control_hz"], CONFIG.tick_hz)
            self.assertEqual(len(data["results"]), 2)
            self.assertTrue(all(result["steps"] == 4 for result in data["results"]))
            self.assertTrue((output / "live-race.json").is_file())
            self.assertFalse(any(path.suffix == ".json" and path.name != "live-race.json" for path in output.iterdir()))
        finally:
            publisher.close()

    def test_real_websocket_receives_hello_states_and_end(self) -> None:
        publisher = LiveStatePublisher(port=0)
        result: list[object] = []

        def run() -> None:
            try:
                result.append(run_live_onnx_race(
                    [RaceEntrant("a", self.artifact_a), RaceEntrant("b", self.artifact_b)],
                    TRACK,
                    CONFIG,
                    publisher=publisher,
                    options=LiveRaceOptions(max_steps=12, policy_budget_ms=500, pace=True),
                ))
            except BaseException as exc:  # surfaced below
                result.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        self.assertTrue(publisher._ready.wait(5), "live publisher did not start")

        async def receive() -> list[dict]:
            from websockets.asyncio.client import connect

            messages = []
            async with connect(publisher.url) as websocket:
                async for raw in websocket:
                    messages.append(parse_live_message(raw))
                    if messages[-1]["kind"] == "synapse.live.end":
                        break
            return messages

        messages = asyncio.run(receive())
        thread.join(10)
        publisher.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(result), 1)
        if isinstance(result[0], BaseException):
            raise result[0]
        self.assertEqual(messages[0]["kind"], "synapse.live.hello")
        self.assertEqual(messages[-1]["kind"], "synapse.live.end")
        states = [message for message in messages if message["kind"] == "synapse.live.state"]
        self.assertGreaterEqual(len(states), 2)
        self.assertEqual(states[0]["tick"], 0)
        self.assertEqual(states[-1]["tick"], 12)
        self.assertEqual(states[-1]["cars"][0]["status"], "dnf_timeout")

    def test_protocol_rejects_stale_sequences_and_read_only_action_shape(self) -> None:
        hello = {
            "kind": "synapse.live.hello", "schema_version": 1, "run_id": "run",
            "physics_hz": 30, "control_hz": 30, "track": {"meta": {}, "centerline": []}, "entrants": [{}],
        }
        parsed = parse_live_message(json.dumps(hello))
        self.assertEqual(parsed["run_id"], "run")
        state = {
            "kind": "synapse.live.state", "schema_version": 1, "run_id": "run", "seq": 1,
            "tick": 1, "sim_time_s": 1 / 30,
            "cars": [{
                "participant_id": "a", "status": "running", "x": 0, "y": 0, "heading": 0,
                "speed": 0, "lateral_velocity": 0, "yaw_rate": 0, "steer": 0,
                "steering_command": 0, "throttle_brake": 0, "events": [],
            }],
        }
        parse_live_message(json.dumps(state), {"run_id": "run", "seq": 0})
        with self.assertRaises(ValueError):
            parse_live_message(json.dumps(state), {"run_id": "run", "seq": 1})


if __name__ == "__main__":
    unittest.main()
