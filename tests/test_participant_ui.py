from __future__ import annotations

import unittest
import hashlib
import http.client
import importlib.util
import io
import os
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from threading import Event, Lock, Thread
import time
from unittest.mock import Mock, patch

import numpy as np

from racing_event.config import SimulationConfig
from racing_event.observation import TrainingContract
from racing_event.reward import SIGNAL_NAMES
from racing_event.trajectory import engine_code_hash, load_replay
from tools import participant_ui as ui
from tools.participant_ui import (
    LIVE_TELEMETRY_PREFIX,
    PAGE,
    ROOT,
    TrainingProcess,
    build_live_package,
    build_onnx_package,
    build_live_race_command,
    build_training_command,
    parse_live_telemetry_line,
    parse_update_line,
    public_maps,
)
from tests.test_model_artifact import write_artifact


class ParticipantUiTests(unittest.TestCase):
    def test_standalone_cli_starts_without_pythonpath_from_another_directory(self) -> None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "tools/participant_ui.py"), "--port", str(port)],
                cwd=directory, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        self.fail(process.communicate()[0])
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                    try:
                        connection.request("GET", "/api/status")
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.loads(response.read())["state"], "idle")
                        return
                    except ConnectionRefusedError:
                        time.sleep(.05)
                    finally:
                        connection.close()
                self.fail("standalone dashboard did not become ready")
            finally:
                process.terminate()
                process.communicate(timeout=5)

    def test_update_line_is_parsed_for_dashboard_metrics(self) -> None:
        line = "update 3: steps 768/10000 return=-2.500 lap=0.125 speed=21.250 m/s crash=0.250 off_track=0.375"
        self.assertEqual(
            parse_update_line(line),
            {
                "update": 3,
                "budget_used": 768,
                "budget_steps": 10000,
                "return": -2.5,
                "lap": 0.125,
                "speed": 21.25,
                "crash": 0.25,
                "off_track": 0.375,
            },
        )

    def test_update_line_exposes_checkpoint_zone_progress(self) -> None:
        line = "update 4: steps 1024/10000 return=1.000 lap=0.250 speed=20.000 m/s crash=0.000 off_track=0.100 zones=6.500"
        self.assertEqual(parse_update_line(line)["zones"], 6.5)

    def test_command_is_workspace_bound_and_uses_existing_cli(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory).resolve()
            command = build_training_command({"output_dir": str(output_dir), "trainer": "examples/participant_train.py", "steps": 16})
        self.assertTrue(command[1].endswith("tools/train_participant.py"))
        self.assertIn("--device", command)
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertIn("--trainer", command)
        self.assertTrue(command[command.index("--trainer") + 1].endswith("/tools/run_reference_baseline.py"))
        self.assertIn("--output-dir", command)
        self.assertEqual(Path(command[command.index("--output-dir") + 1]), output_dir)
        with self.assertRaises(ValueError):
            build_training_command({"output_dir": "../../outside"})

    def test_page_contains_participant_contract_and_controls(self) -> None:
        for text in (
            "Start / resume participant training",
            "Steps this run",
            'id="processor"',
            "GPU · CUDA",
            "Spend 1 iteration · add 1,000,000 steps to this model",
            "Pause freezes the trainer process",
            "Reward function · signal weights",
            "Forward-facing ray sensors",
            "Watch artifact for map",
            "Set up / repair Python",
            "/api/status",
            'id="ray-diagram"',
            'role="img"',
            "renderRayDiagram",
            'id="ray-summary"',
            "The cyan line is 0° forward",
            "curvature_matched_speed",
            "Measures abrupt steering changes; larger means rougher",
            '["ray-count", "ray-angles"].forEach(id => $(id).addEventListener("input", renderRayDiagram))',
            'viewBox="0 0 640 430"',
            'id="ray-callouts"',
            "ray-label-box",
            "ray-endpoint",
            "car-wheel",
            "Positive angles sweep left",
            "Start live heat",
            "Shared race process",
            "model-file sizes",
            "Build ONNX submission package",
            "package_manifest.json",
            "/api/package",
            "Training maps (all public)",
            "client-budget-callout",
        ):
            self.assertIn(text, PAGE)
        self.assertNotIn("Cumulative budget", PAGE)

    def test_live_telemetry_line_parser_accepts_ticks_and_rejects_untrusted_lines(self) -> None:
        payload = {
            "kind": "live_tick",
            "tick": 4,
            "entrants": [{"participant_id": "alpha", "model_mb": 1.25}],
        }
        self.assertEqual(parse_live_telemetry_line(LIVE_TELEMETRY_PREFIX + json.dumps(payload)), payload)
        self.assertIsNone(parse_live_telemetry_line("ordinary trainer output"))
        self.assertIsNone(parse_live_telemetry_line(LIVE_TELEMETRY_PREFIX + "not-json"))
        self.assertIsNone(parse_live_telemetry_line(LIVE_TELEMETRY_PREFIX + json.dumps({"kind": "other", "entrants": []})))

    def test_live_race_command_is_workspace_bound_and_emits_telemetry(self) -> None:
        artifact = ROOT / "artifacts" / "onnx-imitation"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch.object(ui, "_ready_python", return_value=Path(sys.executable)):
            output = Path(directory) / "heat"
            command = build_live_race_command({
                "participant_a": "alpha",
                "participant_b": "beta",
                "artifact_a": str(artifact),
                "artifact_b": str(artifact),
                "race_output": str(output),
                "race_max_steps": "12",
                "race_budget_ms": "7.5",
                "race_port": "8777",
                "race_late_policy": "trusted_live",
                "race_isolate": True,
            })
        self.assertIn("--telemetry-stdout", command)
        self.assertIn("--isolate", command)
        self.assertEqual(command[command.index("--max-steps") + 1], "12")
        self.assertEqual(command[command.index("--policy-budget-ms") + 1], "7.5")
        self.assertEqual(command[command.index("--port") + 1], "8777")
        with self.assertRaisesRegex(ValueError, "different"):
            build_live_race_command({"participant_a": "same", "participant_b": "same", "artifact_a": str(artifact), "artifact_b": str(artifact)})

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "optional ONNX packaging checks require requirements-evaluator.lock",
    )
    def test_live_package_builds_valid_submission_and_matching_runtime_inputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / "source"
            source.mkdir()
            shutil.copy(ROOT / "config" / "simulation.json", source / "run_config.json")
            write_artifact(source / "artifact")
            output = Path(directory) / "package"
            result = build_live_package({
                "package_source": str(source),
                "package_output": str(output),
                "package_track": "tracks/train/monaco.json",
                "package_participant": "packaged-alpha",
            })
            self.assertEqual(result["source_kind"], "numpy_artifact")
            self.assertTrue(Path(result["submission_dir"]).joinpath("model.onnx").is_file())
            self.assertTrue(Path(result["config_path"]).is_file())
            self.assertTrue(Path(result["track_path"]).is_file())
            manifest = json.loads((output / "package_manifest.json").read_text())
            self.assertEqual(manifest["recommended_command"][1], "tools/run_live_onnx.py")
            self.assertEqual(manifest["track_id"], "monaco")
            with self.assertRaisesRegex(ValueError, "new directory"):
                build_live_package({
                    "package_source": str(source),
                    "package_output": str(output),
                    "package_track": "tracks/train/monaco.json",
                })

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
        "optional ONNX packaging checks require requirements-evaluator.lock",
    )
    def test_model_only_package_contains_no_track_or_replay(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            source = Path(directory) / "source"
            source.mkdir()
            shutil.copy(ROOT / "config" / "simulation.json", source / "run_config.json")
            write_artifact(source / "artifact")
            output = Path(directory) / "submission-package"
            result = build_onnx_package({
                "package_source": str(source),
                "package_output": str(output),
                "package_participant": "submission-alpha",
            })
            self.assertTrue(Path(result["submission_dir"]).joinpath("model.onnx").is_file())
            self.assertFalse((output / "tracks").exists())
            self.assertFalse((output / "replays").exists())
            manifest = json.loads((output / "package_manifest.json").read_text())
            self.assertEqual(manifest["kind"], "synapse_onnx_submission_package")
            self.assertNotIn("track_id", manifest)

    def test_process_resource_snapshot_reports_a_process_tree(self) -> None:
        snapshot = ui._process_resource_snapshot(os.getpid())
        self.assertIn(snapshot["scope"], {"process_tree", "child_processes"})
        self.assertGreaterEqual(snapshot["pids"], 1)

    def test_ui_writes_weighted_reward_and_ray_config_snapshots(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory).resolve()
            command = build_training_command({
                "output_dir": str(output_dir),
                "trainer": "examples/participant_train.py",
                "reward_weights": {"progress": "4.5", "crash": "-20"},
                "ray_count": "3",
                "ray_angles_deg": "-30,0,30",
                "steps": 1,
            })
            reward_path = Path(command[command.index("--reward") + 1])
            config_path = Path(command[command.index("--config") + 1])
            reward_data = json.loads(reward_path.read_text())
            self.assertEqual(reward_data["weights"]["progress"], 4.5)
            self.assertEqual(set(reward_data["weights"]), set(SIGNAL_NAMES))
            self.assertEqual(json.loads(config_path.read_text())["sensors"]["ray_angles_deg"], [-30.0, 0.0, 30.0])
            identity = json.loads((output_dir / "run_identity.json").read_text())
            self.assertTrue(identity["reward_sha256"])

    def test_ui_preserves_non_sorted_ray_order_in_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory).resolve()
            build_training_command({
                "output_dir": str(output_dir),
                "trainer": "examples/participant_train.py",
                "ray_count": 3,
                "ray_angles_deg": "30,-30,0",
                "steps": 1,
            })
            config = json.loads((output_dir / "run_config.json").read_text())
            identity = json.loads((output_dir / "run_identity.json").read_text())
        self.assertEqual(config["sensors"]["ray_angles_deg"], [30.0, -30.0, 0.0])
        self.assertEqual(identity["ray_angles_deg"], [30.0, -30.0, 0.0])
        self.assertEqual(identity["ray_count"], 3)

    def test_ui_rejects_ray_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            with self.assertRaisesRegex(ValueError, "ray_count"):
                build_training_command({
                    "output_dir": directory,
                    "trainer": "examples/participant_train.py",
                    "ray_count": 2,
                    "ray_angles_deg": "-30,0,30",
                })
            with self.assertRaisesRegex(ValueError, "ray_count"):
                build_training_command({
                    "output_dir": directory,
                    "trainer": "examples/participant_train.py",
                    "ray_count": 3.5,
                    "ray_angles_deg": "-30,0,30",
                })

    def test_ui_accepts_exact_ray_boundaries_and_rejects_out_of_range(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory).resolve()
            build_training_command({
                "output_dir": str(output_dir),
                "trainer": "examples/participant_train.py",
                "ray_count": 3,
                "ray_angles_deg": "-90,0,90",
                "steps": 1,
            })
            self.assertEqual(
                json.loads((output_dir / "run_config.json").read_text())["sensors"]["ray_angles_deg"],
                [-90.0, 0.0, 90.0],
            )
            with self.assertRaisesRegex(ValueError, r"\[-90, 90\]"):
                build_training_command({
                    "output_dir": str(Path(directory) / "out-of-range"),
                    "trainer": "examples/participant_train.py",
                    "ray_count": 3,
                    "ray_angles_deg": "-90,0,90.01",
                })

    def test_ui_does_not_overwrite_an_existing_run_with_new_contract_inputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory).resolve()
            form = {
                "output_dir": str(output_dir),
                "trainer": "examples/participant_train.py",
                "reward_weights": {"progress": "4"},
                "ray_count": 3,
                "ray_angles_deg": "-30,0,30",
                "steps": 1,
            }
            build_training_command(form)
            before = {name: (output_dir / name).read_bytes() for name in ("run_config.json", "reward_config.json", "run_identity.json")}
            changed = dict(form)
            changed["ray_angles_deg"] = "-90,0,90"
            with self.assertRaisesRegex(ValueError, r"different (config_hash|ray_angles_deg)"):
                build_training_command(changed)
            after = {name: (output_dir / name).read_bytes() for name in before}
        self.assertEqual(after, before)

    def test_public_map_selector_has_two_development_maps(self) -> None:
        maps = public_maps()
        self.assertEqual(len(maps), 2)
        self.assertGreater(len(maps[0]["centerline"]), 3)
        self.assertEqual(
            {item["id"] for item in maps},
            {"monaco", "spa"},
        )

    def test_training_position_metric_preserves_nested_car_locations(self) -> None:
        line = 'SYNAPSE_METRIC {"kind":"training_progress","training_position":{"kind":"training_position","update":2,"cars":[{"lane":0,"track_id":"monaco","x":1.5,"y":-2.0,"heading":0.2,"progress_fraction":0.1,"zones_reached":2}]}}'
        metric = ui.parse_metric_line(line)
        self.assertEqual(metric["training_position"]["cars"][0]["track_id"], "monaco")

    def test_numeric_inputs_reject_fractional_boolean_and_infinite_values(self) -> None:
        for value in (1.5, True, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_training_command({"steps": value})

    def test_missing_paths_are_rejected_before_start(self) -> None:
        with self.assertRaisesRegex(ValueError, "trainer does not exist"):
            build_training_command({"trainer": "artifacts/does-not-exist.py"})
        with self.assertRaisesRegex(ValueError, "config does not exist"):
            build_training_command({"config": "artifacts/does-not-exist.json"})

    def test_stop_reaches_a_terminal_stopped_state(self) -> None:
        process = TrainingProcess()
        process.start_command(
            [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(10)"],
            kind="test",
        )
        for _ in range(20):
            if process.snapshot()["running"]:
                break
            time.sleep(0.01)
        self.assertTrue(process.snapshot()["running"])
        process.stop()
        for _ in range(100):
            snapshot = process.snapshot()
            if not snapshot["running"]:
                break
            time.sleep(0.02)
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["state"], "stopped")

    def test_stop_terminates_the_trainer_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "child.pid"
            code = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(10)"
            )
            process = TrainingProcess()
            process.start_command([sys.executable, "-c", code, str(pid_path)], kind="tree-test")
            for _ in range(100):
                if pid_path.exists():
                    break
                time.sleep(0.02)
            self.assertTrue(pid_path.exists())
            child_pid = int(pid_path.read_text())
            process.stop()
            for _ in range(150):
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.02)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

    def test_stop_survives_trailing_metrics_and_successful_exit(self) -> None:
        process = TrainingProcess()
        code = (
            "import signal, time, sys\n"
            "def stop(*_):\n"
            " print('update 1: steps 2/16 return=0 lap=0 speed=0 m/s crash=0 off_track=0', flush=True)\n"
            " time.sleep(.1)\n"
            " sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "print('ready', flush=True)\n"
            "time.sleep(10)\n"
        )
        process.start_command([sys.executable, "-c", code], kind="test")
        self.addCleanup(process.stop)
        wait_for(lambda: "ready" in process.snapshot()["lines"])
        process.stop()
        wait_for(lambda: not process.snapshot()["running"])
        snapshot = process.snapshot()
        self.assertEqual(snapshot["metrics"]["budget_used"], 2)
        self.assertEqual(snapshot["state"], "stopped")
        self.assertEqual(snapshot["returncode"], 0)
        self.assertIsNone(snapshot["error"])

    def test_stale_output_and_completion_cannot_change_new_process(self) -> None:
        process = TrainingProcess()
        process.start_command([sys.executable, "-c", "import time; time.sleep(10)"], kind="new")
        self.addCleanup(process.stop)
        before = process.snapshot()
        old = Mock(stdout=io.StringIO("old trailing output\n"))
        old.wait.return_value = 0
        process._read_output(old)
        self.assertEqual(process.snapshot(), before)
        process.stop()
        wait_for(lambda: not process.snapshot()["running"])

    def test_second_start_is_rejected_before_snapshot_preparation(self) -> None:
        process = TrainingProcess()
        process.start_command([sys.executable, "-c", "import time; time.sleep(10)"], kind="test")
        self.addCleanup(process.stop)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "must-not-be-created"
            with patch.object(ui, "build_training_command", wraps=build_training_command) as prepare:
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    process.start({"output_dir": str(output), "ray_angles_deg": "90,0,-90"})
                prepare.assert_not_called()
            self.assertFalse(output.exists())
        process.stop()
        wait_for(lambda: not process.snapshot()["running"])

    def test_spawn_failure_is_terminal_and_can_be_retried(self) -> None:
        process = TrainingProcess()
        with self.assertRaises(OSError):
            process.start_command([str(ROOT / "missing-python")], kind="test")
        self.assertEqual(process.snapshot()["state"], "failed")
        self.assertFalse(process.snapshot()["running"])
        process.start_command([sys.executable, "-c", "print('retried')"], kind="test")
        wait_for(lambda: not process.snapshot()["running"])
        self.assertEqual(process.snapshot()["state"], "completed")

    def test_malformed_metrics_do_not_drop_process_supervision(self) -> None:
        process = TrainingProcess()
        bad_line = "update 1: steps 2/16 return=nan lap=0 speed=0 m/s crash=0 off_track=0"
        self.assertIsNone(parse_update_line(bad_line))
        process.start_command([sys.executable, "-c", f"import time; print({bad_line!r}, flush=True); time.sleep(.2)"], kind="test")
        self.addCleanup(process.stop)
        wait_for(lambda: bad_line in process.snapshot()["lines"])
        self.assertTrue(process.snapshot()["running"])
        wait_for(lambda: not process.snapshot()["running"])
        self.assertEqual(process.snapshot()["state"], "completed")
        self.assertEqual(process.snapshot()["metrics"], {})

    def test_pygame_probe_cannot_mistake_hidden_sdl_surface_for_visible_playback(self) -> None:
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            with self.assertRaisesRegex(ValueError, "SDL selected a headless driver"):
                ui._ready_python(require_pygame=True, headless=False)
            self.assertTrue(ui._ready_python(require_pygame=True, headless=True).is_file())

    @unittest.skipUnless(shutil.which("node"), "Node is optional; install it for dashboard JS checks")
    def test_dashboard_frontend_behavior(self) -> None:
        result = subprocess.run(
            [shutil.which("node"), "--test", "tests/participant_dashboard.test.cjs"],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def wait_for(predicate, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for process/request state")
        time.sleep(.01)


def write_watch_run(root: Path) -> dict[str, object]:
    """A three-ray participant-owned linear model, trained on a different map."""
    data = json.loads((ROOT / "config/simulation.json").read_text())
    data["sensors"]["ray_angles_deg"] = [90, 0, -90]
    data["episode"]["max_steps"] = 3
    (root / "run_config.json").write_text(json.dumps(data))
    config = SimulationConfig.from_file(root / "run_config.json")
    artifact = root / "artifact"
    artifact.mkdir()
    track = next(item for item in public_maps() if item["id"] == "monaco")
    contract = TrainingContract.for_config(config, track_ids=[track["id"]], track_hashes=[track["content_hash"]])
    contract.save(artifact / "training_contract.json")
    size = contract.observation.size
    np.savez(artifact / "weights.npz", w=np.zeros((2, size), dtype=np.float32), b=np.zeros(2, dtype=np.float32))
    manifest = {
        "format": "synapse_numpy_mlp", "format_version": 1, "environment_code_hash": engine_code_hash(),
        "observation": {"schema_version": "1.0", "size": size, "dtype": "float32"},
        "action": {"schema_version": "1.0", "size": 2, "dtype": "float32", "bounds": [[-1, 1], [-1, 1]], "output_activation": "tanh"},
        "preprocessing": {"kind": "identity"},
        "layers": [{"name": "linear", "weights": "w", "bias": "b", "activation": "tanh"}],
        "weights_sha256": hashlib.sha256((artifact / "weights.npz").read_bytes()).hexdigest(),
        "training_contract_sha256": hashlib.sha256((artifact / "training_contract.json").read_bytes()).hexdigest(),
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    return {"source": "artifact", "renderer": "browser", "map_id": "monaco", "output_dir": str(root), "artifact_dir": str(artifact)}


class ParticipantUiApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.form = write_watch_run(self.root)
        for name, value in (("PROCESS", TrainingProcess()), ("RACE_PROCESS", TrainingProcess()), ("RENDER_PROCESS", TrainingProcess()), ("REPLAY_OUTPUTS", {}), ("ACTION_LOCK", Lock())):
            patcher = patch.object(ui, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.server = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self) -> None:
        ui.PROCESS.stop()
        ui.RACE_PROCESS.stop()
        ui.RENDER_PROCESS.stop()
        wait_for(lambda: not ui.PROCESS.snapshot()["running"] and not ui.RACE_PROCESS.snapshot()["running"] and not ui.RENDER_PROCESS.snapshot()["running"])
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, payload=None, *, method="POST", headers=None, raw=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=20)
        body = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else b"")
        connection.request(method, path, body, headers or {"Content-Type": "application/json"})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def post(self, path, form):
        status, _, body = self.request(path, form)
        return status, json.loads(body)

    def test_generic_watch_exports_saved_custom_rays_without_rewriting_contract(self) -> None:
        artifact = Path(self.form["artifact_dir"])
        before = {path.name: path.read_bytes() for path in artifact.iterdir()}
        status, result = self.post("/api/watch", self.form)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["source"], "artifact")
        self.assertEqual(result["renderer"], "browser")
        self.assertEqual(ui.RENDER_PROCESS.snapshot()["state"], "idle")
        replay = load_replay(result["replay_path"])
        config = SimulationConfig.from_file(self.root / "run_config.json")
        self.assertEqual(replay["engine"]["config_hash"], TrainingContract.config_hash_for(config))
        self.assertEqual(replay["result"]["steps"], 3)
        self.assertEqual(before, {path.name: path.read_bytes() for path in artifact.iterdir()})
        code, headers, html = self.request(result["browser_url"], method="GET")
        self.assertEqual(code, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b'id="replay-data"', html)
        code, headers, download = self.request(result["download_url"], method="GET")
        self.assertEqual(code, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertEqual(json.loads(download), replay)
        # A later build has distinct URLs; the first remains readable.
        status, again = self.post("/api/watch", self.form)
        self.assertEqual(status, 200, again)
        self.assertNotEqual(again["browser_url"], result["browser_url"])
        self.assertEqual(self.request(result["browser_url"], method="GET")[0], 200)
        Path(result["replay_path"]).write_text("{}")
        self.assertEqual(self.request(result["download_url"], method="GET")[0], 400)

    def test_reference_is_latest_saved_and_leaves_original_contract_intact(self) -> None:
        from racing_event.ppo import FixedMLPActorCritic
        from racing_event.observation import RunningObservationNormalizer

        config = SimulationConfig.from_file(self.root / "run_config.json")
        contract, _ = TrainingContract.load(Path(self.form["artifact_dir"]) / "training_contract.json")
        normalizer = RunningObservationNormalizer(contract.observation.size)
        normalizer.update(np.ones((2, contract.observation.size)))
        self.assertTrue(normalizer.training)
        contract.save(self.root / "training_contract.json", normalizer=normalizer)
        original = (self.root / "training_contract.json").read_bytes()
        FixedMLPActorCritic(observation_size=contract.observation.size).save_artifact(self.root / "policy.pt")
        status, result = self.post("/api/watch", {**self.form, "source": "reference"})
        self.assertEqual(status, 200, result)
        self.assertEqual(result["source_label"], "Latest saved reference artifact")
        self.assertEqual(load_replay(result["replay_path"])["engine"]["config_hash"], TrainingContract.config_hash_for(config))
        self.assertEqual((self.root / "training_contract.json").read_bytes(), original)
        self.assertFalse(list((self.root / "replays").glob(".reference-contract.*")))
        # Even with a valid reference beside it, a bad generic artifact must fail.
        (Path(self.form["artifact_dir"]) / "manifest.json").write_text("{}")
        status, error = self.post("/api/watch", self.form)
        self.assertEqual(status, 400)
        self.assertIn("artifact", error["error"])

    def test_explicit_source_renderer_and_saved_config_are_required(self) -> None:
        for key, value in (("source", None), ("source", "scripted"), ("renderer", None), ("renderer", "unity"), ("map_id", "../outside"), ("output_dir", None), ("artifact_dir", None)):
            with self.subTest(key=key, value=value):
                status, result = self.post("/api/watch", {**self.form, key: value})
                self.assertEqual(status, 400, result)
        (self.root / "run_config.json").unlink()
        status, result = self.post("/api/watch", self.form)
        self.assertEqual(status, 400)
        self.assertIn("saved run_config.json", result["error"])
        self.assertEqual(ui.REPLAY_OUTPUTS, {})

    def test_config_mismatch_is_rejected_without_changing_generic_artifact(self) -> None:
        contract = Path(self.form["artifact_dir"]) / "training_contract.json"
        before = contract.read_bytes()
        config = json.loads((self.root / "run_config.json").read_text())
        config["sensors"]["ray_angles_deg"] = [-90, 0, 90]
        (self.root / "run_config.json").write_text(json.dumps(config))
        status, result = self.post("/api/watch", self.form)
        self.assertEqual(status, 400, result)
        self.assertEqual(contract.read_bytes(), before)
        self.assertEqual(ui.REPLAY_OUTPUTS, {})

    def test_headless_pygame_is_real_one_frame_check_with_fallback_links(self) -> None:
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}):
            status, result = self.post("/api/watch", {**self.form, "renderer": "pygame"})
            self.assertEqual(status, 200, result)
            self.assertTrue(result["headless"])
            self.assertNotIn("renderer_error", result)
            wait_for(lambda: not ui.RENDER_PROCESS.snapshot()["running"], timeout=10)
        self.assertEqual(ui.RENDER_PROCESS.snapshot()["state"], "completed")
        self.assertEqual(ui.RENDER_PROCESS.snapshot()["returncode"], 0)
        self.assertEqual(self.request(result["browser_url"], method="GET")[0], 200)

    def test_missing_pygame_readiness_keeps_explicit_renderer_and_download(self) -> None:
        # NumPy and Pygame imports work here, but opening the selected SDL surface
        # cannot. This proves the readiness probe goes beyond module availability.
        with patch.dict(os.environ, {"SDL_VIDEODRIVER": "synapse-invalid-driver", "DISPLAY": ":synapse-test"}):
            status, result = self.post("/api/watch", {**self.form, "renderer": "pygame"})
        self.assertEqual(status, 200, result)
        self.assertEqual(result["renderer"], "pygame")
        self.assertEqual(result["renderer_state"], "unavailable")
        self.assertIn("readiness check failed", result["renderer_error"])
        self.assertIn("synapse-invalid-driver", result["renderer_error"])
        self.assertFalse(result["headless"])
        self.assertEqual(self.request(result["download_url"], method="GET")[0], 200)

    def test_arbitrary_trainer_is_rejected_before_start(self) -> None:
        trainer = self.root / "trainer.py"
        trainer.write_text("import time\nprint('participant ready', flush=True)\ntime.sleep(10)\n")
        output = self.root / "training"
        status, result = self.post("/api/start", {
            "trainer": str(trainer), "output_dir": str(output), "steps": 1,
            "ray_count": 3, "ray_angles_deg": "90,0,-90", "reward_weights": {"progress": 4},
        })
        self.assertEqual(status, 400, result)
        self.assertIn("always uses tools/run_reference_baseline.py", result["error"])
        self.assertFalse(output.exists())

    def test_live_race_api_supervises_command_and_exposes_telemetry(self) -> None:
        command = [sys.executable, "-c", "import time; print('Live WebSocket endpoint: ws://127.0.0.1:8777/live/v1', flush=True); print('LIVE_TELEMETRY {\"kind\":\"live_tick\",\"entrants\":[]}', flush=True); time.sleep(10)"]
        with patch.object(ui, "build_live_race_command", return_value=command):
            status, result = self.post("/api/race/start", {"ignored": True})
        self.assertEqual(status, 200, result)
        wait_for(lambda: ui.RACE_PROCESS.snapshot()["telemetry"].get("kind") == "live_tick")
        snapshot = ui.RACE_PROCESS.snapshot()
        self.assertTrue(snapshot["running"])
        self.assertEqual(snapshot["telemetry"]["kind"], "live_tick")
        self.assertTrue(any(line.startswith("Live WebSocket endpoint:") for line in snapshot["lines"]))
        stop_status, _ = self.post("/api/race/stop", {})
        self.assertEqual(stop_status, 200)
        wait_for(lambda: not ui.RACE_PROCESS.snapshot()["running"])

    def test_watch_does_not_export_or_spawn_while_renderer_is_active(self) -> None:
        ui.RENDER_PROCESS.start_command([sys.executable, "-c", "import time; time.sleep(10)"], kind="renderer-test")
        with patch.object(ui, "build_watch_replay") as export:
            status, result = self.post("/api/watch", self.form)
        self.assertEqual(status, 400, result)
        self.assertIn("close the current Pygame", result["error"])
        export.assert_not_called()

    def test_request_length_and_encoding_fail_promptly_for_all_post_routes(self) -> None:
        for route in ("/api/start", "/api/watch", "/api/package", "/api/setup", "/api/stop", "/api/race/start", "/api/race/stop"):
            for length in ("-1", "16385", "nonsense", "999999999999999999"):
                with self.subTest(route=route, length=length):
                    code, _, body = self.request(route, headers={"Content-Length": length}, raw=b"")
                    self.assertEqual(code, 400, body)
        for raw in (b"\xff", b"[]", b"{"):
            self.assertEqual(self.request("/api/start", raw=raw)[0], 400)
        self.assertEqual(self.request("/api/watch", headers={"Transfer-Encoding": "chunked"}, raw=b"0\r\n\r\n")[0], 400)
        self.assertFalse(ui.PROCESS.snapshot()["running"])

    def test_cross_origin_and_rebound_host_cannot_control_or_read_dashboard(self) -> None:
        for headers in (
            {"Origin": "https://untrusted.example"},
            {"Origin": "null"},
            {"Host": f"untrusted.example:{self.server.server_port}"},
            {"Sec-Fetch-Site": "cross-site"},
        ):
            with self.subTest(headers=headers):
                code, _, _ = self.request("/api/setup", headers=headers)
                self.assertEqual(code, 403)
                self.assertFalse(ui.PROCESS.snapshot()["running"])
        code, _, _ = self.request("/api/status", method="GET", headers={"Host": "untrusted.example"})
        self.assertEqual(code, 403)
        code, _, _ = self.request("/api/stop", headers={"Origin": f"http://127.0.0.1:{self.server.server_port}"})
        self.assertEqual(code, 200)

    def test_watch_and_start_are_serialized_and_duplicates_rejected(self) -> None:
        entered, release = Event(), Event()
        def export(_form):
            entered.set()
            self.assertTrue(release.wait(5))
            return {"ok": True}
        responses = []
        with patch.object(ui, "build_watch_replay", side_effect=export) as build:
            first = Thread(target=lambda: responses.append(self.post("/api/watch", self.form)))
            first.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.post("/api/watch", self.form)[0], 400)
                output = self.root / "must-not-be-created"
                self.assertEqual(self.post("/api/start", {"output_dir": str(output)})[0], 400)
                self.assertFalse(output.exists())
                _, _, body = self.request("/api/status", method="GET")
                self.assertTrue(json.loads(body)["action_busy"])
                self.assertEqual(build.call_count, 1)
            finally:
                release.set()
                first.join(timeout=5)
        self.assertEqual(responses[0][0], 200)

    def test_workspace_escape_and_symlinked_inputs_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            link = self.root / "outside"
            link.symlink_to(outside, target_is_directory=True)
            for artifact in (outside, str(link), "../../outside"):
                status, result = self.post("/api/watch", {**self.form, "artifact_dir": artifact})
                self.assertEqual(status, 400, result)
                self.assertIn("workspace", result["error"])
            (self.root / "replays").symlink_to(outside, target_is_directory=True)
            self.assertEqual(self.post("/api/watch", self.form)[0], 400)
            (self.root / "replays").unlink()
            (self.root / "run_config.json").unlink()
            (self.root / "run_config.json").symlink_to(Path(outside) / "config.json")
            self.assertEqual(self.post("/api/watch", self.form)[0], 400)
            with self.assertRaisesRegex(ValueError, "workspace"):
                build_training_command({"output_dir": str(self.root)})
            self.assertEqual(self.request("/replays/../../config/simulation.json", method="GET")[0], 404)


if __name__ == "__main__":
    unittest.main()
