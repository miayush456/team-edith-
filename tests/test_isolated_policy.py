"""Model-only worker checks, offline by default.

Local subprocess tests below are JSONL *protocol* tests, not real container
verification. Only DockerIntegrationTests uses Docker, and it requires an
explicit opt-in and an already-built local image (it never builds or pulls):

    SYNAPSE_TEST_DOCKER=1 SYNAPSE_TEST_DOCKER_IMAGE=synapse-evaluator:v1 \
        .venv/bin/python -m unittest tests.test_isolated_policy -v
"""
from __future__ import annotations

import hashlib
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from racing_event import isolated_policy
from racing_event.evaluation import evaluate_participant_artifact
from racing_event.observation import RunningObservationNormalizer
from racing_event.submission import load_submission
from racing_event.vector_env import VectorizedRacingEnv
from tests.test_model_artifact import CONFIG, FLOWING, write_artifact


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "inference_worker.py"
LINE_LIMIT = 16 * 1024
HAS_ONNX = bool(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"))


def artifact_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, raw in sorted(files.items()):
        digest.update(name.encode() + b"\0" + raw)
    return digest.hexdigest()


# A deliberately faulty protocol peer. This does not replace the actual worker
# in the successful round-trip tests. It only supplies otherwise hard-to-trigger
# output/exit/deadline failures over real OS pipes.
FAULTY_PEER = r"""
import json
import sys
import time

mode, digest, response = sys.argv[1:]

def send(raw):
    raw += '\n'
    if mode == 'fragmented':
        for start in range(0, len(raw), 7):
            sys.stdout.write(raw[start:start + 7])
            sys.stdout.flush()
            time.sleep(0.002)
    else:
        sys.stdout.write(raw)
        sys.stdout.flush()

if mode == 'startup_exit':
    sys.exit(3)
if mode == 'startup_timeout':
    sys.stdin.buffer.read()
    sys.exit(0)
if mode == 'startup_oversize':
    send('x' * (16 * 1024 + 1))
elif mode == 'ready_response':
    send(response)
else:
    send(json.dumps({'ready': True, 'artifact_sha256': digest}))

request = sys.stdin.buffer.readline()
if not request:
    sys.exit(0)
if mode == 'action_exit':
    sys.exit(3)
if mode == 'action_timeout':
    sys.stdin.buffer.read()
elif mode == 'stubborn':
    time.sleep(60)
elif mode == 'action_oversize':
    send('x' * (16 * 1024 + 1))
elif mode == 'partial_response':
    sys.stdout.write('{"sequence":')
    sys.stdout.flush()
elif mode == 'partial_exit':
    sys.stdout.write('{"sequence":')
    sys.stdout.flush()
    sys.exit(3)
elif mode == 'partial_oversize':
    sys.stdout.write('x' * (16 * 1024 + 1))
    sys.stdout.flush()
elif mode == 'replayed_sequence':
    send(response)
    if sys.stdin.buffer.readline():
        send(response)
else:
    send(response)
sys.stdin.buffer.read()
"""


class ArtifactTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Keep production's temporary snapshots out of the checkout, including
        # when construction fails before docker_command is reached.
        self.runtime = self.root / "organizer"
        self.enterContext(mock.patch.object(isolated_policy, "ROOT", self.runtime))
        self.source = self.root / "submission"
        self.layers = [(
            "actor",
            np.linspace(-0.08, 0.12, 46, dtype=np.float32).reshape(2, 23),
            np.asarray([-0.2, 0.3], dtype=np.float32),
            "tanh",
        )]
        write_artifact(self.source, layers=self.layers)
        self.observation = np.linspace(-1, 1, 23, dtype=np.float32)[None, :]
        self.addCleanup(self.close_source_descriptors)

    def source_descriptors(self):
        result = {}
        directory = Path("/proc/self/fd")
        if directory.is_dir():
            for name in os.listdir(directory):
                try:
                    target = os.readlink(directory / name)
                except FileNotFoundError:
                    continue  # listdir's own descriptor has already closed.
                if target.startswith(str(self.root) + os.sep):
                    result[int(name)] = target
        return result

    def close_source_descriptors(self):
        # Keep a production descriptor-leak regression from contaminating other
        # tests. Only descriptors pointing into this test's private tree qualify.
        for fd in self.source_descriptors():
            os.close(fd)

    def assert_no_snapshots(self) -> None:
        parent = self.runtime / "artifacts" / "isolated-inputs"
        self.assertEqual(list(parent.iterdir()) if parent.exists() else [], [])


class LocalProtocolTestCase(ArtifactTestCase):
    """Real local subprocesses; Docker launch/removal boundaries are patched."""

    def setUp(self) -> None:
        super().setUp()
        self.launches: list[tuple[Path, str]] = []
        self.processes: list[subprocess.Popen] = []
        self.command_factory = lambda snapshot: [
            sys.executable, "-u", str(WORKER), "--artifact-dir", str(snapshot),
        ]
        self.before_launch = lambda snapshot: None
        self.command = self.enterContext(mock.patch.object(
            isolated_policy, "docker_command", side_effect=self.local_command,
        ))
        # Even cleanup must never contact a Docker daemon in protocol tests.
        self.removal = self.enterContext(mock.patch.object(
            isolated_policy.subprocess, "run",
            return_value=subprocess.CompletedProcess(["docker", "rm"], 0),
        ))
        real_popen = subprocess.Popen

        def start_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            self.processes.append(process)
            return process

        self.enterContext(mock.patch.object(isolated_policy.subprocess, "Popen", side_effect=start_process))
        self.enterContext(mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}))
        self.addCleanup(self.reap_processes)

    def local_command(self, snapshot, name, image, memory_mb):
        snapshot = Path(snapshot)
        self.launches.append((snapshot, name))
        self.before_launch(snapshot)
        return self.command_factory(snapshot)

    def reap_processes(self) -> None:
        # A failing assertion must not leave a test worker behind.
        for process in self.processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

    def open_policy(self, source=None, **kwargs):
        options = {"startup_timeout": 10, "action_timeout": 1}
        options.update(kwargs)
        policy = isolated_policy.IsolatedPolicy(source or self.source, **options)
        self.addCleanup(policy.close)
        return policy

    def use_faulty_peer(self, mode: str, response=None) -> None:
        raw = response if isinstance(response, str) else json.dumps(response)

        def command(snapshot):
            files = {path.name: path.read_bytes() for path in snapshot.iterdir()}
            return [sys.executable, "-u", "-c", FAULTY_PEER, mode, artifact_digest(files), raw]

        self.command_factory = command

    def assert_closed(self) -> None:
        self.assert_no_snapshots()
        for process in self.processes:
            self.assertIsNotNone(process.poll(), "worker was not reaped")
            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.stdout.closed)
        for snapshot, name in self.launches:
            self.assertFalse(snapshot.exists(), "copied model was not removed")
            self.assertTrue(any(
                call.args and call.args[0][:2] == ["docker", "rm"]
                and "--force" in call.args[0] and name in call.args[0]
                for call in self.removal.call_args_list
            ), "container removal was not attempted")

    def assert_rejected_before_launch(self, source) -> None:
        with self.assertRaises((ValueError, OSError)):
            self.open_policy(source)
        self.command.assert_not_called()
        self.assertEqual(self.processes, [])
        self.assert_no_snapshots()


class DockerCommandTests(unittest.TestCase):
    @staticmethod
    def option_values(command, option):
        values = []
        for index, value in enumerate(command):
            if value == option:
                values.append(command[index + 1])
            elif value.startswith(option + "="):
                values.append(value.split("=", 1)[1])
        return values

    def test_safe_command_mounts_only_the_frozen_model_and_never_tracks(self):
        snapshot = Path(tempfile.gettempdir()) / "isolated model"
        image = "synapse-evaluator:v1"
        command = isolated_policy.docker_command(snapshot, "test-worker", image, 512)
        self.assertEqual(command[:2], ["docker", "run"])
        self.assertIn("--rm", command)
        self.assertIn("-i", command)
        self.assertIn("--read-only", command)
        for option, expected in (
            ("--pull", "never"), ("--network", "none"), ("--cap-drop", "ALL"),
            ("--security-opt", "no-new-privileges:true"), ("--cpus", "1"),
            ("--memory", "512m"), ("--memory-swap", "512m"),
            ("--pids-limit", "64"), ("--ipc", "none"),
        ):
            with self.subTest(option=option):
                self.assertIn(expected, self.option_values(command, option))
        user = self.option_values(command, "--user")
        self.assertEqual(len(user), 1)
        self.assertTrue(all(part not in {"0", "root"} for part in user[0].split(":")))
        self.assertIn("nofile=64:64", self.option_values(command, "--ulimit"))
        self.assertIn("core=0:0", self.option_values(command, "--ulimit"))
        self.assertEqual(set(self.option_values(command, "--env")),
                         {"OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1"})
        tmpfs = self.option_values(command, "--tmpfs")
        self.assertEqual(len(tmpfs), 1)
        self.assertTrue(tmpfs[0].startswith("/tmp:"))
        self.assertTrue({"noexec", "nosuid", "size=16m"}.issubset(tmpfs[0].split(":", 1)[1].split(",")))
        mounts = self.option_values(command, "--mount")
        self.assertEqual(len(mounts), 1, "no source tree, track or Docker socket mounts")
        fields = mounts[0].split(",")
        self.assertIn("type=bind", fields)
        self.assertIn(f"src={snapshot}", fields)
        self.assertIn("dst=/model", fields)
        self.assertIn("readonly", fields)
        for forbidden in ("--privileged", "--volume", "-v", "--volumes-from", "--device", "--env-file"):
            self.assertFalse(any(value == forbidden or value.startswith(forbidden + "=") for value in command))
        self.assertFalse(any(str(ROOT / "tracks") in value for value in command))
        self.assertEqual(command[command.index("--artifact-dir") + 1], "/model")
        self.assertGreater(command.index("--artifact-dir"), command.index(image))

    def test_command_rejects_unsafe_image_path_and_memory_arguments(self):
        defaults = {"snapshot": Path(tempfile.gettempdir()) / "model", "name": "test-worker",
                    "image": "synapse-evaluator:v1", "memory_mb": 512}
        cases = [
            ("image", value) for value in ("", "--privileged", "bad image", "image;id", "image\nother")
        ] + [
            ("snapshot", Path(tempfile.gettempdir()) / value) for value in ("model,other", "model\nother")
        ] + [
            ("memory_mb", value) for value in (0, 127, 4097, True, 512.0, "512")
        ]
        for key, value in cases:
            with self.subTest(argument=key, value=value):
                with self.assertRaises(ValueError):
                    isolated_policy.docker_command(**{**defaults, key: value})
        for memory in (128, 4096):
            with self.subTest(memory=memory):
                self.assertIn(f"{memory}m", self.option_values(
                    isolated_policy.docker_command(**{**defaults, "memory_mb": memory}), "--memory"))


class WorkerProtocolTests(LocalProtocolTestCase):
    """Run tools/inference_worker.py locally; these do not verify Docker."""

    def test_protocol_actual_worker_matches_in_process_inference_repeatedly(self):
        expected = load_submission(self.source)
        with self.open_policy() as policy:
            self.assertEqual(policy.artifact_sha256, expected.artifact_sha256)
            self.assertEqual(policy.observation_size, expected.observation_size)
            for observation in (self.observation, -self.observation, self.observation):
                action = policy(observation)
                self.assertEqual(action.dtype, np.dtype("float32"))
                np.testing.assert_array_equal(action, expected(observation))
        self.assert_closed()

    def test_protocol_copied_bytes_and_hash_survive_source_mutation(self):
        expected = load_submission(self.source)
        original = {path.name: path.read_bytes() for path in self.source.iterdir()}

        def inspect_and_mutate_source(snapshot):
            self.assertNotEqual(snapshot.resolve(), self.source.resolve())
            self.assertEqual({path.name: path.read_bytes() for path in snapshot.iterdir()}, original)
            for path in snapshot.iterdir():
                self.assertFalse(path.is_symlink())
                self.assertEqual(path.stat().st_mode & 0o222, 0)
                self.assertNotEqual(path.stat().st_ino, (self.source / path.name).stat().st_ino)
            for path in self.source.iterdir():
                path.write_bytes(b"changed after snapshot")

        self.before_launch = inspect_and_mutate_source
        with self.open_policy() as policy:
            self.assertEqual(policy.artifact_sha256, artifact_digest(original))
            np.testing.assert_array_equal(policy(self.observation), expected(self.observation))
        self.assert_closed()
        self.assertTrue(self.source.is_dir())
        self.assertEqual((self.source / "weights.npz").read_bytes(), b"changed after snapshot")

    def test_protocol_live_workers_have_independent_snapshot_lifetimes(self):
        expected = load_submission(self.source)
        first = self.open_policy()
        second = self.open_policy()
        first_snapshot, first_name = self.launches[0]
        second_snapshot, second_name = self.launches[1]
        self.assertNotEqual(first_snapshot, second_snapshot)
        self.assertNotEqual(first_name, second_name)
        self.assertEqual(first.artifact_sha256, second.artifact_sha256)

        first.close()
        self.assertFalse(first_snapshot.exists())
        self.assertTrue(second_snapshot.is_dir())
        self.assertIsNone(self.processes[1].poll())
        np.testing.assert_array_equal(second(self.observation), expected(self.observation))
        second.close()
        self.assert_closed()

    def test_protocol_affine_preprocessing_and_frozen_normalizer_match(self):
        normalizer = RunningObservationNormalizer(23)
        normalizer.update(np.stack((np.zeros(23), np.ones(23))).astype(np.float32))
        normalizer.freeze()
        cases = (
            ({"kind": "affine", "mean": [0.2] * 23, "scale": [0.5] * 23, "clip": [1.0] * 23}, None),
            ({"kind": "contract_normalizer"}, normalizer),
        )
        for index, (preprocessing, frozen) in enumerate(cases):
            with self.subTest(preprocessing=preprocessing["kind"]):
                source = self.root / f"preprocessed-{index}"
                write_artifact(source, layers=self.layers, preprocessing=preprocessing, normalizer=frozen)
                expected = load_submission(source)
                before = (source / "training_contract.json").read_bytes()
                with self.open_policy(source) as policy:
                    state = policy.normalizer.state_dict() if policy.normalizer is not None else None
                    for _ in range(2):
                        np.testing.assert_array_equal(policy(self.observation), expected(self.observation))
                    self.assertEqual(state, policy.normalizer.state_dict() if policy.normalizer is not None else None)
                self.assertEqual((source / "training_contract.json").read_bytes(), before)
                self.assert_closed()

    def test_protocol_proxy_supports_trusted_holdout_evaluation(self):
        expected = evaluate_participant_artifact(
            VectorizedRacingEnv(FLOWING, CONFIG, reward=None, n_envs=1),
            load_submission(self.source), max_steps=4,
        )
        with self.open_policy() as policy:
            first = evaluate_participant_artifact(
                VectorizedRacingEnv(FLOWING, CONFIG, reward=None, n_envs=1), policy, max_steps=4,
            )
            second = evaluate_participant_artifact(
                VectorizedRacingEnv(FLOWING, CONFIG, reward=None, n_envs=1), policy, max_steps=4,
            )
            self.assertEqual(first.to_dict(), expected.to_dict())
            self.assertEqual(second.to_dict(), expected.to_dict())
            self.assertEqual(first.total_reward, 0)
        self.assert_closed()

    def test_evaluator_cli_path_records_isolation_and_matches_local_results(self):
        from tools.evaluate_participant import evaluate
        args = argparse.Namespace(
            artifact_dir=self.source, tracks=[ROOT / "tests/fixtures/tracks/synthetic_flowing.json"],
            config=ROOT / "config/simulation.json", repeats=2, max_steps=4, isolate=False,
            worker_image="synapse-evaluator:v1", worker_memory_mb=512, startup_timeout=10, action_timeout=1,
        )
        local = evaluate(args)
        args.isolate = True
        isolated = evaluate(args)
        self.assertEqual(isolated["tracks"], local["tracks"])
        self.assertEqual(isolated["artifact_sha256"], local["artifact_sha256"])
        self.assertEqual(isolated["inference"]["mode"], "isolated_docker")
        self.assertEqual(local["inference"]["mode"], "trusted_local_preview")
        self.assert_closed()

    @unittest.skipUnless(HAS_ONNX, "optional ONNX protocol test requires requirements-evaluator.lock")
    def test_protocol_actual_onnx_worker_matches_numpy(self):
        from tools.export_onnx import export

        target = self.root / "onnx-submission"
        export(self.source, target)
        expected = load_submission(self.source)
        with self.open_policy(target) as policy:
            self.assertEqual(policy.artifact_sha256, load_submission(target).artifact_sha256)
            for observation in (self.observation, -self.observation):
                np.testing.assert_allclose(policy(observation), expected(observation), atol=1e-7)
        self.assert_closed()

    @unittest.skipUnless(HAS_ONNX, "optional ONNX protocol test requires requirements-evaluator.lock")
    def test_protocol_onnx_worker_preserves_embedded_preprocessing(self):
        from tools.export_onnx import export

        normalizer = RunningObservationNormalizer(23)
        normalizer.update(np.stack((np.zeros(23), np.ones(23))).astype(np.float32))
        normalizer.freeze()
        source = self.root / "onnx-normalized-source"
        target = self.root / "onnx-normalized-target"
        write_artifact(source, layers=self.layers,
                       preprocessing={"kind": "contract_normalizer"}, normalizer=normalizer)
        export(source, target)
        expected = load_submission(source)
        with self.open_policy(target) as policy:
            self.assertIsNone(policy.normalizer, "ONNX preprocessing must not be applied twice")
            for observation in (self.observation, -self.observation):
                np.testing.assert_allclose(policy(observation), expected(observation), atol=1e-7)
        self.assert_closed()

    def run_worker(self, request: bytes, source=None):
        process = subprocess.Popen(
            [sys.executable, "-u", str(WORKER), "--artifact-dir", str(source or self.source)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(request, timeout=10)
        return process.returncode, stdout.splitlines(), stderr

    def test_protocol_raw_jsonl_ready_sequence_and_clean_eof(self):
        request = b"".join(json.dumps({"sequence": sequence, "observations": observation.tolist()}).encode() + b"\n"
                           for sequence, observation in enumerate((self.observation, -self.observation), 1))
        code, lines, stderr = self.run_worker(request)
        expected = load_submission(self.source)
        self.assertEqual(code, 0)
        self.assertEqual(stderr, b"")
        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[0]), {"ready": True, "artifact_sha256": expected.artifact_sha256})
        for sequence, observation in enumerate((self.observation, -self.observation), 1):
            response = json.loads(lines[sequence])
            self.assertEqual(set(response), {"sequence", "actions"})
            self.assertEqual(response["sequence"], sequence)
            np.testing.assert_array_equal(response["actions"], expected(observation))
        self.assertTrue(all(len(line) + 1 <= LINE_LIMIT for line in lines))

    def test_protocol_raw_worker_clean_eof_after_ready_without_requests(self):
        code, lines, stderr = self.run_worker(b"")
        self.assertEqual(code, 0)
        self.assertEqual(stderr, b"")
        self.assertEqual([json.loads(line) for line in lines], [
            {"ready": True, "artifact_sha256": load_submission(self.source).artifact_sha256},
        ])

    def test_protocol_raw_worker_startup_rejection_has_no_ready_or_payload_leak(self):
        marker = "DO_NOT_ECHO_ARTIFACT_CONTENTS"
        (self.source / "manifest.json").write_text('{"' + marker + '":')
        code, lines, stderr = self.run_worker(b"")
        self.assertEqual(code, 2)
        self.assertEqual(stderr, b"")
        self.assertEqual(len(lines), 1)
        response = json.loads(lines[0])
        self.assertEqual(set(response), {"error"})
        self.assertIsInstance(response["error"], str)
        self.assertLess(len(lines[0]), 256)
        for private in (marker.encode(), str(self.source).encode(), b"Traceback"):
            self.assertNotIn(private, lines[0])

    def test_protocol_raw_worker_rejects_bad_requests_without_payload_leaks(self):
        valid = {"sequence": 1, "observations": self.observation.tolist()}
        cases = {
            "wrong_sequence": json.dumps({**valid, "sequence": 2}).encode() + b"\n",
            "extra_field": json.dumps({**valid, "private_marker": "DO_NOT_ECHO"}).encode() + b"\n",
            "bad_shape": json.dumps({**valid, "observations": [[0.0]]}).encode() + b"\n",
            "nonfinite": json.dumps({**valid, "observations": [[float("nan")] * 23]}).encode() + b"\n",
            "not_object": b"[]\n",
            "malformed": b'{"DO_NOT_ECHO":\n',
            "incomplete": json.dumps(valid).encode(),
            "oversized": b"x" * (LINE_LIMIT + 1) + b"\n",
        }
        for case, request in cases.items():
            with self.subTest(case=case):
                code, lines, stderr = self.run_worker(request)
                self.assertEqual(code, 2)
                self.assertEqual(stderr, b"")
                self.assertEqual(len(lines), 2)
                self.assertTrue(json.loads(lines[0])["ready"])
                error = json.loads(lines[1])
                self.assertEqual(set(error), {"error"})
                self.assertIsInstance(error["error"], str)
                self.assertLess(len(lines[1]), 256)
                for private in (b"DO_NOT_ECHO", str(self.source).encode(), b"Traceback", b"observations"):
                    self.assertNotIn(private, lines[1])

    def test_protocol_raw_worker_rejects_duplicate_request_sequence(self):
        request = json.dumps({"sequence": 1, "observations": self.observation.tolist()}).encode() + b"\n"
        code, lines, _ = self.run_worker(request * 2)
        self.assertEqual(code, 2)
        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[1])["sequence"], 1)
        self.assertEqual(set(json.loads(lines[2])), {"error"})

    def test_protocol_raw_worker_requires_integer_sequence_type(self):
        for sequence in (True, 1.0):
            with self.subTest(sequence=sequence, type=type(sequence).__name__):
                request = json.dumps({"sequence": sequence, "observations": self.observation.tolist()}).encode() + b"\n"
                code, lines, _ = self.run_worker(request)
                self.assertEqual(code, 2)
                self.assertEqual(set(json.loads(lines[-1])), {"error"})


class SnapshotRejectionTests(LocalProtocolTestCase):
    def copy_source(self, name):
        destination = self.root / name
        shutil.copytree(self.source, destination)
        return destination

    def test_untrusted_contract_cannot_allocate_unbounded_host_normalizer(self):
        path = self.source / "training_contract.json"
        contract = json.loads(path.read_text())
        contract["trainer"]["normalizer"] = {"size": 10**12}
        path.write_text(json.dumps(contract))
        with mock.patch("racing_event.observation.RunningObservationNormalizer.from_state_dict") as construct:
            self.assert_rejected_before_launch(self.source)
            construct.assert_not_called()

    def test_snapshot_rejects_symlink_directory_and_symlink_ancestor(self):
        direct = self.root / "linked-submission"
        direct.symlink_to(self.source, target_is_directory=True)
        ancestor = self.root / "linked-parent"
        ancestor.symlink_to(self.root, target_is_directory=True)
        for source in (direct, ancestor / self.source.name):
            with self.subTest(source=source):
                self.assert_rejected_before_launch(source)

    def test_snapshot_rejects_missing_or_non_directory_submission(self):
        source_file = self.root / "not-a-directory"
        source_file.write_text("not an artifact")
        for source in (self.root / "missing-submission", source_file):
            with self.subTest(source=source):
                self.assert_rejected_before_launch(source)

    def test_snapshot_rejects_each_symlink_file(self):
        for name in ("manifest.json", "training_contract.json", "weights.npz", "model.onnx"):
            with self.subTest(name=name):
                source = self.copy_source("symlink-" + name)
                target = "weights.npz" if name == "model.onnx" else name
                (source / target).unlink()
                (source / name).symlink_to(self.source / target)
                self.assert_rejected_before_launch(source)

    def test_snapshot_rejects_nonregular_files_without_waiting_for_fifo_writer(self):
        for kind in ("directory", "fifo"):
            for name in ("manifest.json", "training_contract.json", "weights.npz", "model.onnx"):
                with self.subTest(kind=kind, name=name):
                    source = self.copy_source(kind + "-" + name)
                    path = source / name
                    (source / ("weights.npz" if name == "model.onnx" else name)).unlink()
                    if kind == "fifo":
                        os.mkfifo(path)
                    else:
                        path.mkdir()
                    self.assert_rejected_before_launch(source)

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "descriptor cleanup check requires Linux procfs")
    def test_snapshot_nonregular_rejection_does_not_leak_file_descriptors(self):
        source = self.copy_source("directory-descriptor")
        (source / "weights.npz").unlink()
        (source / "weights.npz").mkdir()

        before = self.source_descriptors()
        self.assert_rejected_before_launch(source)
        leaked = {fd: target for fd, target in self.source_descriptors().items()
                  if before.get(fd) != target}
        self.assertEqual(leaked, {}, "rejected submission left source descriptors open")

    def test_snapshot_rejects_extra_and_missing_files_before_launch(self):
        for name in ("manifest.json", "training_contract.json", "weights.npz"):
            with self.subTest(missing=name):
                source = self.copy_source("missing-" + name)
                (source / name).unlink()
                self.assert_rejected_before_launch(source)
        for name in ("reward.py", "private-track.json", "model.onnx"):
            with self.subTest(extra=name):
                source = self.copy_source("extra-" + name)
                (source / name).write_text("raise RuntimeError('must never execute')\n")
                self.assert_rejected_before_launch(source)
        source = self.copy_source("extra-directory")
        (source / ".hidden-inputs").mkdir()
        (source / ".hidden-inputs" / "private-track.json").write_text("must not be mounted")
        self.assert_rejected_before_launch(source)

    def test_snapshot_rejects_oversized_json_and_model_files_before_launch(self):
        for name, limit in (("manifest.json", 256 * 1024), ("training_contract.json", 256 * 1024),
                            ("weights.npz", 64 * 1024 * 1024), ("model.onnx", 64 * 1024 * 1024)):
            with self.subTest(name=name):
                source = self.copy_source("oversized-" + name)
                if name == "model.onnx":
                    (source / "weights.npz").unlink()
                # Sparse files exercise fstat limits without allocating 64 MiB.
                with (source / name).open("wb") as stream:
                    stream.truncate(limit + 1)
                self.assert_rejected_before_launch(source)

    def test_protocol_actual_worker_rejects_tampered_artifact_and_cleans_snapshot(self):
        for name in ("weights.npz", "training_contract.json"):
            with self.subTest(name=name):
                source = self.copy_source("tampered-" + name)
                with (source / name).open("ab") as stream:
                    stream.write(b"\n")
                with self.assertRaises(ValueError):
                    self.open_policy(source)
                self.assert_closed()


class ResponseFailureProtocolTests(LocalProtocolTestCase):
    """Faulty local subprocess peers test the proxy, not container isolation."""

    def test_protocol_startup_timeout_exit_and_oversize_cleanup(self):
        for mode, exception in (("startup_timeout", TimeoutError), ("startup_exit", ValueError),
                                ("startup_oversize", ValueError)):
            with self.subTest(mode=mode):
                self.use_faulty_peer(mode)
                with self.assertRaises(exception):
                    self.open_policy(startup_timeout=0.2 if mode == "startup_timeout" else 5)
                self.assert_closed()

    def test_protocol_requires_ready_message_to_identify_exact_copied_artifact(self):
        for ready in ({"ready": True, "artifact_sha256": "0" * 64}, {"ready": True},
                      {"ready": False, "artifact_sha256": "0" * 64}, {"error": "rejected"}):
            with self.subTest(ready=ready):
                self.use_faulty_peer("ready_response", ready)
                with self.assertRaises(ValueError):
                    self.open_policy()
                self.assert_closed()

    def test_protocol_action_timeout_exit_oversize_and_unterminated_response_cleanup(self):
        for mode, exception in (("action_timeout", TimeoutError), ("action_exit", ValueError),
                                ("action_oversize", ValueError), ("partial_response", TimeoutError),
                                ("partial_exit", ValueError),
                                ("partial_oversize", ValueError)):
            with self.subTest(mode=mode):
                self.use_faulty_peer(mode)
                policy = self.open_policy(action_timeout=0.2)
                with self.assertRaises(exception):
                    policy(self.observation)
                self.assert_closed()

    def test_protocol_rejects_bad_json_error_nonobject_and_wrong_response_fields(self):
        responses = ("not-json", "null", [], {"error": "untrusted payload"},
                     {"sequence": 1}, {"sequence": 1, "actions": [[0, 0]], "extra": True})
        for response in responses:
            with self.subTest(response=response):
                self.use_faulty_peer("response", response)
                policy = self.open_policy()
                with self.assertRaises(ValueError):
                    policy(self.observation)
                self.assert_closed()

    def test_protocol_accepts_fragmented_ready_and_action_jsonl(self):
        self.use_faulty_peer("fragmented", {"sequence": 1, "actions": [[-0.25, 0.5]]})
        with self.open_policy() as policy:
            np.testing.assert_array_equal(policy(self.observation), [[-0.25, 0.5]])
        self.assert_closed()

    def test_protocol_rejects_replayed_response_after_successful_action(self):
        self.use_faulty_peer("replayed_sequence", {"sequence": 1, "actions": [[0.0, 0.0]]})
        policy = self.open_policy()
        np.testing.assert_array_equal(policy(self.observation), [[0.0, 0.0]])
        with self.assertRaises(ValueError):
            policy(-self.observation)
        self.assert_closed()

    def test_protocol_rejects_wrong_sequence(self):
        for sequence in (0, 2, "1", 1.5, None):
            with self.subTest(sequence=sequence):
                self.use_faulty_peer("response", {"sequence": sequence, "actions": [[0.0, 0.0]]})
                policy = self.open_policy()
                with self.assertRaises(ValueError):
                    policy(self.observation)
                self.assert_closed()

    def test_protocol_requires_integer_response_sequence_type(self):
        for sequence in (True, 1.0):
            with self.subTest(sequence=sequence, type=type(sequence).__name__):
                self.use_faulty_peer("response", {"sequence": sequence, "actions": [[0.0, 0.0]]})
                with self.open_policy() as policy:
                    with self.assertRaises(ValueError):
                        policy(self.observation)
                self.assert_closed()

    def test_protocol_rejects_nonfinite_out_of_range_and_misshaped_actions(self):
        actions = ([[float("nan"), 0.0]], [[0.0, float("inf")]], [[-float("inf"), 0.0]],
                   [[1.01, 0.0]], [[0.0, -1.01]], [], [0.0, 0.0], [[0.0]],
                   [[0.0, 0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [["invalid", 0.0]])
        for action in actions:
            with self.subTest(actions=action):
                self.use_faulty_peer("response", {"sequence": 1, "actions": action})
                policy = self.open_policy()
                with self.assertRaises(ValueError):
                    policy(self.observation)
                self.assert_closed()

    def test_protocol_accepts_exact_action_bounds(self):
        self.use_faulty_peer("response", {"sequence": 1, "actions": [[-1.0, 1.0]]})
        with self.open_policy() as policy:
            np.testing.assert_array_equal(policy(self.observation), [[-1.0, 1.0]])
        self.assert_closed()

    def test_protocol_rejects_actions_just_outside_bounds(self):
        for action in ([[1.00000001, 0.0]], [[0.0, -1.00000001]]):
            with self.subTest(actions=action):
                self.use_faulty_peer("response", {"sequence": 1, "actions": action})
                with self.open_policy() as policy:
                    with self.assertRaises(ValueError):
                        policy(self.observation)
                self.assert_closed()

    def test_protocol_invalid_observations_fail_and_close_worker(self):
        observations = (np.zeros(23, dtype=np.float32), np.zeros((2, 23), dtype=np.float32),
                        np.zeros((1, 22), dtype=np.float32), np.zeros((1, 23), dtype=np.int32),
                        np.full((1, 23), np.nan, dtype=np.float32),
                        np.full((1, 23), np.inf, dtype=np.float32))
        for observation in observations:
            with self.subTest(shape=observation.shape, dtype=observation.dtype, first=observation.flat[0]):
                policy = self.open_policy()
                with self.assertRaises(ValueError):
                    policy(observation)
                self.assert_closed()

    def test_invalid_deadlines_do_not_launch_or_leave_snapshot(self):
        for option in ("startup_timeout", "action_timeout"):
            for value in (0, -1, 301, True, float("nan"), float("inf")):
                with self.subTest(option=option, value=value):
                    with self.assertRaises(ValueError):
                        self.open_policy(**{option: value})
                    self.command.assert_not_called()
                    self.assert_no_snapshots()

    def test_process_launch_failure_removes_snapshot(self):
        with mock.patch.object(isolated_policy.subprocess, "Popen", side_effect=FileNotFoundError("no docker")):
            with self.assertRaises(FileNotFoundError):
                self.open_policy()
        self.assertEqual(self.processes, [])
        self.assertTrue(self.launches)
        self.assert_no_snapshots()
        self.assertTrue(all(not snapshot.exists() for snapshot, _ in self.launches))

    def test_command_construction_failure_removes_snapshot_before_launch(self):
        self.command.side_effect = ValueError("invalid worker command")
        with self.assertRaises(ValueError):
            self.open_policy()
        self.command.assert_called_once()
        self.assertEqual(self.processes, [])
        self.assert_no_snapshots()

    def test_protocol_cleanup_tolerates_container_removal_failures(self):
        for error in (FileNotFoundError("no docker"), subprocess.TimeoutExpired(["docker", "rm"], 10)):
            with self.subTest(error=type(error).__name__):
                self.removal.side_effect = error
                self.use_faulty_peer("action_exit")
                policy = self.open_policy()
                with self.assertRaises(ValueError):
                    policy(self.observation)
                policy.close()
                self.assert_closed()

    def test_protocol_context_exception_and_repeated_close_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, "test body"):
            with self.open_policy() as policy:
                policy(self.observation)
                raise RuntimeError("test body")
        policy.close()
        policy.close()
        self.assert_closed()

    def test_protocol_cleanup_kills_and_reaps_worker_ignoring_stdin_eof(self):
        self.use_faulty_peer("stubborn")
        policy = self.open_policy(action_timeout=0.2)
        with self.assertRaises(TimeoutError):
            policy(self.observation)
        self.assert_closed()
        self.assertEqual(self.processes[-1].returncode, -signal.SIGKILL)


@unittest.skipUnless(os.environ.get("SYNAPSE_TEST_DOCKER") == "1",
                     "real Docker integration is opt-in: SYNAPSE_TEST_DOCKER=1; requires a local image")
class DockerIntegrationTests(ArtifactTestCase):
    def setUp(self):
        super().setUp()
        # Host Docker can see the shared checkout, not this container's /tmp.
        parent = ROOT / "artifacts"
        parent.mkdir(exist_ok=True)
        runtime = tempfile.TemporaryDirectory(prefix="docker-protocol-", dir=parent)
        self.addCleanup(runtime.cleanup)
        self.runtime = Path(runtime.name)
        self.enterContext(mock.patch.object(isolated_policy, "ROOT", self.runtime))

    def test_opt_in_real_docker_worker_round_trip(self):
        self.assertIsNotNone(shutil.which("docker"), "Docker opt-in requires the Docker CLI and daemon")
        image = os.environ.get("SYNAPSE_TEST_DOCKER_IMAGE", "synapse-evaluator:v1")
        expected = load_submission(self.source)
        # No mocked Docker boundary here. docker_command enforces --pull=never
        # and --network=none, so this check cannot download an absent image.
        with isolated_policy.IsolatedPolicy(self.source, image=image) as policy:
            self.assertEqual(policy.artifact_sha256, expected.artifact_sha256)
            np.testing.assert_array_equal(policy(self.observation), expected(self.observation))
        self.assert_no_snapshots()


if __name__ == "__main__":
    unittest.main()
