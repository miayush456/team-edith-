"""Run the optional, dependency-free JS checks through the project test entry."""

from pathlib import Path
import shutil
import subprocess
import unittest


class ReplayFrontendTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is optional; install it to run playback JS checks")
    def test_playback_logic_and_dom_controls(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [shutil.which("node"), "--test", "tests/replay_frontend.test.cjs"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("# fail 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
