"""Create/check the local SYNAPSE participant Python environment.

The script keeps dependencies inside the project `.venv`. It is intentionally
small and explicit so a participant who starts the UI with system Python gets a
clear setup path instead of a raw ``ModuleNotFoundError``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
REQUIREMENTS = ROOT / "requirements.lock"
OPTIONAL_REQUIREMENTS = (ROOT / "requirements-evaluator.lock", ROOT / "requirements-live.lock")


def _python_has_dependencies(python: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import os; os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'; "
                "import numpy, pygame, onnx, onnxruntime, websockets; "
                "import torch; "
                "print(f'numpy={numpy.__version__} pygame={pygame.version.ver} torch={torch.__version__} onnx={onnx.__version__} onnxruntime={onnxruntime.__version__} websockets={websockets.__version__}')",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    detail = (result.stdout or result.stderr).strip()
    return result.returncode == 0, detail


def _require_supported_python() -> None:
    if os.name != "posix":
        raise RuntimeError(
            "SYNAPSE's local participant tooling currently requires POSIX/Linux "
            "file locking; native Windows is unsupported. Use WSL or a Linux environment."
        )
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "SYNAPSE supports Python 3.11; run this helper with python3.11 "
            "so the project virtual environment uses the pinned runtime"
        )


def setup_environment() -> str:
    _require_supported_python()
    if VENV_PYTHON.exists():
        ready, detail = _python_has_dependencies(VENV_PYTHON)
        if ready:
            return f"SYNAPSE environment ready: {detail}"
    print(f"Creating project environment with {sys.executable} ...")
    builder = venv.EnvBuilder(with_pip=True, clear=False)
    builder.create(VENV_DIR)
    if not VENV_PYTHON.exists():
        raise RuntimeError(f"virtual environment did not create {VENV_PYTHON}")
    print("Installing pinned SYNAPSE environment and reference dependencies ...")
    subprocess.run(
        [str(VENV_PYTHON), "-m", "pip", "install", "--requirement", str(REQUIREMENTS)],
        cwd=ROOT,
        check=True,
    )
    for optional in OPTIONAL_REQUIREMENTS:
        print(f"Installing optional evaluator/live dependencies from {optional.name} ...")
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", "--requirement", str(optional)],
            cwd=ROOT,
            check=True,
        )
    ready, detail = _python_has_dependencies(VENV_PYTHON)
    if not ready:
        raise RuntimeError(f"dependency check failed: {detail}")
    return f"SYNAPSE environment ready: {detail}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only check the project environment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        _require_supported_python()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.check:
        if not VENV_PYTHON.exists():
            raise SystemExit(f"SYNAPSE environment is missing. Run: {sys.executable} tools/setup_participant.py")
        ready, detail = _python_has_dependencies(VENV_PYTHON)
        if not ready:
            raise SystemExit(f"SYNAPSE environment is incomplete: {detail}\nRun: {sys.executable} tools/setup_participant.py")
        print(f"SYNAPSE environment ready: {detail}")
        return
    try:
        print(setup_environment())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "SYNAPSE setup failed. Check network access and Python 3.11, then retry "
            f"'{sys.executable} tools/setup_participant.py'. Details: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
