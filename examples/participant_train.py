"""Compatibility wrapper for the single SYNAPSE reference PPO trainer.

All participant training uses ``tools/run_reference_baseline.py`` so the nine
dashboard reward weights are applied consistently. Use ``--model`` to supply
a custom actor architecture; change the reference baseline when changing PPO
behavior.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_reference_baseline import main

if __name__ == "__main__":
    main()
