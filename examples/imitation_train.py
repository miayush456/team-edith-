"""Starting point for PyTorch imitation from the scripted teacher."""

from tools.train_scripted_imitation import train

# Use the existing CLI for a complete teacher-data collection and training
# loop. Replace the fixed actor with participant_code/model.py for BYOA PPO.
if __name__ == "__main__":
    import sys
    print("Run: .venv/bin/python tools/train_scripted_imitation.py --output-dir artifacts/runs/imitation")
