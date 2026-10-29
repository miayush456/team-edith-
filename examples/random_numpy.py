"""Sketch of a custom NumPy algorithm using the SYNAPSE SDK facade."""

import numpy as np
import synapse


args = synapse.parse_args()
env = synapse.make_env(args)
weights = np.zeros((2, env.observation_size), dtype=np.float32)
bias = np.asarray([0.0, 0.5], dtype=np.float32)
for _ in range(max(1, args.steps // max(1, args.rollout_length))):
    observation, _ = env.reset(seed=args.seed)
    action = np.tanh(observation @ weights.T + bias)
    env.step(action)

# Convert NumPy layers to the declarative artifact with racing_event.save_layers
# for a non-Torch submission; the SDK save facade is intended for Torch actors.
print("random-search template collected budgeted transitions")
