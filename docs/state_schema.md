# SYNAPSE state and artifact schema

The default observation is `float32[23]`: nine self-state/control values, one
value per configured boundary ray, four curvature lookaheads, and local width.
All fields are statically normalized or clipped to their published bounds.
`obs_size = 10 + ray_count + lookahead_count`; ray order is part of the contract.

The action is `float32[2]` in `[-1, 1]`: `steering` followed by signed
`throttle_brake` (positive throttle, zero coast, negative brake).

Training contracts record observation fields, ray angles, config hash, public
training track IDs/hashes, deterministic evaluation settings, and the reward
config hash. Reference training always applies all ten published reward
signals; zero is the explicit way to disable one signal. ONNX artifacts embed any trainer-side normalizer in the
graph; the submitted contract therefore contains no separate normalizer.

The safe NumPy artifact contains exactly `manifest.json`,
`training_contract.json`, and `weights.npz`. The manifest hashes the latter two
files and declares a sequential dense graph with `identity`, `relu`, or `tanh`
activations. The final layer must be two `tanh` outputs.
