# Participant code template

`model.py` is the smallest pluggable-architecture example. `build_actor(obs_size)`
returns a PyTorch module whose two outputs are **pre-tanh** steering and signed
throttle/brake means. SYNAPSE supplies the critic during PPO training and adds
the final `tanh` when exporting to ONNX.

Only the validated ONNX graph is evaluated by the organizer. Model and trainer
files execute locally during training/checking, so treat them as code you own
and review. The supported runtime layer set is listed in
[`docs/submission_runtime.md`](../docs/submission_runtime.md).
