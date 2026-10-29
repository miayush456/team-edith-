# Submission runtime profile

The live submission is a model-only ONNX artifact containing exactly
`manifest.json`, `training_contract.json`, and `model.onnx`. The graph has one
float32 input named `observations` with shape `[1, N]` and one float32 output
named `actions` with shape `[1, 2]`. Opset 17 and IR version 10 are required;
the final node is `Tanh`.

The whitelist is intentionally small: `Gemm`, `MatMul`, `Add`, `Sub`, `Mul`,
`Div`, `Min`, `Max`, `Relu`, `Tanh`, `Sigmoid`, `Clip`, `Identity`, and `Cast`.
Models are bounded to 128 nodes, one million parameters, static rank-two tensors,
and bounded tensor allocations. Convolution, recurrence, attention, arbitrary
custom operators, normalization modules, and dynamic shapes are not in the
current profile. Skip connections built from `Add` are supported.

The checker enforces this exact profile, deterministic CPU inference, and
Torch/ONNX parity. The organizer evaluator should still run untrusted packages
in the model-only isolated worker; graph validation is not a security sandbox.

An imported model is marked `source_kind: imported` and records that its
training-map provenance is a participant claim. The evaluator never imports
participant Python, reward callbacks, pickle files, or trainer checkpoints.
