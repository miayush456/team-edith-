# Participant guide

This guide expands the README's first-run path for the reference PPO and the
bring-your-own-architecture workflow.

## Architecture choices

Use the reference baseline at `tools/run_reference_baseline.py`; omit `--model`
for the bundled fixed actor or pass `--model participant_code/model.py` for a
custom architecture. A model file defines `build_actor(obs_size)`
and may define `build_critic(obs_size)`. The actor returns pre-tanh means; the
platform owns the tanh-normal action distribution and critic. A deterministic
replay/evaluation uses `tanh(mean)`.

```bash
.venv/bin/python tools/train_participant.py \
  --model participant_code/model.py \
  --hparams examples/ppo_hparams.json \
  --reward examples/baseline_reward.json \
  --track-dir tracks/train --config config/simulation.json \
  --output-dir artifacts/runs/custom-001 --steps 256 \
  --budget-steps 10000 --n-envs 2 --rollout-length 32 --seed 17
```

The model file hash and parameter count are stored in the run identity. Changing
the model, rays, config, reward, or hparams requires a new output directory.

## Exact reference resume and reward forks

Reuse the same directory with the same identity to resume. To start a fresh
reward experiment from a saved actor, use `--init-from` and a new output
directory. A fork starts a new optimizer and consumes one client allocation.

## Model checker

Run the same checker used by the dashboard before spending training transitions:

```bash
make check-model MODEL=participant_code/model.py
```

It imports the model in a bounded subprocess, validates shape/finite outputs,
exports and validates ONNX, checks Torch/ONNX parity, counts parameters/nodes,
and measures single-thread latency against the 20 ms live budget.

## SDK (advanced local research)

The participant workflow above is the supported event-training path. The
optional `synapse` facade is for local algorithm experiments; it reserves
active-lane transitions before every step, but it does not replace the
reference baseline's enforced reward/PPO path.

## Selection and evaluation

Public-map selection is diagnostic only. Use the dashboard's both-map evaluation
or evaluate Monaco and Spa separately. It is not event ranking and does not
spend training transitions.
