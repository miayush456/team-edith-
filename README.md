# SYNAPSE Participant Manual

Welcome. This file is the beginner-friendly path through the SYNAPSE racing
package. You do **not** need previous racing, reinforcement-learning, or
command-line experience to follow it. Start at step 1 and copy the commands
exactly. Later sections explain what each command does and how to replace the
starter model with your own training program.

> **Important idea:** Python is the authority. Your model chooses two actions;
> Python applies those actions to the physics, decides whether the car crashed
> or finished, and writes the result. The renderer only displays recorded
> state. It cannot steer a car, change a reward, or change a result.

![Annotated first-run map: the trainer script, public maps, and output folder](docs/assets/participant-start.svg)

## What you will do

You will complete this sequence:

1. Open a terminal in this repository.
2. Install the local Python environment and optional evaluator/live dependencies.
3. Choose an actor architecture. The reference PPO trainer is already included.
4. Run a **tiny** training experiment so failures are cheap.
5. Follow the live rollout position in the dashboard, then replay what the car actually did.
6. Change one parameter at a time and repeat.
7. Export the best saved model to the validated ONNX format.
8. Build a model-only ONNX submission package.
9. Run two submissions against each other in a Python-authoritative live heat.

The local public maps are:

```text
tracks/train/monaco.json
tracks/train/spa.json
```

They are the only participant-facing training maps in this checkout. The map
manifest is `tracks/train_manifest.json`.

## 1. The four words you need to know

| Word | Plain-English meaning | Example in this repository |
|---|---|---|
| **Reference trainer** | The single supported training path. It applies the ten reward weights, runs PPO, and saves the result. | `tools/run_reference_baseline.py` |
| **Actor model** | Optional participant-owned PyTorch architecture supplied to the reference trainer. | `participant_code/model.py` |
| **Run directory** | A folder containing one training attempt and its saved weights/metadata. | `artifacts/participant/manual-monaco` |
| **Artifact** | A model saved in a format that the local evaluator can read. | `policy.pt` + `policy.json` + `training_contract.json` |
| **ONNX submission package** | A new folder containing `model.onnx`, its contract, matching config snapshot, hashes, and provenance. It contains no track or replay. | `artifacts/onnx-packages/team-a` |

The word **model script** can be confusing. In this manual, `model.py` defines
the actor architecture; `tools/run_reference_baseline.py` is the program that
trains it. The trained weights are saved later in the run directory. You do not
point `--model` at a `.pt`, `.onnx`, or `.npz` file.

## 2. Open the repository correctly

### 2.1 Find the repository folder

Open a terminal:

- Linux: open **Terminal**.
- macOS: open **Terminal**.
- Windows: use WSL, Git Bash, or the Linux environment supplied for this
  project.

Change directory to the folder that contains this README. For example:

```bash
cd /path/to/Synapse
```

Check that you are in the right place:

```bash
pwd
ls
```

You should see names such as `README.md`, `tools`, `tracks`, `config`, and
`examples`. If `ls` does not show those names, use `cd` again. All relative
paths in the commands below are relative to this repository root.

### 2.2 Check Python

SYNAPSE supports Python 3.11:

```bash
python3.11 --version
```

The result should begin with `Python 3.11`. If your computer has another
Python version as well, always use `python3.11` for setup and `.venv/bin/python`
for project commands.

### 2.3 Create the project environment

Run:

```bash
python3.11 tools/setup_participant.py
```

This creates `.venv/` and installs the locked CPU dependencies. A GPU is not
required. The training UI has a separate CUDA processor option and checks GPU
readiness when selected. If setup succeeds, check the environment with:

```bash
.venv/bin/python --version
.venv/bin/python -c "import numpy; print('NumPy is ready')"
```

You can also use the dashboard's **Set up / repair Python** button later. Do
not delete `.venv` between every experiment; it is the shared project
environment, not a model output folder.

### 2.4 Validate the public maps

Run:

```bash
.venv/bin/python tools/build_training_manifest.py
```

You should see `validated 2 public training maps` and a list containing
`monaco` and `spa`. This command checks geometry, hashes, widths, boundaries,
and the published training contract. It does not train a model.

## 3. Choose the actor architecture

Every participant training run uses the same reference PPO entrypoint:

```text
tools/run_reference_baseline.py
```

This is intentional. It is the single place where SYNAPSE loads the
environment, applies the ten UI reward weights, collects budgeted rollouts,
updates PPO, and writes the checkpoint. The fixed starter actor and every
custom actor use this same path. The compatibility wrapper
`examples/participant_train.py` delegates to it, but you do not need to copy or
replace that wrapper.

To change the network architecture, provide a model file:

```text
participant_code/
  model.py                 # build_actor(obs_size), optional build_critic(obs_size)
```

The launcher imports `build_actor(obs_size)` locally, wraps it with the
reference PPO distribution and critic, and passes it to
`tools/run_reference_baseline.py`. The actor returns two pre-tanh means in
steering/throttle-brake order. See [`participant_code/model.py`](participant_code/model.py).

If you want to change PPO behavior, change the reference implementation in
`tools/run_reference_baseline.py` (and its explicitly imported PPO helper when
an algorithm-level primitive changes). Do not create a second trainer: the
launcher rejects arbitrary trainer paths so the UI reward weights cannot be
bypassed.

### 3.1 Reference training inputs

`tools/train_participant.py` forwards these options to the reference baseline:

| Argument | What it tells your script | Example |
|---|---|---|
| `--track-dir` | Directory containing the complete public `*.json` map set. | `tracks/train` |
| `--config` | The simulation and sensor configuration file. | `config/simulation.json` |
| `--output-dir` | Where your script must save the run. | `artifacts/runs/team-a` |
| `--steps` | Maximum transitions requested in this invocation. | `256` |
| `--budget-steps` | Persistent maximum allowed by this local run ledger (up to 1,000,000). | `1000000` |
| `--n-envs` | Number of parallel environment lanes requested. | `2` |
| `--rollout-length` | Number of transitions collected per lane before an update. | `32` |
| `--seed` | Reproducibility seed. | `17` |
| `--device` | PPO processor (`cpu` or `cuda`); the simulator remains CPU-side. | `cpu` |
| `--reward` | The ten reward weights used by every PPO rollout. | `examples/baseline_reward.json` |
| `--model` | Optional workspace-local `model.py` exposing `build_actor(obs_size)`. | `participant_code/model.py` |
| `--hparams` | Optional JSON snapshot of PPO hyperparameters. | `examples/ppo_hparams.json` |

The launcher supplies the complete public map set, the simulation contract,
the persistent transition budget, the selected ten-weight reward file, and
the optional actor/hyperparameter files. `--trainer` remains only as a
compatibility flag for the reference path; arbitrary Python trainers are
rejected. The default PPO and reward behavior are therefore reproducible from
one script.

![Annotated trainer structure: parse arguments, load data, update the model, and save an artifact](docs/assets/training-script-shape.svg)

### 3.2 What your trainer should save

For the included reference path, the run directory must eventually contain:

```text
your-run/
  policy.pt
  policy.json
  training_contract.json
```

The reference trainer may also create checkpoint generations, metrics,
`training_report.md`, `budget.json`, and `run_identity.json`. Keep the entire
run directory; the extra files are useful evidence and resume state.

For the safe declarative NumPy path, the artifact directory must contain
**exactly** these three regular files:

```text
numpy-artifact/
  manifest.json
  training_contract.json
  weights.npz
```

The NumPy manifest describes a sequential MLP. Supported activations are
`identity`, `relu`, and `tanh`; the final layer has two `tanh` outputs. Hashes
must match the exact files. The loader rejects extra files, symlinks, object
arrays, non-finite values, bad shapes, and unsupported layers. The full
declarative contract is in [`docs/state_schema.md`](docs/state_schema.md).

If you use a framework or architecture that is not the bundled reference
model, do not assume that a raw framework checkpoint can be packaged. Convert
it to the supported declarative artifact or produce an ONNX graph that obeys
the [ONNX profile](docs/submission_runtime.md#onnx-profile-v1). The evaluator
does not import your training Python, load pickle files, or execute your reward
code.

### 3.3 Optional progress output

The dashboard can display a progress line when your trainer prints lines like:

```text
update 1: steps 128/2048 return=12.4 lap=n/a speed=18.2 m/s crash=0 off_track=0.03
```

The exact metrics are yours. The important part is that your program writes
normal text to standard output, flushes periodically, and exits with status 0
when training succeeds. A non-zero exit code is shown as a failed run.

## 4. Run a tiny first experiment

Start small. A 256-transition run is for checking paths and file formats, not
for producing a fast racer.

### 4.1 Train on both public maps

This command runs the reference PPO baseline with its default actor:

```bash
.venv/bin/python tools/train_participant.py \
  --reward examples/baseline_reward.json \
  --track-dir tracks/train \
  --config config/simulation.json \
  --output-dir artifacts/runs/first-check \
  --steps 256 \
  --budget-steps 1000000 \
  --n-envs 2 \
  --rollout-length 32 \
  --seed 17
```

Read the command from left to right:

1. `.venv/bin/python` uses the installed project environment.
2. `tools/train_participant.py` is the safe launcher around the reference baseline.
3. `--reward ...` selects the ten weights used by PPO for every transition.
4. `--track-dir ...` exposes Monaco and Spa.
5. `--config ...` selects the sensor/physics configuration.
6. `--output-dir ...` selects a new folder for this experiment.
7. `--steps 256` makes this invocation short.
8. `--budget-steps 1000000` creates the local transition ceiling.
9. `--n-envs 2` asks for two parallel lanes.
10. `--rollout-length 32` asks for 32 transitions per lane per update.
11. `--seed 17` makes this attempt reproducible.

To use a custom actor while keeping the same PPO and reward path, add:

```bash
  --model participant_code/model.py
```

When it succeeds, inspect the output:

```bash
find artifacts/runs/first-check -maxdepth 2 -type f -printf '%P\n' | sort
```

For the reference run you should see `policy.pt`, `policy.json`, and
`training_contract.json` among the results. If you do not, stop and fix the
reference baseline before spending more transitions.

### 4.2 Training always uses the complete public set

The participant launcher requires `tracks/train` to contain every public
training map. In this release that means both Monaco and Spa. It rejects a
directory containing only Monaco or only Spa, because a model trained on one
map can learn that map's geometry instead of learning a route-following policy
that transfers to unseen competition maps.

Use this for participant training:

```bash
--track-dir tracks/train
```

The circuit selectors later in this manual are intentionally separate. Choose
one map when you replay, evaluate, package, or race; do not choose one map for
the training run. If you want to inspect Monaco alone, use the evaluator or
replay command with `--track tracks/train/monaco.json` after training.

### 4.3 Understand steps and budget

These two numbers are different:

- `--steps` is the amount requested **this time you start the command**.
- `--budget-steps` is the persistent local ceiling for the run directory.

For example, `--steps 256 --budget-steps 1000000` requests 256 transitions and
allows future reference-starter resumes to spend up to 1,000,000 accepted
transitions in that run ledger. `n_envs` lanes are charged individually. A
vector call with two active lanes costs two transitions.

The local onboarding budget is a prototype fairness mechanism, not a claim
that the final event budget has been frozen. Do not edit `budget.json` to try to
refund work. The reference baseline owns the training budget accounting for
every actor architecture.

### 4.4 Client retrain cap

The participant launcher allows **12 training iterations** on one client. Each
iteration funds up to **1,000,000 additional transitions**. A new output
directory spends one iteration; resuming that same directory does not. When a
run needs more training, pause it and choose **Spend 1 iteration · add
1,000,000 steps to this model**. This raises that checkpoint's transition cap
and continues from its actor, optimizer, normalizer, environment state, and
random-number state. If training already completed, the same action resumes
from the last saved checkpoint.
The step target is normally frozen once training starts; this explicit extension
is the only UI action that raises it, and it is recorded against one of the 12
iterations. The extension preserves the current PPO settings and continues
training.

Use **Pause** and **Continue** to suspend and resume the live trainer process.
Pause freezes the process in place, including its optimizer and active rollout;
Continue resumes that same process state. Stop terminates immediately. If Stop
interrupts an update after its environment steps were reserved, exact checkpoint
resume may be refused to avoid silently reusing charged transitions; use
Pause/Continue when you need an interruption-free hold.

An allocation is identified by its output directory. Reusing that same
directory to resume a compatible run does not consume another allocation; a
new output directory is a new retrain and uses one slot. The local ledger is:

```text
.synapse/client_training_budget.json
```

The dashboard shows the remaining iteration count. This is a client-side
operational guard, not organizer-grade proof: someone who controls the client
can delete or edit local files. Organizer-side evaluation must enforce the
actual competition budget separately.

When upgrading from a lower client cap, SYNAPSE raises the ledger cap while
preserving its allocation history. Existing output folders keep the budget
assigned when they were first allocated; choose a new output folder to start a
run with the 1,000,000-step limit. Spend an iteration to extend an existing
checkpoint instead of creating a new output folder when you want more training
for that model.

### 4.5 Research-based starting point for training

SYNAPSE provides a compact, low-dimensional state vector with car telemetry,
ray distances, and track lookahead, plus two bounded continuous controls. Start
with a feed-forward MLP rather than a vision CNN or recurrent network. Two
64-unit hidden layers are a sensible baseline; compare two 128-unit layers if
validation shows underfitting. Stable-Baselines3 uses two 64-unit layers as a
standard PPO MLP starting point for 1D observations. Racing studies use a range
of policy and sensor designs; none establishes the best architecture for
SYNAPSE.

Use **1,000,000 environment transitions as an initial experiment ceiling**, not
a promise of convergence. Continuous-control performance varies substantially
by environment and reward design. The reference trainer now uses shuffled
64-transition PPO minibatches, 10 update epochs, and a `3e-4` learning rate;
these are stronger starting settings, not tuned winners. The dashboard's
default two environments and 32 steps per environment still collect 64
transitions per update. Compare completion, distance, lap time, speed, crashes,
and off-track time across several seeds. Training does not stop early when
progress plateaus: it uses the requested transition count unless you pause or
stop it. Plan the cap and check progress through the live map and metrics.

Each run evaluates policy snapshots in the background and retains the best
deterministic replay it has evaluated at `best_replay.json` in the run folder.
The replay panel's **Training run · best captured replay** source plays it
directly, including for a custom actor, without packaging an artifact first.
Replay capture can lag behind training when evaluation is slower than updates;
the file represents the best evaluated snapshot, not necessarily the newest
policy. It is for inspection and does not select or replace the trained policy.

The PPO and reward settings are part of a run's identity. Keep them unchanged
when resuming an exact checkpoint. If you deliberately change the PPO setup,
reward weights, sensors, or actor architecture, start a new output directory so
the experiment is recorded as a separate run.

References: [Stable-Baselines3 policy network guidance](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html),
[Stable-Baselines3 PPO parameters](https://stable-baselines3.readthedocs.io/en/v1.0/modules/ppo.html),
[a LiDAR neural-controller verification case study](https://arxiv.org/abs/1910.11309).

The training UI defaults to CPU and also offers **GPU · CUDA**. GPU selection
places the policy, PPO update tensors, and optimizer on CUDA; the simulator and
rollout collection remain CPU-side. The trainer checks CUDA availability in the
selected Python environment and reports the missing PyTorch/CUDA/driver/device
component instead of silently falling back to CPU. CPU and GPU checkpoints have
different runtime identities, so resume a run on its original processor.

Teacher demonstrations used by the optional scripted/imitation helpers are
local diagnostic data and are not event submissions. Participant environment
transitions collected through `tools/train_participant.py` or the `synapse`
SDK are the budgeted unit; the SDK reserves them before each active-lane step.
Deleting `.synapse/` or re-cloning resets the local guard.

## 5. Use the dashboard instead of typing commands

The dashboard is optional. It is a local form around the same launcher; it is
not a second simulator or a second model.

Start it from the repository root:

```bash
.venv/bin/python tools/participant_ui.py --port 8766
```

Open this address in a browser on the same computer:

```text
http://127.0.0.1:8766/
```

Keep the terminal open while the dashboard is running. It launches local code
and is intentionally bound to loopback; do not expose it to the internet.

### 5.1 Fill in the training form

In the training panel:

1. **Reference training script:** leave
   `tools/run_reference_baseline.py` selected. It is fixed so the ten UI
   weights are used by every PPO rollout.
2. **Actor model:** optionally enter `participant_code/model.py` to change the
   network architecture without changing PPO or reward handling.
3. **Training maps:** leave `tracks/train`; this always means Monaco and Spa
   together. The training launcher rejects a one-map directory.
4. **Output directory:** type a new folder such as
   `artifacts/runs/dashboard-first`.
5. **Steps:** enter `256` for the first smoke run.
6. **Budget steps:** the default ceiling is `1000000`; training uses the full
   requested transition count unless stopped or extended.
7. **Parallel environments:** enter `1` or `2` while learning the workflow.
8. **Rollout length:** enter `32` for a small test.
9. **Seed:** enter `17`. Use a different seed only for a deliberately new
   experiment.
10. **Reward function:** tune the ten weights; the reference baseline applies
    them directly to each training transition.
11. **Ray sensors:** leave the default nine angles in place until the baseline
    works.
12. Press **Start / resume participant training** once.

Before launching, the dashboard writes `run_config.json`, `reward_config.json`,
and `run_identity.json` beside the run. The reward file records the selected
signal weights. Checkpoint weight gives a one-time bonus for each of 16 zones
reached per episode; its starter weight is one. During training, the status
panel plots the latest rollout position for each parallel environment. It also displays the remaining client
retrain allocations. This is why dashboard-created runs are convenient for
later replay and packaging.

While training, use the **Latest driven position** map to follow each parallel
environment. Select a lane to inspect its car on Monaco or Spa; the forward
marker, lap-distance, speed, and time-in-episode readout update after each PPO
rollout. To watch a complete deterministic rollout from an evaluated policy,
use **Training run · best captured replay** in the replay panel. This works
with custom architectures and does not require packaging a participant
artifact. The best replay is saved at the run folder's `best_replay.json` path;
the adjacent `best_replay_summary.json` records the captured update and result.
The map holds the lane's last driven position if an episode ends at the rollout
boundary,
so it does not jump to a randomized spawn before the trainer can inspect the
crash or final lap position. Going off the road onto runoff does not itself
respawn the car; it can recover. Barrier contact, lap completion, driving too
far backward, getting stuck, or reaching the episode step limit ends that
episode and starts another when collection continues.

## 6. Sensors and actions: what the model sees and does

With the default configuration, every observation is a vector of 23 `float32`
values:

```text
9 driving-state values
9 boundary-ray values
4 curvature-lookahead values
1 local-width value
```

The exact default order is:

```text
speed, longitudinal_velocity, lateral_velocity, yaw_rate,
lateral_offset, heading_error, previous_steering,
previous_longitudinal, steering_angle,
ray_0 ... ray_8,
curvature_lookahead_0 ... curvature_lookahead_3,
local_width
```

The full field table, ranges, and units are in
[`docs/state_schema.md`](docs/state_schema.md). The observations are
track-relative: there is no absolute x/y position, absolute heading, track ID,
or absolute lap-progress value.

The model returns exactly two finite numbers in `[-1, 1]`, in this order:

```text
action[0] = steering
action[1] = throttle_brake
```

For `throttle_brake`, positive means throttle, zero means coast, and negative
means braking. Do not swap the two outputs. Do not return NaN or infinity.

### 6.1 Optional ray changes

The UI allows 1–16 unique ray angles from `-90` through `90` degrees. The
angle list must include `0`, the forward center ray. For example:

```text
-30,0,30
```

creates three rays and therefore a 17-value observation (`10 + 3 + 4`). The
default list is:

```text
0,-15,15,-30,30,-60,60,-90,90
```

Changing rays changes the observation size and the configuration hash. Start a
new output directory after changing them. Do not resume a run that was trained
with a different ray order. The contract is designed to reject that mismatch
instead of silently feeding the model the wrong numbers.

## 7. Reward design and parameter tuning

SYNAPSE publishes raw transition signals, and the reference baseline converts
the ten dashboard weights into the PPO reward on every training transition.
The checkpoint signal fires once the first time an episode reaches each of 16
equal-distance zones. Its default reward is +1 per newly reached zone, adding a
progress milestone bonus on top of the dense percentage-of-lap progress signal.
The dashboard reports average zones reached during each training update,
alongside lap completion and return.
This keeps reward tuning visible, reproducible, and shared by the default and
custom-architecture paths. Participant training does not provide a second
reward callback or a second trainer; change `tools/run_reference_baseline.py`
when the reference PPO behavior itself needs to change.

The starter reward file is:

```json
{
  "schema_version": 1,
  "weights": {
    "progress": 12.0,
    "speed": 0.5,
    "curvature_matched_speed": 0.25,
    "off_track": -1.5,
    "crash": -10.0,
    "stuck": -8.0,
    "lap_complete": 20.0,
    "time": -0.15,
    "steer_smoothness": -0.05
  }
}
```

Plain-English meanings:

| Signal | Meaning | Typical symptom when too weak |
|---|---|---|
| `progress` | Reward for moving forward along the route. | The car does not try to advance. |
| `speed` | Reward related to forward speed. | The car is safe but very slow. |
| `curvature_matched_speed` | Encourages speed appropriate to the bend. | The car enters corners too fast. |
| `off_track` | Penalty for leaving the bounded road. | The car cuts across runoff. |
| `crash` | Penalty for a crash/contact event. | The car repeatedly hits barriers. |
| `stuck` | Penalty for failing to move. | The car sits at the start. |
| `lap_complete` | Bonus when the episode completes its lap target. | The car gives up before a lap. |
| `time` | Small pressure against wasting time. | The car drives safely but never commits. |
| `steer_smoothness` | Penalty for abrupt steering changes. | The car oscillates left/right. |

### 7.1 The recommended tuning loop

![Annotated one-change-at-a-time tuning loop](docs/assets/tuning-loop.svg)

Use this loop instead of changing five values and guessing which one helped:

1. **Baseline:** run 256 transitions with the starter reward, seed 17, default
   rays, and the complete public training set (`tracks/train`). Replay or
   evaluate one map at a time after training.
2. **Inspect:** build a replay. Look for a crash, a stuck start, excessive
   steering oscillation, or a car that is safe but motionless.
3. **Change one input:** for example, change only `off_track` from `-1.5` to
   `-3.0`. The reference baseline receives the resulting weight through
   `reward_config.json`; the actor architecture and PPO stay fixed.
4. **Use a new output directory:** for example,
   `artifacts/runs/off-track-stronger`. This prevents incompatible run identity
   files from being mixed together.
5. **Retrain:** keep all other values, map choices, and seed the same for the
   first comparison.
6. **Replay and evaluate:** compare the actual driving, not only the printed
   training return.
7. **Keep or revert:** record the change in a notebook or text file, then try
   the next single change.
8. **Check both maps:** a change that helps Monaco but makes Spa crash is not a
   reliable improvement.

### 7.2 Example: stronger off-track penalty

Copy the reward file, edit one number, and run a fresh directory:

```bash
cp examples/baseline_reward.json participant_code/reward_off_track.json
# Edit only "off_track": -1.5 to "off_track": -3.0

.venv/bin/python tools/train_participant.py \
  --reward participant_code/reward_off_track.json \
  --track-dir tracks/train \
  --config config/simulation.json \
  --output-dir artifacts/runs/off-track-stronger \
  --steps 2048 --budget-steps 1000000 \
  --n-envs 2 --rollout-length 32 --seed 17
```

Do not reuse `artifacts/runs/first-check` for a different reward file. A saved
run identity deliberately rejects that combination.

### 7.3 Example: compare two seeds

Keep the reward and maps fixed and change only the seed and output folder:

```bash
.venv/bin/python tools/train_participant.py \
  --reward examples/baseline_reward.json \
  --track-dir tracks/train --config config/simulation.json \
  --output-dir artifacts/runs/seed-17 \
  --steps 2048 --budget-steps 1000000 --n-envs 2 --rollout-length 32 --seed 17

.venv/bin/python tools/train_participant.py \
  --reward examples/baseline_reward.json \
  --track-dir tracks/train --config config/simulation.json \
  --output-dir artifacts/runs/seed-23 \
  --steps 2048 --budget-steps 1000000 --n-envs 2 --rollout-length 32 --seed 23
```

Two seeds are two separate experiments. They are not two models inside one
run directory.

### 7.4 Safe starter values for reference PPO

For the reference trainer, begin here:

| Parameter | First smoke | Small experiment | Why |
|---|---:|---:|---|
| `steps` | 256 | 2,048 or more | Invocation work, not the lifetime ledger. |
| `budget-steps` | 1,000,000 | 1,000,000 | Local per-run transition ceiling. |
| `n-envs` | 1 | 2 or 4 | More lanes collect more data per update. |
| `rollout-length` | 16 or 32 | 32 or 64 | Must multiply with `n-envs` to at most 8,192. |
| `seed` | 17 | 17, then 23 | Repeatability and variability checks. |
| `learning-rate` | `0.0003` | `0.0003` | PPO actor and critic optimizer rate. |
| `update-epochs` | `2` for a smoke | `10` default | Each epoch shuffles into 64-sample minibatches. |

Change one row at a time. A large `n-envs * rollout-length` product is not
automatically better; it uses more memory and makes a short smoke run harder
to interpret.

## 8. Replay the model with the renderer

The renderer is a viewer. It consumes a validated replay produced by Python;
it never runs the environment itself.

### 8.1 Dashboard replay

After a dashboard training run:

1. Keep **Output directory** set to the exact run directory.
2. Choose **Training run · latest saved actor** for either the bundled actor
   or a custom `model.py` actor.
3. Choose `Monaco` or `Spa`.
4. Choose **Pygame** for a local window or **Browser** for a browser replay.
5. Click **Build / watch replay**.
6. Use **Open browser fallback** or **Download replay JSON** if the machine has
   no desktop display.

The selected run must contain `run_config.json`. This file records the actual
ray order and other settings. The dashboard refuses to guess a default when it
is missing. You can watch or evaluate a custom actor directly from its training
run; ONNX packaging is not required for local iteration.

### 8.2 Command-line replay for a reference run

Create the replay directory first:

```bash
mkdir -p artifacts/replay
.venv/bin/python tools/export_replay.py \
  --weights artifacts/runs/first-check/policy.pt \
  --metadata artifacts/runs/first-check/policy.json \
  --contract artifacts/runs/first-check/training_contract.json \
  --track tracks/train/monaco.json \
  --config config/simulation.json \
  --participant-id first-check \
  --output artifacts/replay/first-check-monaco.json
```

Open a local Pygame viewer:

```bash
.venv/bin/python tools/run_pygame.py \
  --replay artifacts/replay/first-check-monaco.json
```

On a machine without a display, render one frame:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv/bin/python tools/run_pygame.py \
  --replay artifacts/replay/first-check-monaco.json --headless
```

For a dependency-free browser file:

```bash
.venv/bin/python tools/build_replay.py \
  artifacts/replay/first-check-monaco.json \
  --output artifacts/replay/first-check-monaco.html

.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \
  --directory artifacts/replay
```

Open `http://127.0.0.1:8765/first-check-monaco.html`. Playback controls,
recorded telemetry, events, the car pose, and the final Python result come from
the replay. A DNF is not silently turned into a lap.

## 9. Evaluate before packaging

For a reference artifact, run the deterministic evaluator twice:

```bash
.venv/bin/python tools/evaluate_policy.py \
  --weights artifacts/runs/first-check/policy.pt \
  --metadata artifacts/runs/first-check/policy.json \
  --contract artifacts/runs/first-check/training_contract.json \
  --track tracks/train/monaco.json \
  --config config/simulation.json \
  --repeats 2 \
  --output artifacts/evaluation/first-check-monaco.json
```

Useful statuses are `finished`, `dnf_crash`, `dnf_timeout`, `dnf_stuck`, and
`dnf_backward`. The evaluator uses a fixed start and a frozen preprocessing
mapping. Repeated runs should have the same trajectory digest. A training
return is not a race score; inspect `status`, lap time when finished, off-track
time, and the replay.

For a safe NumPy artifact, use:

```bash
.venv/bin/python tools/evaluate_participant.py \
  --artifact-dir path/to/numpy-artifact \
  --track tracks/train/monaco.json \
  --config config/simulation.json \
  --repeats 2 \
  --output artifacts/evaluation/numpy-monaco.json
```

Repeat the command with `tracks/train/spa.json` before deciding which run is
better.

## 10. Export ONNX

The live interface uses ONNX, not a trainer script and not a raw Torch
checkpoint. Install the optional evaluator dependencies once:

```bash
.venv/bin/python -m pip install -r requirements-evaluator.lock
```

### 10.1 Export the bundled reference artifact

The output directory must be new:

```bash
mkdir -p artifacts/submissions
.venv/bin/python tools/export_onnx.py \
  --reference \
  --source artifacts/runs/first-check \
  --output artifacts/submissions/first-check
```

Check that it contains exactly:

```text
artifacts/submissions/first-check/
  manifest.json
  training_contract.json
  model.onnx
```

The exporter validates the graph before returning. It embeds any saved
preprocessing so the evaluator does not apply it a second time.

### 10.2 Export a declarative NumPy artifact

For a directory containing exactly `manifest.json`, `training_contract.json`,
and `weights.npz`, omit `--reference`:

```bash
.venv/bin/python tools/export_onnx.py \
  --source artifacts/numpy-artifact \
  --output artifacts/submissions/numpy-model
```

Do not submit the trainer, reward file, Python source, pickle, or Torch
checkpoint as the model-only live submission. They are useful for training
provenance but are not the ONNX runtime input.

The current profile expects one float32 input named `observations` with shape
`[1, N]` and one float32 `actions` output with shape `[1, 2]`. The final output
is `tanh`, and the allowed graph operators and limits are listed in
[`docs/submission_runtime.md`](docs/submission_runtime.md).

## 11. Build an ONNX submission package

Use a **new** output directory every time. This creates the model package that
the live runner consumes. It does not select or copy a circuit:

```bash
.venv/bin/python tools/package_live_submission.py \
  --model-only \
  --source artifacts/runs/first-check \
  --output artifacts/onnx-packages/team-a \
  --participant-id team-a
```

The source run must contain the recognized reference files. For a NumPy source,
pass the directory that contains its three exact artifact files instead.
If the source is an imported ONNX artifact without a saved `run_config.json`,
also pass its matching config explicitly:

```bash
  --config artifacts/runs/external/run_config.json
```

The package looks like this:

```text
artifacts/onnx-packages/team-a/
  README.md
  package_manifest.json       # hashes and provenance
  config/simulation.json       # matching saved config
  submission/
    manifest.json
    training_contract.json
    model.onnx
  provenance/
    run_config.json            # when available
    reward_config.json         # when available
    run_identity.json          # when available
    training_report.md         # when available
```

The package builder checks the model contract against the copied config and
records SHA-256 hashes. It does not copy a track, replay, training Python, Torch
checkpoint, or reward callback into the model submission. During a live heat,
the runner receives the event/local-test map and config separately and invokes
the ONNX model for fresh actions on every tick.

![Annotated package flow: saved run to ONNX package to Python live heat](docs/assets/package-live.svg)

The dashboard can perform the same operation:

1. Open **Build ONNX submission**.
2. Select the trained run or model source.
3. Select the matching saved config. Imported ONNX sources require this when
   they do not contain `run_config.json`.
4. Enter a new package output directory.
5. Enter a simple submission ID such as `team-a`.
6. Click **Build ONNX submission package**.
7. Confirm that the result shows a `submission/` directory and a
   `package_manifest.json`.

The dashboard does not choose a map for the submission or place it into a race
slot. In **04 · Local live heat / inference test**, enter the local test track,
config, and the package's `submission/` directory. An organizer live runner
will provide competition-time maps separately.

## 12. Run two models against each other

The local heat is a ghost race: entrants share the same Python track/config but
do not collide with one another. Python calls each model, validates its action,
advances physics, and records the result.

### 12.1 Direct live runner

Install live dependencies:

```bash
.venv/bin/python -m pip install -r requirements-live.lock
```

Run two packages for a short local heat:

```bash
.venv/bin/python tools/run_live_onnx.py \
  --entrant team-a artifacts/onnx-packages/team-a/submission \
  --entrant team-b artifacts/onnx-packages/team-b/submission \
  --track tracks/train/monaco.json \
  --config config/simulation.json \
  --output-dir artifacts/live-results/monaco-team-a-team-b \
  --port 0 \
  --max-steps 240 \
  --policy-budget-ms 20 \
  --late-policy strict_scored \
  --telemetry-stdout
```

The runner prints a WebSocket address such as:

```text
ws://127.0.0.1:32949/live/v1
```

The final report is:

```text
artifacts/live-results/monaco-team-a-team-b/live-race.json
```

`strict_scored` marks an entrant `dnf_inference` when an action is late,
missing, invalid, or fails. `trusted_live` may hold the previous action for a
local visual demo, but it is not the strict scored behavior:

```bash
--late-policy trusted_live
```

### 12.2 Connect the read-only renderer

In another terminal, serve the renderer folder:

```bash
.venv/bin/python -m http.server 8765 \
  --bind 127.0.0.1 --directory renderer
```

Open the printed live URL by adding the WebSocket query parameter:

```text
http://127.0.0.1:8765/?live=ws://127.0.0.1:32949/live/v1
```

Replace the WebSocket address with the one printed by your own heat. The
browser receives `hello`, ordered state packets, and an `end` packet. It cannot
send actions back to Python. Slow browser rendering can drop intermediate
display packets without changing the race.

### 12.3 Dashboard live heat

The dashboard's **Grid control / live heat** panel has the same controls:

1. Put the two `submission/` directories into the two racer slots.
2. Give each slot a participant ID.
3. Select the same track and matching config for the heat.
4. Choose the action budget and tick limit.
5. Keep `strict_scored` for a meaningful timing check.
6. Click **Start live heat**.
7. Read latency, deadline misses, inference failures, and final status.

The terminal panel also reports shared live-process RSS, process count, and
thread count. It intentionally labels these as shared resources.

## 13. Optional reference resume and fine-tuning

The reference PPO has a checkpoint-aware command for experiments. This is the
same training path used by the default and custom-architecture runs.

To continue the same reference run, reuse the same output directory, reward,
config, maps, ray order, seed, and budget identity. Change only the invocation
limit when you understand the checkpoint rules:

```bash
.venv/bin/python tools/train_participant.py \
  --reward examples/baseline_reward.json \
  --track-dir tracks/train \
  --config config/simulation.json \
  --output-dir artifacts/runs/first-check \
  --steps 256 --budget-steps 1000000 \
  --n-envs 2 --rollout-length 32 --seed 17
```

For the reference-only best-selection and warm-start controls, read
[`docs/participant_guide.md`](docs/participant_guide.md). A selected `best/`
artifact is not automatically the dashboard's latest artifact; check which
directory you export.

## 14. Final checks before calling a model ready

Run the project checks from the repository root:

```bash
make test
node --test tests/*.test.cjs
node --check renderer/app.js
node --check renderer/live_source.js
make render
git diff --check
```

For a map/contract check:

```bash
.venv/bin/python tools/build_training_manifest.py
.venv/bin/python -m py_compile racing_event/*.py tools/*.py
```

For a practical participant check, do all of these:

- train a tiny run;
- replay it on Monaco;
- replay or evaluate it on Spa;
- export ONNX;
- package it into a new directory;
- run a short two-entrant heat; and
- inspect `live-race.json` for inference failures and deadline misses.

The renderer check is successful when Python creates a valid replay and Pygame
or the browser displays it. A renderer does not prove that the model is good;
the replay and evaluator result are the evidence.

## 15. Troubleshooting

### “participant training always uses tools/run_reference_baseline.py”

The participant launcher intentionally rejects arbitrary trainer scripts. Leave
the reference entrypoint selected, put architecture changes in
`participant_code/model.py`, and edit `tools/run_reference_baseline.py` when
you need to change PPO behavior.

### “track_dir is not a directory” or “contains no JSON tracks”

Pass a folder, not a single file:

```bash
--track-dir tracks/train
```

For one map, create a folder and copy the JSON map into it as shown in
section 4.2.

### “saved run_config.json is required”

The dashboard needs the exact saved sensor/config file to replay a run. Train
through the dashboard, or create a run directory first and pass that exact file
as `--config`:

```bash
mkdir -p artifacts/runs/cli-run
cp config/simulation.json artifacts/runs/cli-run/run_config.json
```

Use the same config file that trained the model, especially if you changed ray
count or order. A mismatch changes the model's observation contract.

Then use `--config artifacts/runs/cli-run/run_config.json` during training.
Keep the file in the run directory for replay and packaging.

### “config mismatch” or a ray-order error

You changed the config or ray list while reusing a run. Start a new output
directory. A model trained with a different observation size must not be fed
the old contract.

### “live package output must be a new directory”

Choose a new package name. For example, change
`team-a-monaco` to `team-a-monaco-v2`. This avoids silently overwriting a
package whose hashes have already been recorded.

### The replay says `dnf_crash`, `dnf_stuck`, or `dnf_timeout`

That is a real Python result, not necessarily a tool failure. Open the replay,
check the first event, and tune one thing at a time. A tiny 256-step smoke run
may end in a DNF simply because it is too short to finish a lap.

### The live result says `dnf_inference`

Check the entrant's `deadline_misses` and `inference_failures` in
`live-race.json`. Confirm that `model.onnx` is valid, that its input is
`[1, N]`, that its output is `[1, 2]`, and that all actions are finite and in
`[-1, 1]`. Use `trusted_live` only to diagnose a local display; use
`strict_scored` for the meaningful test.

### The Pygame window does not appear

Pygame opens on the machine running Python. On a headless machine, use the
browser replay or a one-frame check:

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  .venv/bin/python tools/run_pygame.py \
  --replay artifacts/replay/first-check-monaco.json --headless
```

### A port is already in use

Use another unused port for the dashboard, renderer, or live runner. Passing
`--port 0` to the live runner asks the operating system to choose a free live
WebSocket port; copy the printed address into the renderer URL.

## 16. What is authoritative and what is provisional

Python owns physics, observations, actions, termination, timing, live
deadlines, and results. Pygame and the browser are read-only presentation
layers. Participants own their model, optimizer, preprocessing, training
algorithm, and reward implementation.

This checkout contains a working engineering profile. Event-wide budget rules,
held-out maps, final ranking, final dates, and the organizer's production
isolation image still require release decisions. The ONNX runtime profile and
isolation boundaries are documented in
[`docs/submission_runtime.md`](docs/submission_runtime.md); read that document
before treating a local preview as an event submission.

## 17. Platform and security notes

The local participant tooling targets POSIX/Linux (native Windows is
unsupported; use WSL or Linux) and Python 3.11. `model.py`, trainer scripts,
and reward callbacks are participant code and execute locally. The evaluator
does not import them; it accepts only the validated ONNX submission and should
run it in the model-only isolated worker.

## 18. Useful reference documents

- [`docs/participant_guide.md`](docs/participant_guide.md) — exact reference
  resume, warm-start, selection, replay, and dashboard details.
- [`docs/state_schema.md`](docs/state_schema.md) — observation/action fields,
  bounds, artifacts, and normalization.
- [`docs/submission_runtime.md`](docs/submission_runtime.md) — ONNX graph
  profile, package contents, trusted preview, and isolated evaluation.
- [`docs/replay_contract.md`](docs/replay_contract.md) — replay validation and
  renderer boundary.
- [`docs/track_sources.md`](docs/track_sources.md) — Monaco/Spa provenance and
  deterministic map transformation.

When in doubt, first run the smallest command that exercises the part you are
changing. A 256-step training run, one replay, and a short live heat usually
find a path or contract error before a long experiment spends time.
