PYTHON ?= .venv/bin/python

.PHONY: test render e1 e2 participant-setup participant-maps participant-smoke participant-ui evaluate-participant evaluate-both onnx-race live-onnx-race ui-browser-check check-model first-run

test:
	$(PYTHON) -m unittest discover -s tests -v

render:
	mkdir -p artifacts/replay
	$(PYTHON) tools/export_replay.py --scripted --track tracks/train/monaco.json --config config/simulation.json --steps 120 --participant-id pygame-smoke --output artifacts/replay/pygame-smoke.json
	SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy $(PYTHON) tools/run_pygame.py --replay artifacts/replay/pygame-smoke.json --headless

e1:
	$(PYTHON) tools/benchmark_vector_env.py --steps 10 --n-envs 1 2 4 8 16 --output reports/e1_vectorized_benchmark.md

e2:
	$(PYTHON) tools/run_e2_baseline.py --output reports/e2_baseline_convergence.md

participant-maps:
	$(PYTHON) tools/build_training_manifest.py

participant-setup:
	python3.11 tools/setup_participant.py

participant-smoke:
	$(PYTHON) tools/train_participant.py --steps 256 --budget-steps 10000 --output-dir artifacts/runs/run-001

participant-ui:
	$(PYTHON) tools/participant_ui.py

# Optional: install requirements-ui-qa.lock and Chromium before using this.
ui-browser-check:
	$(PYTHON) tools/check_ui_browser.py

check-model:
	$(PYTHON) tools/check_model.py $(MODEL)

first-run:
	python3.11 tools/setup_participant.py
	$(PYTHON) tools/build_training_manifest.py
	RUN_DIR=$$($(PYTHON) -c 'from tools.participant_ui import default_paths; print(default_paths()["output_dir"])'); $(PYTHON) tools/train_participant.py --steps 256 --budget-steps 10000 --n-envs 2 --rollout-length 32 --output-dir $$RUN_DIR
	@echo "SYNAPSE participant dashboard: http://127.0.0.1:8766/"

evaluate-participant:
	$(PYTHON) tools/evaluate_participant.py --artifact-dir submission --track tracks/train/monaco.json --config config/simulation.json --repeats 2

evaluate-both:
	$(PYTHON) tools/evaluate_both.py --source $(SOURCE)

# Pass CLI options with: make onnx-race ARGS='--entrant team-a submission-a ...'
onnx-race:
	$(PYTHON) tools/run_onnx_race.py $(ARGS)

# Pass CLI options with: make live-onnx-race ARGS='--entrant team-a submission-a ...'
live-onnx-race:
	$(PYTHON) tools/run_live_onnx.py $(ARGS)
