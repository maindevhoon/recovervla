# AGENTS.md

## Project

RecoverVLA is an entry for the Intel Physical AI online challenge. The target is
an end-to-end, simulation-first bimanual manipulation system using two simulated
SO-101 arms in MuJoCo. It must interpret natural-language instructions and
camera observations, coordinate both arms, and demonstrate robust table-setting
behavior.

## Required technical direction

- Use MuJoCo as the primary simulator.
- Use a learned Physical AI policy for final task execution. OMPL, MoveIt, ROS,
  inverse kinematics, or scripted trajectories may generate demonstrations, but
  must not replace the learned policy in the final pipeline.
- Store demonstrations in LeRobot-compatible format where practical.
- Prefer a compact ACT policy as the first deployable baseline. Pretrained VLA
  models may be used for language or visual reasoning when they add measurable
  value.
- Keep model interfaces explicit: camera tensors, robot state, language/task
  representation, action chunks, normalization statistics, and camera names.
- Export supported inference components to OpenVINO IR and benchmark them on an
  Intel system. Preserve a PyTorch reference path for correctness comparisons.
- Target Intel Core Ultra Series 2/3 CPU, iGPU, and NPU where available.

## Evaluation requirements

- Make every experiment reproducible with explicit seeds and configuration.
- Evaluate at least 10 randomized seeds.
- Randomize object placement, mass, friction, lighting, background, and relevant
  geometry without making the task physically invalid.
- Report task success, subtask success, recovery count, latency, throughput,
  precision, and selected inference device.
- Compare optimized outputs against the PyTorch reference and document any
  accuracy or task-success regression.
- Do not claim real simulation, learned-policy, hardware, or benchmark results
  when a component is still a stub or dry run.

## Repository expectations

- Keep setup and execution commands current in `README.md`.
- Prefer small, reviewable changes with tests for deterministic logic.
- Keep large datasets, model weights, generated videos, and build artifacts out
  of Git. Reference their download locations and checksums instead.
- Never commit credentials, API keys, access tokens, private URLs, or personal
  data.
- Respect the licenses of all models, datasets, meshes, textures, and other
  third-party assets, and document their provenance.
- Keep the final environment container-friendly and runnable on Intel hardware.

## Verification

User constraint: edit files locally, but run no simulation, tests, training, or
fine-tuning on the Mac. Execute verification on the remote Linux host. Training
and fine-tuning target L4; Kaggle/Colab CUDA GPUs are allowed fallbacks. Never add
assistant contributor tags or Co-authored-by trailers to commits.

For the current lightweight control-plane milestone, run:

```bash
PYTHONPATH=src python3 -m recovervla.evaluation --seeds 10
python3 -m unittest discover -s tests -v
```

As MuJoCo, training, and OpenVINO components are added, include focused smoke
tests for each and one documented end-to-end command.
