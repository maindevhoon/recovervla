# Remote runbook

Execution status: code authored, no local tests, simulation, training, or fine-tuning run.
The expert is an initial implementation requiring physical tuning on the remote host.
Do not purchase extended training time until collection yields successful episodes.

## L4 host

Use a Linux L4 machine with CUDA drivers and persistent storage. All commands below
run there, from the repository directory, never on the Mac.

```bash
pip install -e '.[sim,train]'
python scripts/fetch_assets.py
export MUJOCO_GL=egl
RECOVERVLA_REMOTE_TESTS=1 python -m unittest discover -s tests -v
python -m recovervla.collect --output datasets/smoke --episodes 1 --max-attempts 10
```

Inspect `datasets/smoke/report.json`. IK, contact, or transfer failures require
adjusting the scene/grasp waypoints in code and collecting into a new directory.
Position-only IK does not yet enforce grasp orientation or collision-free paths.
The current bottleneck is physical expert validation, not GPU training speed.
`python scripts/validate_expert.py` is the physics-only gate and does not create a
renderer. GitHub Actions workflow `expert-gate` runs that command on Ubuntu.

After a successful smoke run, collect independent training and validation roots:

```bash
python -m recovervla.collect --output datasets/train --episodes 40 --max-attempts 200 --start-seed 100
python -m recovervla.collect --output datasets/validation --episodes 10 --max-attempts 100 --start-seed 500
python scripts/validate_dataset.py --root datasets/validation --repo-id maindevhoon/recovervla-demonstrations
python scripts/train_act.py --dataset-root datasets/train --output outputs/act-first --steps 5000
```

The training launcher validates the training root before starting. Separate roots
keep validation observations out of normalization statistics. Seeds 10000–10009
are reserved for the later learned-policy evaluation, which is not implemented yet.
`--checkpoint` accepts a compatible 12-action ACT checkpoint for fine-tuning.
It does not convert an incompatible ALOHA checkpoint or fine-tune SmolVLA.
Retain `outputs/` on persistent storage; LeRobot saves every 1000 steps.

## Docker on L4

```bash
docker build -t recovervla .
docker run --gpus all -it --shm-size=8g -v /workspace/data:/workspace/recovervla/datasets -v /workspace/outputs:/workspace/recovervla/outputs recovervla
```

## Kaggle / Colab fallback

Use a GPU notebook, clone this repository in its remote filesystem, and install
the same extras. Run the above shell commands in notebook cells with `!` prefixes.
Set `os.environ['MUJOCO_GL'] = 'egl'` before importing MuJoCo. Notebook GPUs and
session lengths vary; use persistent dataset/checkpoint storage and keep batch size
at 8 or lower if memory is limited. On a non-L4 GPU explicitly add:

```bash
python scripts/train_act.py --dataset-root datasets/train --output outputs/act-first --allow-other-gpu
```

No credentials are stored in these scripts and no datasets or weights are published
automatically. Download or copy remote results before terminating a rented instance.

## Remaining acceptance work

- Execute remote scene loading, camera rendering, and vessel retention checks.
- Tune contact grasps, orientation-constrained pouring, and collision checks.
- Produce complete successful demonstrations across varied positions.
- Add learned-policy closed-loop evaluation and held-out success metrics.
- Add OpenVINO export, numerical comparison, and Intel device benchmarks.

The implementation uses LeRobot 0.4.4's official v3 writer, superseding the draft
handwritten v2.1 format. Scene jitter currently spans ±1 cm; 4 mm granular particles
fit in the bottle. Background/geometry randomization and full dataset provenance
validation remain follow-up work. No success rate or Intel performance is claimed.
