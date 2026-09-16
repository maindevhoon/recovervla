# Kaggle fast path

Use this path when time is limited. It deliberately stops at the first failed
gate instead of spending GPU quota on an invalid dataset. All execution is on
Kaggle Linux; do not run these commands on the Mac.

## Notebook settings

- Accelerator: GPU (T4 x2 or P100; one GPU is sufficient)
- Internet: enabled (required for the repository, robot assets, and packages)
- Persistence: save a notebook version after every successful gate

## Cell 1 — bootstrap and inspect the assigned GPU

```python
import os, subprocess, sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
repo = Path("/kaggle/working/recovervla")
if not repo.exists():
    subprocess.run(["git", "clone", "https://github.com/maindevhoon/recovervla.git", str(repo)], check=True)
os.chdir(repo)
subprocess.run(["git", "pull", "--ff-only"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[sim,train]"], check=True)
if not Path("artifacts/so101/manifest.json").exists():
    subprocess.run([sys.executable, "scripts/fetch_assets.py"], check=True)

import torch
assert torch.cuda.is_available(), "Enable a Kaggle GPU before continuing"
print(torch.cuda.get_device_name(0))
```

## Cell 2 — remote scene gate

```python
subprocess.run([sys.executable, "scripts/render_scene.py", "--seed", "0"], check=True)
from IPython.display import Image, display
display(Image(filename="artifacts/preview/hero.png", width=720))
```

## Cell 3 — fast physics-only expert gate

```python
subprocess.run([
    sys.executable, "-u", "scripts/validate_expert.py", "--attempts", "3",
    "--timeout", "120", "--report", "/kaggle/working/full-physics.json"
], check=True, timeout=450)
```

This skips camera rendering and video encoding, making failed-waypoint tuning much
faster. `validate_expert.py` constructs the scene without a MuJoCo renderer, so the
gate does not need EGL. It streams waypoint timings and contacts and returns
nonzero if any requested seed fails. Continue only after the complete physical
sequence passes. On GitHub, `expert-gate` runs the same script on Ubuntu.

For drawer regression only, add `--scope drawer --attempts 10`. A drawer-only
pass is not permission to start full-task collection. `--timeout` bounds the
expert per episode, while the outer subprocess timeout also covers scene setup.
The collector has a separate `--episode-timeout` (default 600 seconds) because
rendering and encoding are included. Never capture all subprocess output while
debugging: use `-u` and streamed stdout so failures remain visible.

For independent drawer-seed logs and a summary report, use
`python scripts/validate_batch.py --scope drawer --attempts 10 --workers 2 --output /kaggle/working/drawer-check`.
The 100–109 drawer batch passed on Kaggle. The same seeds then passed a
plate-scope lift using the central pinch tab. That still does not validate
placement, mug grasp, pouring, or an encoded demonstration.

Run physics and rendering regressions remotely:

```python
os.environ["RECOVERVLA_REMOTE_TESTS"] = "1"
os.environ["PYTHONPATH"] = str(repo / "src")
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
```

## Cell 4 — one encoded demonstration gate

```python
subprocess.run([
    sys.executable, "-m", "recovervla.collect",
    "--output", "/kaggle/working/datasets/smoke",
    "--episodes", "1", "--max-attempts", "10", "--start-seed", "100"
], check=True)
print(Path("/kaggle/working/datasets/smoke/report.json").read_text())
```

Do not continue if this cell produces no successful episode. Inspect the report,
tune the expert, and collect into a new output directory.

## Cell 5 — small time-boxed dataset

```python
subprocess.run([
    sys.executable, "-m", "recovervla.collect",
    "--output", "/kaggle/working/datasets/train20",
    "--episodes", "20", "--max-attempts", "100", "--start-seed", "200"
], check=True)
```

## Cell 6 — short ACT baseline

```python
subprocess.run([
    sys.executable, "scripts/train_act.py",
    "--dataset-root", "/kaggle/working/datasets/train20",
    "--output", "/kaggle/working/outputs/act-5k",
    "--steps", "5000", "--batch-size", "8", "--allow-other-gpu"
], check=True)
```

Before ending the session, use **Save Version** with outputs enabled. Kaggle's
working directory is otherwise temporary. The 5k run is a pipeline baseline,
not final evidence of policy performance.
