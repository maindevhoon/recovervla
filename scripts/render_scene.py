"""Render the initial dining scene on a remote Linux runtime, without training."""
import argparse
import os
from pathlib import Path
import sys

if sys.platform != "linux":
    raise SystemExit("Run this preview in Colab or another remote Linux runtime, not on the Mac.")
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco
from PIL import Image
from recovervla.sim.build_scene import build

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, default=Path("artifacts/preview"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
xml, variation = build(Path("artifacts/so101"), args.seed)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=900, width=1200)
try:
    for camera in ("hero", "overhead", "scene", "left_wrist", "right_wrist"):
        renderer.update_scene(data, camera=camera)
        path = args.output / f"{camera}.png"
        Image.fromarray(renderer.render()).save(path)
        print(path)
finally:
    renderer.close()
print("Static initial-state preview only; no physics rollout or policy evaluation.")
