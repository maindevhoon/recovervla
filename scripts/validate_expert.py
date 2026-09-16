"""Fast remote physics gate without camera rendering or dataset encoding."""
import argparse
import json
from pathlib import Path
import platform
import sys

if platform.system() != "Linux":
    raise SystemExit("Run expert validation on remote Linux, not on the Mac")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recovervla.evaluation import REFERENCE_INSTRUCTION
from recovervla.planner import TableSettingPlanner
from recovervla.sim.expert import Expert
from recovervla.sim.scene import Scene

parser = argparse.ArgumentParser()
parser.add_argument("--robot-dir", type=Path, default=Path("artifacts/so101"))
parser.add_argument("--start-seed", type=int, default=100)
parser.add_argument("--attempts", type=int, default=3)
args = parser.parse_args()
rows = []
for seed in range(args.start_seed, args.start_seed + args.attempts):
    scene = Scene(args.robot_dir, seed, render=False)
    expert = Expert(scene, None)
    row = {"seed": seed, "success": False}
    try:
        expert.run(TableSettingPlanner().plan(REFERENCE_INSTRUCTION))
    except RuntimeError as error:
        row["error"] = str(error)
    else:
        row["success"] = True
    finally:
        row["completed_skills"] = expert.history
        row["snapshot"] = scene.snapshot()
        scene.close()
        rows.append(row)
print(json.dumps(rows, indent=2))
if not any(row["success"] for row in rows):
    raise SystemExit("No successful physics-only expert attempt")
