"""Fast remote physics gate without camera rendering or dataset encoding."""
import argparse
import json
from pathlib import Path
import platform
import sys
import math
from time import monotonic

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
parser.add_argument("--timeout", type=float, default=120)
parser.add_argument("--scope", choices=("full", "place", "drawer", "plate", "bottle"), default="full")
parser.add_argument("--report", type=Path)
args = parser.parse_args()
if args.attempts < 1 or args.start_seed < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
    parser.error("Require positive attempts and timeout, and nonnegative start seed")
rows = []
for seed in range(args.start_seed, args.start_seed + args.attempts):
    started = monotonic()
    def progress(event, **fields):
        print(json.dumps({"seed": seed, "elapsed": round(monotonic() - started, 3),
                          "event": event, **fields}), flush=True)
    progress("scene_start")
    scene = Scene(args.robot_dir, seed, render=False)
    progress("scene_ready", snapshot=scene.snapshot())
    expert = Expert(scene, None, progress=progress, timeout=args.timeout)
    row = {"seed": seed, "scope": args.scope, "success": False}
    try:
        if args.scope in ("drawer", "plate"):
            expert._open_drawer()
            expert.history.append("open_drawer:top_drawer")
            if args.scope == "plate":
                expert.grasp("left", "plate")
                expert.history.append("grasp:plate")
        elif args.scope == "bottle":
            expert.grasp("left", "bottle")
            expert.history.append("grasp:bottle")
        elif args.scope == "place":
            expert.run(TableSettingPlanner.place_plan())
        else:
            expert.run(TableSettingPlanner().plan(REFERENCE_INSTRUCTION))
    except (RuntimeError, TimeoutError) as error:
        row["error"] = str(error)
    else:
        row["success"] = True
    finally:
        row["completed_skills"] = expert.history
        row["elapsed_seconds"] = round(monotonic() - started, 3)
        row["snapshot"] = scene.snapshot()
        scene.close()
        rows.append(row)
        progress("attempt_finished", **row)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
if not all(row["success"] for row in rows):
    raise SystemExit(f"{sum(row['success'] for row in rows)}/{len(rows)} {args.scope} attempts passed")
