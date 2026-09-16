"""Remote reachability probe; does not move live scene objects."""
import platform
import sys
from pathlib import Path
import json

if platform.system() != "Linux":
    raise SystemExit("Run on remote Linux")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recovervla.sim.scene import Scene
from recovervla.sim.ik import solve

scene = Scene(Path("artifacts/so101"), 100, render=False)
try:
    print("support", {name: scene.body(name).tolist() for name in ("plate", "mug", "bottle")}, flush=True)
    for x in (-.12, -.08, -.04, 0):
        for direction in ([0, 0, -1], [1, 0, -1], [1, 0, -.5]):
            try:
                for y, z in ((.022, .139), (.022, .099), (-.075, .095)):
                    solve(scene, "left", [x, y, z], approach=direction)
            except RuntimeError as error:
                print(json.dumps({"x": x, "direction": direction, "error": str(error)}), flush=True)
            else:
                print(json.dumps({"x": x, "direction": direction, "reachable": True}), flush=True)
finally:
    scene.close()
