"""Remote-only IK reachability probe for scene-layout tuning."""
from pathlib import Path
import platform
import sys

if platform.system() != "Linux":
    raise SystemExit("Run workspace probing on remote Linux, not on the Mac")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recovervla.sim.ik import solve
from recovervla.sim.scene import Scene

scene = Scene(Path("artifacts/so101"), 100)
reachable = []
for x in (-.28, -.24, -.20, -.16, -.12, -.08):
    for y in (-.02, .02, .06, .10, .14, .18):
        for z in (.07, .10, .13, .16, .20):
            try:
                solve(scene, "left", (x, y, z))
            except RuntimeError:
                continue
            reachable.append((x, y, z))
scene.close()
print(reachable)
if not reachable:
    raise SystemExit("No reachable targets in probe grid")
