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
targets = [(-.06, .06, .10), (-.04, .06, .10), (0, .06, .10),
           (-.10, .10, .10), (-.10, .14, .10), (-.10, 0, .10),
           (-.16, .14, .10), (-.06, .14, .10)]
reachable = []
for target in targets:
    try:
        solve(scene, "left", target)
    except RuntimeError as error:
        print(error)
        continue
    reachable.append(target)
scene.close()
print(reachable)
if not reachable:
    raise SystemExit("No reachable targets in probe grid")
