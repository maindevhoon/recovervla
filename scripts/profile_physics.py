"""Compare contact solver settings on remote Linux from identical initial state."""
import platform
import sys
from pathlib import Path
from time import perf_counter
import json

if platform.system() != "Linux":
    raise SystemExit("Run on remote Linux")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import mujoco
import numpy as np
from recovervla.sim.build_scene import build
from recovervla.sim.schema import JOINTS

xml, _ = build(Path("artifacts/so101"), 100)
model = mujoco.MjModel.from_xml_string(xml)
for noslip, solver in ((3, mujoco.mjtSolver.mjSOL_NEWTON),
                       (0, mujoco.mjtSolver.mjSOL_NEWTON),
                       (0, mujoco.mjtSolver.mjSOL_CG)):
    model.opt.noslip_iterations = noslip
    model.opt.solver = solver
    data = mujoco.MjData(model)
    for name in JOINTS:
        aid = model.actuator(name).id
        value = model.actuator_ctrlrange[aid, 1] if name.endswith("_gripper") else 0
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        data.ctrl[aid] = value
    start = perf_counter()
    for _ in range(340):
        mujoco.mj_step(model, data)
    print(json.dumps({"noslip": noslip, "solver": int(solver), "seconds": perf_counter() - start,
                      "sim_seconds": data.time, "contacts": data.ncon,
                      "finite": bool(np.isfinite(data.qpos).all()),
                      "minimum_contact_distance": float(min(data.contact.dist, default=0))}), flush=True)
