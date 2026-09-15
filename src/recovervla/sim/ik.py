"""Bounded position IK on scratch data; never changes live object state."""
import numpy as np
import mujoco


def solve(scene, arm, target, max_iterations=200, tolerance=.004):
    model = scene.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = scene.data.qpos
    indices = np.arange(0, 5) if arm == "left" else np.arange(6, 11)
    qadr, dadr = scene.qadr[indices], scene.dadr[indices]
    site_id = model.site(arm + "_gripperframe").id
    jac = np.zeros((3, model.nv))
    for _ in range(max_iterations):
        mujoco.mj_forward(model, scratch)
        error = np.asarray(target) - scratch.site_xpos[site_id]
        if np.linalg.norm(error) < tolerance:
            result = scene.command.copy()
            result[indices] = scratch.qpos[qadr]
            return result
        mujoco.mj_jacSite(model, scratch, jac, None, site_id)
        j = jac[:, dadr]
        delta = j.T @ np.linalg.solve(j @ j.T + .001 * np.eye(3), error)
        scratch.qpos[qadr] = np.clip(scratch.qpos[qadr] + np.clip(delta, -.08, .08),
                                    scene.limits[indices, 0], scene.limits[indices, 1])
    raise RuntimeError(f"IK could not reach {arm} target {np.asarray(target).tolist()}")
