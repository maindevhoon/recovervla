"""Bounded position IK on scratch data; never changes live object state."""
import numpy as np
import mujoco


def solve(scene, arm, target, max_iterations=200, tolerance=.004, wrist_roll=None):
    model = scene.model
    indices = np.arange(0, 5) if arm == "left" else np.arange(6, 11)
    qadr, dadr = scene.qadr[indices], scene.dadr[indices]
    site_id = model.site(arm + "_gripperframe").id
    jac = np.zeros((3, model.nv))
    current = scene.data.qpos[qadr].copy()
    # A straight/zero SO-101 pose is close to a Jacobian singularity. These
    # deterministic bent-arm seeds cover elbow-up/down configurations without
    # making reproducibility depend on a random optimizer.
    seeds = [current]
    if wrist_roll is not None:
        preferred = current.copy()
        preferred[4] = wrist_roll
        seeds.insert(0, preferred)
    for lift, elbow, wrist in ((-.8, .9, .7), (.8, -.9, -.7),
                               (-1.2, 1.3, .5), (1.2, -1.3, -.5)):
        seed = current.copy()
        seed[1:4] = (lift, elbow, wrist)
        if wrist_roll is not None:
            seed[4] = wrist_roll
        seeds.append(seed)
    best_error = float("inf")
    for seed in seeds:
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = scene.data.qpos
        scratch.qpos[qadr] = np.clip(seed, scene.limits[indices, 0], scene.limits[indices, 1])
        for _ in range(max_iterations):
            # IK needs transforms and joint axes only. Full mj_forward also
            # solves contacts for the vessels and 60 free particles on every
            # iteration, including unreachable candidates.
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            error = np.asarray(target) - scratch.site_xpos[site_id]
            norm = float(np.linalg.norm(error))
            best_error = min(best_error, norm)
            if norm < tolerance:
                result = scene.command.copy()
                result[indices] = scratch.qpos[qadr]
                return result
            mujoco.mj_jacSite(model, scratch, jac, None, site_id)
            j = jac[:, dadr]
            delta = j.T @ np.linalg.solve(j @ j.T + .002 * np.eye(3), error)
            scratch.qpos[qadr] = np.clip(scratch.qpos[qadr] + np.clip(delta, -.08, .08),
                                        scene.limits[indices, 0], scene.limits[indices, 1])
    raise RuntimeError(f"IK could not reach {arm} target {np.asarray(target).tolist()}; "
                       f"best error {best_error:.4f} m")
