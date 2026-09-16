"""Bounded position IK on scratch data; never changes live object state."""
import numpy as np
import mujoco


def solve(scene, arm, target, max_iterations=200, tolerance=.004, wrist_roll=None,
          approach=None, wrist_flex=None, align=.94):
    model = scene.model
    indices = np.arange(0, 5) if arm == "left" else np.arange(6, 11)
    qadr, dadr = scene.qadr[indices], scene.dadr[indices]
    site_id = model.site(arm + "_gripperframe").id
    jac = np.zeros((3, model.nv))
    jacrot = np.zeros_like(jac)
    direction = None if approach is None else np.asarray(approach, dtype=float)
    if direction is not None:
        direction = direction / np.linalg.norm(direction)
    current = scene.data.qpos[qadr].copy()
    # A straight/zero SO-101 pose is close to a Jacobian singularity. These
    # deterministic bent-arm seeds cover elbow-up/down configurations without
    # making reproducibility depend on a random optimizer.
    seeds = [current]
    if wrist_roll is not None or wrist_flex is not None:
        preferred = current.copy()
        if wrist_flex is not None:
            preferred[3] = wrist_flex
        if wrist_roll is not None:
            preferred[4] = wrist_roll
        seeds.insert(0, preferred)
    for lift, elbow, wrist in ((-.8, .9, .7), (.8, -.9, -.7),
                               (-1.2, 1.3, .5), (1.2, -1.3, -.5)):
        seed = current.copy()
        seed[1:4] = (lift, elbow, wrist)
        if wrist_flex is not None:
            seed[3] = wrist_flex
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
            axis = scratch.site_xmat[site_id].reshape(3, 3)[:, 0]
            # Five-joint SO-101 cannot independently set arbitrary tool yaw.
            # A downward approach cone is sufficient for tray clearance.
            aligned = direction is None or np.dot(axis, direction) > align
            if norm < tolerance and aligned:
                result = scene.command.copy()
                result[indices] = scratch.qpos[qadr]
                return result
            mujoco.mj_jacSite(model, scratch, jac, jacrot, site_id)
            j = jac[:, dadr]
            if direction is not None and not aligned:
                # Site X points from the wrist toward the fingertips.
                # Aligning one axis constrains two rotational DOFs. Leave
                # rotation around that axis free: SO-101 has only five arm
                # joints, so a third rotational constraint overconstrains IK.
                projector = np.eye(3) - np.outer(axis, axis)
                j = np.vstack((j, .1 * projector @ jacrot[:, dadr]))
                error = np.r_[error, .1 * np.cross(axis, direction)]
            delta = j.T @ np.linalg.solve(j @ j.T + .0001 * np.eye(len(error)), error)
            scratch.qpos[qadr] = np.clip(scratch.qpos[qadr] + np.clip(delta, -.08, .08),
                                        scene.limits[indices, 0], scene.limits[indices, 1])
            if wrist_flex is not None:
                scratch.qpos[qadr[3]] = np.clip(
                    wrist_flex, scene.limits[indices[3], 0], scene.limits[indices[3], 1])
    raise RuntimeError(f"IK could not reach {arm} target {np.asarray(target).tolist()}; "
                       f"best error {best_error:.4f} m")
