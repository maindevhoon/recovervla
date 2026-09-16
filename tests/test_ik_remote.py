"""Real model IK regressions, explicitly enabled on remote Linux only."""
import os
import unittest


@unittest.skipUnless(os.environ.get("RECOVERVLA_REMOTE_TESTS") == "1", "Remote IK test not enabled")
class RemoteIKTests(unittest.TestCase):
    def test_kinematic_jacobian_matches_full_forward_and_preserves_live_state(self):
        from pathlib import Path
        import mujoco
        import numpy as np
        from recovervla.sim.scene import Scene
        from recovervla.sim.ik import solve

        scene = Scene(Path("artifacts/so101"), 100, render=False)
        try:
            before = scene.data.qpos.copy()
            scratch = mujoco.MjData(scene.model)
            scratch.qpos[:] = before
            scratch.qpos[scene.qadr[:5]] = [.4, -.5, .8, .4, .5]
            sid = scene.model.site("left_gripperframe").id
            jac_full = np.zeros((3, scene.model.nv))
            jac_fast = np.zeros_like(jac_full)
            mujoco.mj_forward(scene.model, scratch)
            point = scratch.site_xpos[sid].copy()
            mujoco.mj_jacSite(scene.model, scratch, jac_full, None, sid)
            mujoco.mj_kinematics(scene.model, scratch)
            mujoco.mj_comPos(scene.model, scratch)
            mujoco.mj_jacSite(scene.model, scratch, jac_fast, None, sid)
            np.testing.assert_allclose(jac_fast, jac_full, atol=1e-12)
            result = solve(scene, "left", point)
            scratch.qpos[scene.qadr] = result
            mujoco.mj_kinematics(scene.model, scratch)
            self.assertLess(np.linalg.norm(scratch.site_xpos[sid] - point), .004)
            np.testing.assert_array_equal(scene.data.qpos, before)
        finally:
            scene.close()
