import unittest
from unittest.mock import patch

try:
    import mujoco  # noqa: F401
    import numpy as np
    from recovervla.sim.expert import Expert
except ImportError:
    np = None
    Expert = None


class FakeScene:
    def __init__(self):
        self.command = np.zeros(12)
        self.command[[5, 11]] = 1.0
        self.limits = np.zeros((12, 2))
        self.limits[:, 0] = -0.17453
        self.limits[:, 1] = 1.74533

    def step(self, action):
        self.command = np.asarray(action, dtype=float).copy()


@unittest.skipUnless(Expert is not None, "numpy/mujoco extra not installed")
class ExpertCommandTests(unittest.TestCase):
    def test_expired_budget_stops_before_advancing_physics(self):
        scene = FakeScene()
        initial = scene.command.copy()
        with patch("recovervla.sim.expert.monotonic", return_value=10):
            expert = Expert(scene, None, timeout=1)
        with patch("recovervla.sim.expert.monotonic", return_value=12):
            with self.assertRaises(TimeoutError):
                expert.grip("left", True)
        np.testing.assert_array_equal(initial, scene.command)

    def test_grip_uses_actuator_limits(self):
        scene = FakeScene()
        expert = Expert(scene, None)
        expert.grip("left", True)
        self.assertAlmostEqual(scene.command[5], scene.limits[5, 0])
        expert.grip("left", False)
        self.assertAlmostEqual(scene.command[5], scene.limits[5, 1])
        expert.grip("right", True)
        self.assertAlmostEqual(scene.command[11], scene.limits[11, 0])
        expert.grip("right", False)
        self.assertAlmostEqual(scene.command[11], scene.limits[11, 1])

    def test_pour_tilts_wrist_flex_not_roll(self):
        # A top-down neck grasp has tool X pointing down. Wrist roll spins
        # the bottle; wrist flex tips it into the mug.
        scene = FakeScene()
        scene.command[3] = 0.2
        scene.command[4] = 0.1
        tilted = Expert(scene, None).pour_command()
        self.assertGreater(tilted[3], 0.2)
        self.assertAlmostEqual(tilted[4], 0.1)
        self.assertLessEqual(tilted[3], scene.limits[3, 1])


if __name__ == "__main__":
    unittest.main()
