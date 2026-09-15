import unittest
from recovervla.sim.schema import JOINTS, features


class ContractTests(unittest.TestCase):
    def test_motor_contract(self):
        self.assertEqual(len(set(JOINTS)), 12)
        self.assertEqual(features()["action"]["shape"], (12,))
        self.assertEqual(len(features()), 5)

    def test_scene_on_remote(self):
        import os
        from pathlib import Path
        if os.environ.get("RECOVERVLA_REMOTE_TESTS") != "1":
            self.skipTest("Remote simulation test not enabled")
        from recovervla.sim.scene import Scene
        scene = Scene(Path("artifacts/so101"), 0)
        try:
            self.assertEqual(scene.model.nu, 12)
            self.assertEqual(scene.observe()["observation.images.scene"].shape, (128, 128, 3))
            self.assertEqual(len(scene.qadr), 12)
        finally:
            scene.close()
