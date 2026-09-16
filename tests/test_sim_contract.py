import unittest
from recovervla.sim.schema import JOINTS, features


class ContractTests(unittest.TestCase):
    def test_remote_scene_supports_plate_across_ten_seeds(self):
        import os
        from pathlib import Path
        if os.environ.get("RECOVERVLA_REMOTE_TESTS") != "1":
            self.skipTest("Remote simulation test not enabled")
        from recovervla.sim.scene import Scene
        for seed in range(10):
            with self.subTest(seed=seed):
                scene = Scene(Path("artifacts/so101"), seed, render=False)
                try:
                    # Plate must remain on the tray, not sink onto/through
                    # the tabletop as the native cylinder collider did.
                    self.assertGreater(scene.body("plate")[2], .057)
                    self.assertLess(scene.body("plate")[2], .065)
                    self.assertGreater(scene.body("mug")[2], .026)
                    self.assertGreater(scene.body("bottle")[2], .026)
                finally:
                    scene.close()

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
