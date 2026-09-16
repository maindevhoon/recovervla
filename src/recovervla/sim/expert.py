"""Initial contact-based expert. Requires remote grasp and orientation tuning."""
import numpy as np
from .ik import solve
from .schema import FPS
from ..types import Skill


class Expert:
    def __init__(self, scene, record):
        self.scene, self.record = scene, record
        self.history = []
        self.zones = {"plate": np.array([-.04, -.12, .033]), "mug": np.array([.09, -.12, .033])}

    def move(self, command, seconds=1.5):
        start = self.scene.command.copy()
        steps = max(round(seconds * FPS), int(np.ceil(np.max(abs(command - start)) / .025)))
        for fraction in np.linspace(1 / steps, 1, steps):
            action = start + fraction * (command - start)
            if self.record is not None:
                frame = self.scene.observe()
                frame["action"] = action.astype(np.float32)
                self.record(frame)
            self.scene.step(action)

    def reach(self, arm, target, wrist_roll=None):
        self.move(solve(self.scene, arm, target, wrist_roll=wrist_roll))

    def grip(self, arm, closed):
        command = self.scene.command.copy()
        index = 5 if arm == "left" else 11
        command[index] = self.scene.limits[index, 0 if closed else 1]
        self.move(command, .5)

    def _open_drawer(self):
        point = self.scene.site("drawer_grasp")
        self.grip("left", False)
        try:
            self.reach("left", point + np.array([0.0, 0.0, 0.05]))
        except RuntimeError:
            self.reach("left", point + np.array([0.0, -0.04, 0.02]))
        grasped = False
        for roll in (None, -1.2, 1.2, -2.0, 2.0, 0.6, -0.6):
            try:
                self.grip("left", False)
                self.reach("left", point, wrist_roll=roll)
                self.grip("left", True)
            except RuntimeError:
                continue
            if (self.scene.contact("left", "drawer") or
                    float(self.scene.data.joint("drawer_slide").qpos[0]) > 0.004):
                grasped = True
                break
        if not grasped:
            raise RuntimeError(f"Drawer grasp made no contact: {self.scene.snapshot()}")
        for _ in range(12):
            if float(self.scene.data.joint("drawer_slide").qpos[0]) >= 0.05:
                self.grip("left", False)
                return
            handle = self.scene.site("drawer_grasp")
            try:
                self.reach("left", handle + np.array([0.0, -0.012, 0.0]))
            except RuntimeError as error:
                raise RuntimeError(
                    f"Drawer pull IK failed: {error}; {self.scene.snapshot()}"
                ) from error
        raise RuntimeError(f"Drawer did not open by contact: {self.scene.snapshot()}")

    def grasp(self, arm, target):
        point = self.scene.site(target + "_grasp")
        self.grip(arm, False)
        self.reach(arm, point + [0, 0, .06])
        self.reach(arm, point)
        self.grip(arm, True)
        before = self.scene.body(target)[2]
        self.reach(arm, point + [0, 0, .06])
        if self.scene.body(target)[2] < before + .02 or not self.scene.contact(arm, target):
            raise RuntimeError(f"Physical grasp failed: {arm} {target}")

    def run(self, plan):
        for action in plan.actions:
            arm, target = action.arm.value, action.target
            if action.skill == Skill.OPEN_DRAWER:
                self._open_drawer()
            elif action.skill == Skill.GRASP:
                self.grasp(arm, target)
            elif action.skill == Skill.PLACE:
                offset = self.scene.site(arm + "_gripperframe") - self.scene.body(target)
                self.reach(arm, self.zones[target] + offset + [0, 0, .06])
                self.reach(arm, self.zones[target] + offset)
                self.grip(arm, False)
                self.reach(arm, self.scene.site(arm + "_gripperframe") + [0, 0, .06])
            elif action.skill == Skill.POUR:
                self.reach("left", self.scene.body("mug") + [0, 0, .16])
                command = self.scene.command.copy()
                command[4] = np.clip(command[4] + 1.5, *self.scene.limits[4])
                self.move(command, 2)
                self.move(command, 2)
                if self.scene.contained() < 20:
                    raise RuntimeError("Particle transfer below threshold")
            elif action.skill == Skill.VERIFY:
                self.move(self.scene.command.copy(), 1)
                for name, zone in self.zones.items():
                    if np.linalg.norm(self.scene.body(name)[:2] - zone[:2]) > .04:
                        raise RuntimeError(f"Final placement failed: {name}")
                    if abs(self.scene.body(name)[2] - zone[2]) > .02:
                        raise RuntimeError(f"Object is not resting on table: {name}")
                if self.scene.data.body("mug").xmat.reshape(3, 3)[2, 2] < .95 or self.scene.contained() < 20:
                    raise RuntimeError("Mug not upright or transferred particles lost")
            else:
                raise NotImplementedError(action.skill)
            self.history.append(f"{action.skill.value}:{target}")
