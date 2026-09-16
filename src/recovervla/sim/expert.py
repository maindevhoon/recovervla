"""Initial contact-based expert. Requires remote grasp and orientation tuning."""
import numpy as np
from time import monotonic
from .ik import solve
from .schema import FPS
from ..types import Skill


class Expert:
    def __init__(self, scene, record, progress=None, timeout=120):
        self.scene, self.record = scene, record
        self.history = []
        self.progress = progress
        self.deadline = monotonic() + timeout

        self.zones = {"plate": np.array([-.04, -.12, .033]), "mug": np.array([.09, -.12, .033])}

    def report(self, event, **fields):
        if self.progress:
            self.progress(event, **fields)

    def check_budget(self):
        if monotonic() >= self.deadline:
            raise TimeoutError("Expert wall-clock budget exhausted")

    def move(self, command, seconds=1.5):
        start = self.scene.command.copy()
        steps = max(round(seconds * FPS), int(np.ceil(np.max(abs(command - start)) / .025)))
        for fraction in np.linspace(1 / steps, 1, steps):
            self.check_budget()
            action = start + fraction * (command - start)
            if self.record is not None:
                frame = self.scene.observe()
                frame["action"] = action.astype(np.float32)
                self.record(frame)
            self.scene.step(action)

    def reach(self, arm, target, wrist_roll=None, approach=None):
        self.check_budget()
        start = monotonic()
        self.report("reach_start", arm=arm, target=np.asarray(target).tolist())
        command = solve(self.scene, arm, target, wrist_roll=wrist_roll, approach=approach)
        self.report("ik_ready", seconds=monotonic() - start)
        self.move(command)
        self.report("reach_end", seconds=monotonic() - start, snapshot=self.scene.snapshot())

    def grip(self, arm, closed):
        command = self.scene.command.copy()
        index = 5 if arm == "left" else 11
        command[index] = self.scene.limits[index, 0 if closed else 1]
        self.move(command, .5)

    def _open_drawer(self):
        # Pinching the thin handle failed remotely: the fingertip site reached
        # the handle, then pulled 1 cm in -Y with no contact. Sweep a closed
        # gripper from behind the handle through it so the jaws paddle the lip.
        point = np.asarray(self.scene.site("drawer_grasp"), float)
        behind = point + np.array([0.0, 0.022, 0.004])
        front = point + np.array([0.0, -0.075, 0.0])
        last_error = None
        for roll in (None,):
            self.report("drawer_attempt", wrist_roll=roll)
            try:
                self.grip("left", False)
                self.reach("left", behind + np.array([0.0, 0.0, 0.06]), approach=[0, 0, -1])
                self.reach("left", behind, approach=[0, 0, -1])
                self.grip("left", True)
                for frac in np.linspace(0.2, 1.0, 6):
                    self.reach("left", behind + frac * (front - behind), approach=[0, 0, -1])
                    if float(self.scene.data.joint("drawer_slide").qpos[0]) >= 0.05:
                        self.grip("left", False)
                        return
            except RuntimeError as error:
                self.report("drawer_attempt_failed", error=str(error))
                last_error = error
                continue
        raise RuntimeError(
            f"Drawer did not open by contact: {self.scene.snapshot()}; last={last_error}"
        )

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
            self.check_budget()
            self.report("skill_start", skill=action.skill.value, target=action.target)
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
