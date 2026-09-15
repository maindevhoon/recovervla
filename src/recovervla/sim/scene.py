import numpy as np
import mujoco
from .build_scene import build
from .schema import CAMERAS, FPS, JOINTS, RESOLUTION


class Scene:
    def __init__(self, robot_dir, seed):
        xml, self.variation = build(robot_dir, seed)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(name).id for name in JOINTS]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.dadr = self.model.jnt_dofadr[self.joints]
        self.aids = [self.model.actuator(name).id for name in JOINTS]
        self.limits = self.model.actuator_ctrlrange[self.aids].copy()
        self.command = np.zeros(12)
        self.command[[5, 11]] = 1.0
        self.data.qpos[self.qadr] = self.command
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=RESOLUTION, width=RESOLUTION)
        for _ in range(20):
            self.step(self.command)

    def close(self):
        self.renderer.close()

    def observe(self):
        frame = {"observation.state": self.data.qpos[self.qadr].astype(np.float32).copy()}
        for camera in CAMERAS:
            self.renderer.update_scene(self.data, camera=camera)
            frame[f"observation.images.{camera}"] = self.renderer.render().copy()
        return frame

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Invalid 12-D action")
        self.command = np.clip(action, self.limits[:, 0], self.limits[:, 1])
        self.data.ctrl[self.aids] = self.command
        for _ in range(round(1 / FPS / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("Non-finite physics")

    def site(self, name):
        return self.data.site(name).xpos.copy()

    def body(self, name):
        return self.data.body(name).xpos.copy()

    def contact(self, arm, target):
        for contact in self.data.contact:
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     for g in (contact.geom1, contact.geom2)]
            if any(n.startswith(arm + "_") and "jaw" in n for n in names) and any(n.startswith(target + "_") for n in names):
                return True
        return False

    def contained(self):
        origin = self.body("mug")
        rotation = self.data.body("mug").xmat.reshape(3, 3)
        particles = np.array([self.body(f"water_{i:02}") for i in range(60)])
        local = (particles - origin) @ rotation
        return int(((np.linalg.norm(local[:, :2], axis=1) < .024) &
                    (local[:, 2] > .007) & (local[:, 2] < .051)).sum())
