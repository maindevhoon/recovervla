import numpy as np
import mujoco
from .build_scene import build
from .schema import CAMERAS, FPS, JOINTS, RESOLUTION


class Scene:
    def __init__(self, robot_dir, seed, render=True):
        xml, self.variation = build(robot_dir, seed)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.joints = [self.model.joint(name).id for name in JOINTS]
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.dadr = self.model.jnt_dofadr[self.joints]
        self.aids = [self.model.actuator(name).id for name in JOINTS]
        self.limits = self.model.actuator_ctrlrange[self.aids].copy()
        self.command = np.zeros(12)
        # Start fully open using the SO-101 actuator limits, not a unit guess.
        self.command[[5, 11]] = self.limits[[5, 11], 1]
        self.data.qpos[self.qadr] = self.command
        mujoco.mj_forward(self.model, self.data)
        self.renderer = None
        if render:
            self.renderer = mujoco.Renderer(self.model, height=RESOLUTION, width=RESOLUTION)
        for _ in range(20):
            self.step(self.command)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def observe(self):
        if self.renderer is None:
            raise RuntimeError("Scene was created without a renderer")
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

    def _on_gripper(self, arm, geom_id):
        gripper = self.model.body(arm + "_gripper").id
        body = int(self.model.geom_bodyid[geom_id])
        while body > 0:
            if body == gripper:
                return True
            body = int(self.model.body_parentid[body])
        return False

    def contact(self, arm, target):
        # Unnamed jaw meshes still belong to the gripper body tree; name
        # matching on "jaw" missed those contacts on remote validation.
        for contact in self.data.contact:
            geoms = (contact.geom1, contact.geom2)
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     for g in geoms]
            if any(self._on_gripper(arm, g) for g in geoms) and any(n.startswith(target + "_") for n in names):
                return True
        return False

    def contained(self):
        origin = self.body("mug")
        rotation = self.data.body("mug").xmat.reshape(3, 3)
        particles = np.array([self.body(f"water_{i:02}") for i in range(60)])
        local = (particles - origin) @ rotation
        return int(((np.linalg.norm(local[:, :2], axis=1) < .024) &
                    (local[:, 2] > .007) & (local[:, 2] < .051)).sum())

    def snapshot(self):
        gripper = self.site("left_gripperframe")
        handle = self.site("drawer_grasp")
        contacts = []
        for contact in self.data.contact[:min(int(self.data.ncon), 8)]:
            contacts.append([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                             for g in (contact.geom1, contact.geom2)])
        return {
            "drawer_qpos": float(self.data.joint("drawer_slide").qpos[0]),
            "handle": handle.tolist(),
            "left_gripper": gripper.tolist(),
            "handle_distance": float(np.linalg.norm(gripper - handle)),
            "drawer_contact": self.contact("left", "drawer"),
            "left_gripper_cmd": float(self.command[5]),
            "ncon": int(self.data.ncon),
            "contacts": contacts,
        }
