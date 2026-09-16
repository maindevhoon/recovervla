"""Construct a dual SO-101 scene from a pinned upstream XML and local meshes."""
from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from .dining import decorate

REFERENCE_ATTRS = {"name", "class", "childclass", "joint", "joint1", "joint2", "body",
                  "body1", "body2", "site", "site1", "site2", "mesh", "material",
                  "tendon", "actuator", "target", "camera"}


def vector(values):
    return " ".join(str(float(value)) for value in values)


def geom(parent, name, kind, size, pos=(0, 0, 0), **kwargs):
    return ET.SubElement(parent, "geom", name=name, type=kind, size=vector(size),
                         pos=vector(pos), **{key: str(value) for key, value in kwargs.items()})


def vessel(world, name, pos, radius, height):
    body = ET.SubElement(world, "body", name=name, pos=vector(pos))
    ET.SubElement(body, "freejoint", name=name + "_free")
    geom(body, name + "_floor", "cylinder", (radius + .004, .003), mass="0.04")
    # Overlapping tangential boxes form a watertight collision ring.
    for i in range(20):
        angle = 2 * math.pi * i / 20
        geom(body, f"{name}_wall_{i}", "box", (.004, radius * .18, height / 2),
             ((radius + .004) * math.cos(angle), (radius + .004) * math.sin(angle), height / 2),
             euler=f"0 0 {angle}", mass="0.003", rgba="0.3 0.6 0.9 1")
    ET.SubElement(body, "site", name=name + "_grasp", pos=f"0 0 {height / 2}")
    return body


def build(robot_dir: Path, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    root = ET.Element("mujoco", model="recovervla")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast",
                  iterations="50", cone="elliptic", impratio="10", solver="CG",
                  noslip_iterations="0")
    # MuJoCo 3.3.7 native cylinder/box contacts let the plate tunnel through
    # the tray in our isolated drop test. The legacy collider preserves support.
    ET.SubElement(root.find("option"), "flag", nativeccd="disable")
    assets = ET.SubElement(root, "asset")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", friction="0.8 0.005 0.0001")
    world = ET.SubElement(root, "worldbody")
    actuators = ET.SubElement(root, "actuator")
    # The upstream zero pose extends along local +X. Face both arms toward
    # the shared workspace, rather than toward opposite table edges (+/-Y).
    for side, x, yaw in (("left", -.26, 0), ("right", .26, math.pi)):
        arm = ET.parse(robot_dir / "so101.xml").getroot()
        for mesh in arm.findall("./asset/mesh"):
            mesh.set("name", mesh.get("name", Path(mesh.attrib["file"]).stem))
            mesh.set("file", str((robot_dir / "assets" / mesh.attrib["file"]).resolve()))
        for node in arm.iter():
            for key in REFERENCE_ATTRS & node.attrib.keys():
                node.set(key, side + "_" + node.attrib[key])
        for node in arm.findall("./asset/*"):
            assets.append(deepcopy(node))
        for node in arm.findall("./default/*"):
            # Global mesh default can be shared; named defaults remain isolated.
            if node.tag == "default":
                defaults.append(deepcopy(node))
        base = arm.find("./worldbody/body")
        base.set("pos", f"{x} 0 0.025")
        base.set("quat", vector((math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))))
        for camera in base.iter("camera"):
            camera.set("name", side + "_wrist")
        world.append(base)
        for node in arm.findall("./actuator/*"):
            actuators.append(deepcopy(node))
    ET.SubElement(world, "light", pos="0 0 1.5", diffuse=vector([rng.uniform(.6, 1)] * 3))
    ET.SubElement(world, "camera", name="scene", pos="0 -0.85 0.8", xyaxes="1 0 0 0 0.68 0.73")
    geom(world, "table", "box", (.45, .35, .025), rgba="0.5 0.35 0.2 1")
    decorate(root, world, assets)
    drawer = ET.SubElement(world, "body", name="drawer", pos="0 0.08 0.05")
    ET.SubElement(drawer, "joint", name="drawer_slide", type="slide", axis="0 -1 0",
                  range="0 .09", damping="0.2")
    geom(drawer, "drawer_floor", "box", (.08, .07, .006), mass="0.08")
    geom(drawer, "drawer_back", "box", (.08, .006, .02), (0, .07, .02), mass="0.02")
    for side in (-1, 1):
        geom(drawer, f"drawer_side_{side}", "box", (.006, .07, .02), (side * .08, 0, .02), mass="0.02")
    # Lift the handle above the drawer lip and into the SO-101's reachable
    # near-table workspace. The former 6.5 cm world height was 2.65 cm beyond
    # the closest deterministic IK solution on remote MuJoCo validation.
    # A wide box plus a front lip gives the gripper a paddle surface. Contact
    # still has to move the joint; there is no hidden weld.
    handle_kwargs = dict(mass="0.025", friction="2.5 .05 .001",
                         condim="6", solref="0.01 1", priority="1")
    geom(drawer, "drawer_handle", "box", (.04, .012, .016), (0, -.08, .045), **handle_kwargs)
    geom(drawer, "drawer_handle_lip", "box", (.04, .006, .02), (0, -.098, .04), **handle_kwargs)
    ET.SubElement(drawer, "site", name="drawer_grasp", pos="0 -.08 .045")
    positions = {"plate": (0, .08, .061), "mug": (.16, -.05, .032), "bottle": (-.12, -.08, .032)}
    sampled = {}
    for name, position in positions.items():
        pos = np.array(position) + np.r_[rng.uniform(-.01, .01, 2), 0]
        sampled[name] = pos.tolist()
        if name == "plate":
            body = ET.SubElement(world, "body", name=name, pos=vector(pos))
            ET.SubElement(body, "freejoint", name=name + "_free")
            geom(body, "plate_geom", "cylinder", (.04, .004), mass="0.035", rgba="0.9 0.9 0.9 1")
            ET.SubElement(body, "site", name="plate_grasp", pos="0 -.034 0")
        else:
            body = vessel(world, name, pos, .028 if name == "mug" else .023, .055 if name == "mug" else .10)
        mass_scale, friction = rng.uniform(.8, 1.2), rng.uniform(.6, 1.1)
        for node in body.findall("geom"):
            node.set("mass", str(float(node.attrib["mass"]) * mass_scale))
            node.set("friction", f"{friction} .005 .0001")
    bottle = np.array(sampled["bottle"])
    for i in range(60):
        layer, offset = divmod(i, 6)
        a = offset * math.pi / 3
        pos = bottle + [.012 * math.cos(a), .012 * math.sin(a), .009 + layer * .0085]
        body = ET.SubElement(world, "body", name=f"water_{i:02}", pos=vector(pos))
        ET.SubElement(body, "freejoint")
        geom(body, f"water_geom_{i}", "sphere", (.004,), mass="0.0003", rgba="0.1 0.3 1 1")
    for name, x in (("fork", -.05), ("knife", 0), ("napkin", .05)):
        geom(world, name, "box", (.008, .025, .002), (x, -.20, .028), rgba="0.8 0.8 0.8 1")
    return ET.tostring(root, encoding="unicode"), {"seed": seed, "positions": sampled}
