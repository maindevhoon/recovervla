MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
JOINTS = tuple(f"{side}_{motor}" for side in ("left", "right") for motor in MOTORS)
CAMERAS = ("scene", "left_wrist", "right_wrist")
FPS = 20
RESOLUTION = 128


def features():
    result = {key: {"dtype": "float32", "shape": (12,), "names": list(JOINTS)}
              for key in ("observation.state", "action")}
    result.update({f"observation.images.{name}": {
        "dtype": "video", "shape": (RESOLUTION, RESOLUTION, 3),
        "names": ["height", "width", "channels"]} for name in CAMERAS})
    return result
