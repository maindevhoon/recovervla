"""Decode every frame using the same LeRobot version as training."""
import argparse
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from recovervla.sim.schema import CAMERAS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", required=True)
    a = p.parse_args()
    ds = LeRobotDataset(a.repo_id, root=a.root, video_backend="pyav")
    if len(ds) == 0:
        raise ValueError("Empty dataset")
    for i in range(len(ds)):
        frame = ds[i]
        for name in ("observation.state", "action"):
            if frame[name].shape != (12,) or not torch.isfinite(frame[name]).all():
                raise ValueError(f"Invalid {name} at frame {i}")
        for camera in CAMERAS:
            value = frame[f"observation.images.{camera}"]
            if value.shape != (3, 128, 128) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid camera {camera} at frame {i}")
    print(f"Validated {len(ds)} frames / {ds.num_episodes} episodes")


if __name__ == "__main__":
    main()
