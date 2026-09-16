"""Record a physical expert rollout and render it to MP4 on remote Linux.

This is a visualization of the demonstration generator, not a learned-policy
evaluation. Failed attempts are deliberately retained and labelled in JSON.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if sys.platform != "linux":
    raise SystemExit("Run this script on Kaggle/Colab/Linux, never on the Mac.")
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco

from recovervla.evaluation import REFERENCE_INSTRUCTION
from recovervla.planner import TableSettingPlanner
from recovervla.sim.expert import Expert
from recovervla.sim.scene import Scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot-dir", type=Path, default=Path("artifacts/so101"))
    parser.add_argument("--camera", default="hero")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--sample-every", type=int, default=2,
                        help="Capture one frame per N 20 Hz control steps")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.seed < 0 or min(args.width, args.height, args.sample_every,
                            args.video_fps, args.timeout) < 1:
        parser.error("Seed must be nonnegative; dimensions and rates must be positive")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required to encode MP4 on the remote host")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    scene = Scene(args.robot_dir, args.seed, render=False)
    poses = [scene.data.qpos.copy()]
    stages = ["initial"]
    events = []
    stage = "initial"
    steps = 0
    original_step = scene.step

    def capture_step(action):
        nonlocal steps
        original_step(action)
        steps += 1
        if steps % args.sample_every == 0:
            poses.append(scene.data.qpos.copy())
            stages.append(stage)

    def progress(event, **fields):
        nonlocal stage
        if event == "skill_start":
            stage = f"{fields['skill']}:{fields['target']}"
            print(f"stage {stage}", flush=True)
        if event == "skill_start" or event.endswith("failed"):
            events.append({"event": event, "step": steps,
                           "fields": fields if event == "skill_start" else
                           {"error": str(fields.get("error", ""))}})

    scene.step = capture_step
    expert = Expert(scene, record=None, progress=progress, timeout=args.timeout)
    error = None
    try:
        expert.run(TableSettingPlanner().plan(REFERENCE_INSTRUCTION))
    except (RuntimeError, TimeoutError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"rollout stopped: {error}", flush=True)
    finally:
        scene.step = original_step
        if steps % args.sample_every:
            poses.append(scene.data.qpos.copy())
            stages.append(stage)

    report = {"seed": args.seed, "success": error is None,
              "error": error, "completed_skills": expert.history,
              "last_stage": stage, "control_steps": steps,
              "video_frames": len(poses), "sample_every": args.sample_every,
              "video_fps": args.video_fps, "camera": args.camera,
              "video": str(args.output), "events": events,
              "note": "Expert demonstration visualization; not a learned-policy rollout."}
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"report {report_path}", flush=True)

    renderer = None
    encoder = None
    try:
        renderer = mujoco.Renderer(scene.model, height=args.height, width=args.width)
        replay = mujoco.MjData(scene.model)
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size",
                   f"{args.width}x{args.height}", "-framerate", str(args.video_fps),
                   "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(args.output)]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        for index, pose in enumerate(poses):
            replay.qpos[:] = pose
            mujoco.mj_forward(scene.model, replay)
            renderer.update_scene(replay, camera=args.camera)
            encoder.stdin.write(renderer.render().tobytes())
            if index and index % 250 == 0:
                print(f"rendered {index}/{len(poses)} frames", flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
        print(f"video {args.output} ({len(poses)} frames)", flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if renderer is not None:
            renderer.close()
        scene.close()


if __name__ == "__main__":
    main()
