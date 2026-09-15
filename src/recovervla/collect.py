"""Remote-only collection through LeRobot's official v3 dataset writer."""
import argparse
import json
import platform
from pathlib import Path
from .evaluation import REFERENCE_INSTRUCTION
from .planner import TableSettingPlanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-dir", type=Path, default=Path("artifacts/so101"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="maindevhoon/recovervla-demonstrations")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Collection is configured for remote Linux hosts")
    if args.episodes < 1 or args.max_attempts < args.episodes or args.start_seed < 0:
        parser.error("Require positive episodes, max-attempts >= episodes, nonnegative seed")
    if args.start_seed <= 10009 and args.start_seed + args.max_attempts > 10000:
        parser.error("Seeds 10000–10009 are reserved for evaluation")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; choose a new output")
    repo = Path(__file__).resolve().parents[2]
    if output.is_relative_to(repo) and not output.is_relative_to(repo / "datasets"):
        parser.error("In-repository output must be under datasets/")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from .sim.schema import features, FPS
    from .sim.scene import Scene
    from .sim.expert import Expert
    ds = LeRobotDataset.create(args.repo_id, FPS, features(), root=output,
                               robot_type="so101_bimanual", vcodec="libx264")
    report = {"executor": "mujoco-scripted-expert", "angles": "radian", "attempts": [],
              "disclaimer": "Scripted demonstration collection; remote physics validation required",
              "evaluation_seeds": list(range(10000, 10010))}
    successes = 0
    try:
        for seed in range(args.start_seed, args.start_seed + args.max_attempts):
            scene = None
            row = {"seed": seed, "success": False}
            expert = None
            try:
                scene = Scene(args.robot_dir, seed)
                row["variation"] = scene.variation
                def record(frame):
                    ds.add_frame({**frame, "task": REFERENCE_INSTRUCTION})
                expert = Expert(scene, record)
                expert.run(TableSettingPlanner().plan(REFERENCE_INSTRUCTION))
            except RuntimeError as error:
                row["error"] = str(error)
                ds.clear_episode_buffer()
            else:
                # Writer errors abort collection, rather than masquerading as physics failures.
                ds.save_episode()
                row.update(success=True, episode_index=successes)
                successes += 1
            finally:
                row["completed_skills"] = expert.history if expert else []
                if scene is not None:
                    scene.close()
                report["attempts"].append(row)
                (output / "report.json").write_text(json.dumps(report, indent=2))
            if successes == args.episodes:
                break
    finally:
        ds.finalize()
    if successes < args.episodes:
        raise SystemExit(f"Only {successes}/{args.episodes} successful episodes; inspect report.json")


if __name__ == "__main__":
    main()
