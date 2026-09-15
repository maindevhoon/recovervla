from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict

from .planner import TableSettingPlanner
from .types import Action, RunResult, SceneVariation, TaskPlan


REFERENCE_INSTRUCTION = (
    "Open the top drawer, pick up the plate, place it on the table, pick up "
    "the mug, and pour water into the mug."
)


def make_variation(seed: int) -> SceneVariation:
    rng = random.Random(seed)
    return SceneVariation(
        seed=seed,
        position_offset_m=round(rng.uniform(-0.08, 0.08), 4),
        friction=round(rng.uniform(0.35, 1.20), 4),
        object_mass_scale=round(rng.uniform(0.75, 1.25), 4),
        light_intensity=round(rng.uniform(0.60, 1.40), 4),
    )


class DryRunExecutor:
    """Deterministic placeholder for the future MuJoCo skill executor."""

    def execute(self, plan: TaskPlan, scene: SceneVariation) -> RunResult:
        failures: list[str] = []
        retries = 0
        completed = 0
        for index, action in enumerate(plan.actions):
            attempts = 0
            while not self._succeeds(action, scene, index, attempts):
                if attempts >= action.max_retries:
                    failures.append(f"{action.skill}:{action.target}")
                    return RunResult(scene.seed, False, completed, len(plan.actions), retries, failures)
                attempts += 1
                retries += 1
            completed += 1
        return RunResult(scene.seed, True, completed, len(plan.actions), retries, failures)

    @staticmethod
    def _succeeds(action: Action, scene: SceneVariation, index: int, attempt: int) -> bool:
        difficulty = abs(scene.position_offset_m) * 4 + abs(scene.friction - 0.7)
        difficulty += abs(scene.object_mass_scale - 1.0)
        threshold = 1.1 + attempt * 0.45
        return difficulty + index * 0.01 <= threshold


def evaluate(seeds: int) -> dict[str, object]:
    plan = TableSettingPlanner().plan(REFERENCE_INSTRUCTION)
    executor = DryRunExecutor()
    runs = [executor.execute(plan, make_variation(seed)) for seed in range(seeds)]
    successes = sum(run.success for run in runs)
    return {
        "executor": "dry-run",
        "disclaimer": "Not a MuJoCo or hardware result",
        "seed_count": seeds,
        "successes": successes,
        "success_rate": successes / seeds if seeds else 0.0,
        "runs": [asdict(run) for run in runs],
        "intel_benchmark": {
            "device": None,
            "precision": None,
            "latency_ms": None,
            "throughput_fps": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    payload = json.dumps(evaluate(args.seeds), indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)


if __name__ == "__main__":
    main()

