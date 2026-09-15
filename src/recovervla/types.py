from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Arm(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


class Skill(StrEnum):
    OPEN_DRAWER = "open_drawer"
    GRASP = "grasp"
    PLACE = "place"
    HANDOFF = "handoff"
    POUR = "pour"
    VERIFY = "verify"


@dataclass(frozen=True)
class Action:
    skill: Skill
    arm: Arm
    target: str
    destination: str | None = None
    verify: str | None = None
    max_retries: int = 1


@dataclass(frozen=True)
class TaskPlan:
    instruction: str
    actions: tuple[Action, ...]

    def validate(self) -> None:
        if not self.actions:
            raise ValueError("a task plan must contain at least one action")
        if not any(action.arm is Arm.BOTH for action in self.actions):
            raise ValueError("the plan must contain a genuinely bimanual action")
        if self.actions[-1].skill is not Skill.VERIFY:
            raise ValueError("the plan must end with final-state verification")


@dataclass(frozen=True)
class SceneVariation:
    seed: int
    position_offset_m: float
    friction: float
    object_mass_scale: float
    light_intensity: float


@dataclass
class RunResult:
    seed: int
    success: bool
    completed_actions: int
    total_actions: int
    retries: int
    failures: list[str] = field(default_factory=list)

