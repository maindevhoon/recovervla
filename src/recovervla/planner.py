from __future__ import annotations

from .types import Action, Arm, Skill, TaskPlan


class TableSettingPlanner:
    """Produces an auditable skill plan for the challenge's reference task."""

    def plan(self, instruction: str) -> TaskPlan:
        normalized = " ".join(instruction.lower().split())
        required = ("drawer", "plate", "mug", "pour")
        missing = [word for word in required if word not in normalized]
        if missing:
            raise ValueError(f"instruction is missing required concepts: {', '.join(missing)}")

        actions = (
            Action(Skill.OPEN_DRAWER, Arm.LEFT, "top_drawer", verify="drawer_open"),
            Action(Skill.GRASP, Arm.LEFT, "plate", verify="plate_grasped"),
            Action(Skill.PLACE, Arm.LEFT, "plate", "table_place_zone", "plate_placed"),
            Action(Skill.GRASP, Arm.RIGHT, "mug", verify="mug_grasped"),
            Action(Skill.GRASP, Arm.LEFT, "bottle", verify="bottle_grasped"),
            Action(Skill.POUR, Arm.BOTH, "bottle", "mug", "pour_complete", max_retries=2),
            Action(Skill.PLACE, Arm.RIGHT, "mug", "table_mug_zone", "mug_placed"),
            Action(Skill.VERIFY, Arm.BOTH, "table_setting", verify="task_complete", max_retries=0),
        )
        plan = TaskPlan(instruction=instruction, actions=actions)
        plan.validate()
        return plan

