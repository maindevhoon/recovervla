import unittest

from recovervla.evaluation import REFERENCE_INSTRUCTION, evaluate, make_variation
from recovervla.planner import TableSettingPlanner
from recovervla.types import Arm, Skill


class PlannerTests(unittest.TestCase):
    def test_plan_is_bimanual_and_verified(self) -> None:
        plan = TableSettingPlanner().plan(REFERENCE_INSTRUCTION)
        self.assertTrue(any(action.arm is Arm.BOTH for action in plan.actions))
        self.assertIs(plan.actions[-1].skill, Skill.VERIFY)

    def test_incomplete_instruction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TableSettingPlanner().plan("Put the plate down")

    def test_variations_are_reproducible(self) -> None:
        self.assertEqual(make_variation(7), make_variation(7))
        self.assertNotEqual(make_variation(7), make_variation(8))

    def test_ten_seed_report_has_benchmark_schema(self) -> None:
        report = evaluate(10)
        self.assertEqual(report["seed_count"], 10)
        self.assertEqual(len(report["runs"]), 10)
        self.assertIn("latency_ms", report["intel_benchmark"])


if __name__ == "__main__":
    unittest.main()
