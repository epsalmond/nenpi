"""Shared polling/recipe fixture: the contract burn-governor vendors."""
import json
from pathlib import Path
import unittest

from nenpi.command_classification import RECIPES, classify_steps

FIXTURE = Path(__file__).parent / "fixtures" / "command-classification.json"


class SharedFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())

    def test_every_case(self):
        defaults = self.fixture["defaults"]
        for case in self.fixture["cases"]:
            options = dict(defaults, **case.get("options", {}))
            with self.subTest(case=case["name"]):
                self.assertEqual(len(case["steps"]), len(case["expect"]))
                got = classify_steps(case["steps"], **options)
                for index, (actual, expected) in enumerate(zip(got, case["expect"])):
                    self.assertEqual(actual, expected, "%s step %d" % (case["name"], index + 1))

    def test_contract_is_classification_only(self):
        allowed = {"polling", "kind", "recipe_id"}
        kinds = {"pure_poll", "watch", "none"}
        for case in self.fixture["cases"]:
            for expected in case["expect"]:
                self.assertEqual(set(expected), allowed, case["name"])
                self.assertIn(expected["kind"], kinds)
                self.assertEqual(expected["polling"], expected["kind"] != "none")
                self.assertIn(expected["recipe_id"], set(RECIPES) | {None})
        self.assertEqual(set(self.fixture["recipes"]), set(RECIPES))

    def test_every_recipe_and_exclusion_is_covered(self):
        names = {case["name"] for case in self.fixture["cases"]}
        recipes = {e["recipe_id"] for case in self.fixture["cases"] for e in case["expect"]}
        kinds = {e["kind"] for case in self.fixture["cases"] for e in case["expect"]}
        self.assertEqual(recipes - {None}, set(RECIPES))
        self.assertEqual(kinds, {"pure_poll", "watch", "none"})
        for required in ("excluded-recommended-scripts", "excluded-watch-flag-and-gh-run-watch",
                         "excluded-sleeps-inside-scripts", "edit-between-repeats-resets-window",
                         "normal-exploration-distinct-reads-and-greps"):
            self.assertIn(required, names)


if __name__ == "__main__":
    unittest.main()
