"""Shared polling/recipe fixture: the contract burn-governor vendors."""
import json
from pathlib import Path
import time
import unittest

from nenpi.activity_model import Operation, Response
from nenpi.command_classification import RECIPES, classify_steps, shell_step, tool_step
from nenpi.polling_report import build, mechanical_runs
from tests import test_auto_report as auto_fixtures
from tests.test_drain import Harness, claude_tool_output_line, claude_user_prompt_line

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


PR = "98765"
SECRET_RESULT = "SENTINEL-RESULT-TEXT"


class PollingReport(Harness):
    loop = auto_fixtures.AutoReport.loop
    session = auto_fixtures.AutoReport.session

    def claude_session(self, calls, sid="claude-poll"):
        """calls: (tool name, input dict, result text or None) per model response."""
        now = time.time() - 3000
        lines = [claude_user_prompt_line(now, sid, text="ship it")]
        for i, (name, tool_input, result) in enumerate(calls):
            lines.append(json.dumps({
                "type": "assistant", "sessionId": sid, "cwd": "/repo",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(now + 2 + i * 3)),
                "requestId": "req-%d" % i,
                "message": {"id": "msg-%d" % i, "role": "assistant", "model": "claude-opus-5",
                    "content": [{"type": "tool_use", "id": "call-%d" % i, "name": name, "input": tool_input}],
                    "usage": {"input_tokens": 1000, "cache_read_input_tokens": 99_000,
                              "cache_creation_input_tokens": 0, "output_tokens": 10}}}))
            if result is not None:
                lines.append(claude_tool_output_line(now + 3 + i * 3, sid, "call-%d" % i, result))
        self.write_claude("poll.jsonl", lines)

    def test_claude_checks_loop_is_pure_poll_and_wait_for_status(self):
        check = ("Bash", {"command": "gh pr checks " + PR}, SECRET_RESULT + " pending 1m%ds")
        self.claude_session([check] * 4)
        report = self.run_json("polling", "--harness", "claude", "--json")
        self.assertEqual(report["summary"]["responses"], 2)
        self.assertEqual(report["summary"]["input_tokens"], 200_000)
        self.assertEqual(report["by_kind"]["pure_poll"]["responses"], 2)
        self.assertEqual(report["by_recipe"]["wait-for-status"]["responses"], 2)
        top = report["top_patterns"][0]
        self.assertEqual((top["pattern"], top["recipe_id"], top["runs"]), ("gh pr checks", "wait-for-status", 1))
        self.assertGreater(report["summary"]["weighted_units"], 0)
        self.assertEqual(report["advisory_once_per_session_pattern"]["after_first_advisory"]["responses"], 1)
        sweep = {row["threshold"]: row["responses"] for row in report["threshold_sweep"]}
        self.assertEqual(sweep, {2: 3, 3: 2, 4: 1, 5: 0})
        prompt = report["sessions"][0]["prompts"][0]
        self.assertEqual((prompt["polling_responses"], prompt["recipes"]), (2, {"wait-for-status": 2}))
        text = json.dumps(report)
        self.assertNotIn(PR, text)
        self.assertNotIn(SECRET_RESULT, text)
        for cache in (self.root / "cache" / "activities-v5").glob("*.json"):
            self.assertNotIn(PR, cache.read_text())
            self.assertNotIn(SECRET_RESULT, cache.read_text())

    def test_changing_results_are_watch_and_edits_reset(self):
        check = "gh run view " + PR
        self.claude_session([
            ("Bash", {"command": check}, "queued"), ("Bash", {"command": check}, "in_progress"),
            ("Edit", {"file_path": "/repo/a.py", "old_string": "a", "new_string": "b"}, "ok"),
            ("Bash", {"command": check}, "done"), ("Bash", {"command": check}, "done"),
            ("Bash", {"command": check}, "x"), ("Bash", {"command": check}, "y")])
        report = self.run_json("polling", "--harness", "claude", "--json")
        self.assertEqual(report["by_kind"]["pure_poll"]["responses"], 1)
        self.assertEqual(report["by_kind"]["watch"]["responses"], 1)

    def test_codex_process_waits_and_activities_json_section(self):
        self.loop(5)
        report = self.run_json("polling", "--harness", "codex", "--json")
        self.assertEqual(report["by_kind"]["pure_poll"]["responses"], 3)
        self.assertEqual(report["top_patterns"][0]["pattern"], "Waiting for processes")
        self.assertIsNone(report["top_patterns"][0]["recipe_id"])
        activities = self.run_json("activities", "--harness", "codex", "--json")
        section = activities["polling"]
        self.assertEqual(section["summary"]["responses"], 3)
        self.assertEqual(section["sessions"][0]["prompts"][0]["polling_responses"], 3)

    def test_human_report_is_small(self):
        self.claude_session([("Bash", {"command": "gh pr checks " + PR}, "pending")] * 3)
        result = self.run_tool("polling", "--harness", "claude")
        self.assertEqual(result.returncode, 0, result.stderr)
        text = result.stdout.decode()
        self.assertIn("wait-for-status", text)
        self.assertLess(len(text.splitlines()), 40)
        self.assertNotIn(PR, text)


def _response(i, ops, results=(), prompt=1):
    return Response("claude", "acct", "s", "t", prompt, "p", float(i), "r%d" % i, "m",
                    1000, 900, 1, 1.0, {}, operations=ops, results=list(results))


class OfflineScores(unittest.TestCase):
    def test_mechanical_runs_need_distinct_small_unedited_steps(self):
        families = ["git status", "gh pr view", "git log", "ls", "gh api", "git diff", "rg"]
        ops = [[Operation("x", step=shell_step(c + " %d" % i, "/r"))] for i, c in enumerate(families)]
        rows = mechanical_runs([_response(i, op, [(i, 10)]) for i, op in enumerate(ops)], 5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["distinct_ratio"], 1.0)
        self.assertEqual(rows[0]["responses"], 7)
        edit = [Operation("Editing code", step=tool_step("Edit", {"file_path": "a"}))]
        broken = [_response(i, op, [(i, 10)]) for i, op in enumerate(ops[:4] + [edit] + ops[4:])]
        self.assertEqual(mechanical_runs(broken, 5), [])
        large = [_response(i, op, [(i, 5000)]) for i, op in enumerate(ops)]
        self.assertEqual(mechanical_runs(large, 5), [])

    def test_mechanical_score_stays_out_of_flags(self):
        ops = [[Operation("x", step=shell_step("git status %d" % i, "/r"))] for i in range(8)]
        report = build([_response(i, op, [(i, 10)]) for i, op in enumerate(ops)], sweep=False)
        self.assertEqual(report["summary"]["responses"], 0)
        self.assertNotIn("mechanical_runs", report)


if __name__ == "__main__":
    unittest.main()
