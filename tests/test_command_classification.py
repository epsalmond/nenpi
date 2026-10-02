"""Shared polling/recipe fixture: the contract burn-governor vendors."""
import json
from pathlib import Path
import time
import unittest

from nenpi.activity_model import Operation, Response
from nenpi.command_classification import RECIPES, classify_steps, shell_step, tool_step
from nenpi.polling_report import build, mechanical_runs
from tests import test_auto_report as auto_fixtures
from tests.test_drain import (Harness, claude_tool_output_line, claude_user_prompt_line, codex_session_meta_line,
                              codex_task_started_line, codex_tool_call_line, codex_tool_output_line,
                              codex_turn_context_line, codex_usage_record_line)

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
                         "normal-exploration-distinct-reads-and-greps",
                         "excluded-recommended-scripts-via-interpreters-and-wrappers",
                         "excluded-recommended-scripts-in-compound-commands", "excluded-wrapped-wait-loops",
                         "pr-reads-during-review-are-not-status-checks", "tail-sources-are-log-paths-only",
                         "codex-exec-call-is-one-step"):
            self.assertIn(required, names)


PR = "98765"
SECRET_RESULT = "SENTINEL-RESULT-TEXT"


class Labels(unittest.TestCase):
    """Labels reach caches and reports, so they hold allowlisted names only."""

    CASES = {
        "git -C acme-secret-repo status": "git status",
        "git -c user.name=acme-name commit -m acme-message": "git commit",
        "git acme-alias": "git",
        "kubectl -n acme-prod get pods": "kubectl get",
        "kubectl --context acme-ctx logs acme-pod": "kubectl logs",
        "docker --context acme-ctx ps": "docker ps",
        "docker logs --tail 50 acme-web": "docker logs",
        "npm --prefix acme-dir run acme-task": "npm run",
        "cargo +acme-toolchain build": "cargo build",
        "make acme-target": "make",
        "./scripts/acme-payroll-export --since 1d": "script",
        "~/bin/acme-payroll-export": "script",
        "acme-payroll-export --since 1d": "shell",
        "bash ./acme-deploy.sh": "script",
        "python3 tools/acme_report.py": "script",
        "sudo -u acme-user journalctl -u acme-unit": "journalctl",
        "timeout 30 acme-tool": "shell",
        "gh acme-alias acme-arg": "gh",
        "gh pr view acme-branch --json body": "gh pr view",
        "gh -R acme/secret pr checks 12": "gh pr checks",
        "gh api repos/acme/secret/pulls/1/comments": "gh api",
        "nohup scripts/wait-for-status acme-pr &": "wait-for-status",
        "bash ~/git/acme/scripts/query-logs --container acme-web": "query-logs",
        "tail -n 20 /var/log/acme.log": "tail",
        "(acme-tool --flag)": "shell",
    }

    def test_labels_hold_no_argument_values(self):
        for command, expected in self.CASES.items():
            with self.subTest(command=command):
                label = shell_step(command, "/acme-cwd").label
                self.assertEqual(label, expected)
                self.assertNotIn("acme", label)


class CodexExec(Harness):
    """A code-mode exec call is one step however many tools its code calls."""

    def exec_session(self, calls):
        now = time.time() - 3000
        sid = "c0de0000-1111-2222-3333-444444444444"
        lines = [codex_session_meta_line(now, sid, "/repo"), codex_turn_context_line(now + 1, "gpt-6-astra"),
                 codex_task_started_line(now + 1)]
        for i, code in enumerate(calls):
            ts, cid = now + 2 + i * 3, "exec-%d" % i
            lines += [codex_tool_call_line(ts, cid, "exec", item="custom_tool_call", command=code),
                      codex_usage_record_line(ts + .1, sid, input_tokens=100_000, cached_input_tokens=99_000,
                                              output_tokens=10, turn_id="t"),
                      codex_tool_output_line(ts + .2, cid, "build pending", item="custom_tool_call_output")]
        self.write_codex("rollout-%s.jsonl" % sid, lines, day=now)

    CODE = "".join("await tools.exec_command({cmd: 'gh pr checks %s'});\n" % PR for _ in range(3))

    def test_one_exec_with_repeated_tool_calls_does_not_flag_itself(self):
        self.exec_session([self.CODE])
        report = self.run_json("polling", "--harness", "codex", "--json")
        self.assertEqual(report["summary"]["responses"], 0)

    def test_repeated_exec_calls_still_flag(self):
        self.exec_session([self.CODE] * 3)
        report = self.run_json("polling", "--harness", "codex", "--json")
        self.assertEqual(report["summary"]["responses"], 1)
        self.assertEqual(report["by_kind"]["pure_poll"]["responses"], 1)
        self.assertEqual(report["by_recipe"]["wait-for-status"]["responses"], 1)
        self.assertEqual(report["top_patterns"][0]["pattern"], "gh pr checks")


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
        caches = list((self.root / "cache" / "activities-v6").glob("*.json"))
        self.assertTrue(caches)
        for cache in caches:
            self.assertNotIn(PR, cache.read_text())
            self.assertNotIn(SECRET_RESULT, cache.read_text())

    def test_private_names_stay_out_of_report_and_cache(self):
        commands = ["git -C acme-secret-repo status", "kubectl -n acme-prod get pods",
                    "./scripts/acme-payroll-export --since 1d"]
        self.claude_session([("Bash", {"command": c}, "same") for c in commands for _ in range(3)])
        report = self.run_json("polling", "--harness", "claude", "--json")
        self.assertEqual(report["summary"]["responses"], 3)
        self.assertEqual({row["pattern"] for row in report["top_patterns"]}, {"git status", "kubectl get", "script"})
        self.assertNotIn("acme", json.dumps(report))
        for cache in (self.root / "cache" / "activities-v6").glob("*.json"):
            self.assertNotIn("acme", cache.read_text())

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
