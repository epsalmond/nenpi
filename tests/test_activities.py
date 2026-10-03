"""Activity parity, response accounting, and conservative hypothetical savings."""
import dataclasses
import json
import time

from nenpi.activity_model import Response, exec_operations, operation, shell_operation
from nenpi.activity_report import bucket_responses
from nenpi.context_savings import estimate_savings
from tests import test_auto_report as auto_fixtures
from tests.test_drain import Harness, claude_assistant_line, claude_user_prompt_line


def response(i, harness="codex", **changes):
    value = Response(harness, "." + harness, "session", "thread", 1, "project", i,
        "r%d" % i, "model", 100_000, 99_000, 10, 1,
        {"input": 10, "cached_input": 1} if harness == "codex" else
        {"input": 10, "cache_read": 1, "cache_write_5m": 12.5, "cache_write_1h": 20})
    return dataclasses.replace(value, **changes)


class Activities(Harness):
    loop = auto_fixtures.AutoReport.loop
    session = auto_fixtures.AutoReport.session

    def test_default_polling_matches_model_calls(self):
        self.loop(400)
        report = self.run_json("--json")
        self.assertEqual(report["scope"]["since"], "72h")
        bucket = report["activities"][0]
        self.assertEqual(bucket["activity"], "Waiting for processes")
        self.assertEqual(bucket["responses"], 400)
        self.assertEqual(bucket["repeated_responses"], 399)
        self.assertEqual(bucket["repeated_input_tokens"], 39_900_000)
        self.assertNotIn("session_id:1", json.dumps(report))

    def test_claude_bash_polling_parity(self):
        now = time.time() - 1000
        sid = "claude-loop"
        lines = [claude_user_prompt_line(now, sid, text="check CI")]
        for i in range(4):
            record = json.loads(claude_assistant_line(now + 2 + i * 3, sid,
                message_id="msg-%d" % i, input_tokens=1000, cache_read=99_000))
            record["message"]["content"] = [{"type": "tool_use", "id": "call-%d" % i,
                "name": "Bash", "input": {"command": "gh run view 123 --json status"}}]
            lines.append(json.dumps(record))
        self.write_claude("loop.jsonl", lines)
        report = self.run_json("activities", "--harness", "claude", "--json")
        bucket = report["activities"][0]
        self.assertEqual(bucket["activity"], "Waiting for CI")
        self.assertEqual(bucket["responses"], 4)
        self.assertEqual(bucket["repeated_responses"], 3)
        self.assertEqual(sum(b["responses"] for b in report["activities"]), report["summary"]["responses"])

    def test_same_target_not_shared_between_threads(self):
        op = shell_operation("gh run view 123")
        report = bucket_responses([response(1, operations=[op]), response(2, operations=[op]),
            response(3, operations=[op], thread="other"), response(4, operations=[op], prompt=2)])
        self.assertEqual(report[0]["repeated_responses"], 1)

    def test_mixed_response_is_never_charged_twice_or_all_repetition(self):
        ops = [shell_operation("gh run view 123"), shell_operation("cat app.py")]
        report = bucket_responses([response(1, operations=ops), response(2, operations=ops)])
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["input_tokens"], 200_000)
        self.assertEqual(report[0]["repeated_responses"], 0)

    def test_literal_wrappers_and_targets(self):
        first = shell_operation("rtk proxy gh run view 123 --json status")
        self.assertEqual(first, shell_operation("gh run watch 123"))
        self.assertNotEqual(first.target, shell_operation("gh run view 456").target)
        self.assertEqual(exec_operations("await tools.write_stdin({session_id:123})"),
                         operation("write_stdin", {"session_id": 123}))
        self.assertFalse(exec_operations("await tools.write_stdin({...args})")[0].target)

    def test_hypothetical_policy_protects_recent_outputs(self):
        self.assertEqual(estimate_savings([response(1, results=[(1.5, 20_000)]),
            response(2, context=125_000)]), [])

    def test_hypothetical_cache_cost_parity_and_reset(self):
        def history(harness):
            return [response(1, harness, context=20_000, results=[(1.5, 10_000)]),
                *[response(i, harness, context=60_000 + i * 100, cached=60_000,
                    cache_write_kind="cache_write_5m") for i in range(2, 62)]]
        codex = estimate_savings(history("codex"))[0]
        claude = estimate_savings(history("claude"))[0]
        self.assertEqual(codex["reduction_tokens"], 9880)
        self.assertEqual(claude["reduction_tokens"], 9880)
        self.assertGreater(codex["net_units_saved"], claude["net_units_saved"])
        self.assertGreater(claude["break_even_calls"], codex["break_even_calls"])
        rows = history("codex")
        rows[1].reset = True
        self.assertEqual(estimate_savings(rows), [])
        applied = [{"session_id": "session", "thread_id": "thread", "prompt": 1}]
        self.assertEqual(estimate_savings(history("codex"), applied), [])
        self.assertTrue(estimate_savings(history("claude"), applied))

    def test_cold_avoids_rebuild_and_credits_first_write(self):
        rows = [response(1, context=20_000, results=[(1.5, 10_000)]),
                response(2, context=60_000, cached=0)]
        estimate = estimate_savings(rows)[0]
        self.assertEqual(estimate["break_even_calls"], 1)
        self.assertEqual(estimate["cache_write_tokens"], 0)
        self.assertAlmostEqual(estimate["net_read_equivalent_tokens_saved"], 98_800)

    def test_cache_contains_no_commands_and_invalidates_on_append(self):
        self.loop(4)
        first = self.run_json("--json")
        second = self.run_json("--json")
        self.assertEqual(first, second)
        caches = list((self.root / "cache" / "activities-v7").glob("*.json"))
        self.assertTrue(caches)
        for cache in caches:
            text = cache.read_text()
            self.assertNotIn("write_stdin", text)
            self.assertNotIn("session_id", text)
        self.loop(6)
        updated = self.run_json("--json")
        self.assertEqual(updated["summary"]["responses"], 10)
        self.assertEqual(updated["activities"][0]["responses"], 10)

    def test_dynamic_input_does_not_become_a_status_poll(self):
        op = exec_operations("await tools.write_stdin({session_id:123, chars: input})")[0]
        self.assertFalse(op.polling)
        op = exec_operations("await tools.wait_agent({...args})")[0]
        self.assertFalse(op.target)
        op = exec_operations('await tools.wait_agent({ids:["a","b"]})')[0]
        self.assertEqual(op.target, operation("wait_agent", {"ids": ["a", "b"]})[0].target)

    def test_literal_chains_preserve_status_target_and_mixed_work(self):
        op = shell_operation("cd /repo && rtk proxy gh run view 123 --json status | jq .status")
        self.assertEqual(op.activity, "Waiting for CI")
        self.assertTrue(op.target)
        mixed = shell_operation("gh run view 123; git commit -m fix")
        self.assertEqual(mixed.activity, "Mixed activity")
        self.assertFalse(mixed.polling)

    def test_activity_filter_and_followup_keep_scope(self):
        self.loop(4)
        report = self.run_json("activities", "--activity", "processes", "--since", "7d", "--json")
        self.assertEqual(report["summary"]["responses"], 4)
        self.assertEqual(report["summary"]["input_tokens"], 400_000)
        command = report["activities"][0]["locations"][0]["inspect"]
        self.assertIn("--harness codex", command)
        self.assertIn("--since 7d", command)
        self.assertIn("--prompt 1", command)

    def test_default_render_highlights_repetition_and_is_ascii_safe(self):
        from nenpi.activity_report import build_report
        from nenpi import drain as d
        from nenpi.report_render import render_activities
        import re
        self.loop(4)
        args = d.build_parser().parse_args(["activities", "--since", "7d"])
        with self.env_applied():
            report = build_report(d.prepare(args))
        args.width, args.ascii, args.color = 60, True, "always"
        colored = render_activities(report, args)
        self.assertIn("\x1b[", colored)
        args.color = "never"
        plain = render_activities(report, args)
        self.assertEqual(re.sub(r"\x1b\[[0-9;]*m", "", colored), plain)
        self.assertTrue(plain.isascii())
        self.assertIn("Repeated activity", plain)
        self.assertNotIn("Most expensive", plain)
        self.assertNotIn("See sessions", plain)

    def test_claude_session_gets_hypothetical_savings_without_shake(self):
        from tests.test_drain import iso
        now = time.time() - 1000
        sid = "claude-savings"
        first = json.loads(claude_assistant_line(now + 1, sid, "first", input_tokens=20_000))
        first["message"]["content"] = [{"type": "tool_use", "id": "read", "name": "Read", "input": {"file_path": "/project/file"}}]
        result = {"type": "user", "sessionId": sid, "timestamp": iso(now + 2),
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "read", "content": "x" * 40_000}]}}
        lines = [claude_user_prompt_line(now, sid), json.dumps(first), json.dumps(result)]
        for i in range(3, 63):
            lines.append(claude_assistant_line(now + i, sid, "later-%d" % i, input_tokens=1000, cache_read=60_000 + i * 100))
        self.write_claude("savings.jsonl", lines)
        report = self.run_json("auto", "--harness", "claude", "--session", sid, "--json")
        self.assertEqual(report["shake"]["events"], [])
        self.assertEqual(len(report["estimated_shake_savings"]), 1)
        estimate = report["estimated_shake_savings"][0]
        self.assertEqual(estimate["harness"], "claude")
        self.assertEqual(estimate["reduction_tokens"], 9880)
        self.assertGreater(estimate["net_units_saved"], 0)

    def test_repeated_reads_stop_after_edits(self):
        read = shell_operation("rg needle src")
        edit = operation("apply_patch", {})[0]
        rows = [response(1, operations=[read]), response(2, operations=[read]),
                response(3, operations=[edit]), response(4, operations=[read])]
        buckets = bucket_responses(rows)
        self.assertEqual(buckets[0]["repeated_responses"], 1)

    def test_project_filters_and_followups(self):
        self.loop(4)
        yes = self.run_json("--project", "/project", "--json")
        self.assertEqual(yes["summary"]["responses"], 4)
        self.assertIn("--project /project", yes["next"][0]["cmd"])
        no = self.run_json("--exclude-project", "/project", "--json")
        self.assertEqual(no["summary"]["responses"], 0)
        self.assertEqual(no["estimated_shake_savings"], [])
        no = self.run_json("--project", "/project", "--exclude-project", "/project", "--json")
        self.assertEqual(no["summary"]["responses"], 0)

    def test_dynamic_read_targets_are_not_repetition(self):
        op = exec_operations("await tools.read_file({path: file, offset: 1})")[0]
        self.assertFalse(op.target)

    def test_project_directory_boundaries(self):
        from nenpi.project_filter import matches
        self.assertTrue(matches("/home/eric/arcade/apps/web", "/home/eric/arcade"))
        self.assertFalse(matches("/home/eric/arcade-old", "/home/eric/arcade"))
        self.assertTrue(matches("/home/eric/arcade", "arc*"))

    def test_missing_tool_record_is_not_unknown_command(self):
        buckets = bucket_responses([response(1), response(2, operations=operation("mystery", {}))])
        self.assertEqual({b["activity"] for b in buckets}, {"No linked tool call", "Other / unknown"})

    def test_project_filter_preserves_attribution_and_claude_parity(self):
        from nenpi import drain as d
        self.loop(4)
        now = time.time() - 1000
        lines = [claude_user_prompt_line(now, "claude-project", cwd="/project/app")]
        record = json.loads(claude_assistant_line(now + 1, "claude-project", "msg-project", input_tokens=1000))
        record["cwd"] = "/project/app"
        lines.append(json.dumps(record))
        self.write_claude("project.jsonl", lines)
        with self.env_applied():
            parser = d.build_parser()
            full = d.prepare(parser.parse_args(["activities"]))
            selected = d.prepare(parser.parse_args(["activities", "--project", "/project"]))
        self.assertEqual(set(selected.scan.sessions), set(full.scan.sessions))
        self.assertEqual([i.sessions for i in full.intervals], [i.sessions for i in selected.intervals])
        report = self.run_json("--project", "/project", "--json")
        self.assertEqual(report["summary"]["responses"], 5)
        self.assertEqual(self.run_json("--exclude-project", "/project", "--json")["summary"]["responses"], 0)

    def test_repetition_locations_precede_larger_nonrepeated_usage(self):
        op = shell_operation('gh run view 42')
        rows = [response(1, context=900000, operations=[op], thread='large'),
                response(2, context=1000, operations=[op], thread='loop'),
                response(3, context=1000, operations=[op], thread='loop')]
        locations = bucket_responses(rows)[0]['locations']
        self.assertEqual(locations[0]['thread_id'], 'loop')
        self.assertEqual(locations[0]['repeated_input_tokens'], 1000)

    def test_opaque_breakdowns_partition_response_input(self):
        rows = [response(1, operations=operation('mystery_tool', {})),
                response(2, operations=[shell_operation('rg needle .'), shell_operation('pytest')])]
        buckets = bucket_responses(rows)
        for bucket in buckets:
            self.assertEqual(sum(g['input_tokens'] for g in bucket['operation_groups']), bucket['input_tokens'])
        self.assertIn('mystery_tool', str(buckets))
        self.assertIn('Reading/searching code + Running builds/tests', str(buckets))
        shell = shell_operation('rg needle .; pytest')
        self.assertEqual(shell.detail, 'Reading/searching code + Running builds/tests')

    def test_compact_default_has_truncation_without_explanatory_blocks(self):
        from nenpi import drain as d
        from nenpi.report_render import render_activities
        self.loop(4)
        report = self.run_json('--json')
        report['total_activities'] += 18
        args = d.build_parser().parse_args(['activities', '--no-color', '--width', '80'])
        output = render_activities(report, args)
        self.assertIn('+18 more (--top 0)', output)
        self.assertNotIn('Projects', output)
        self.assertNotIn('Context reduction and payback', output)
        self.assertNotIn('See sessions', output)
        self.assertLess(len(output.splitlines()), 20)
