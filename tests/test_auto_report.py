"""An expensive polling loop must remain visible despite tiny tool results."""

import json
import time

from tests.test_drain import (
    Harness, claude_assistant_line, claude_user_prompt_line,
    codex_session_meta_line, codex_subagent_meta_line,
    codex_task_started_line, codex_tool_call_line, codex_tool_output_line,
    codex_turn_context_line, codex_usage_record_line, iso,
)


class AutoReport(Harness):
    session = "a4000000-1111-2222-3333-444444444444"

    def loop(self, count=400, *, thread=None, offset=0, tie=False, model="gpt-6-astra"):
        now = time.time() - 7200 + offset
        sid = thread or self.session
        lines = [
            codex_subagent_meta_line(now, sid, self.session, "/project") if thread else codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, model),
            codex_task_started_line(now + 1),
        ]
        for i in range(count):
            ts = now + 2 + i * 3
            cid = "poll-%s-%d" % (sid, i)
            lines += [
                codex_tool_call_line(ts, cid, "exec", item="custom_tool_call", command='await tools.write_stdin({session_id:1});'),
                codex_usage_record_line(ts + .1, sid, input_tokens=100_000,
                                        cached_input_tokens=99_000, output_tokens=10,
                                        turn_id=None if thread else "loop"),
                codex_tool_output_line(ts + (.1 if tie else .2), cid, "ok", item="custom_tool_call_output"),
            ]
        self.write_codex("rollout-%s.jsonl" % sid, lines, day=now)
        return now

    def report(self, *extra):
        return self.run_json("auto", "--json", *extra)

    def test_400_tiny_results_surface_repeated_context(self):
        self.loop()
        result = self.report()
        summary = result["summary"]
        self.assertEqual(summary["responses"], 400)
        self.assertEqual(summary["input_tokens"], 40_000_000)
        self.assertEqual(summary["cached_input_tokens"], 39_600_000)
        self.assertEqual(summary["uncached_input_tokens"], 400_000)
        self.assertEqual(summary["tool_calls"], 400)
        self.assertEqual(summary["result_chars"], 800)
        self.assertEqual(summary["small_result_responses"], 399)
        self.assertEqual(summary["small_result_max_streak"], 399)
        self.assertEqual(summary["small_result_result_chars"], 798)
        self.assertEqual(summary["small_result_cached_input_tokens"], 399 * 99_000)
        self.assertAlmostEqual(result["findings"][0]["values"]["small_result_weighted_share"], 399 / 400)
        self.assertEqual(result["findings"][0]["kind"], "small_results")
        self.assertIn("400 responses", result["findings"][0]["evidence"][0])
        self.assertEqual(result["scope"]["since"], "7d")
        self.assertIn("--explain", result["next"][0]["cmd"])
        self.assertIn("--since 7d", result["next"][0]["cmd"])
        self.assertNotIn("write_stdin", str(result))

    def test_single_large_result_does_not_hide_repetition(self):
        now = self.loop()
        other = "b4000000-1111-2222-3333-444444444444"
        self.write_codex("rollout-large.jsonl", [
            codex_session_meta_line(now, other, "/other"),
            codex_turn_context_line(now + 1, "gpt-6-astra"),
            codex_task_started_line(now + 1),
            codex_tool_call_line(now + 2, "large", "exec"),
            codex_usage_record_line(now + 3, other, input_tokens=200_000, cached_input_tokens=0, output_tokens=100),
            codex_tool_output_line(now + 4, "large", "x" * 200_000),
        ], day=now)
        result = self.report()
        self.assertEqual(result["findings"][0]["session_id"], self.session)
        large = result["findings"][1]
        self.assertEqual(large["values"]["result_chars"], 200_000)
        self.assertEqual(large["values"]["small_result_responses"], 0)
        self.assertEqual(large["kind"], "usage")

    def test_root_and_child_reconcile_without_thread_total_overlap(self):
        self.loop(20)
        self.loop(10, thread="c4000000-1111-2222-3333-444444444444", offset=2)
        result = self.report("--session", self.session, "--prompt", "1")
        summary = result["summary"]
        self.assertEqual(summary["responses"], 30)
        self.assertEqual(summary["root_responses"], 20)
        self.assertEqual(summary["subagent_responses"], 10)
        self.assertEqual(summary["input_tokens"], 3_000_000)
        self.assertEqual(summary["small_result_max_streak"], 19)
        self.assertEqual(summary["small_result_responses"], 28)
        threads = result["findings"][0]["threads"]
        self.assertEqual(sum(t["input_tokens"] for t in threads), summary["input_tokens"])

    def test_equal_timestamps_are_not_claimed_as_consuming_response_evidence(self):
        self.loop(20, tie=True)
        result = self.report()
        self.assertEqual(result["summary"]["responses"], 20)
        self.assertEqual(result["summary"]["small_result_responses"], 0)
        self.assertEqual(result["summary"]["unlinked_tool_results"], 20)

    def test_tool_free_responses_and_prompt_boundaries_do_not_extend_a_run(self):
        now = self.loop(20)
        path = next(self.codex_sessions.rglob("rollout-%s.jsonl" % self.session))
        with path.open("a") as f:
            for line in [
                codex_usage_record_line(now + 63, self.session, input_tokens=100_000, cached_input_tokens=99_000, output_tokens=10, turn_id="loop"),
                codex_usage_record_line(now + 64, self.session, input_tokens=100_000, cached_input_tokens=99_000, output_tokens=10, turn_id="loop"),
                codex_turn_context_line(now + 65, "gpt-6-astra"),
                codex_task_started_line(now + 65),
                codex_tool_output_line(now + 66, "boundary", "ok"),
                codex_usage_record_line(now + 67, self.session, input_tokens=100_000, cached_input_tokens=99_000, output_tokens=10, turn_id="next"),
            ]:
                f.write(line + "\n")
        result = self.report("--session", self.session, "--prompt", "1")
        self.assertEqual(result["summary"]["responses"], 22)
        self.assertEqual(result["summary"]["small_result_responses"], 20)
        self.assertEqual(result["summary"]["small_result_max_streak"], 20)
        second = self.report("--session", self.session, "--prompt", "2")
        self.assertEqual(second["summary"]["responses"], 1)
        self.assertEqual(second["summary"]["small_result_responses"], 0)
        self.assertEqual(second["summary"]["unlinked_tool_results"], 1)

    def test_unknown_weights_do_not_hide_input_burden(self):
        self.loop(20, model="unknown-future-model")
        result = self.report()
        self.assertEqual(result["summary"]["responses"], 20)
        self.assertEqual(result["summary"]["unweighted_responses"], 20)
        self.assertEqual(result["summary"]["small_result_unweighted_responses"], 19)
        self.assertEqual(result["findings"][0]["kind"], "small_results")
        self.assertIsNone(result["findings"][0]["share"])

    def test_empty_data_and_invalid_prompt_scope(self):
        result = self.report()
        self.assertEqual(result["summary"]["responses"], 0)
        self.assertEqual(result["findings"], [])
        self.assertTrue(result["next"])
        invalid = self.run_tool("auto", "--prompt", "2")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("requires --session", invalid.stderr.decode())

    def test_window_scope_and_warm_cache_are_stable(self):
        self.loop(20)
        first = self.report()
        second = self.report()
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(first["findings"], second["findings"])
        self.assertEqual(self.report("--since", "1h")["summary"]["responses"], 0)

    def test_mixed_harness_weights_are_not_added_and_empty_hints_keep_scope(self):
        now = self.loop(2)
        self.write_claude("other.jsonl", [
            claude_user_prompt_line(now, "claude-other"),
            claude_assistant_line(now + 2, "claude-other", "msg-1", input_tokens=100, cache_read=1000, output_tokens=20),
        ])
        result = self.report("--harness", "all")
        self.assertIsNone(result["summary"]["weighted_units"])
        self.assertEqual(set(result["summary"]["weighted_units_by_harness"]), {"codex", "claude"})
        root = str(self.home / ".codex")
        empty = self.report("--codex-root", root, "--account", "no-such-account", "--since", "1h")
        command = empty["next"][0]["cmd"]
        self.assertIn("--codex-root " + root, command)
        self.assertIn("--account no-such-account", command)
        self.assertIn("--harness codex", command)

    def test_auto_connects_applied_shake_to_cache_effect_and_current_context(self):
        now = self.loop(20)
        path = next(self.codex_sessions.rglob("rollout-%s.jsonl" % self.session))
        with path.open("a") as f:
            f.write(json.dumps({"timestamp": iso(now + 65), "type": "compacted",
                                "message": "[shake] context reduced surgically",
                                "replacement_history": []}) + "\n")
            f.write(codex_usage_record_line(now + 67, self.session, input_tokens=50_000,
                                           cached_input_tokens=1000, output_tokens=10, turn_id="loop") + "\n")
        result = self.report("--session", self.session, "--prompt", "1")
        self.assertEqual(result["shake"]["coverage"]["applied_markers_found"], 1)
        event = result["shake"]["events"][0]
        self.assertEqual(event["effect"], "context_lower_after_marker")
        self.assertEqual(event["before_context_tokens"], 100_000)
        self.assertEqual(event["after_context_tokens"], 50_000)
        self.assertEqual(event["after_uncached_input_tokens"], 49_000)
        self.assertNotIn("shake_candidate", result["findings"][0])
        self.assertTrue(any("uncached input" in e and "Shake" in e for e in result["findings"][0]["evidence"]))
