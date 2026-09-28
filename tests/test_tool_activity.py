"""Usage explanations must reconcile responses, not assume one call per turn."""

import json
import time
import unittest
from unittest.mock import patch

import nenpi.tool_activity as tool_activity
from nenpi.drain import command_shape
from nenpi.tool_activity import _safe_example, exec_activity
from tests.test_drain import (
    Harness, codex_session_meta_line, codex_subagent_meta_line,
    codex_task_started_line, codex_tool_call_line, codex_tool_output_line,
    codex_turn_context_line, codex_usage_record_line,
)


class StaticExecActivity(unittest.TestCase):
    def test_batch_and_wrappers_expose_underlying_commands(self):
        names, examples = exec_activity('''
            await Promise.all([
                tools.exec_command({cmd: "rtk proxy sh -c 'git status --short'"}),
                tools.exec_command({cmd: 'rg -n needle src'}),
                tools.write_stdin({session_id: 7, chars: ""})
            ]);
        ''')
        self.assertEqual(names, ["process poll", "shell: git status --short", "shell: rg -n <arg> <arg>"])
        self.assertEqual(len(examples), 2)

    def test_comments_strings_and_dynamic_commands_are_not_executed_or_unwrapped(self):
        names, examples = exec_activity('''
            // tools.apply_patch("not a call")
            const text = "tools.apply_patch('not a call either')";
            const pattern = /tools.fake()/;
            /* tools.exec_command({cmd:'ignored'}) */
            await tools.exec_command({cmd: 'git ' + action});
            await tools.exec_command({cmd: `rg ${needle} src`});
            await tools.write_stdin({chars: input, session_id: 7});
        ''')
        self.assertEqual(names, ["process input/dynamic", "shell: dynamic command"])
        self.assertEqual(examples, [])
        self.assertEqual(exec_activity("const f = tools[name]; await f(args);")[0], ["exec: unclassified code"])
        for argument in ("{...options}", "{session_id:7, chars}", "options"):
            self.assertEqual(exec_activity("await tools.write_stdin(" + argument + ");")[0], ["process input/dynamic"])

    def test_literal_template_and_input_are_distinct_from_polling(self):
        names, examples = exec_activity('await tools.exec_command({"cmd": `git diff`}); await tools.write_stdin({chars:"y\\n"});')
        self.assertEqual(names, ["process input/dynamic", "shell: git diff"])
        self.assertEqual(examples, ["git diff"])
        self.assertEqual(command_shape('rtk proxy bash -lc "git status"'), "git status")

    def test_nested_arrays_and_member_access_do_not_hide_literal_properties(self):
        names, examples = exec_activity('''
            await tools.exec_command({
                cmd: "git status",
                cwd: process.cwd(),
                args: [1, {nested: true}]
            });
            await tools.write_stdin({session_id: 7, chars: "", metadata: {tags: ["x"]}});
        ''')
        self.assertEqual(names, ["process poll", "shell: git status"])
        self.assertEqual(examples, ["git status"])

    def test_common_auth_values_are_redacted_but_useful_paths_remain(self):
        command = (
            "curl -H 'Authorization: Bearer header-secret' "
            "--api-key='flag-secret' --token option-secret "
            "OPENAI_API_KEY=env-secret --password \"phrase secret\" "
            "https://example.test/private-resource"
        )
        example = _safe_example(command, 500)
        for secret in ("header-secret", "flag-secret", "option-secret", "env-secret", "phrase secret"):
            self.assertNotIn(secret, example)
        self.assertIn("<redacted>", example)
        self.assertIn("https://example.test/private-resource", example)


class ToolActivityReport(Harness):
    session = "23000000-1111-2222-3333-444444444444"

    def fixture(self):
        now = time.time() - 3600
        self.write_codex("rollout-activity.jsonl", [
            codex_session_meta_line(now, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5-codex"),
            codex_task_started_line(now + 1),
            codex_tool_call_line(now + 2, "search", "exec", item="custom_tool_call",
                                 command='await tools.exec_command({cmd:"rg -n UNIQUE_SEARCH src"});'),
            codex_tool_call_line(now + 2, "poll", "exec", item="custom_tool_call",
                                 command='await tools.write_stdin({session_id:7, chars:""});'),
            codex_tool_output_line(now + 3, "search", "result", item="custom_tool_call_output"),
            codex_tool_output_line(now + 3, "poll", "", item="custom_tool_call_output"),
            codex_usage_record_line(now + 4, self.session, input_tokens=5000,
                                    cached_input_tokens=4000, output_tokens=50, turn_id="one"),
            codex_turn_context_line(now + 10, "gpt-5-codex"),
            codex_task_started_line(now + 10),
            codex_tool_call_line(now + 11, "edit", "exec", item="custom_tool_call",
                                 command='await tools.apply_patch("UNIQUE_PATCH_BODY");'),
            codex_tool_output_line(now + 12, "edit", "done", item="custom_tool_call_output"),
            codex_usage_record_line(now + 13, self.session, input_tokens=8000,
                                    cached_input_tokens=6000, output_tokens=200, turn_id="two"),
        ], day=now)
        return now

    def report(self, *extra):
        return self.run_json("tools", "--harness", "codex", "--session", self.session,
                             "--explain", "--json", *extra)

    def test_shared_response_counts_once_and_prompt_scope_reconciles(self):
        self.fixture()
        payload = self.report("--prompt", "1")
        activity = payload["activity"]
        self.assertEqual(activity["tool_calls"], 2)
        self.assertEqual(activity["explained_calls"], 2)
        self.assertEqual(activity["model_responses"], 1)
        self.assertEqual(activity["unmatched_calls"], 0)
        self.assertEqual(len(activity["groups"]), 1)
        group = activity["groups"][0]
        self.assertEqual(group["uncached_input_tokens"], 1000)
        self.assertEqual(group["cached_input_tokens"], 4000)
        self.assertEqual(group["output_tokens"], 50)
        self.assertEqual(group["activity"], "process poll + shell: rg -n <arg> <arg>")
        self.assertEqual(group["tool_family"], "exec")
        self.assertEqual(group["result_chars"], 6)
        self.assertEqual(group["est_result_tokens"], 1.5)
        self.assertEqual(activity["tool_families"][0]["tool_family"], "exec")
        self.assertEqual(activity["tool_families"][0]["model_responses"], 1)
        self.assertAlmostEqual(activity["tool_families"][0]["weighted_units"], activity["weighted_units"])
        self.assertEqual(activity["result_chars"], 6)
        self.assertEqual(activity["explained_result_chars"], 6)
        self.assertEqual(activity["unexplained_result_chars"], 0)
        self.assertEqual(activity["poll_only_calls"], 0)
        second = self.report("--prompt", "2")["activity"]
        self.assertEqual(second["tool_calls"], 1)
        self.assertEqual(second["groups"][0]["activity"], "apply_patch")
        full = self.report()["activity"]
        self.assertEqual(full["model_responses"], 2)
        self.assertAlmostEqual(full["weighted_units"], activity["weighted_units"] + second["weighted_units"])

    def test_default_stays_private_and_explanation_does_not_cache_inputs(self):
        self.fixture()
        plain = self.run_json("tools", "--session", self.session, "--prompt", "1", "--json")
        self.assertNotIn("activity", plain)
        self.assertNotIn("UNIQUE_SEARCH", json.dumps(plain))
        self.assertTrue(any("--explain" in hint["cmd"] for hint in plain["next"]))
        first = self.report()["activity"]
        self.assertEqual(first, self.report()["activity"])
        cache = "\n".join(p.read_text() for p in (self.root / "cache").rglob("*.json"))
        self.assertNotIn("UNIQUE_SEARCH", cache)
        self.assertNotIn("UNIQUE_PATCH_BODY", cache)
        self.assertNotIn("UNIQUE_PATCH_BODY", json.dumps(first))

    def test_missing_response_id_remains_unpriced(self):
        now = self.fixture()
        usage = json.loads(codex_usage_record_line(now + 22, self.session, input_tokens=9000,
                                                  cached_input_tokens=8000, output_tokens=10, turn_id="two"))
        usage["payload"].pop("response_id")
        path = next(self.codex_sessions.rglob("rollout-activity.jsonl"))
        with path.open("a") as f:
            for line in [
                codex_tool_call_line(now + 20, "noid", "exec", item="custom_tool_call",
                                     command='await tools.write_stdin({session_id:7});'),
                codex_tool_output_line(now + 21, "noid", "", item="custom_tool_call_output"),
                json.dumps(usage),
            ]:
                f.write(line + "\n")
        activity = self.report()["activity"]
        self.assertEqual(activity["explained_calls"], 4)
        self.assertEqual(activity["unmatched_calls"], 1)
        self.assertEqual(activity["model_responses"], 2)
        self.assertEqual(activity["poll_only_calls"], 1)

    def test_subagent_response_is_not_charged_to_root_twice(self):
        now = self.fixture()
        thread = "24000000-1111-2222-3333-444444444444"
        self.write_codex("rollout-child.jsonl", [
            codex_subagent_meta_line(now + 2, thread, self.session, "/project"),
            codex_turn_context_line(now + 2, "gpt-5-codex"),
            codex_tool_call_line(now + 3, "child", "exec", item="custom_tool_call",
                                 command='await tools.exec_command({cmd:"git diff"});'),
            codex_tool_output_line(now + 4, "child", "diff", item="custom_tool_call_output"),
            codex_usage_record_line(now + 5, thread, input_tokens=2000,
                                    cached_input_tokens=1000, output_tokens=20),
        ], day=now)
        activity = self.report("--prompt", "1")["activity"]
        self.assertEqual(activity["model_responses"], 2)
        self.assertEqual(activity["tool_calls"], 3)
        self.assertEqual(sum(g["subagent_calls"] for g in activity["groups"]), 1)
        self.assertEqual(sum(g["uncached_input_tokens"] for g in activity["groups"]), 2000)

    def test_usage_without_tool_calls_is_in_scope_coverage_denominator(self):
        now = self.fixture()
        path = next(self.codex_sessions.rglob("rollout-activity.jsonl"))
        with path.open("a") as f:
            f.write(codex_usage_record_line(
                now + 30, self.session, input_tokens=10000,
                cached_input_tokens=9000, output_tokens=100, turn_id="one",
            ) + "\n")
        activity = self.report("--prompt", "1")["activity"]
        self.assertEqual(activity["model_responses"], 1)
        self.assertEqual(activity["scope_model_responses"], 2)
        self.assertEqual(activity["uncovered_model_responses"], 1)
        self.assertEqual(activity["coverage_model_responses"], 0.5)
        self.assertEqual(activity["scope_cached_input_tokens"], 13000)
        self.assertEqual(activity["scope_uncached_input_tokens"], 2000)
        self.assertIsNone(activity["coverage_weighted_usage"])

    def test_missing_usage_boundary_does_not_price_an_earlier_batch(self):
        now = time.time() - 3600
        self.write_codex("rollout-boundary.jsonl", [
            codex_session_meta_line(now, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5-codex"),
            codex_task_started_line(now + 1),
            codex_tool_call_line(now + 2, "first", "exec", item="custom_tool_call",
                                 command='await tools.write_stdin({session_id:7, chars:""});'),
            codex_tool_output_line(now + 3, "first", "ok", item="custom_tool_call_output"),
            codex_turn_context_line(now + 4, "gpt-5-codex"),
            codex_tool_call_line(now + 5, "second", "exec", item="custom_tool_call",
                                 command='await tools.write_stdin({session_id:7, chars:""});'),
            codex_tool_output_line(now + 6, "second", "ok", item="custom_tool_call_output"),
            codex_usage_record_line(now + 7, self.session, input_tokens=5000,
                                    cached_input_tokens=4000, output_tokens=50,
                                    turn_id="one"),
        ], day=now)
        activity = self.report("--prompt", "1")["activity"]
        self.assertEqual(activity["tool_calls"], 2)
        self.assertEqual(activity["explained_calls"], 2)
        self.assertEqual(activity["model_responses"], 1)
        self.assertEqual(activity["unpriced_tool_call_responses"], 1)
        self.assertEqual(activity["unmatched_calls"], 1)
        self.assertEqual(activity["groups"][0]["uncached_input_tokens"], 1000)
        self.assertEqual(activity["groups"][0]["cached_input_tokens"], 4000)

    def test_outputs_after_usage_record_do_not_split_the_next_response_batch(self):
        now = time.time() - 3600
        self.write_codex("rollout-output-order.jsonl", [
            codex_session_meta_line(now, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5-codex"),
            codex_task_started_line(now + 1),
            codex_tool_call_line(now + 2, "one", "exec", item="custom_tool_call",
                                 command='await tools.write_stdin({session_id:7, chars:""});'),
            codex_usage_record_line(now + 3, self.session, input_tokens=5000,
                                    cached_input_tokens=4000, output_tokens=50,
                                    turn_id="one"),
            codex_tool_output_line(now + 4, "one", "one", item="custom_tool_call_output"),
            codex_tool_call_line(now + 5, "two", "exec", item="custom_tool_call",
                                 command='await tools.write_stdin({session_id:7, chars:""});'),
            codex_tool_call_line(now + 5, "three", "exec", item="custom_tool_call",
                                 command='await tools.write_stdin({session_id:7, chars:""});'),
            codex_usage_record_line(now + 6, self.session, input_tokens=6000,
                                    cached_input_tokens=5000, output_tokens=60,
                                    turn_id="one"),
            codex_tool_output_line(now + 7, "two", "two", item="custom_tool_call_output"),
            codex_tool_output_line(now + 7, "three", "three", item="custom_tool_call_output"),
        ], day=now)
        activity = self.report("--prompt", "1")["activity"]
        self.assertEqual(activity["tool_calls"], 3)
        self.assertEqual(activity["model_responses"], 2)
        self.assertEqual(activity["unmatched_calls"], 0)
        self.assertEqual(activity["unpriced_tool_call_responses"], 0)
        self.assertEqual(activity["groups"][0]["tool_calls"], 3)
        self.assertEqual(activity["groups"][0]["model_responses"], 2)
        self.assertEqual(activity["groups"][0]["result_chars"], 11)

    def test_large_result_body_is_not_decoded_for_response_boundary_detection(self):
        now = time.time() - 3600
        sentinel = "RESULT_BODY_MUST_NOT_BE_DECODED"
        self.write_codex("rollout-large-output.jsonl", [
            codex_session_meta_line(now, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5-codex"),
            codex_task_started_line(now + 1),
            codex_tool_call_line(now + 2, "large", "exec", item="custom_tool_call",
                                 command='await tools.write_stdin({session_id:7, chars:""});'),
            codex_tool_output_line(now + 3, "large", sentinel, item="custom_tool_call_output"),
            codex_usage_record_line(now + 4, self.session, input_tokens=5000,
                                    cached_input_tokens=4000, output_tokens=50,
                                    turn_id="one"),
        ], day=now)
        path = next(self.codex_sessions.rglob("rollout-large-output.jsonl"))
        original_loads = json.loads

        def guarded_loads(raw, *args, **kwargs):
            if isinstance(raw, str) and sentinel in raw:
                raise AssertionError("tool output body was decoded")
            return original_loads(raw, *args, **kwargs)

        with patch.object(tool_activity.json, "loads", side_effect=guarded_loads):
            batches = list(tool_activity._source_activity(path))
        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0][0])

    def test_mixed_outer_families_share_one_exclusive_response_row(self):
        now = time.time() - 3600
        self.write_codex("rollout-mixed.jsonl", [
            codex_session_meta_line(now, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5-codex"),
            codex_task_started_line(now + 1),
            codex_tool_call_line(now + 2, "exec", "exec", item="custom_tool_call",
                                 command='await tools.write_stdin({session_id:7, chars:""});'),
            codex_tool_call_line(now + 2, "message", "send_message",
                                 item="custom_tool_call", namespace="collaboration",
                                 command='{"target":"agent","message":"status"}'),
            codex_tool_output_line(now + 3, "exec", "ok", item="custom_tool_call_output"),
            codex_tool_output_line(now + 3, "message", "", item="custom_tool_call_output"),
            codex_usage_record_line(now + 4, self.session, input_tokens=9000,
                                    cached_input_tokens=8000, output_tokens=100,
                                    turn_id="one"),
        ], day=now)
        activity = self.report("--prompt", "1")["activity"]
        self.assertEqual(activity["model_responses"], 1)
        self.assertEqual(len(activity["tool_families"]), 1)
        family = activity["tool_families"][0]
        self.assertEqual(family["tool_family"], "collaboration.send_message + exec")
        self.assertEqual(family["tool_calls"], 2)
        self.assertEqual(family["model_responses"], 1)
        group = activity["groups"][0]
        self.assertEqual(group["tool_family"], family["tool_family"])
        self.assertEqual(group["activity"], "collaboration.send_message + process poll")
        self.assertEqual(group["weighted_units"], family["weighted_units"])

    def test_four_hundred_small_poll_results_expose_repeated_input_usage(self):
        now = time.time() - 3600
        lines = [
            codex_session_meta_line(now, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5-codex"),
            codex_task_started_line(now + 1),
        ]
        for index in range(400):
            stamp = now + 2 + index * 4
            call_id = "poll-%03d" % index
            lines.extend([
                codex_tool_call_line(
                    stamp, call_id, "exec", item="custom_tool_call",
                    command='await tools.write_stdin({session_id:7, chars:""});',
                ),
                codex_tool_output_line(
                    stamp + 1, call_id, "ok", item="custom_tool_call_output",
                ),
                codex_usage_record_line(
                    stamp + 2, self.session, input_tokens=80000,
                    cached_input_tokens=79000, output_tokens=50, turn_id="loop",
                ),
            ])
        # One large result provides the counterpart: high output volume in one
        # response, with the same input footprint as each polling response.
        lines.extend([
            codex_tool_call_line(now + 1602, "large", "search", item="function_call",
                                 command="rg -n needle src"),
            codex_tool_output_line(now + 1603, "large", "x" * 200000),
            codex_usage_record_line(now + 1604, self.session, input_tokens=80000,
                                    cached_input_tokens=79000, output_tokens=50,
                                    turn_id="loop"),
        ])
        self.write_codex("rollout-four-hundred.jsonl", lines, day=now)
        payload = self.report("--prompt", "1")
        activity = payload["activity"]
        self.assertEqual(activity["tool_calls"], 401)
        self.assertEqual(activity["model_responses"], 401)
        self.assertEqual(activity["scope_model_responses"], 401)
        self.assertEqual(activity["uncovered_model_responses"], 0)
        self.assertEqual(activity["poll_only_calls"], 400)
        self.assertEqual(activity["scope_unweighted_responses"], 401)
        self.assertEqual(activity["matched_unweighted_responses"], 401)
        poll = next(row for row in activity["groups"] if row["activity"] == "process poll")
        search = next(row for row in activity["groups"] if row["activity"] == "search")
        self.assertEqual(poll["tool_calls"], 400)
        self.assertEqual(poll["model_responses"], 400)
        self.assertEqual(poll["result_chars"], 800)
        self.assertEqual(poll["cached_input_tokens"], 400 * 79000)
        self.assertEqual(poll["uncached_input_tokens"], 400 * 1000)
        self.assertEqual(poll["unweighted_responses"], 400)
        self.assertEqual(search["tool_calls"], 1)
        self.assertEqual(search["model_responses"], 1)
        self.assertEqual(search["result_chars"], 200000)
        self.assertEqual(search["unweighted_responses"], 1)
        self.assertGreater(poll["cached_input_tokens"], search["cached_input_tokens"])
        self.assertLess(poll["est_result_tokens"], search["est_result_tokens"])
        self.assertAlmostEqual(
            sum(row["weighted_units"] for row in activity["tool_families"]),
            activity["weighted_units"],
        )
        self.assertAlmostEqual(
            sum(row["weighted_units"] for row in activity["groups"]),
            activity["weighted_units"],
        )


if __name__ == "__main__":
    unittest.main()
