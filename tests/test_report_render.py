import unittest
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

from nenpi.report_render import _command_text, _table, _wrap_parts, render_activity, render_auto
from nenpi.theme import DEFAULT_COLORS


class ReportRenderTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(no_color=True, ascii=True, width=60, top=5)

    def sample_report(self):
        return {
            "scope": {"harness": "codex", "session_id": "01234567", "prompt": 2},
            "findings": [{"session_id": "01234567", "prompt": 2, "cwd": "project",
                "title": "Small results, repeated large context", "severity": "high",
                "values": {"responses": 400, "input_tokens": 40000000,
                    "cached_input_tokens": 39200000, "uncached_input_tokens": 800000,
                    "weighted_units": 100, "small_result_responses": 398,
                    "small_result_input_tokens": 39800000,
                    "small_result_cached_input_tokens": 39000000,
                    "small_result_uncached_input_tokens": 800000,
                    "small_result_weighted_share": 0.72, "small_result_max_streak": 211},
                "next": [{"cmd": "nenpi tools --session 01234567 --prompt 2 --explain", "why": "verbose explanation"}],
            }],
            "shake": {"events": [{"session_id": "01234567", "prompt": 2,
                "thread_id": "01234567", "role": "subagent", "agent_name": "Ampere", "before_context_tokens": 100000,
                "after_context_tokens": 50000, "reduction_tokens": 50000,
                "before_uncached_input_tokens": 1000, "after_uncached_input_tokens": 49000,
                "subsequent_response_count": 196,
                "payback": {"status": "estimated", "read_tokens_saved": 9800000,
                    "break_even_calls": 9, "break_even_reached": True,
                    "extra_uncached_tokens": 48000, "net_read_equivalent_tokens_saved": 9368000}}]},
        }

    def test_auto_uses_compact_tables(self):
        args = SimpleNamespace(no_color=True, ascii=False, width=120)
        output = render_auto(self.sample_report(), args)
        self.assertIn("Tiny output → large read", output)
        self.assertIn("Shake · subagents of this prompt", output)
        self.assertIn("━", output)
        self.assertIn("39.8M", output)
        self.assertIn("72.0%", output)
        self.assertIn("100.0K → 50.0K", output)
        self.assertIn("(−50.0K)", output)
        self.assertIn("48.0K", output)
        self.assertIn("Post-Shake Turns", output)
        self.assertIn("01234567 Ampere", output)
        self.assertIn("196", output)
        self.assertIn("9.4M", output)
        self.assertIn("Break-even turn*", output)
        self.assertEqual(output.count("* Cache write:"), 1)
        self.assertNotIn("Est.", output)
        self.assertLess(len(output.splitlines()), 26)
        for redundant in ("HIGH", "repeated large context", "responses following",
                          "not measured", "not tokens saved", "0 responses lack",
                          "ratio", "verbose explanation", "transform stats"):
            self.assertNotIn(redundant, output)

    def test_auto_theme_and_plain_output_match(self):
        args = SimpleNamespace(no_color=False, color="always", ascii=True, width=60)
        with patch.dict(os.environ, {"NO_COLOR": "1"}), \
                patch("nenpi.drain.load_theme", return_value=("codex", DEFAULT_COLORS.copy())):
            output = render_auto(self.sample_report(), args)
        for code in ("32", "39"):
            self.assertIn("\033[" + code + "m", output)
        plain = render_auto(self.sample_report(), self.args)
        self.assertEqual(re.sub(r"\x1b\[[0-9;]*m", "", output), plain)
        self.assertTrue(plain.isascii())
        self.assertTrue(all(len(line) <= 60 for line in plain.splitlines()
                            if "nenpi tools" not in line))

    def test_net_cost_is_negative_and_red(self):
        report = self.sample_report()
        report["shake"]["events"][0]["payback"]["net_read_equivalent_tokens_saved"] = -348000
        report["shake"]["events"][0]["payback"]["break_even_reached"] = False
        report["shake"]["events"][0]["subsequent_response_count"] = 5
        with patch("nenpi.drain.load_theme", return_value=("codex", DEFAULT_COLORS.copy())):
            output = render_auto(report, SimpleNamespace(no_color=False, color="always", width=140))
        self.assertRegex(output, r"\x1b\[31m *-348.0K")
        self.assertNotIn("--348", output)
        self.assertRegex(output, r"\x1b\[31m *5\x1b\[0m")
        self.assertRegex(output, r"\x1b\[39m *9\x1b\[0m")

    def test_missing_payback_is_not_zero(self):
        report = self.sample_report()
        report["shake"]["events"][0]["payback"] = {"status": "unknown_weights"}
        output = render_auto(report, self.args)
        self.assertIn("unknown weights", output)
        self.assertNotIn("0 calls", output)
        report["findings"][0]["values"]["unweighted_responses"] = 4
        self.assertIn("4 calls lack model weights", render_auto(report, self.args))

    def test_codex_table_geometry_and_numeric_alignment(self):
        from nenpi.drain import Painter
        lines = []
        _table(lines, ["A", "Count"], [[("x", "data"), ("1", "data")],
                                     [("y", "data"), ("22", "data")]],
               80, Painter(False), indent="")
        self.assertEqual(lines, [" A    Count", "━━━  ━━━━━━━", " x        1", " y       22"])

    def test_commands_preserve_shell_quoting_with_separate_roles(self):
        from nenpi.drain import Painter
        command = "nenpi tools --codex-root '/a path' --prompt 2"
        with patch("nenpi.drain.load_theme", return_value=("codex", DEFAULT_COLORS.copy())):
            output = _command_text(command, Painter(True, force_color=True))
        self.assertEqual(re.sub(r"\x1b\[[0-9;]*m", "", output), command)
        self.assertIn("\033[38;2;137;180;250mnenpi", output)
        self.assertIn("\033[38;2;235;160;172mcodex-root", output)
        self.assertIn("\033[38;2;166;227;161m'/a path'", output)

    def test_roles_survive_wrapping_and_theme_overrides(self):
        from nenpi.drain import Painter
        colors = DEFAULT_COLORS | {"label": "blue", "data": "magenta",
                                   "instruction": "red", "total": "cyan"}
        parts = [("Input: ", "label"), ("123 responses with large context ", "data"),
                 ("40M total", "total")]
        with patch("nenpi.drain.load_theme", return_value=("codex", colors)):
            painter = Painter(True, force_color=True)
        lines = []
        _wrap_parts(lines, parts, 28, painter, "  ")
        output = "\n".join(lines)
        self.assertIn("\033[34mInput: ", output)
        self.assertIn("\033[35m", output)
        self.assertIn("\033[1m\033[36m", output)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
        self.assertEqual(" ".join(plain.split()), "".join(p[0] for p in parts))
        self.assertTrue(all(len(line) <= 28 for line in plain.splitlines()))

    def test_activity_shows_outer_and_nested_views_with_usage_denominator(self):
        activity = {
            "tool_calls": 400, "explained_calls": 390, "model_responses": 390,
            "unmatched_calls": 10, "weighted_units": 500.0, "scope_weighted_units": 520.0,
            "scope_model_responses": 400,
            "coverage_model_responses": 0.975, "coverage_weighted_usage": 0.96,
            "result_chars": 8200, "est_result_tokens": 2050,
            "explained_result_chars": 8100, "unexplained_result_chars": 100,
            "scope_unweighted_responses": 25, "matched_unweighted_responses": 5,
            "unpriced_tool_call_responses": 3,
            "tool_families": [{"tool_family": "exec", "activity": "exec", "tool_calls": 390,
                               "model_responses": 390, "result_chars": 8000,
                               "cached_input_tokens": 39200000,
                               "uncached_input_tokens": 800000, "output_tokens": 24000,
                               "weighted_units": 480.0, "unweighted_responses": 1},
                              {"tool_family": "collaboration.send_message",
                               "activity": "collaboration.send_message", "tool_calls": 10,
                               "model_responses": 10, "result_chars": 100,
                               "cached_input_tokens": 1000, "uncached_input_tokens": 100,
                               "output_tokens": 50, "weighted_units": 20.0},
                              {"tool_family": "unlinked tool calls", "activity": "unlinked tool calls",
                               "tool_calls": 1, "model_responses": 0, "result_chars": 100,
                               "cached_input_tokens": 0, "uncached_input_tokens": 0,
                               "output_tokens": 0, "weighted_units": 0.0}],
            "groups": [{"tool_family": "exec", "activity": "process poll", "tool_calls": 390,
                        "model_responses": 390, "result_chars": 8000,
                        "cached_input_tokens": 39200000, "uncached_input_tokens": 800000,
                        "output_tokens": 24000, "weighted_units": 480.0,
                        "unmatched_calls": 0},
                       {"tool_family": "collaboration.send_message",
                        "activity": "collaboration.send_message", "tool_calls": 10,
                        "model_responses": 10, "result_chars": 100,
                        "cached_input_tokens": 1000, "uncached_input_tokens": 100,
                        "output_tokens": 50, "weighted_units": 20.0},
                       {"tool_family": "unlinked tool calls", "activity": "tool activity unlinked",
                        "tool_calls": 1, "model_responses": 0, "result_chars": 100,
                        "cached_input_tokens": 0, "uncached_input_tokens": 0,
                        "output_tokens": 0, "weighted_units": 0.0}],
            "poll_only_calls": 390,
            "note": "Static patterns associate complete responses, not a tool's causal share.",
        }

        output = render_activity(activity, self.args)
        normalized = " ".join(output.split())

        self.assertIn("390 parsed into activities", normalized)
        self.assertIn("response coverage 390 / 400 (97.5%)", normalized)
        self.assertIn("matched weighted-usage coverage 96.0%", normalized)
        self.assertIn("Tool families and their activity", normalized)
        self.assertIn("all selected tool results: 8,200 chars (~2.0K tokens estimated", normalized)
        self.assertIn("result characters in mapped calls: 8,100; from calls without a source/call match: 100", normalized)
        self.assertIn("1. exec 390 calls", normalized)
        self.assertIn("1. process poll", normalized)
        self.assertIn("result text 8,000 chars", normalized)
        self.assertIn("input processed 40.0M (39.2M cached / 800.0K uncached)", normalized)
        self.assertIn("nested rows are included in their family total", normalized)
        self.assertEqual(normalized.count("collaboration.send_message"), 1)
        self.assertIn("unlinked tool calls", normalized)
        self.assertNotIn("tool activity unlinked", normalized)
        self.assertIn("25 of 400 scoped responses lack model weights; 5 matched responses are unweighted", normalized)
        self.assertIn("3 tool-call batches had no unique usage record", normalized)
        self.assertIn("390 calls reference only process polls", normalized)
        self.assertNotIn("\033[", output)


if __name__ == "__main__":
    unittest.main()
