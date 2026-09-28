"""Shake reports separate persisted transform estimates from request evidence."""

import json
import sqlite3
import time
import unittest

from nenpi.shake_report import estimate_payback
from datetime import datetime, timezone

from tests.test_drain import (
    Harness, codex_session_meta_line, codex_subagent_meta_line,
    codex_task_started_line, codex_turn_context_line,
    codex_usage_record_line,
)


def shake_line(epoch, kind="manual", replacement="SENSITIVE SHAKE HISTORY"):
    suffix = {
        "manual": "",
        "automatic": " (automatic)",
        "automatic_cold_resume": " (automatic, cold resume)",
        "automatic_escalated": " (automatic, escalated)",
    }[kind]
    timestamp = datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")
    return json.dumps({
        "type": "compacted",
        "timestamp": timestamp,
        "payload": {
            "message": "[shake] context reduced surgically" + suffix,
            "replacement_history": [{"role": "assistant", "content": replacement}],
        },
        "window_id": "shake-window",
    })


class ShakeReport(Harness):
    session = "d5000000-1111-2222-3333-444444444444"

    def analyze(self, session=None, prompt=None):
        args = [
            "auto", "--harness", "codex", "--since", "7d", "--json",
            "--codex-root", str(self.home / ".codex"),
        ]
        if session:
            args.extend(["--session", session])
        if prompt is not None:
            args.extend(["--prompt", str(prompt)])
        return self.run_json(*args)["shake"]

    def logs_db(self, rows):
        path = self.home / ".codex" / "logs_2.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE logs (id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, "
                "ts_nanos INTEGER NOT NULL, level TEXT NOT NULL, target TEXT NOT NULL, "
                "feedback_log_body TEXT, module_path TEXT, file TEXT, line INTEGER, "
                "thread_id TEXT, process_uuid TEXT, estimated_bytes INTEGER)"
            )
            connection.executemany(
                "INSERT INTO logs (ts, ts_nanos, level, target, feedback_log_body, thread_id) "
                "VALUES (?, ?, ?, ?, ?, ?)", rows,
            )
            connection.commit()
        finally:
            connection.close()
        return path

    @staticmethod
    def shake_log(epoch, thread, trigger="manual"):
        whole = int(epoch)
        nanos = int((epoch - whole) * 1_000_000_000)
        message = (
            'session_loop{thread.id=%s}:turn{model=gpt-5-codex}: '
            '[shake] context reduced surgically trigger="%s" mode="elide" '
            "tool_outputs_elided=2 blocks_elided=1 images_dropped=0 "
            "thinking_dropped=0 tokens_freed=32000"
        ) % (thread, trigger)
        return (whole, nanos, "INFO", "codex_core::shake", message, thread)

    def test_applied_marker_reports_context_and_cache_separately(self):
        now = time.time() - 3600
        sid = self.session
        marker_time = now + 3.5
        path = self.write_codex("rollout-shake.jsonl", [
            codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, sid, input_tokens=100_000,
                                    cached_input_tokens=90_000, output_tokens=80),
            shake_line(marker_time, replacement="PRIVATE_REPLACEMENT_CONTENT"),
            codex_usage_record_line(now + 4, sid, input_tokens=60_000,
                                    cached_input_tokens=0, output_tokens=70),
            codex_usage_record_line(now + 5, sid, input_tokens=55_000,
                                    cached_input_tokens=40_000, output_tokens=60),
        ], day=now)
        db = self.logs_db([self.shake_log(marker_time, sid)])
        mtime_before = db.stat().st_mtime_ns

        report = self.analyze(sid, 1)

        self.assertEqual(len(report["events"]), 1)
        event = report["events"][0]
        self.assertEqual(event["status"], "applied")
        self.assertEqual(event["kind"], "manual")
        self.assertEqual(event["before_context_tokens"], 100_000)
        self.assertEqual(event["after_context_tokens"], 60_000)
        self.assertEqual(event["reduction_tokens"], 40_000)
        self.assertEqual(event["before_cached_input_tokens"], 90_000)
        self.assertEqual(event["after_cached_input_tokens"], 0)
        self.assertEqual(event["before_uncached_input_tokens"], 10_000)
        self.assertEqual(event["after_uncached_input_tokens"], 60_000)
        self.assertFalse(event["cross_prompt"])
        self.assertTrue(event["weighted_comparable"])
        self.assertEqual(event["subsequent_response_count"], 2)
        self.assertEqual(event["subsequent_input_tokens"], 115_000)
        self.assertEqual(event["payback"]["status"], "estimated")
        self.assertEqual(event["payback"]["read_tokens_saved"], 80_000)
        self.assertEqual(event["stats"]["tokens_freed_estimate"], 32_000)
        self.assertEqual(event["stats"]["tool_outputs_elided"], 2)
        self.assertEqual(report["coverage"]["stats_records_matched"], 1)
        self.assertEqual(db.stat().st_mtime_ns, mtime_before)
        self.assertNotIn("PRIVATE_REPLACEMENT_CONTENT", json.dumps(report))
        self.assertNotIn("SENSITIVE SHAKE HISTORY", json.dumps(report))
        self.assertIn("not guaranteed quota savings", " ".join(report["notes"]))
        self.assertEqual(path.exists(), True)

    def test_marker_without_stats_or_followup_is_unknown_not_no_run(self):
        now = time.time() - 3600
        sid = self.session
        self.write_codex("rollout-shake-last.jsonl", [
            codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, sid, input_tokens=80_000,
                                    cached_input_tokens=75_000, output_tokens=50),
            shake_line(now + 3, "automatic_cold_resume"),
        ], day=now)

        report = self.analyze(sid, 1)

        self.assertEqual(len(report["events"]), 1)
        event = report["events"][0]
        self.assertEqual(event["kind"], "automatic_cold_resume")
        self.assertEqual(event["status"], "applied")
        self.assertIsNone(event["after_context_tokens"])
        self.assertEqual(event["payback"]["status"], "missing_or_ambiguous_usage")
        self.assertIsNone(event["reduction_tokens"])
        self.assertIsNone(event["stats"])
        self.assertEqual(event["subsequent_response_count"], 0)
        self.assertEqual(report["coverage"]["markers_without_stats"], 1)
        self.assertEqual(report["coverage"]["no_op_status"], "unknown_without_a_persisted_skip/no-op record")
        self.assertIn("no-op", report["notes"][0])

    def test_cross_prompt_same_model_is_not_weight_comparable(self):
        now = time.time() - 3600
        sid = self.session
        self.write_codex("rollout-shake-boundary.jsonl", [
            codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, sid, input_tokens=100_000,
                                    cached_input_tokens=90_000, output_tokens=60,
                                    turn_id="turn-before"),
            shake_line(now + 3),
            codex_task_started_line(now + 4),
            codex_usage_record_line(now + 5, sid, input_tokens=70_000,
                                    cached_input_tokens=0, output_tokens=70,
                                    turn_id="turn-after"),
        ], day=now)

        report = self.analyze(sid, 2)

        self.assertEqual(len(report["events"]), 1)
        event = report["events"][0]
        self.assertTrue(event["cross_prompt"])
        self.assertEqual(event["payback"]["status"], "prompt_changed")
        self.assertFalse(event["weighted_comparable"])
        self.assertIsNone(event["weighted_delta_units"])
        self.assertEqual(event["before_context_tokens"], 100_000)
        self.assertEqual(event["after_context_tokens"], 70_000)

    def test_root_and_child_markers_stay_on_their_threads(self):
        now = time.time() - 3600
        child = "e5000000-1111-2222-3333-444444444444"
        root_lines = [
            codex_session_meta_line(now, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, self.session, input_tokens=100_000,
                                    cached_input_tokens=90_000, output_tokens=50),
            shake_line(now + 3),
            codex_usage_record_line(now + 4, self.session, input_tokens=70_000,
                                    cached_input_tokens=0, output_tokens=50),
        ]
        child_lines = [
            codex_subagent_meta_line(now + 1, child, self.session, "/project"),
            codex_turn_context_line(now + 2, "gpt-5-codex"),
            codex_task_started_line(now + 2),
            codex_usage_record_line(now + 3.1, child, input_tokens=200_000,
                                    cached_input_tokens=195_000, output_tokens=20),
            shake_line(now + 3.5, "automatic"),
            codex_usage_record_line(now + 4.2, child, input_tokens=150_000,
                                    cached_input_tokens=0, output_tokens=20),
        ]
        self.write_codex("rollout-root-shake.jsonl", root_lines, day=now)
        self.write_codex("rollout-child-shake.jsonl", child_lines, day=now)

        report = self.analyze(self.session)

        self.assertEqual(len(report["events"]), 2)
        root_event, child_event = sorted(report["events"], key=lambda event: event["thread_id"])
        by_thread = {event["thread_id"]: event for event in report["events"]}
        self.assertEqual(by_thread[self.session]["reduction_tokens"], 30_000)
        self.assertEqual(by_thread[child]["reduction_tokens"], 50_000)
        self.assertEqual(by_thread[child]["role"], "subagent")
        self.assertEqual(by_thread[child]["payback"]["status"], "unknown_weights")
        self.assertEqual(root_event["session_id"], child_event["session_id"])

    def test_agent_name_survives_inherited_root_metadata_and_warm_cache(self):
        now = time.time() - 3600
        child = "e5000000-1111-2222-3333-444444444444"
        meta = json.loads(codex_subagent_meta_line(now, child, self.session, "/project"))
        meta["payload"]["agent_nickname"] = "Ampere"
        self.write_codex("rollout-named-child.jsonl", [
            json.dumps(meta),
            codex_session_meta_line(now + 0.5, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, child, input_tokens=100000,
                                    cached_input_tokens=90000, output_tokens=20),
            shake_line(now + 3),
            codex_usage_record_line(now + 4, child, input_tokens=50000,
                                    cached_input_tokens=1000, output_tokens=20),
        ], day=now)
        for _ in range(2):
            event = self.analyze()["events"][0]
            self.assertEqual(event["agent_name"], "Ampere")
            self.assertEqual(event["thread_id"], child)

    def test_cold_resume_does_not_charge_expired_cache_to_shake(self):
        now = time.time() - 7200
        sid = self.session
        self.write_codex("rollout-cold-resume.jsonl", [
            codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, sid, input_tokens=100444,
                                    cached_input_tokens=99968, output_tokens=20),
            shake_line(now + 4322, "automatic_cold_resume"),
            codex_usage_record_line(now + 4323, sid, input_tokens=72014,
                                    cached_input_tokens=17024, output_tokens=20),
        ], day=now)
        event = self.analyze(sid, 1)["events"][0]
        self.assertEqual(event["payback"]["cache_baseline"], "presumed_expired")
        self.assertEqual(event["payback"]["break_even_calls"], 1)
        self.assertEqual(event["payback"]["cache_rebuild_premium_units"], 0)
        self.assertEqual(event["payback"]["first_call_uncached_tokens_avoided"], 28430)
        self.assertEqual(event["after_uncached_input_tokens"], 54990)
        self.assertAlmostEqual(event["idle_before_shake_seconds"], 4320, delta=.002)

    def test_model_change_after_shake_withholds_payback(self):
        now = time.time() - 3600
        sid = self.session
        self.write_codex("rollout-model-change.jsonl", [
            codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, sid, input_tokens=100000,
                                    cached_input_tokens=90000, output_tokens=20),
            shake_line(now + 3),
            codex_usage_record_line(now + 4, sid, input_tokens=50000,
                                    cached_input_tokens=1000, output_tokens=20),
            codex_turn_context_line(now + 5, "gpt-6-astra"),
            codex_usage_record_line(now + 6, sid, input_tokens=51000,
                                    cached_input_tokens=50000, output_tokens=20),
        ], day=now)
        event = self.analyze(sid, 1)["events"][0]
        self.assertEqual(event["payback"]["status"], "model_changed")

    def test_absent_marker_does_not_claim_shake_never_ran(self):
        now = time.time() - 3600
        self.write_codex("rollout-no-shake-marker.jsonl", [
            codex_session_meta_line(now, self.session, "/project"),
            codex_turn_context_line(now + 1, "gpt-5-codex"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, self.session, input_tokens=50_000,
                                    cached_input_tokens=45_000, output_tokens=30),
        ], day=now)

        report = self.analyze(self.session, 1)

        self.assertEqual(report["events"], [])
        self.assertEqual(report["coverage"]["observed_run_status"], "no_applied_marker_observed")
        self.assertIn("never-run states are not distinguishable", report["notes"][0])

    def test_one_stats_row_is_not_guessed_between_two_nearby_markers(self):
        now = time.time() - 3600
        sid = self.session
        first, second = now + 3, now + 8
        self.write_codex("rollout-close-shakes.jsonl", [
            codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, sid, input_tokens=100_000,
                                    cached_input_tokens=90_000, output_tokens=50),
            shake_line(first),
            shake_line(second),
            codex_usage_record_line(now + 9, sid, input_tokens=70_000,
                                    cached_input_tokens=0, output_tokens=50),
        ], day=now)
        self.logs_db([self.shake_log(second, sid)])

        report = self.analyze(sid, 1)

        self.assertEqual(len(report["events"]), 2)
        self.assertTrue(all(event["stats"] is None for event in report["events"]))
        self.assertTrue(all(event["intervening_shake_count"] == 2 for event in report["events"]))
        self.assertTrue(all(event["effect"] == "multiple_shakes_between_requests"
                            for event in report["events"]))
        self.assertTrue(all(event["reduction_tokens"] is None for event in report["events"]))
        self.assertTrue(all(not event["weighted_comparable"] for event in report["events"]))

    def test_distinct_markers_at_the_same_timestamp_have_a_joint_effect(self):
        now = time.time() - 3600
        sid = self.session
        self.write_codex("rollout-same-time-shakes.jsonl", [
            codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, sid, input_tokens=100_000,
                                    cached_input_tokens=90_000, output_tokens=50),
            shake_line(now + 3, "manual"),
            shake_line(now + 3, "automatic"),
            codex_usage_record_line(now + 4, sid, input_tokens=70_000,
                                    cached_input_tokens=0, output_tokens=50),
        ], day=now)
        report = self.analyze(sid, 1)
        self.assertEqual(len(report["events"]), 2)
        self.assertTrue(all(event["intervening_shake_count"] == 2 for event in report["events"]))
        self.assertTrue(all(event["effect"] == "multiple_shakes_between_requests" for event in report["events"]))
        self.assertTrue(all(event["reduction_tokens"] is None for event in report["events"]))

    def test_nested_replacement_message_cannot_impersonate_top_level_marker(self):
        now = time.time() - 3600
        sid = self.session
        timestamp = datetime.fromtimestamp(now + 3, timezone.utc).isoformat().replace("+00:00", "Z")
        nested_only = json.dumps({
            "type": "compacted",
            "timestamp": timestamp,
            "payload": {
                "replacement_history": [{"message": "[shake] context reduced surgically"}],
            },
        })
        self.write_codex("rollout-nested-marker.jsonl", [
            codex_session_meta_line(now, sid, "/project"),
            codex_turn_context_line(now + 1, "gpt-5.6-luna"),
            codex_task_started_line(now + 1),
            codex_usage_record_line(now + 2, sid, input_tokens=80_000,
                                    cached_input_tokens=70_000, output_tokens=40),
            nested_only,
        ], day=now)

        report = self.analyze(sid, 1)

        self.assertEqual(report["events"], [])


class PaybackTests(unittest.TestCase):
    def test_rebuild_premium_and_first_call_savings(self):
        result = estimate_payback(190201, 85754 - 1523, 196, 10, 1)
        self.assertEqual(result["read_tokens_saved"], 37279396)
        self.assertAlmostEqual(result["net_read_equivalent_tokens_saved"], 36521317)
        self.assertEqual(result["break_even_calls"], 4)
        self.assertAlmostEqual(result["cache_rebuild_premium_units"], 0.758079)
        self.assertAlmostEqual(result["net_input_units_saved"], 36.521317)
        self.assertTrue(result["break_even_reached"])

    def test_cold_resume_scores_avoided_uncached_first_read(self):
        result = estimate_payback(28430, 54514, 5, 100, 10, cold_resume=True)
        self.assertEqual(result["break_even_calls"], 1)
        self.assertEqual(result["cache_rebuild_premium_units"], 0)
        self.assertEqual(result["read_tokens_saved"], 142150)
        self.assertAlmostEqual(result["net_read_equivalent_tokens_saved"], 398020)
        self.assertAlmostEqual(result["net_input_units_saved"], 3.9802)

    def test_no_rebuild_and_pending_payback(self):
        self.assertEqual(estimate_payback(50000, -100, 20, 10, 1)["break_even_calls"], 1)
        pending = estimate_payback(50000, 50000, 2, 10, 1)
        self.assertEqual(pending["break_even_calls"], 9)
        self.assertFalse(pending["break_even_reached"])
        self.assertLess(pending["net_input_units_saved"], 0)
        self.assertEqual(estimate_payback(50000, 50000, 9, 10, 1)["net_input_units_saved"], 0)

    def test_unsupported_or_absent_benefit(self):
        self.assertEqual(estimate_payback(0, 100, 2, 10, 1)["status"], "no_reduction")
        self.assertEqual(estimate_payback(100, 100, 2, 10, 0)["status"], "unsupported_weights")
