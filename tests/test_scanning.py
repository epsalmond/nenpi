"""Contracts for the shared synchronous scanner API."""

from __future__ import annotations

import json
import multiprocessing
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

import nenpi.drain as QD
import nenpi.scanning as scanning
from tests.test_drain import Harness, claude_assistant_line, claude_user_prompt_line
from nenpi.scanning import (
    CancellationToken,
    ScanCancelled,
    ScanStatus,
    serialized_cache,
)


def write_cache_shard(root: str, name: str) -> None:
    path = Path(root) / name
    cache = QD.Cache(Path(root) / "cache", False)
    entry = QD.FileIndex("claude")
    entry.sessions[name] = QD.SessionSummary("claude", name)
    cache.entries[str(path)] = entry
    cache.mark(path)
    cache.flush()


def hold_cache_lock(root: str, ready: Any, release: Any) -> None:
    with serialized_cache(Path(root)):
        ready.set()
        release.wait(2)


class ScannerApi(Harness):
    def test_prepare_reports_phases_and_counts(self) -> None:
        now = 1_700_000_000.0
        self.write_claude(
            "one.jsonl",
            [claude_assistant_line(now, "session-one", "message-one", output_tokens=3)],
        )
        statuses: List[ScanStatus] = []
        args = QD.build_parser().parse_args(["sessions", "--harness", "claude", "--json"])
        with self.env_applied():
            analysis = QD.prepare(args, progress=statuses.append)

        self.assertEqual(len(analysis.scan.events["claude"]), 1)
        self.assertEqual(
            [status.phase for status in statuses],
            ["discovery", "discovery", "scanning", "scanning", "scanning", "analysis", "done"],
        )
        self.assertEqual(statuses[-1].files_seen, 1)
        self.assertEqual(statuses[-1].files_parsed, 1)
        self.assertEqual(statuses[-1].cache_misses, 1)
        self.assertGreaterEqual(statuses[-1].elapsed, 0.0)

    def test_cancel_token_stops_before_scan(self) -> None:
        token = CancellationToken()
        token.cancel()
        args = QD.build_parser().parse_args(["sessions", "--harness", "claude"])
        with self.env_applied(), self.assertRaises(ScanCancelled):
            QD.prepare(args, cancellation=token)

    def test_cancel_callable_stops_inside_a_large_file(self) -> None:
        now = 1_700_000_000.0
        lines = [
            claude_assistant_line(now + index, "long-session", "message-%d" % index)
            for index in range(5000)
        ]
        self.write_claude("long.jsonl", lines)
        calls = [0]

        def should_cancel() -> bool:
            calls[0] += 1
            return calls[0] > 100

        args = QD.build_parser().parse_args(["sessions", "--harness", "claude"])
        with self.env_applied(), self.assertRaises(ScanCancelled):
            QD.prepare(args, cancellation=should_cancel)
        self.assertGreater(calls[0], 100)

    def test_json_mode_keeps_result_stdout_clean(self) -> None:
        now = 1_700_000_000.0
        self.write_claude(
            "json.jsonl",
            [claude_assistant_line(now, "json-session", "json-message", output_tokens=1)],
        )
        result = self.run_tool("sessions", "--harness", "claude", "--json")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        payload = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(payload["command"], "sessions")
        events = [json.loads(line) for line in result.stderr.decode().splitlines() if line]
        self.assertTrue(events)
        self.assertEqual(events[-1]["phase"], "done")
        self.assertEqual({event["phase"] for event in events}, {"discovery", "scanning", "analysis", "done"})

    def test_scan_sources_uses_exact_enabled_roots(self) -> None:
        now = 1_700_000_000.0
        self.write_claude(
            "enabled.jsonl",
            [claude_assistant_line(now, "enabled-session", "enabled-message")],
            project="enabled",
        )
        self.write_claude(
            "ignored.jsonl",
            [claude_assistant_line(now, "ignored-session", "ignored-message")],
            project="ignored",
        )
        progress: List[Any] = []
        with self.env_applied():
            result = scanning.scan_sources(
                [
                    ("claude", str(self.claude_projects / "enabled")),
                    ("claude", str(self.claude_projects / "enabled")),
                ],
                progress=lambda done, total: progress.append((done, total)),
            )
        self.assertEqual([row["session_id"] for row in result["sessions"]], ["enabled-session"])
        self.assertEqual(result["files_seen"], 1)
        self.assertEqual(progress[-1], (1, 1))

    def test_scan_sources_returns_prompt_metrics_and_reductions_without_content(self) -> None:
        now = 1_700_000_000.0
        session = "detail-session"
        lines = [
            claude_user_prompt_line(
                now, session, text="the short label\nPRIVATE PROMPT CONTENT"
            )
        ]
        for index, context in enumerate([100_000, 200_000, 300_000, 40_000, 45_000, 50_000, 55_000]):
            lines.append(
                claude_assistant_line(
                    now + index + 1,
                    session,
                    "detail-%d" % index,
                    cache_read=context,
                    output_tokens=100,
                )
            )
        self.write_claude("detail.jsonl", lines)
        statuses: List[ScanStatus] = []
        with self.env_applied():
            result = scanning.scan_sources(
                [("claude", str(self.claude_projects))], status_callback=statuses.append
            )

        self.assertEqual(len(result["prompts"]), 1)
        prompt = result["prompts"][0]
        self.assertGreater(prompt["turns"], 0)
        self.assertGreater(prompt["input_tokens"], 0)
        self.assertEqual(
            prompt["context_growth"], prompt["context_peak"] - prompt["context_start"]
        )
        self.assertEqual(prompt["session_id"], session)
        self.assertEqual(len(result["reductions"]), 1)
        reduction = result["reductions"][0]
        self.assertEqual(reduction["session_id"], session)
        self.assertEqual(reduction["kind"], "unmarked")
        self.assertEqual(result["sessions"][0]["reductions"], [reduction])
        self.assertEqual(result["sessions"][0]["prompt_details"], [prompt])
        # The redacted one-line label is deliberately exposed (#8, #26); the
        # rest of the prompt never leaves the transcript.
        self.assertEqual(prompt["label"], "the short label")
        self.assertNotIn("PRIVATE PROMPT CONTENT", json.dumps(result))
        self.assertEqual(statuses[-1].phase, "done")

    def test_scan_sources_propagates_cancellation(self) -> None:
        args = [("claude", str(self.claude_projects))]
        with self.env_applied(), self.assertRaises(ScanCancelled):
            scanning.scan_sources(args, cancelled=lambda: True)


class CacheSerialization(unittest.TestCase):
    def test_waiting_for_cache_lock_is_cancellable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("fork")
            ready = context.Event()
            release = context.Event()
            process = context.Process(
                target=hold_cache_lock, args=(str(Path(directory) / "cache"), ready, release)
            )
            process.start()
            self.assertTrue(ready.wait(2))
            token = CancellationToken()
            errors = []

            def wait_for_lock() -> None:
                try:
                    with serialized_cache(Path(directory) / "cache", token):
                        pass
                except ScanCancelled:
                    errors.append(True)

            thread = threading.Thread(target=wait_for_lock)
            thread.start()
            time.sleep(0.1)
            token.cancel()
            thread.join(2)
            release.set()
            process.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [True])
            self.assertEqual(process.exitcode, 0)

    def test_concurrent_flushes_leave_valid_shards_and_no_shared_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("fork")
            names = ["same-file", "same-file", "other-file", "third-file"]
            processes = [
                context.Process(target=write_cache_shard, args=(directory, name))
                for name in names
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            shards = list((Path(directory) / "cache" / ("v%d" % QD.CACHE_SCHEMA)).rglob("*.json"))
            self.assertEqual(len(shards), 3)
            for shard in shards:
                json.loads(shard.read_text(encoding="utf-8"))
            self.assertEqual(list((Path(directory) / "cache").rglob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
