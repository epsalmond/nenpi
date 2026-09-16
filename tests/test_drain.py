#!/usr/bin/env python3
"""Contract tests for quota-drain.

Every fixture here is synthetic. Real transcripts hold prompts and customer
data and must never be copied into this repository.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
QUOTA_DRAIN = SCRIPT_DIR / "quota-drain"
LOADER = importlib.machinery.SourceFileLoader("quota_drain_contract", str(QUOTA_DRAIN))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC and SPEC.loader
QD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = QD
SPEC.loader.exec_module(QD)


def iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


def claude_assistant_line(
    epoch: float,
    session_id: str,
    message_id: str,
    *,
    model: str = "claude-opus-5",
    cwd: str = "/home/agent/project",
    input_tokens: int = 0,
    cache_read: int = 0,
    cache_write_5m: int = 0,
    cache_write_1h: int = 0,
    output_tokens: int = 0,
    sidechain: bool = False,
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "sessionId": session_id,
            "cwd": cwd,
            "version": "2.1.273",
            "timestamp": iso(epoch),
            "requestId": "req_" + message_id,
            "isSidechain": sidechain,
            "message": {
                "id": message_id,
                "role": "assistant",
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": cache_write_5m + cache_write_1h,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": cache_write_5m,
                        "ephemeral_1h_input_tokens": cache_write_1h,
                    },
                    "output_tokens": output_tokens,
                    "output_tokens_details": {"thinking_tokens": 0},
                    "iterations": [{"input_tokens": input_tokens, "output_tokens": output_tokens}],
                },
            },
        }
    )


def claude_cost_state_line(session_id: str, model: str, usage: Dict[str, int]) -> str:
    return json.dumps(
        {
            "type": "cost-state",
            "sessionId": session_id,
            "totalCostUSD": 1.25,
            "totalDuration": 1000,
            "totalAPIDuration": 500,
            "modelUsage": {model: usage},
        }
    )


def codex_session_meta_line(epoch: float, session_id: str, cwd: str) -> str:
    return json.dumps(
        {
            "type": "session_meta",
            "timestamp": iso(epoch),
            "payload": {
                "id": session_id,
                "session_id": session_id,
                "timestamp": iso(epoch),
                "cwd": cwd,
                "originator": "codex-tui",
                "cli_version": "0.51.0",
                "model_provider": "openai",
                "source": "cli",
            },
        }
    )


def codex_turn_context_line(epoch: float, model: str) -> str:
    return json.dumps(
        {"type": "turn_context", "timestamp": iso(epoch), "payload": {"model": model}}
    )


def codex_usage_record_line(
    epoch: float,
    session_id: str,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    cache_write_input_tokens: int = 0,
    thread_total: Optional[Dict[str, int]] = None,
    turn_id: Optional[str] = None,
) -> str:
    usage = {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": input_tokens + output_tokens,
    }
    payload = {
        "thread_id": session_id,
        "session_id": session_id,
        "response_id": "resp-%f" % epoch,
        "usage": usage,
        "turn_token_usage": usage,
        "thread_token_usage": thread_total or usage,
    }
    if turn_id is not None:
        payload["turn_id"] = turn_id
        payload["root_turn_id"] = turn_id
    return json.dumps({"type": "token_usage_record", "timestamp": iso(epoch),
                       "payload": payload})


def claude_user_prompt_line(
    epoch: float,
    session_id: str,
    *,
    cwd: str = "/home/agent/project",
    text: str = "do the thing",
) -> str:
    return json.dumps(
        {
            "type": "user",
            "sessionId": session_id,
            "cwd": cwd,
            "timestamp": iso(epoch),
            "isSidechain": False,
            "message": {"role": "user", "content": text},
        }
    )


def claude_tool_result_line(epoch: float, session_id: str) -> str:
    """A fan-out step, not a prompt: same `type: user`, different shape."""
    return json.dumps(
        {
            "type": "user",
            "sessionId": session_id,
            "timestamp": iso(epoch),
            "isSidechain": False,
            "toolUseResult": {"stdout": "ok"},
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        }
    )


def codex_task_started_line(epoch: float) -> str:
    return json.dumps(
        {"type": "event_msg", "timestamp": iso(epoch), "payload": {"type": "task_started"}}
    )


def codex_compacted_line(epoch: float, window_number: int = 2) -> str:
    """Top-level fields, no payload wrapper; history fields are never read."""
    return json.dumps(
        {
            "type": "compacted",
            "timestamp": iso(epoch),
            "window_number": window_number,
            "window_id": "win-%d" % window_number,
            "previous_window_id": "win-%d" % (window_number - 1),
            "compaction_response_id": "resp-compact",
            "message": "SENSITIVE PROMPT TEXT",
            "replacement_history": ["SENSITIVE"],
            "guardian_history": ["SENSITIVE"],
            "retained_context": "SENSITIVE",
        }
    )


def codex_subagent_meta_line(epoch: float, thread_id: str, session_id: str, cwd: str) -> str:
    return json.dumps(
        {
            "type": "session_meta",
            "timestamp": iso(epoch),
            "payload": {
                "id": thread_id,
                "session_id": session_id,
                "parent_thread_id": session_id,
                "timestamp": iso(epoch),
                "cwd": cwd,
                "originator": "codex-tui",
                "cli_version": "0.51.0",
                "source": {"subagent": {"thread_spawn": {"parent_thread_id": session_id,
                                                         "depth": 1}}},
            },
        }
    )


def codex_token_count_line(
    epoch: float,
    *,
    total: Optional[Dict[str, int]] = None,
    rate_limits: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {"type": "token_count", "info": None, "rate_limits": rate_limits}
    if total is not None:
        payload["info"] = {
            "total_token_usage": {
                "input_tokens": total.get("input_tokens", 0),
                "cached_input_tokens": total.get("cached_input_tokens", 0),
                "cache_write_input_tokens": 0,
                "output_tokens": total.get("output_tokens", 0),
                "reasoning_output_tokens": 0,
                "total_tokens": total.get("input_tokens", 0) + total.get("output_tokens", 0),
            },
            "last_token_usage": {},
            "model_context_window": 400000,
        }
    return json.dumps({"type": "event_msg", "timestamp": iso(epoch), "payload": payload})


def rate_limits(used_percent: float, resets_at: int, window_minutes: int = 300) -> Dict[str, Any]:
    return {
        "limit_id": "codex",
        "limit_name": None,
        "primary": {
            "used_percent": used_percent,
            "window_minutes": window_minutes,
            "resets_at": resets_at,
        },
        "secondary": None,
        "credits": {"has_credits": False, "unlimited": False, "balance": "0"},
        "plan_type": "pro",
        "rate_limit_reached_type": None,
        "spend_control_reached": None,
    }


class Harness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.home = self.root / "home"
        self.claude_projects = self.home / ".claude" / "projects"
        self.codex_sessions = self.home / ".codex" / "sessions"
        self.claude_projects.mkdir(parents=True)
        self.codex_sessions.mkdir(parents=True)
        self.environment = dict(os.environ)
        self.environment.update(
            {
                "QUOTA_DRAIN_HOME_DIR": str(self.home),
                "QUOTA_DRAIN_CACHE_DIR": str(self.root / "cache"),
                "QUOTA_DRAIN_STATE_DIR": str(self.root / "state"),
                "QUOTA_DRAIN_CONFIG_DIR": str(self.root / "config"),
                "TZ": "UTC",
            }
        )
        self.addCleanup(self.temp.cleanup)

    def write_claude(self, name: str, lines: Sequence[str], project: str = "proj") -> Path:
        path = self.claude_projects / project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
        return path

    def write_codex(self, name: str, lines: Sequence[str], day: Optional[float] = None) -> Path:
        stamp = datetime.fromtimestamp(day or time.time(), timezone.utc)
        path = self.codex_sessions / stamp.strftime("%Y/%m/%d") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
        return path

    def run_tool(self, *arguments: str, stdin: Optional[bytes] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(QUOTA_DRAIN)] + list(arguments),
            check=False,
            input=stdin,
            capture_output=True,
            env=self.environment,
            timeout=120,
        )

    def run_json(self, *arguments: str) -> Dict[str, Any]:
        result = self.run_tool(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        return json.loads(result.stdout.decode("utf-8"))

    def session_by_short_id(self, payload: Dict[str, Any], short: str) -> Dict[str, Any]:
        for row in payload["sessions"]:
            if row["short_id"] == short:
                return row
        raise AssertionError("session %s not in %s" % (short, [r["short_id"] for r in payload["sessions"]]))


class ClaudeParsing(Harness):
    def test_repeated_usage_lines_count_once(self) -> None:
        now = time.time() - 600
        session = "aaaaaaaa-1111-2222-3333-444444444444"
        line = claude_assistant_line(
            now, session, "msg_dup", input_tokens=100, cache_read=2000, output_tokens=50
        )
        # The CLI writes the same usage object once per streamed content block.
        self.write_claude("s1.jsonl", [line, line, line])
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        row = self.session_by_short_id(payload, "aaaaaaaa")
        self.assertEqual(row["requests"], 1)
        self.assertEqual(row["tokens"]["input"], 100)
        self.assertEqual(row["tokens"]["cache_read"], 2000)
        self.assertEqual(row["tokens"]["output"], 50)

    def test_distinct_messages_all_count(self) -> None:
        now = time.time() - 600
        session = "bbbbbbbb-1111-2222-3333-444444444444"
        self.write_claude(
            "s2.jsonl",
            [
                claude_assistant_line(now, session, "msg_a", output_tokens=10),
                claude_assistant_line(now + 1, session, "msg_b", output_tokens=20),
            ],
        )
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        row = self.session_by_short_id(payload, "bbbbbbbb")
        self.assertEqual(row["requests"], 2)
        self.assertEqual(row["tokens"]["output"], 30)

    def test_subagent_usage_rolls_up_into_parent(self) -> None:
        now = time.time() - 600
        session = "cccccccc-1111-2222-3333-444444444444"
        self.write_claude(
            "s3.jsonl", [claude_assistant_line(now, session, "msg_main", output_tokens=100)]
        )
        self.write_claude(
            "subagents/agent-77.jsonl",
            [
                claude_assistant_line(
                    now + 5, session, "msg_sub", output_tokens=400, sidechain=True
                )
            ],
        )
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        row = self.session_by_short_id(payload, "cccccccc")
        self.assertEqual(row["tokens"]["output"], 500)
        self.assertEqual(row["subagent_tokens"]["output"], 400)
        self.assertEqual(row["subagent_requests"], 1)
        self.assertGreater(row["subagent_weighted_units"], 0.0)
        self.assertLess(row["subagent_weighted_units"], row["weighted_units"])

    def test_dollar_equivalent_uses_list_prices(self) -> None:
        now = time.time() - 600
        session = "dddddddd-1111-2222-3333-444444444444"
        self.write_claude(
            "s4.jsonl",
            [
                claude_assistant_line(
                    now,
                    session,
                    "msg_price",
                    model="claude-opus-5",
                    input_tokens=1_000_000,
                    cache_read=1_000_000,
                    cache_write_5m=1_000_000,
                    cache_write_1h=1_000_000,
                    output_tokens=1_000_000,
                )
            ],
        )
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        row = self.session_by_short_id(payload, "dddddddd")
        self.assertAlmostEqual(row["weighted_units"], 5.0 + 0.5 + 6.25 + 10.0 + 25.0, places=6)

    def test_cache_read_weight_override(self) -> None:
        now = time.time() - 600
        session = "eeeeeeee-1111-2222-3333-444444444444"
        self.write_claude(
            "s5.jsonl",
            [claude_assistant_line(now, session, "msg_cr", cache_read=2_000_000)],
        )
        payload = self.run_json(
            "sessions", "--harness", "claude", "--json", "--claude-cache-read-weight", "1.0"
        )
        row = self.session_by_short_id(payload, "eeeeeeee")
        self.assertAlmostEqual(row["weighted_units"], 10.0, places=6)

    def test_unknown_model_is_unweighted(self) -> None:
        now = time.time() - 600
        session = "ffffffff-1111-2222-3333-444444444444"
        self.write_claude(
            "s6.jsonl",
            [
                claude_assistant_line(
                    now, session, "msg_syn", model="<synthetic>", output_tokens=1234
                )
            ],
        )
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        row = self.session_by_short_id(payload, "ffffffff")
        self.assertEqual(row["weighted_units"], 0.0)
        self.assertEqual(row["unweighted_tokens"], 1234)

    def test_dated_model_id_normalizes(self) -> None:
        self.assertEqual(
            QD.normalize_claude_model("claude-haiku-4-5-20251001"), "claude-haiku-4-5"
        )
        self.assertEqual(QD.normalize_claude_model("<synthetic>"), QD.UNWEIGHTED)

    def test_verify_reports_cost_state_delta(self) -> None:
        now = time.time() - 600
        session = "11111111-aaaa-2222-3333-444444444444"
        self.write_claude(
            "s7.jsonl",
            [
                claude_assistant_line(
                    now, session, "msg_v", input_tokens=100, output_tokens=200
                ),
                claude_cost_state_line(
                    session,
                    "claude-opus-5",
                    {
                        "inputTokens": 100,
                        "outputTokens": 200,
                        "cacheReadInputTokens": 0,
                        "cacheCreationInputTokens": 0,
                        "costUSD": 1.0,
                    },
                ),
            ],
        )
        payload = self.run_json("verify", "--harness", "claude", "--json")
        rows = [row for row in payload["rows"] if row["short_id"] == "11111111"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["deduped"]["input"], rows[0]["cost_state"]["input"])
        self.assertEqual(rows[0]["deduped"]["output"], rows[0]["cost_state"]["output"])


class CodexParsing(Harness):
    def test_per_turn_deltas_from_usage_records(self) -> None:
        now = time.time() - 1200
        session = "codex-aaa-1111"
        self.write_codex(
            "rollout-1-aaa.jsonl",
            [
                codex_session_meta_line(now, session, "/home/agent/repo"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 10,
                    session,
                    input_tokens=100_000,
                    cached_input_tokens=40_000,
                    output_tokens=5_000,
                ),
                codex_usage_record_line(
                    now + 20,
                    session,
                    input_tokens=50_000,
                    cached_input_tokens=10_000,
                    output_tokens=1_000,
                ),
            ],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        row = payload["sessions"][0]
        self.assertEqual(row["tokens"]["input"], (100_000 - 40_000) + (50_000 - 10_000))
        self.assertEqual(row["tokens"]["cached_input"], 50_000)
        self.assertEqual(row["tokens"]["output"], 6_000)
        self.assertEqual(row["primary_model"], "gpt-5.6-sol")
        # 0.1 Mtok uncached * 100 + 0.05 Mtok cached * 10 + 0.006 Mtok out * 500
        self.assertAlmostEqual(row["weighted_units"], 10.0 + 0.5 + 3.0, places=6)

    def test_cumulative_fallback_when_no_usage_records(self) -> None:
        now = time.time() - 1200
        session = "codex-bbb-2222"
        self.write_codex(
            "rollout-2-bbb.jsonl",
            [
                codex_session_meta_line(now, session, "/home/agent/legacy"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_token_count_line(
                    now + 10,
                    total={"input_tokens": 10_000, "cached_input_tokens": 0, "output_tokens": 500},
                ),
                codex_token_count_line(
                    now + 20,
                    total={"input_tokens": 30_000, "cached_input_tokens": 0, "output_tokens": 900},
                ),
            ],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        row = payload["sessions"][0]
        self.assertEqual(row["tokens"]["input"], 30_000)
        self.assertEqual(row["tokens"]["output"], 900)

    def test_usage_records_win_over_cumulative_counts(self) -> None:
        now = time.time() - 1200
        session = "codex-ccc-3333"
        self.write_codex(
            "rollout-3-ccc.jsonl",
            [
                codex_session_meta_line(now, session, "/home/agent/both"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 10,
                    session,
                    input_tokens=20_000,
                    cached_input_tokens=0,
                    output_tokens=700,
                ),
                codex_token_count_line(
                    now + 11,
                    total={"input_tokens": 20_000, "cached_input_tokens": 0, "output_tokens": 700},
                ),
            ],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        row = payload["sessions"][0]
        self.assertEqual(row["tokens"]["input"], 20_000)
        self.assertEqual(row["tokens"]["output"], 700)

    def test_cached_input_is_a_subset_of_input(self) -> None:
        tokens = QD.codex_usage_tokens(
            {
                "input_tokens": 1000,
                "cached_input_tokens": 400,
                "cache_write_input_tokens": 90,
                "output_tokens": 30,
                "reasoning_output_tokens": 20,
            }
        )
        self.assertEqual(tokens["input"], 600)
        self.assertEqual(tokens["cached_input"], 400)
        self.assertEqual(tokens["cache_write"], 90)
        # reasoning is already inside output_tokens, so it is not added again
        self.assertEqual(tokens["output"], 30)

    def test_model_aliases_normalize(self) -> None:
        self.assertEqual(QD.normalize_codex_model("Luna"), "gpt-5.6-luna")
        self.assertEqual(QD.normalize_codex_model("gpt-luna"), "gpt-5.6-luna")
        self.assertEqual(QD.normalize_codex_model("pingu-unchained-10"), "pingu-unchained-10")


class MeasuredAttribution(Harness):
    def write_two_concurrent_sessions(self, resets_at: int, second_resets_at: int) -> float:
        now = time.time() - 3600
        session_a = "codex-conc-aaaa"
        session_b = "codex-conc-bbbb"
        self.write_codex(
            "rollout-a.jsonl",
            [
                codex_session_meta_line(now, session_a, "/home/agent/alpha"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                # Opening reading; the interval that follows is what gets split.
                codex_token_count_line(now + 5, rate_limits=rate_limits(10.0, resets_at)),
                # Session A burns 3x what session B burns inside the interval.
                codex_usage_record_line(
                    now + 10, session_a, input_tokens=300_000, cached_input_tokens=0,
                    output_tokens=0
                ),
                codex_token_count_line(
                    now + 30, rate_limits=rate_limits(14.0, second_resets_at)
                ),
            ],
        )
        self.write_codex(
            "rollout-b.jsonl",
            [
                codex_session_meta_line(now, session_b, "/home/agent/beta"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 15, session_b, input_tokens=100_000, cached_input_tokens=0,
                    output_tokens=0
                ),
            ],
        )
        return now

    def test_drain_splits_by_weighted_tokens(self) -> None:
        resets_at = int(time.time()) + 7200
        self.write_two_concurrent_sessions(resets_at, resets_at)
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        rows = dict((row["cwd"].rsplit("/", 1)[-1], row) for row in payload["sessions"])
        self.assertAlmostEqual(rows["alpha"]["drain_percent"], 3.0, places=6)
        self.assertAlmostEqual(rows["beta"]["drain_percent"], 1.0, places=6)
        self.assertFalse(rows["alpha"]["drain_is_estimate"])
        self.assertAlmostEqual(rows["alpha"]["share_of_window_percent"], 75.0, places=6)

    def test_rollover_when_resets_at_changes(self) -> None:
        resets_at = int(time.time()) + 7200
        self.write_two_concurrent_sessions(resets_at, resets_at + 18000)
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        rows = dict((row["cwd"].rsplit("/", 1)[-1], row) for row in payload["sessions"])
        # A new window means the second reading is the whole drain so far, not a
        # difference, so the interval carries 14% rather than 4%.
        total = rows["alpha"]["drain_percent"] + rows["beta"]["drain_percent"]
        self.assertAlmostEqual(total, 14.0, places=6)
        self.assertAlmostEqual(rows["alpha"]["drain_percent"], 10.5, places=6)

    def test_windows_subcommand_lists_peak_and_top_sessions(self) -> None:
        resets_at = int(time.time()) + 7200
        self.write_two_concurrent_sessions(resets_at, resets_at)
        payload = self.run_json("windows", "--harness", "codex", "--json")
        self.assertEqual(len(payload["windows"]), 1)
        window = payload["windows"][0]
        self.assertEqual(window["window_minutes"], 300)
        self.assertAlmostEqual(window["peak_used_percent"], 14.0, places=6)
        self.assertEqual(len(window["top_sessions"]), 2)
        self.assertGreater(
            window["top_sessions"][0]["drain_percent"],
            window["top_sessions"][1]["drain_percent"],
        )


class Calibration(Harness):
    TRUE_WEIGHTS = {
        ("gpt-5.6-sol", "input"): 9.0,
        ("gpt-5.6-sol", "cached_input"): 1.0,
        ("gpt-5.6-sol", "output"): 40.0,
        ("gpt-5.6-terra", "input"): 4.0,
        ("gpt-5.6-terra", "cached_input"): 0.5,
        ("gpt-5.6-terra", "output"): 20.0,
    }

    def build_rollouts(self) -> None:
        now = time.time() - 6 * 3600
        resets_at = int(time.time()) + 7200
        used = 0.0
        lines_by_model = {"gpt-5.6-sol": [], "gpt-5.6-terra": []}  # type: Dict[str, List[str]]
        sessions = {
            "gpt-5.6-sol": "codex-fit-sol-0001",
            "gpt-5.6-terra": "codex-fit-ter-0001",
        }
        for model, session in sessions.items():
            lines_by_model[model].append(codex_session_meta_line(now, session, "/fit/" + model))
            lines_by_model[model].append(codex_turn_context_line(now, model))
        snapshot_lines = []  # type: List[str]
        # A varied token mix per interval keeps the six columns identifiable.
        mixes = [
            (900_000, 100_000, 20_000, 200_000, 50_000, 5_000),
            (200_000, 800_000, 5_000, 900_000, 100_000, 40_000),
            (500_000, 500_000, 50_000, 100_000, 900_000, 10_000),
            (100_000, 200_000, 40_000, 700_000, 200_000, 30_000),
            (800_000, 300_000, 10_000, 300_000, 600_000, 20_000),
            (400_000, 900_000, 30_000, 500_000, 400_000, 50_000),
            (700_000, 400_000, 60_000, 200_000, 700_000, 15_000),
            (300_000, 600_000, 15_000, 800_000, 300_000, 45_000),
            (600_000, 200_000, 35_000, 400_000, 500_000, 25_000),
            (250_000, 750_000, 25_000, 600_000, 150_000, 35_000),
            (950_000, 150_000, 45_000, 100_000, 800_000, 5_000),
            (150_000, 950_000, 55_000, 900_000, 250_000, 55_000),
        ]
        stamp = now
        snapshot_lines.append(codex_token_count_line(stamp, rate_limits=rate_limits(0.0, resets_at)))
        for index, mix in enumerate(mixes):
            sol_uncached, sol_cached, sol_out, ter_uncached, ter_cached, ter_out = mix
            stamp += 60
            lines_by_model["gpt-5.6-sol"].append(
                codex_usage_record_line(
                    stamp,
                    sessions["gpt-5.6-sol"],
                    input_tokens=sol_uncached + sol_cached,
                    cached_input_tokens=sol_cached,
                    output_tokens=sol_out,
                )
            )
            lines_by_model["gpt-5.6-terra"].append(
                codex_usage_record_line(
                    stamp + 1,
                    sessions["gpt-5.6-terra"],
                    input_tokens=ter_uncached + ter_cached,
                    cached_input_tokens=ter_cached,
                    output_tokens=ter_out,
                )
            )
            drain = (
                sol_uncached * self.TRUE_WEIGHTS[("gpt-5.6-sol", "input")]
                + sol_cached * self.TRUE_WEIGHTS[("gpt-5.6-sol", "cached_input")]
                + sol_out * self.TRUE_WEIGHTS[("gpt-5.6-sol", "output")]
                + ter_uncached * self.TRUE_WEIGHTS[("gpt-5.6-terra", "input")]
                + ter_cached * self.TRUE_WEIGHTS[("gpt-5.6-terra", "cached_input")]
                + ter_out * self.TRUE_WEIGHTS[("gpt-5.6-terra", "output")]
            ) / 1_000_000.0
            used += drain
            stamp += 60
            snapshot_lines.append(
                codex_token_count_line(stamp, rate_limits=rate_limits(used, resets_at))
            )
        for model, lines in lines_by_model.items():
            self.write_codex("rollout-fit-%s.jsonl" % model.replace(".", "-"), lines)
        self.write_codex("rollout-fit-snapshots.jsonl", snapshot_lines)

    def test_calibrate_recovers_known_weights(self) -> None:
        self.build_rollouts()
        # One bucket per interval: this exercises the fit itself, not the
        # time bucketing that real whole-percent data needs.
        payload = self.run_json(
            "calibrate", "--harness", "codex", "--json", "--calibrate-bucket-hours", "0.01"
        )
        self.assertGreater(payload["samples"], 6)
        self.assertGreater(payload["r_squared"], 0.99)
        fitted = payload["codex"]["models"]
        for (model, kind), expected in self.TRUE_WEIGHTS.items():
            actual = fitted[model][kind]
            self.assertLess(
                abs(actual - expected) / expected,
                0.05,
                "%s/%s fitted %.4f, expected %.4f" % (model, kind, actual, expected),
            )

    def test_calibrate_persists_and_use_calibrated_reads_it(self) -> None:
        self.build_rollouts()
        self.run_json(
            "calibrate", "--harness", "codex", "--json", "--calibrate-bucket-hours", "0.01"
        )
        fit_file = self.root / "state" / "codex-weights.json"
        self.assertTrue(fit_file.is_file())
        stored = json.loads(fit_file.read_text(encoding="utf-8"))
        self.assertEqual(stored["codex"]["unit"], "percent_per_mtok")
        result = self.run_tool(
            "sessions", "--harness", "codex", "--use-calibrated", "--no-color"
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn("calibrated", result.stdout.decode("utf-8"))

    def test_time_buckets_merge_nearby_intervals(self) -> None:
        self.build_rollouts()
        fine = self.run_json(
            "calibrate", "--harness", "codex", "--json", "--calibrate-bucket-hours", "0.01"
        )
        coarse = self.run_json(
            "calibrate", "--harness", "codex", "--json", "--calibrate-bucket-hours", "24"
        )
        self.assertGreater(fine["samples"], coarse["samples"])
        self.assertEqual(fine["intervals"], coarse["intervals"])
        self.assertEqual(coarse["samples"], 1)

    def test_unusable_fit_is_flagged_and_not_applied(self) -> None:
        self.build_rollouts()
        self.run_json(
            "calibrate", "--harness", "codex", "--json", "--calibrate-bucket-hours", "0.01"
        )
        fit_file = self.root / "state" / "codex-weights.json"
        stored = json.loads(fit_file.read_text(encoding="utf-8"))
        stored["usable"] = False
        fit_file.write_text(json.dumps(stored), encoding="utf-8")
        result = self.run_tool("sessions", "--harness", "codex", "--use-calibrated", "--json")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"worse than the mean", result.stderr)
        self.assertNotIn("calibrated", json.loads(result.stdout)["weight_source"])

    def test_nnls_stays_non_negative(self) -> None:
        matrix = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        target = [-5.0, 2.0, 2.0]
        solution = QD.nnls(matrix, target)
        self.assertEqual(len(solution), 2)
        for value in solution:
            self.assertGreaterEqual(value, 0.0)


class CacheBehaviour(Harness):
    def test_resume_after_append(self) -> None:
        now = time.time() - 600
        session = "22222222-aaaa-2222-3333-444444444444"
        path = self.write_claude(
            "resume.jsonl",
            [
                claude_assistant_line(now, session, "msg_1", output_tokens=100),
                claude_assistant_line(now + 1, session, "msg_2", output_tokens=200),
            ],
        )
        first = self.run_json("sessions", "--harness", "claude", "--json")
        row = self.session_by_short_id(first, "22222222")
        self.assertEqual(row["tokens"]["output"], 300)
        self.assertEqual(first["files_parsed"], 1)

        warm = self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(warm["files_parsed"], 0)
        self.assertEqual(
            self.session_by_short_id(warm, "22222222")["tokens"]["output"], 300
        )

        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                claude_assistant_line(now + 2, session, "msg_3", output_tokens=400) + "\n"
            )
            # A duplicate of the last message id must still be deduped across
            # the resume boundary.
            handle.write(
                claude_assistant_line(now + 2, session, "msg_3", output_tokens=400) + "\n"
            )
        os.utime(path, (time.time(), time.time()))
        second = self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(second["files_parsed"], 1)
        self.assertEqual(
            self.session_by_short_id(second, "22222222")["tokens"]["output"], 700
        )
        self.assertEqual(self.session_by_short_id(second, "22222222")["requests"], 3)

    def test_partial_trailing_line_is_reread(self) -> None:
        now = time.time() - 600
        session = "33333333-aaaa-2222-3333-444444444444"
        path = self.claude_projects / "proj" / "partial.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        complete = claude_assistant_line(now, session, "msg_p1", output_tokens=100)
        trailing = claude_assistant_line(now + 1, session, "msg_p2", output_tokens=250)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(complete + "\n")
            handle.write(trailing[:40])
        first = self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(
            self.session_by_short_id(first, "33333333")["tokens"]["output"], 100
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(complete + "\n")
            handle.write(trailing + "\n")
        os.utime(path, (time.time(), time.time()))
        second = self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(
            self.session_by_short_id(second, "33333333")["tokens"]["output"], 350
        )

    def test_rebuild_cache_reparses(self) -> None:
        now = time.time() - 600
        session = "44444444-aaaa-2222-3333-444444444444"
        self.write_claude(
            "rebuild.jsonl", [claude_assistant_line(now, session, "msg_r", output_tokens=10)]
        )
        self.run_json("sessions", "--harness", "claude", "--json")
        warm = self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(warm["files_parsed"], 0)
        rebuilt = self.run_json("sessions", "--harness", "claude", "--json", "--rebuild-cache")
        self.assertEqual(rebuilt["files_parsed"], 1)
        self.assertEqual(
            self.session_by_short_id(rebuilt, "44444444")["tokens"]["output"], 10
        )


class OutputContracts(Harness):
    def test_sessions_json_schema(self) -> None:
        now = time.time() - 600
        session = "55555555-aaaa-2222-3333-444444444444"
        self.write_claude(
            "schema.jsonl", [claude_assistant_line(now, session, "msg_s", output_tokens=10)]
        )
        payload = self.run_json("sessions", "--json")
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(payload["command"], "sessions")
        for key in ("generated_at", "weight_source", "codex_window", "files_scanned",
                    "files_parsed", "sessions"):
            self.assertIn(key, payload)
        row = payload["sessions"][0]
        for key in (
            "harness",
            "session_id",
            "short_id",
            "cwd",
            "primary_model",
            "start",
            "end",
            "requests",
            "subagent_requests",
            "tokens",
            "subagent_tokens",
            "weighted_units",
            "subagent_weighted_units",
            "unweighted_tokens",
            "drain_percent",
            "drain_is_estimate",
            "share_of_window_percent",
            "window_resets_at",
            "relative_to_harness_peak",
        ):
            self.assertIn(key, row)

    def test_no_color_output_has_no_escapes(self) -> None:
        now = time.time() - 600
        session = "66666666-aaaa-2222-3333-444444444444"
        self.write_claude(
            "plain.jsonl", [claude_assistant_line(now, session, "msg_n", output_tokens=10)]
        )
        result = self.run_tool("sessions", "--no-color", "--width", "120")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        text = result.stdout.decode("utf-8")
        self.assertNotIn("\033[", text)
        self.assertIn("66666666", text)

    def test_timeline_buckets(self) -> None:
        now = time.time() - 7200
        session = "77777777-aaaa-2222-3333-444444444444"
        self.write_claude(
            "tl.jsonl",
            [
                claude_assistant_line(now, session, "msg_t1", output_tokens=1000),
                claude_assistant_line(now + 3700, session, "msg_t2", output_tokens=2000),
            ],
        )
        payload = self.run_json("timeline", "--json", "--bucket", "1h")
        self.assertEqual(payload["bucket_seconds"], 3600)
        self.assertGreaterEqual(len(payload["buckets"]), 2)

    def test_since_prunes_old_sessions(self) -> None:
        old = time.time() - 30 * 86400
        recent = time.time() - 600
        self.write_claude(
            "old.jsonl",
            [claude_assistant_line(old, "88888888-aaaa-1111-2222-333333333333", "msg_o",
                                   output_tokens=10)],
        )
        self.write_claude(
            "new.jsonl",
            [claude_assistant_line(recent, "99999999-aaaa-1111-2222-333333333333", "msg_w",
                                   output_tokens=10)],
        )
        payload = self.run_json("sessions", "--json", "--since", "3d")
        short_ids = [row["short_id"] for row in payload["sessions"]]
        self.assertIn("99999999", short_ids)
        self.assertNotIn("88888888", short_ids)

    def test_long_context_multiplier_is_a_no_op_by_default(self) -> None:
        now = time.time() - 600
        session = "aaaabbbb-cccc-1111-2222-333333333333"
        self.write_claude(
            "lc.jsonl",
            [
                claude_assistant_line(
                    now, session, "msg_lc", input_tokens=300_000, output_tokens=1_000
                )
            ],
        )
        default = self.run_json("sessions", "--json")
        doubled = self.run_json("sessions", "--json", "--long-context-multiplier", "2.0")
        base = self.session_by_short_id(default, "aaaabbbb")["weighted_units"]
        scaled = self.session_by_short_id(doubled, "aaaabbbb")["weighted_units"]
        self.assertAlmostEqual(scaled, base * 2.0, places=6)


class Snapshots(Harness):
    def test_stdin_passthrough_is_byte_for_byte(self) -> None:
        payload = json.dumps(
            {
                "rate_limits": {
                    "five_hour": {"used_percentage": 42.5, "resets_at": "2026-09-16T21:30:00Z"},
                    "seven_day": {"used_percentage": 7.0, "resets_at": "2026-09-23T14:00:00Z"},
                },
                "model": {"display_name": "Opus 5"},
            }
        ).encode("utf-8")
        result = self.run_tool("snapshot", "--stdin", stdin=payload)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, payload)
        logged = (self.root / "state" / "snapshots.jsonl").read_text(encoding="utf-8")
        record = json.loads(logged.strip())
        self.assertEqual(record["source"], "statusline")
        self.assertAlmostEqual(record["windows"]["five_hour"]["utilization_percent"], 42.5)
        self.assertNotIn("accountUuid", logged)

    def test_stdin_passthrough_survives_garbage(self) -> None:
        payload = b"this is not json at all\n"
        result = self.run_tool("snapshot", "--stdin", stdin=payload)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, payload)

    def test_config_snapshot_dedupes_by_fetched_at(self) -> None:
        config = self.home / ".claude" / ".claude.json"
        config.write_text(
            json.dumps(
                {
                    "oauthAccount": {
                        "organizationRateLimitTier": "default_claude_max_20x",
                        "accountUuid": "must-not-be-logged",
                    },
                    "cachedUsageUtilization": {
                        "accountUuid": "must-not-be-logged",
                        "fetchedAtMs": 1789578063180,
                        "utilization": {
                            "five_hour": {
                                "utilization": 10,
                                "resets_at": "2026-09-16T21:30:00+00:00",
                                "limit_dollars": None,
                                "used_dollars": None,
                            },
                            "seven_day": {"utilization": 2, "resets_at": None},
                            "seven_day_opus": None,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        first = self.run_tool("snapshot")
        self.assertEqual(first.returncode, 0)
        self.assertIn(b"appended 1", first.stdout)
        second = self.run_tool("snapshot")
        self.assertIn(b"appended 0", second.stdout)
        logged = (self.root / "state" / "snapshots.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("must-not-be-logged", logged)
        record = json.loads(logged.strip())
        self.assertEqual(record["windows"]["five_hour"]["utilization_percent"], 10.0)
        self.assertNotIn("seven_day_opus", record["windows"])

    def test_tier_shows_in_header_without_account_details(self) -> None:
        config = self.home / ".claude" / ".claude.json"
        config.write_text(
            json.dumps(
                {
                    "oauthAccount": {
                        "organizationRateLimitTier": "default_claude_max_20x",
                        "emailAddress": "must-not-be-printed@example.com",
                    }
                }
            ),
            encoding="utf-8",
        )
        now = time.time() - 600
        self.write_claude(
            "tier.jsonl",
            [
                claude_assistant_line(
                    now, "bbbbcccc-dddd-1111-2222-333333333333", "msg_tier", output_tokens=10
                )
            ],
        )
        result = self.run_tool("sessions", "--no-color")
        text = result.stdout.decode("utf-8")
        self.assertIn("default_claude_max_20x", text)
        self.assertNotIn("must-not-be-printed", text)


class ConfigWeights(Harness):
    def test_config_file_overrides_builtin_weights(self) -> None:
        config = self.root / "config" / "weights.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps(
                {
                    "codex": {
                        "unit": "credit_units_per_mtok",
                        "models": {"gpt-5.6-sol": {"input": 1.0, "cached_input": 0.0,
                                                   "output": 0.0}},
                    }
                }
            ),
            encoding="utf-8",
        )
        now = time.time() - 1200
        session = "codex-cfg-0001"
        self.write_codex(
            "rollout-cfg.jsonl",
            [
                codex_session_meta_line(now, session, "/home/agent/cfg"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 10, session, input_tokens=2_000_000, cached_input_tokens=0,
                    output_tokens=1_000_000
                ),
            ],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        self.assertAlmostEqual(payload["sessions"][0]["weighted_units"], 2.0, places=6)
        self.assertIn("config", payload["weight_source"])


class PromptGrouping(Harness):
    def test_claude_tool_results_do_not_start_a_prompt(self) -> None:
        now = time.time() - 1800
        session = "abcd1111-aaaa-2222-3333-444444444444"
        lines = [claude_user_prompt_line(now, session)]
        for step in range(4):
            lines.append(
                claude_assistant_line(
                    now + 1 + step, session, "msg_p%d" % step, input_tokens=100,
                    cache_read=1000 * (step + 1), output_tokens=50
                )
            )
            lines.append(claude_tool_result_line(now + 1.5 + step, session))
        lines.append(claude_user_prompt_line(now + 100, session, text="second prompt"))
        lines.append(
            claude_assistant_line(now + 101, session, "msg_q0", output_tokens=10)
        )
        self.write_claude("prompts.jsonl", lines)
        payload = self.run_json("prompts", "--session", "abcd1111", "--json")
        self.assertEqual(len(payload["prompts"]), 2)
        self.assertEqual(payload["prompts"][0]["api_turns"], 4)
        self.assertEqual(payload["prompts"][1]["api_turns"], 1)
        self.assertEqual(payload["prompts"][0]["index"], 1)

    def test_claude_subagent_calls_join_the_running_prompt(self) -> None:
        now = time.time() - 1800
        session = "abcd2222-aaaa-2222-3333-444444444444"
        self.write_claude(
            "parent.jsonl",
            [
                claude_user_prompt_line(now, session),
                claude_assistant_line(now + 1, session, "msg_main", output_tokens=100),
            ],
        )
        self.write_claude(
            "subagents/agent-9.jsonl",
            [
                claude_assistant_line(
                    now + 2, session, "msg_sub", output_tokens=400, sidechain=True
                )
            ],
        )
        payload = self.run_json("prompts", "--session", "abcd2222", "--json")
        self.assertEqual(len(payload["prompts"]), 1)
        self.assertEqual(payload["prompts"][0]["api_turns"], 2)
        self.assertEqual(payload["prompts"][0]["subagent_turns"], 1)

    def test_codex_groups_by_turn_id(self) -> None:
        now = time.time() - 1800
        session = "codex-prompt-0001"
        lines = [codex_session_meta_line(now, session, "/home/agent/repo")]
        for turn, calls in enumerate((3, 5)):
            # No task_started line: turn_id alone must carry the grouping.
            lines.append(codex_turn_context_line(now + turn * 100 + 1, "gpt-5.6-sol"))
            for call in range(calls):
                lines.append(
                    codex_usage_record_line(
                        now + turn * 100 + 2 + call,
                        session,
                        input_tokens=50_000,
                        cached_input_tokens=1_000,
                        output_tokens=100,
                        turn_id="turn-%d" % turn,
                    )
                )
        self.write_codex("rollout-prompts.jsonl", lines)
        payload = self.run_json("prompts", "--session", "codex-prompt", "--json")
        self.assertEqual(len(payload["prompts"]), 2)
        self.assertEqual([prompt["api_turns"] for prompt in payload["prompts"]], [3, 5])

    def test_codex_subagent_thread_rolls_into_parent_session(self) -> None:
        now = time.time() - 1800
        session = "codex-parent-0001"
        self.write_codex(
            "rollout-root.jsonl",
            [
                codex_session_meta_line(now, session, "/home/agent/root"),
                codex_task_started_line(now + 1),
                codex_turn_context_line(now + 2, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 3, session, input_tokens=100_000, cached_input_tokens=0,
                    output_tokens=1_000
                ),
            ],
        )
        self.write_codex(
            "rollout-child.jsonl",
            [
                codex_subagent_meta_line(now + 4, "codex-child-0001", session,
                                         "/home/agent/root"),
                codex_turn_context_line(now + 5, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 6, "codex-child-0001", input_tokens=20_000,
                    cached_input_tokens=0, output_tokens=500
                ),
            ],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        self.assertEqual(len(payload["sessions"]), 1)
        row = payload["sessions"][0]
        self.assertEqual(row["tokens"]["input"], 120_000)
        self.assertEqual(row["subagent_tokens"]["input"], 20_000)
        self.assertEqual(row["api_turns"], 2)
        self.assertEqual(row["prompts"], 1)

    def test_resent_share_counts_context_beyond_the_first_turn(self) -> None:
        now = time.time() - 1800
        session = "codex-resent-0001"
        lines = [
            codex_session_meta_line(now, session, "/home/agent/resent"),
            codex_task_started_line(now + 1),
            codex_turn_context_line(now + 2, "gpt-5.6-sol"),
        ]
        for call in range(4):
            lines.append(
                codex_usage_record_line(
                    now + 3 + call, session, input_tokens=1_000_000,
                    cached_input_tokens=0, output_tokens=0
                )
            )
        self.write_codex("rollout-resent.jsonl", lines)
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        row = payload["sessions"][0]
        # Four identical calls: one is the real context, three are re-sends.
        self.assertAlmostEqual(row["resent_share"], 0.75, places=6)


class QuadraticGrowth(Harness):
    def test_quadratic_session_is_recognised(self) -> None:
        now = time.time() - 4 * 3600
        session = "codex-quad-0001"
        lines = [codex_session_meta_line(now, session, "/home/agent/quad")]
        # Prompt k sends k turns of a k-sized context: cost grows as k^2.
        for index in range(1, 13):
            stamp = now + index * 100
            lines.append(codex_task_started_line(stamp))
            lines.append(codex_turn_context_line(stamp + 1, "gpt-5.6-sol"))
            for call in range(index):
                lines.append(
                    codex_usage_record_line(
                        stamp + 2 + call,
                        session,
                        input_tokens=100_000 * index,
                        cached_input_tokens=0,
                        output_tokens=0,
                    )
                )
        self.write_codex("rollout-quad.jsonl", lines)
        payload = self.run_json("prompts", "--session", "codex-quad", "--json")
        growth = payload["growth"]
        self.assertEqual(growth["better"], "quadratic")
        self.assertGreater(growth["quadratic"]["r_squared"], 0.99)
        # units = index^2 * 0.1 Mtok * 100 units/Mtok = 10 * index^2
        self.assertAlmostEqual(growth["quadratic"]["square_term"], 10.0, places=3)
        self.assertGreater(growth["last_20_percent_share"], 0.3)

    def test_flat_session_prefers_linear(self) -> None:
        now = time.time() - 4 * 3600
        session = "codex-flat-0001"
        lines = [codex_session_meta_line(now, session, "/home/agent/flat")]
        for index in range(1, 13):
            stamp = now + index * 100
            lines.append(codex_task_started_line(stamp))
            lines.append(codex_turn_context_line(stamp + 1, "gpt-5.6-sol"))
            lines.append(
                codex_usage_record_line(
                    stamp + 2, session, input_tokens=100_000, cached_input_tokens=0,
                    output_tokens=0
                )
            )
        self.write_codex("rollout-flat.jsonl", lines)
        payload = self.run_json("prompts", "--session", "codex-flat", "--json")
        self.assertEqual(payload["growth"]["better"], "linear")

    def test_fanout_reports_distribution_and_top_prompts(self) -> None:
        now = time.time() - 4 * 3600
        session = "codex-fan-0001"
        lines = [codex_session_meta_line(now, session, "/home/agent/fan")]
        for index, calls in enumerate((1, 2, 8, 30), start=1):
            stamp = now + index * 100
            lines.append(codex_task_started_line(stamp))
            lines.append(codex_turn_context_line(stamp + 1, "gpt-5.6-sol"))
            for call in range(calls):
                lines.append(
                    codex_usage_record_line(
                        stamp + 2 + call, session, input_tokens=60_000,
                        cached_input_tokens=0, output_tokens=0
                    )
                )
        self.write_codex("rollout-fan.jsonl", lines)
        payload = self.run_json("fanout", "--harness", "codex", "--json")
        self.assertEqual(payload["prompts"], 4)
        self.assertEqual(payload["turns_per_prompt"]["max"], 30)
        self.assertEqual(payload["top_prompts"][0]["api_turns"], 30)
        self.assertEqual(payload["top_prompts"][0]["cwd"], "fan")
        self.assertEqual(sum(b["count"] for b in payload["turns_histogram"]), 4)


class ContextReductions(Harness):
    def codex_run(self, session: str, contexts: Sequence[int],
                  compact_after: Optional[int] = None) -> None:
        now = time.time() - 4 * 3600
        lines = [
            codex_session_meta_line(now, session, "/home/agent/red"),
            codex_task_started_line(now + 1),
            codex_turn_context_line(now + 2, "gpt-5.6-sol"),
        ]
        for index, context in enumerate(contexts):
            if compact_after is not None and index == compact_after:
                lines.append(codex_compacted_line(now + 3 + index - 0.5))
            lines.append(
                codex_usage_record_line(
                    now + 3 + index, session, input_tokens=context,
                    cached_input_tokens=0, output_tokens=100
                )
            )
        self.write_codex("rollout-%s.jsonl" % session, lines)

    def test_compacted_record_is_labelled_compact(self) -> None:
        self.codex_run(
            "codex-red-0001",
            [100_000, 200_000, 300_000, 40_000, 45_000, 50_000, 55_000],
            compact_after=3,
        )
        payload = self.run_json("reductions", "--harness", "codex", "--json")
        rows = payload["reductions"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "compact")
        self.assertEqual(rows[0]["before"], 300_000)
        self.assertEqual(rows[0]["after"], 40_000)
        self.assertEqual(rows[0]["removed_tokens"], 260_000)
        self.assertEqual(rows[0]["turns_after"], 3)
        self.assertGreater(rows[0]["saved_units"], 0.0)
        self.assertGreater(rows[0]["saved_units_upper_bound"], rows[0]["saved_units"])

    def test_unmarked_drop_is_reported(self) -> None:
        self.codex_run(
            "codex-red-0002", [100_000, 200_000, 300_000, 40_000, 45_000, 50_000, 55_000]
        )
        payload = self.run_json("reductions", "--harness", "codex", "--json")
        self.assertEqual(len(payload["reductions"]), 1)
        self.assertEqual(payload["reductions"][0]["kind"], "unmarked")

    def test_compacted_history_fields_are_never_stored(self) -> None:
        self.codex_run(
            "codex-red-0003",
            [100_000, 200_000, 300_000, 40_000, 45_000, 50_000, 55_000],
            compact_after=3,
        )
        self.run_json("reductions", "--harness", "codex", "--json")
        cached = ""
        for path in (self.root / "cache").rglob("*.json"):
            cached += path.read_text(encoding="utf-8")
        self.assertNotIn("SENSITIVE", cached)

    def test_growing_session_is_not_a_reduction(self) -> None:
        self.codex_run(
            "codex-red-0004", [20_000, 40_000, 80_000, 160_000, 320_000, 400_000, 450_000]
        )
        payload = self.run_json("reductions", "--harness", "codex", "--json")
        self.assertEqual(payload["reductions"], [])

    def test_a_small_new_session_is_not_a_reduction(self) -> None:
        self.codex_run("codex-red-0005", [300_000, 320_000, 340_000, 360_000, 380_000])
        self.codex_run("codex-red-0006", [9_000, 10_000, 11_000, 12_000, 13_000])
        payload = self.run_json("reductions", "--harness", "codex", "--json")
        self.assertEqual(payload["reductions"], [])

    def test_a_single_small_call_that_bounces_back_is_not_a_reduction(self) -> None:
        # Codex interleaves a second, smaller-context call stream into one
        # thread; those dips must not read as context reductions.
        self.codex_run(
            "codex-red-0007",
            [200_000, 40_000, 210_000, 45_000, 220_000, 50_000, 230_000, 55_000],
        )
        payload = self.run_json("reductions", "--harness", "codex", "--json")
        self.assertEqual(payload["reductions"], [])

    def test_claude_drop_is_detected(self) -> None:
        now = time.time() - 4 * 3600
        session = "abcd3333-aaaa-2222-3333-444444444444"
        lines = [claude_user_prompt_line(now, session)]
        for index, context in enumerate(
            [100_000, 200_000, 300_000, 40_000, 45_000, 50_000, 55_000]
        ):
            lines.append(
                claude_assistant_line(
                    now + 1 + index, session, "msg_r%d" % index,
                    cache_read=context, output_tokens=100
                )
            )
        self.write_claude("reduce.jsonl", lines)
        payload = self.run_json("reductions", "--harness", "claude", "--json")
        self.assertEqual(len(payload["reductions"]), 1)
        self.assertEqual(payload["reductions"][0]["kind"], "unmarked")
        self.assertEqual(payload["reductions"][0]["before"], 300_000)


class OauthSampler(Harness):
    RESPONSE = {
        "five_hour": {"utilization": 15.0, "resets_at": "2026-09-16T21:30:00+00:00",
                      "limit_dollars": None, "used_dollars": None},
        "seven_day": {"utilization": 3.0, "resets_at": "2026-09-23T14:00:00+00:00"},
        "seven_day_opus": None,
        "seven_day_sonnet": None,
        "limits": [
            {"kind": "session", "group": "session", "percent": 15, "severity": "normal",
             "is_active": True, "resets_at": "2026-09-16T21:30:00+00:00",
             "scope": {"model": {"id": None, "display_name": "Fable"}}}
        ],
        "spend": {"percent": 0, "used": {"amount_minor": 0}},
        "extra_usage": {"is_enabled": False},
        "organization": {"uuid": "must-not-be-stored"},
        "account": {"uuid": "must-not-be-stored"},
    }

    def write_credentials(self, name: str = ".claude", expires_in: float = 3600.0) -> Path:
        config = self.home / name
        config.mkdir(parents=True, exist_ok=True)
        (config / ".credentials.json").write_text(
            json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "sk-ant-oat-TESTTOKEN",
                        "expiresAt": int((time.time() + expires_in) * 1000),
                    }
                }
            ),
            encoding="utf-8",
        )
        return config

    def sample(self, args_list: Sequence[str], response: Any = None,
               status: Optional[int] = None) -> List[Dict[str, Any]]:
        import io
        import urllib.error
        import urllib.request

        captured = {}

        class FakeResponse(io.BytesIO):
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["timeout"] = timeout
            if status is not None:
                raise urllib.error.HTTPError(
                    request.full_url, status, "rate limited", {}, None
                )
            body = json.dumps(response if response is not None else self.RESPONSE)
            return FakeResponse(body.encode("utf-8"))

        original = urllib.request.urlopen
        environment = dict(os.environ)
        os.environ.update(
            {
                key: value
                for key, value in self.environment.items()
                if key.startswith("QUOTA_DRAIN_")
            }
        )
        urllib.request.urlopen = fake_urlopen
        stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            QD.main(list(args_list))
        finally:
            sys.stdout = stdout
            urllib.request.urlopen = original
            os.environ.clear()
            os.environ.update(environment)
        self.captured = captured
        path = self.root / "state" / "snapshots.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_sampler_writes_a_snapshot_without_identifiers(self) -> None:
        self.write_credentials()
        rows = self.sample(["snapshot", "--oauth"])
        self.assertEqual(len(rows), 1)
        record = rows[0]
        self.assertEqual(record["source"], "oauth")
        self.assertEqual(record["config_dir"], ".claude")
        self.assertEqual(record["windows"]["five_hour"]["utilization_percent"], 15.0)
        self.assertEqual(record["windows"]["seven_day"]["utilization_percent"], 3.0)
        self.assertNotIn("seven_day_opus", record["windows"])
        self.assertEqual(record["limits"][0]["scope_model"], "Fable")
        self.assertEqual(record["spend_percent"], 0.0)
        serialized = json.dumps(record)
        self.assertNotIn("must-not-be-stored", serialized)
        self.assertNotIn("TESTTOKEN", serialized)

    def test_request_shape(self) -> None:
        self.write_credentials()
        self.sample(["snapshot", "--oauth"])
        self.assertEqual(self.captured["url"], QD.OAUTH_USAGE_URL)
        headers = dict((key.lower(), value) for key, value in self.captured["headers"].items())
        self.assertEqual(headers["authorization"], "Bearer sk-ant-oat-TESTTOKEN")
        self.assertEqual(headers["anthropic-beta"], "oauth-2025-04-20")
        self.assertEqual(headers["content-type"], "application/json")
        self.assertTrue(headers["user-agent"].startswith("claude-code/"))
        self.assertEqual(self.captured["timeout"], QD.OAUTH_TIMEOUT_SECONDS)

    def test_token_never_reaches_disk(self) -> None:
        self.write_credentials()
        self.sample(["snapshot", "--oauth"])
        written = ""
        for path in (self.root / "state").rglob("*"):
            if path.is_file():
                written += path.read_text(encoding="utf-8")
        self.assertNotIn("TESTTOKEN", written)

    def test_minimum_interval_blocks_a_second_call(self) -> None:
        self.write_credentials()
        first = self.sample(["snapshot", "--oauth"])
        second = self.sample(["snapshot", "--oauth"])
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        state = json.loads((self.root / "state" / "oauth-poll.json").read_text(encoding="utf-8"))
        self.assertIn(".claude", state)
        self.assertIn("last_attempt", state[".claude"])

    def test_rate_limit_sets_a_backoff(self) -> None:
        self.write_credentials()
        rows = self.sample(["snapshot", "--oauth"], status=429)
        self.assertEqual(rows, [])
        state = json.loads((self.root / "state" / "oauth-poll.json").read_text(encoding="utf-8"))
        self.assertGreater(state[".claude"]["blocked_until"], time.time())

    def test_expired_token_is_skipped(self) -> None:
        self.write_credentials(expires_in=-10.0)
        rows = self.sample(["snapshot", "--oauth"])
        self.assertEqual(rows, [])
        self.assertNotIn("url", self.captured)


class ClaudeCalibration(Harness):
    def write_snapshots(self, samples: Sequence[Tuple[float, float]]) -> None:
        path = self.root / "state" / "snapshots.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for epoch, percent in samples:
                handle.write(
                    json.dumps(
                        {
                            "source": "oauth",
                            "config_dir": ".claude",
                            "ts": epoch,
                            "windows": {
                                "five_hour": {
                                    "utilization_percent": percent,
                                    "resets_at": "2026-09-16T21:30:00+00:00",
                                }
                            },
                        }
                    )
                    + "\n"
                )

    def test_fit_recovers_the_cache_read_ratio(self) -> None:
        now = time.time() - 6 * 3600
        session = "abcd4444-aaaa-2222-3333-444444444444"
        input_rate, cache_rate = 6.0, 0.6
        lines = [claude_user_prompt_line(now, session)]
        samples = [(now, 0.0)]
        used = 0.0
        mixes = [
            (900_000, 100_000), (200_000, 900_000), (500_000, 400_000),
            (100_000, 800_000), (800_000, 200_000), (300_000, 700_000),
            (700_000, 300_000), (400_000, 600_000), (600_000, 150_000),
            (250_000, 750_000), (950_000, 50_000), (150_000, 950_000),
        ]
        for index, (fresh, cached) in enumerate(mixes):
            stamp = now + (index + 1) * 120
            lines.append(
                claude_assistant_line(
                    stamp - 60, session, "msg_c%d" % index,
                    input_tokens=fresh, cache_read=cached, output_tokens=0
                )
            )
            used += (fresh * input_rate + cached * cache_rate) / 1_000_000.0
            samples.append((stamp, used))
        self.write_claude("calib.jsonl", lines)
        self.write_snapshots(samples)
        payload = self.run_json(
            "calibrate", "--harness", "claude", "--json", "--calibrate-bucket-hours", "0.01"
        )
        fitted = payload["windows"]["five_hour"]["models"]["claude-opus-5"]
        self.assertLess(abs(fitted["input"] - input_rate) / input_rate, 0.05)
        self.assertLess(abs(fitted["cache_read"] - cache_rate) / cache_rate, 0.05)
        self.assertGreater(payload["windows"]["five_hour"]["r_squared"], 0.99)

    def test_calibrate_without_snapshots_explains_itself(self) -> None:
        result = self.run_tool("calibrate", "--harness", "claude")
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"snapshot --oauth", result.stderr)


if __name__ == "__main__":
    unittest.main()
