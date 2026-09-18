"""Contract tests for quota-drain.

Every fixture here is synthetic. Real transcripts hold prompts and customer
data and must never be copied into this repository.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nenpi.config as QC
from nenpi.settings import SourceSettings, migrate_json_store
import nenpi.drain as QD

DRAIN_COMMAND = [sys.executable, "-m", "nenpi.drain"]


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


def claude_tool_use_line(
    epoch: float,
    session_id: str,
    message_id: str,
    calls: Sequence[Tuple[str, str]],
    *,
    input_tokens: int = 0,
    cache_read: int = 0,
    output_tokens: int = 0,
    sidechain: bool = False,
) -> str:
    """An assistant line that issues tool calls: (tool_use_id, tool name)."""
    return json.dumps(
        {
            "type": "assistant",
            "sessionId": session_id,
            "cwd": "/home/agent/project",
            "timestamp": iso(epoch),
            "requestId": "req_" + message_id,
            "isSidechain": sidechain,
            "message": {
                "id": message_id,
                "role": "assistant",
                "model": "claude-opus-5",
                "content": [
                    {"type": "tool_use", "id": call_id, "name": name, "input": {}}
                    for call_id, name in calls
                ],
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": output_tokens,
                },
            },
        }
    )


def claude_tool_output_line(
    epoch: float,
    session_id: str,
    call_id: str,
    payload: Any,
    *,
    sidechain: bool = False,
) -> str:
    return json.dumps(
        {
            "type": "user",
            "sessionId": session_id,
            "timestamp": iso(epoch),
            "isSidechain": sidechain,
            "toolUseResult": {"stdout": "synthetic"},
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call_id, "content": payload}
                ],
            },
        }
    )


def codex_tool_call_line(
    epoch: float,
    call_id: str,
    name: str,
    *,
    item: str = "function_call",
    namespace: Optional[str] = None,
) -> str:
    payload = {"type": item, "id": "item-" + call_id, "call_id": call_id, "name": name}
    if namespace is not None:
        payload["namespace"] = namespace
    if item == "custom_tool_call":
        payload["input"] = "synthetic"
    else:
        payload["arguments"] = "{}"
    return json.dumps(
        {"type": "response_item", "timestamp": iso(epoch), "payload": payload}
    )


def codex_tool_output_line(
    epoch: float, call_id: str, output: Any, *, item: str = "function_call_output"
) -> str:
    return json.dumps(
        {
            "type": "response_item",
            "timestamp": iso(epoch),
            "payload": {
                "type": item,
                "id": "out-" + call_id,
                "call_id": call_id,
                "output": output,
            },
        }
    )


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
        # A host that happens to export these would otherwise change which
        # root the default resolution picks, out from under the fixture.
        self.environment.pop("CLAUDE_CONFIG_DIR", None)
        self.environment.pop("CODEX_HOME", None)
        self.environment.update(
            {
                "NENPI_HOME_DIR": str(self.home),
                "NENPI_CACHE_DIR": str(self.root / "cache"),
                "NENPI_STATE_DIR": str(self.root / "state"),
                "NENPI_CONFIG_DIR": str(self.root / "config"),
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

    @contextlib.contextmanager
    def env_applied(self):
        """Apply the temp-dir overrides to this process.

        In-process calls into the module would otherwise read the real
        ~/.claude, ~/.cache and ~/.local/state of whoever runs the tests.
        """
        saved = dict(os.environ)
        os.environ.pop("CLAUDE_CONFIG_DIR", None)
        os.environ.pop("CODEX_HOME", None)
        os.environ.update(
            dict(
                (key, value)
                for key, value in self.environment.items()
                if key.startswith("NENPI_") or key.startswith("QUOTA_DRAIN_")
            )
        )
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def run_tool(self, *arguments: str, stdin: Optional[bytes] = None,
                extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        env = self.environment
        if extra_env:
            env = dict(self.environment)
            env.update(extra_env)
        return subprocess.run(
            DRAIN_COMMAND + list(arguments),
            check=False,
            input=stdin,
            capture_output=True,
            env=env,
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
        rows = dict((row["cwd"], row) for row in payload["sessions"])
        self.assertAlmostEqual(rows["alpha"]["drain_percent"], 3.0, places=6)
        self.assertAlmostEqual(rows["beta"]["drain_percent"], 1.0, places=6)
        self.assertFalse(rows["alpha"]["drain_is_estimate"])
        self.assertAlmostEqual(rows["alpha"]["share_of_window_percent"], 75.0, places=6)

    def test_rollover_when_resets_at_changes(self) -> None:
        resets_at = int(time.time()) + 7200
        # The new five-hour window opened just before the calls ran, so both
        # sessions are inside it.
        self.write_two_concurrent_sessions(resets_at, int(time.time() - 3600) + 18005)
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        rows = dict((row["cwd"], row) for row in payload["sessions"])
        # A new window means the second reading is the whole drain so far, not a
        # difference, so the interval carries 14% rather than 4%.
        total = rows["alpha"]["drain_percent"] + rows["beta"]["drain_percent"]
        self.assertAlmostEqual(total, 14.0, places=6)
        self.assertAlmostEqual(rows["alpha"]["drain_percent"], 10.5, places=6)

    def test_rollover_does_not_reach_back_past_the_reset(self) -> None:
        # The new window opened after both sessions had finished, so none of
        # its drain can belong to them.
        resets_at = int(time.time()) + 7200
        self.write_two_concurrent_sessions(resets_at, int(time.time()) + 18000 + 3600)
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        for row in payload["sessions"]:
            self.assertIsNone(row["drain_percent"])

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

    def test_windows_json_carries_full_session_ids(self) -> None:
        resets_at = int(time.time()) + 7200
        self.write_two_concurrent_sessions(resets_at, resets_at)
        payload = self.run_json("windows", "--harness", "codex", "--json")
        top = payload["windows"][0]["top_sessions"]
        self.assertEqual(sorted(row["session_id"] for row in top),
                         ["codex-conc-aaaa", "codex-conc-bbbb"])
        self.assertEqual([QD.short_id(row["session_id"]) for row in top],
                         [row["short_id"] for row in top])


class DrainIntervalRules(Harness):
    """Unit-level coverage of build_intervals rules (a)/(b)/(c), #16/#17."""

    def make_row(self, ts: float, used: float, resets_at: Any,
                window_minutes: int = 300) -> Dict[str, Any]:
        return {
            "ts": ts, "used_percent": used, "resets_at": resets_at,
            "window_minutes": window_minutes, "account": "acct",
            "limit_id": "codex", "plan_type": "pro",
        }

    def test_decrease_then_rise_starts_the_interval_at_the_low_reading(self) -> None:
        # 40 -> 30 -> 35: the interval is the 5-point rise from the 30
        # reading, not 0 (stuck at the old high-water mark) and not 5
        # measured back from the 40 reading.
        now = time.time() - 3600
        resets = int(now) + 7200
        rows = [
            self.make_row(now, 40.0, resets),
            self.make_row(now + 600, 30.0, resets),
            self.make_row(now + 1200, 35.0, resets),
        ]
        intervals = QD.build_intervals(rows, 300)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0].drain, 5.0, places=6)
        self.assertEqual(intervals[0].start, now + 600)
        self.assertEqual(intervals[0].end, now + 1200)
        self.assertFalse(intervals[0].rollover)

    def test_jitter_within_tolerance_is_not_a_new_baseline(self) -> None:
        # 40 -> 39 -> 41: the 1-point dip is vendor jitter (<= JITTER_TOLERANCE)
        # and must not become the new baseline - if it did, the following
        # rise would be measured as 39 -> 41 (+2) on top of the 40 already
        # attributed, over-charging by 2. Keeping the baseline at 40 across
        # the dip means the rise to 41 is correctly +1 over the 40 already
        # attributed, and the interval reaches back to the ORIGINAL anchor.
        now = time.time() - 3600
        resets = int(now) + 7200
        rows = [
            self.make_row(now, 40.0, resets),
            self.make_row(now + 600, 39.0, resets),
            self.make_row(now + 1200, 41.0, resets),
        ]
        intervals = QD.build_intervals(rows, 300)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0].drain, 1.0, places=6)
        self.assertEqual(intervals[0].start, now)
        self.assertEqual(intervals[0].end, now + 1200)

    def test_decrease_beyond_tolerance_is_not_jitter(self) -> None:
        now = time.time() - 3600
        resets = int(now) + 7200
        rows = [
            self.make_row(now, 40.0, resets),
            # A drop of 10 is well past JITTER_TOLERANCE (2): a real decrease.
            self.make_row(now + 600, 30.0, resets),
        ]
        intervals = QD.build_intervals(rows, 300)
        self.assertEqual(len(intervals), 0)

    def test_slide_with_unchanged_used_is_not_a_rollover(self) -> None:
        # resets_at drifts (an idle pool re-stamping resets_at = now + 7d)
        # but used_percent does not: no drain evidence, so no interval,
        # regardless of the resets_at churn.
        now = time.time() - 3600
        resets = int(now) + 7200
        rows = [
            self.make_row(now, 40.0, resets),
            self.make_row(now + 600, 40.0, resets + 70),
            self.make_row(now + 1200, 40.0, resets + 140),
        ]
        intervals = QD.build_intervals(rows, 300)
        self.assertEqual(len(intervals), 0)

    def test_idle_zero_used_slide_produces_no_interval(self) -> None:
        now = time.time() - 3600
        resets = int(now) + 604800
        rows = [
            self.make_row(now + index * 70, 0.0, resets + index * 70)
            for index in range(5)
        ]
        intervals = QD.build_intervals(rows, 300)
        self.assertEqual(len(intervals), 0)


class AttributionCap(Harness):
    """CLI-level coverage of the attribution cap and the #17 windows fix."""

    def write_dense_phase(self, base: int, session: str, resets: int) -> None:
        """Three non-rollover intervals at a known 0.1%/unit rate.

        `gpt-5.6-sol` prices `input` at 100.0 credit units per Mtok, so
        100_000 tokens is 10.0 weighted units; each interval's drain is 1.0,
        so drain/units == 0.1 for every one of them - enough samples
        (>= MIN_RATE_INTERVALS) and enough coverage (>= MIN_RATE_UNITS) for
        the pool to measure its own rate instead of falling back.
        """
        lines = [
            codex_session_meta_line(base, session, "/home/agent/dense"),
            codex_turn_context_line(base, "gpt-5.6-sol"),
            codex_token_count_line(base + 5, rate_limits=rate_limits(10.0, resets)),
        ]
        used = 10.0
        for step in range(3):
            offset = 10 + step * 10
            lines.append(
                codex_usage_record_line(
                    base + offset, session, input_tokens=100_000, cached_input_tokens=0,
                    output_tokens=0
                )
            )
            used += 1.0
            lines.append(
                codex_token_count_line(
                    base + offset + 5, rate_limits=rate_limits(used, resets)
                )
            )
        self.write_codex("rollout-dense.jsonl", lines)

    def write_jump(self, tiny_start: int, tiny_session: str, tiny_model: str,
                   tiny_tokens: int, resets: int, used: float) -> None:
        """One rollover interval, far later, with only a tiny session active."""
        self.write_codex(
            "rollout-tiny.jsonl",
            [
                codex_session_meta_line(tiny_start, tiny_session, "/home/agent/tiny"),
                codex_turn_context_line(tiny_start, tiny_model),
                codex_usage_record_line(
                    tiny_start + 5, tiny_session, input_tokens=tiny_tokens,
                    cached_input_tokens=0, output_tokens=0
                ),
                codex_token_count_line(tiny_start + 10, rate_limits=rate_limits(used, resets)),
            ],
        )

    def test_sparse_reading_drain_is_capped_and_the_rest_unattributed(self) -> None:
        # Reproduces #16's 01a09886: a handful of turns land inside a big,
        # far-later jump and would otherwise absorb the whole thing.
        base = int(time.time()) - 4 * 3600
        dense_session = "dddddddd-dense-0000-0000-000000000000"
        resets_a = base + 7200
        self.write_dense_phase(base, dense_session, resets_a)

        tiny_start = base + 3600
        opened = tiny_start - 60
        resets_b = opened + 18000
        tiny_session = "tttttttt-tiny0-0000-0000-000000000000"
        self.write_jump(tiny_start, tiny_session, "gpt-5.6-sol", 6_000, resets_b, 29.0)

        payload = self.run_json("sessions", "--harness", "codex", "--json", "--top", "20")
        tiny_row = self.session_by_short_id(payload, QD.short_id(tiny_session))
        dense_row = self.session_by_short_id(payload, QD.short_id(dense_session))
        # allowed = CAP_FACTOR(3) * rate(0.1) * units(0.6) = 0.18
        self.assertAlmostEqual(tiny_row["drain_percent"], 0.18, places=6)
        self.assertLessEqual(tiny_row["drain_percent"], 3.0 * 0.1 * 0.6 + 1e-9)
        self.assertAlmostEqual(dense_row["drain_percent"], 3.0, places=6)
        # share_of_window's denominator stays interval.drain (29.0), not the
        # sum of what was actually attributed (0.18): the gap between them
        # IS the unattributed figure, not renormalized away.
        self.assertAlmostEqual(
            tiny_row["share_of_window_percent"], 100.0 * 0.18 / 29.0, places=4
        )
        pools = payload["pools"]
        jump_pool = max(pools, key=lambda item: item["peak"])
        self.assertAlmostEqual(jump_pool["peak"], 29.0, places=6)
        self.assertAlmostEqual(jump_pool["unattributed"], 29.0 - 0.18, places=6)

        text = self.run_tool("sessions", "--harness", "codex")
        self.assertEqual(text.returncode, 0, text.stderr.decode("utf-8", "replace"))
        self.assertIn(b"unattributed: ", text.stdout)
        self.assertIn(b"clients not in the scanned roots", text.stdout)

        windows_text = self.run_tool("windows", "--harness", "codex")
        self.assertEqual(windows_text.returncode, 0)
        self.assertIn(b"unattributed 28.8%", windows_text.stdout)

    def test_raw_token_fallback_is_exempt_from_the_cap(self) -> None:
        # A model with no weight prices at ~1e-9 units/token, nowhere near
        # the weighted-unit scale the cap is calibrated against; capping it
        # would zero out a session that really did the work.
        base = int(time.time()) - 4 * 3600
        dense_session = "dddddddd-dense-1111-1111-111111111111"
        resets_a = base + 7200
        self.write_dense_phase(base, dense_session, resets_a)

        tiny_start = base + 3600
        opened = tiny_start - 60
        resets_b = opened + 18000
        exotic_session = "eeeeeeee-exotic-000-000-000000000000"
        self.write_jump(tiny_start, exotic_session, "gpt-9000-nonexistent", 6_000, resets_b, 29.0)

        payload = self.run_json("sessions", "--harness", "codex", "--json", "--top", "20")
        exotic_row = self.session_by_short_id(payload, QD.short_id(exotic_session))
        # Uncapped: the sole session in the interval gets its whole drain.
        self.assertAlmostEqual(exotic_row["drain_percent"], 29.0, places=6)

    def test_rate_fallback_with_too_little_coverage_warns_once(self) -> None:
        # Only one non-rollover interval - below MIN_RATE_INTERVALS - and no
        # calibrated or fitted rate on disk: the pool is left uncapped and
        # warns exactly once.
        base = int(time.time()) - 4 * 3600
        dense_session = "dddddddd-sparse-00-0000-000000000000"
        resets_a = base + 7200
        thin_lines = [
            codex_session_meta_line(base, dense_session, "/home/agent/thin"),
            codex_turn_context_line(base, "gpt-5.6-sol"),
            codex_token_count_line(base + 5, rate_limits=rate_limits(10.0, resets_a)),
            codex_usage_record_line(
                base + 10, dense_session, input_tokens=100_000, cached_input_tokens=0,
                output_tokens=0
            ),
            codex_token_count_line(base + 15, rate_limits=rate_limits(11.0, resets_a)),
        ]
        self.write_codex("rollout-thin.jsonl", thin_lines)

        tiny_start = base + 3600
        opened = tiny_start - 60
        resets_b = opened + 18000
        tiny_session = "tttttttt-fallback-0-0000-000000000000"
        self.write_jump(tiny_start, tiny_session, "gpt-5.6-sol", 6_000, resets_b, 29.0)

        result = self.run_tool("sessions", "--harness", "codex", "--json", "--top", "20")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        payload = json.loads(result.stdout.decode("utf-8"))
        tiny_row = self.session_by_short_id(payload, QD.short_id(tiny_session))
        # Uncapped: not enough non-rollover coverage to trust a measured rate.
        self.assertAlmostEqual(tiny_row["drain_percent"], 29.0, places=6)
        stderr = result.stderr.decode("utf-8", "replace")
        self.assertEqual(stderr.count("no attribution cap"), 1)

    def test_zero_peak_windows_with_drifting_resets_at_collapse(self) -> None:
        # #17: an idle pool re-stamps resets_at by roughly a minute per
        # reading; minute-rounding alone still yields one window per
        # reading. Clustering resets_at within tolerance collapses the
        # whole drifting, always-zero run into one window.
        base = int(time.time()) - 4 * 3600
        session = "iiiiiiii-idle0-0-0000-000000000000"
        lines = [
            codex_session_meta_line(base, session, "/home/agent/idle"),
            codex_turn_context_line(base, "gpt-5.6-sol"),
        ]
        resets = base + 604800
        for index in range(12):
            lines.append(
                codex_token_count_line(
                    base + index * 70, rate_limits=rate_limits(0.0, resets + index * 70)
                )
            )
        self.write_codex("rollout-idle.jsonl", lines)
        payload = self.run_json("windows", "--harness", "codex", "--json")
        self.assertLessEqual(len(payload["windows"]), 1)
        if payload["windows"]:
            self.assertAlmostEqual(payload["windows"][0]["peak_used_percent"], 0.0, places=6)

    def test_prompt_drain_respects_the_session_cap(self) -> None:
        # Follow-on to #16: `attribute()` caps a session's INTERVAL share,
        # but `apply_prompt_drain`'s per-prompt split still multiplied the
        # interval's raw drain, so `prompts --session X` kept showing the
        # uncapped figure even after `sessions` capped it. Two prompts split
        # a capped 0.18% share 2/3-1/3 by their own tokens - never 29.0%.
        base = int(time.time()) - 4 * 3600
        dense_session = "dddddddd-dense-2222-2222-222222222222"
        resets_a = base + 7200
        self.write_dense_phase(base, dense_session, resets_a)

        tiny_start = base + 3600
        opened = tiny_start - 60
        resets_b = opened + 18000
        tiny_session = "tttttttt-tinyp-0000-0000-000000000000"
        self.write_codex(
            "rollout-tiny-prompts.jsonl",
            [
                codex_session_meta_line(tiny_start, tiny_session, "/home/agent/tiny"),
                codex_task_started_line(tiny_start),
                codex_turn_context_line(tiny_start, "gpt-5.6-sol"),
                codex_usage_record_line(
                    tiny_start + 5, tiny_session, input_tokens=4_000,
                    cached_input_tokens=0, output_tokens=0
                ),
                codex_task_started_line(tiny_start + 30),
                codex_turn_context_line(tiny_start + 30, "gpt-5.6-sol"),
                codex_usage_record_line(
                    tiny_start + 35, tiny_session, input_tokens=2_000,
                    cached_input_tokens=0, output_tokens=0
                ),
                codex_token_count_line(tiny_start + 40, rate_limits=rate_limits(29.0, resets_b)),
            ],
        )

        payload = self.run_json("prompts", "--session", QD.short_id(tiny_session), "--json")
        shares = [prompt["drain_percent"] for prompt in payload["prompts"]]
        self.assertEqual(len(shares), 2)
        # allowed = CAP_FACTOR(3) * rate(0.1) * units(0.6) = 0.18, split
        # 4000:2000 by each prompt's own tokens.
        self.assertAlmostEqual(sum(shares), 0.18, places=6)
        self.assertAlmostEqual(shares[0], 0.12, places=6)
        self.assertAlmostEqual(shares[1], 0.06, places=6)

    def test_interval_with_no_local_sessions_is_fully_unattributed(self) -> None:
        # A pool can take a reading with no local session active at all (a
        # client nas cannot see did the work). `attribute()`'s early
        # per-interval bailout (total <= 0) must still record the whole
        # drain as unattributed, or sum(shares) + unattributed == drain
        # breaks for that interval.
        base = int(time.time()) - 4 * 3600
        dense_session = "dddddddd-dense-3333-3333-333333333333"
        resets_a = base + 7200
        self.write_dense_phase(base, dense_session, resets_a)

        ghost_start = base + 3600
        opened = ghost_start - 60
        resets_b = opened + 18000
        self.write_codex(
            "rollout-ghost.jsonl",
            [codex_token_count_line(ghost_start + 10, rate_limits=rate_limits(40.0, resets_b))],
        )

        payload = self.run_json("sessions", "--harness", "codex", "--json", "--top", "20")
        pools = payload["pools"]
        ghost_pool = max(pools, key=lambda item: item["peak"])
        self.assertAlmostEqual(ghost_pool["peak"], 40.0, places=6)
        self.assertAlmostEqual(ghost_pool["attributed"], 0.0, places=6)
        self.assertAlmostEqual(ghost_pool["unattributed"], 40.0, places=6)

        text = self.run_tool("sessions", "--harness", "codex")
        self.assertEqual(text.returncode, 0, text.stderr.decode("utf-8", "replace"))
        self.assertIn(b"unattributed: ", text.stdout)

    def test_exempt_session_does_not_absorb_a_capped_sessions_overflow(self) -> None:
        # #16 could still reproduce if the tiny session ran an unweighted
        # (raw-token-fallback) model: water_fill used to let any session
        # with `allowed is None` sit in the unclamped pool, so it absorbed
        # every clamped session's freed drain instead of `unattributed`.
        base = int(time.time()) - 4 * 3600
        dense_session = "dddddddd-dense-4444-4444-444444444444"
        resets_a = base + 7200
        self.write_dense_phase(base, dense_session, resets_a)

        tiny_start = base + 3600
        opened = tiny_start - 60
        resets_b = opened + 18000
        capped_session = "cccccccc-capped-0000-0000-00000000000"
        exempt_session = "eeeeeeee-exempt-0000-0000-00000000000"
        self.write_codex(
            "rollout-mixed.jsonl",
            [
                codex_session_meta_line(tiny_start, capped_session, "/home/agent/capped"),
                codex_turn_context_line(tiny_start, "gpt-5.6-sol"),
                codex_usage_record_line(
                    tiny_start + 5, capped_session, input_tokens=6_000,
                    cached_input_tokens=0, output_tokens=0
                ),
                codex_session_meta_line(tiny_start + 6, exempt_session, "/home/agent/exempt"),
                codex_turn_context_line(tiny_start + 6, "gpt-9000-nonexistent"),
                codex_usage_record_line(
                    tiny_start + 10, exempt_session, input_tokens=6_000,
                    cached_input_tokens=0, output_tokens=0
                ),
                codex_token_count_line(tiny_start + 15, rate_limits=rate_limits(29.0, resets_b)),
            ],
        )

        payload = self.run_json("sessions", "--harness", "codex", "--json", "--top", "20")
        capped_row = self.session_by_short_id(payload, QD.short_id(capped_session))
        exempt_row = self.session_by_short_id(payload, QD.short_id(exempt_session))
        # capped_session's raw share is 29.0 * 0.6/(0.6 + exempt_units); its
        # cap is 0.18. exempt_session keeps its own raw proportional share
        # (never touched by water-fill); the capped session's freed drain
        # becomes unattributed, not a top-up for the exempt session.
        self.assertAlmostEqual(capped_row["drain_percent"], 0.18, places=6)
        exempt_units = 6_000 * 1e-9
        exempt_raw_share = 29.0 * exempt_units / (0.6 + exempt_units)
        self.assertAlmostEqual(exempt_row["drain_percent"], exempt_raw_share, places=6)

        pools = payload["pools"]
        jump_pool = max(pools, key=lambda item: item["peak"])
        self.assertAlmostEqual(
            jump_pool["unattributed"], 29.0 - 0.18 - exempt_raw_share, places=6
        )

    def test_no_codex_events_at_all_leaves_every_interval_fully_unattributed(self) -> None:
        # `attribute()` used to `return` before the per-interval loop when
        # there were no Codex usage events at all (not just none active in
        # a given interval), leaving `interval.unattributed` at its 0.0
        # default even though real drain was measured. That breaks
        # sum(shares) + unattributed == drain for every interval.
        intervals = [
            QD.Interval(("acct", "lim", "plan", 300), 0.0, 100.0, 12.5, None, False),
            QD.Interval(("acct", "lim", "plan", 300), 100.0, 200.0, 7.0, None, False),
        ]
        weights = QD.Weights({}, [])
        args = argparse.Namespace(long_context_multiplier=1.0, claude_cache_read_weight=None)
        QD.attribute(intervals, [], weights, args)
        for interval in intervals:
            self.assertAlmostEqual(interval.unattributed, interval.drain, places=6)
            self.assertEqual(interval.sessions, {})

    def test_prompt_shares_sum_to_the_whole_session_share(self) -> None:
        # Only main-thread rows carry a prompt key (assigned by
        # `assemble_prompts` before `attribute()` runs); a row that never
        # joined a prompt group still counts toward the session's `shares`
        # total. Dividing a prompt's units by the session's ALL-EVENT units
        # (including that unkeyed row) used to leave the prompt's share
        # short of the session's own attributed share. The denominator must
        # be the session's PROMPT-KEYED units instead.
        interval = QD.Interval(("acct", "lim", "plan", 300), 0.0, 100.0, 10.0, None, False)
        session_id = "sess-1"
        # event without a prompt key (e.g. a sub-thread row outside any
        # prompt group): row length 13, no EVENT_PROMPT slot at all.
        no_prompt_event = [session_id, "m", 10.0, 100_000, 0, 0, 0, 0, 0, 0, "", "", ""]
        # event with a prompt key: row length 14, EVENT_PROMPT set.
        with_prompt_event = (
            [session_id, "m", 20.0, 200_000, 0, 0, 0, 0, 0, 0, "", "", ""] + [1]
        )
        self.assertEqual(len(no_prompt_event), QD.EVENT_PROMPT)
        self.assertEqual(len(with_prompt_event), QD.EVENT_PROMPT + 1)
        weights = QD.Weights({"codex": {"models": {"m": {"input": 1.0}}}}, ["test"])
        args = argparse.Namespace(long_context_multiplier=1.0, claude_cache_read_weight=None)
        QD.attribute([interval], [no_prompt_event, with_prompt_event], weights, args)
        session_share = interval.sessions[session_id]
        self.assertAlmostEqual(session_share, 10.0, places=6)
        prompt_total = sum(
            value for key, value in interval.prompts.items() if key[0] == session_id
        )
        self.assertAlmostEqual(prompt_total, session_share, places=6)


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
        self.assertEqual(payload["schema"], 1)
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

    def test_codex_calibrate_json_names_its_command(self) -> None:
        """Every other --json payload carries `command`; codex calibrate must too."""
        self.build_rollouts()
        payload = self.run_json(
            "calibrate", "--harness", "codex", "--json", "--calibrate-bucket-hours", "0.01"
        )
        self.assertEqual(payload["command"], "calibrate")
        self.assertEqual(payload["harness"], "codex")

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
        self.assertIn(b"stored fit is not usable", result.stderr)
        self.assertNotIn("calibrated", json.loads(result.stdout)["weight_source"])

    def test_calibrated_table_is_all_one_unit(self) -> None:
        self.build_rollouts()
        self.run_json(
            "calibrate", "--harness", "codex", "--json", "--calibrate-bucket-hours", "0.01"
        )
        fit_file = self.root / "state" / "codex-weights.json"
        stored = json.loads(fit_file.read_text(encoding="utf-8"))
        self.assertGreater(stored["fallback_scale"], 0.0)
        self.assertTrue(stored["usable"])
        with self.env_applied():
            calibrated = QD.load_weights(True)
            default = QD.load_weights(False)
        self.assertIn("calibrated", calibrated.source_label)
        self.assertEqual(calibrated.table["codex"]["unit"], "percent_per_mtok")
        # gpt-6-astra never ran in the fixture, so it has no fitted
        # coefficient; it must still be priced on the fitted scale, not left
        # in rate-card credit units.
        astra = calibrated.model_entry("codex", "gpt-6-astra")
        rate_card = default.model_entry("codex", "gpt-6-astra")
        self.assertIsNotNone(astra)
        self.assertLess(astra["input"], rate_card["input"])
        self.assertAlmostEqual(
            astra["input"] / rate_card["input"], stored["fallback_scale"], places=9
        )
        fitted_sol = calibrated.model_entry("codex", "gpt-5.6-sol")["input"]
        # Both models now sit within a couple of orders of magnitude of each
        # other instead of differing by the unit mismatch.
        self.assertLess(max(fitted_sol, astra["input"]) / min(fitted_sol, astra["input"]), 1000.0)

    def test_zero_weighted_model_still_receives_attribution(self) -> None:
        config = self.root / "config" / "weights.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps(
                {
                    "codex": {
                        "unit": "credit_units_per_mtok",
                        "models": {
                            "gpt-5.6-terra": {"input": 0.0, "cached_input": 0.0,
                                              "output": 0.0}
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        now = time.time() - 3600
        resets_at = int(time.time()) + 7200
        self.write_codex(
            "rollout-zero.jsonl",
            [
                codex_session_meta_line(now, "codex-zero-0001", "/home/agent/zero"),
                codex_turn_context_line(now, "gpt-5.6-terra"),
                codex_token_count_line(now + 5, rate_limits=rate_limits(10.0, resets_at)),
                codex_usage_record_line(
                    now + 10, "codex-zero-0001", input_tokens=200_000,
                    cached_input_tokens=0, output_tokens=0
                ),
                codex_token_count_line(now + 30, rate_limits=rate_limits(14.0, resets_at)),
            ],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        row = payload["sessions"][0]
        self.assertEqual(row["weighted_units"], 0.0)
        # A model priced at zero is not a model that drained nothing.
        self.assertAlmostEqual(row["drain_percent"], 4.0, places=6)

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
            "cwd_hash",
            "fork_of",
            "duplicate_turns",
        ):
            self.assertIn(key, row)
        # The full working directory is customer-identifying and must not
        # reach the report or the cache.
        self.assertEqual(row["cwd"], "project")
        self.assertNotIn("/", row["cwd"])
        cached = ""
        for path in (self.root / "cache").rglob("*.json"):
            cached += path.read_text(encoding="utf-8")
        self.assertNotIn("/home/agent/project", cached)

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

    def test_table_fits_120_columns(self) -> None:
        now = time.time() - 600
        self.write_claude(
            "width.jsonl",
            [
                claude_assistant_line(
                    now, "eeee0001-1111-2222-3333-444444444444", "msg_w",
                    input_tokens=1_234_567, cache_read=98_765_432,
                    cache_write_5m=12_345, output_tokens=654_321
                )
            ],
        )
        result = self.run_tool("sessions", "--no-color", "--width", "120")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        for line in result.stdout.decode("utf-8").splitlines():
            self.assertLessEqual(len(line), 120, line)

    def test_ascii_falls_back_when_asked(self) -> None:
        now = time.time() - 600
        self.write_claude(
            "ascii.jsonl",
            [
                claude_assistant_line(
                    now, "eeee0002-1111-2222-3333-444444444444", "msg_a",
                    output_tokens=1000
                )
            ],
        )
        result = self.run_tool("sessions", "--no-color", "--ascii", "--width", "150")
        text = result.stdout.decode("utf-8")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("\u2588", text)
        self.assertIn("#", text)
        # fanout draws its own histograms; they go through the painter too.
        fanout = self.run_tool("fanout", "--no-color", "--ascii", "--width", "150")
        self.assertEqual(fanout.returncode, 0, fanout.stderr.decode("utf-8", "replace"))
        drawn = fanout.stdout.decode("utf-8")
        self.assertNotIn("\u2588", drawn)
        self.assertIn("#", drawn)

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

    def test_stdin_appends_only_on_change(self) -> None:
        payload = json.dumps(
            {"rate_limits": {"five_hour": {"used_percentage": 42.5,
                                           "resets_at": "2026-09-16T21:30:00Z"}}}
        ).encode("utf-8")
        for _ in range(3):
            self.assertEqual(self.run_tool("snapshot", "--stdin", stdin=payload).stdout,
                             payload)
        moved = json.dumps(
            {"rate_limits": {"five_hour": {"used_percentage": 44.0,
                                           "resets_at": "2026-09-16T21:30:00Z"}}}
        ).encode("utf-8")
        self.run_tool("snapshot", "--stdin", stdin=moved)
        logged = (self.root / "state" / "snapshots.jsonl").read_text(encoding="utf-8")
        self.assertEqual(len(logged.strip().splitlines()), 2)

    def _statusline_payload(self, used_percentage: float) -> bytes:
        return json.dumps(
            {"rate_limits": {"five_hour": {"used_percentage": used_percentage,
                                           "resets_at": "2026-09-16T21:30:00Z"}}}
        ).encode("utf-8")

    def test_statusline_dedup_is_per_config_dir(self) -> None:
        # Dedup that ignores the account (#9): comparing only against the
        # newest record regardless of `config_dir` drops a genuine change
        # from one account whenever it coincides with the other account's
        # last-seen value. A: 10% -> B: 30% -> A: 30% (a REAL change for A,
        # from 10% to 30%, that happens to equal B's last reading) -> B: 10%
        # (a real change for B). All four are distinct per-account readings
        # and must all be kept.
        claude_root = self.home / ".claude"
        arcade_root = self.home / ".claude-arcade"
        arcade_root.mkdir(parents=True, exist_ok=True)
        sequence = [
            (claude_root, 10.0),
            (arcade_root, 30.0),
            (claude_root, 30.0),
            (arcade_root, 10.0),
        ]
        for root, used in sequence:
            result = self.run_tool(
                "snapshot", "--stdin", stdin=self._statusline_payload(used),
                extra_env={"CLAUDE_CONFIG_DIR": str(root)},
            )
            self.assertEqual(result.returncode, 0)
        logged = (self.root / "state" / "snapshots.jsonl").read_text(encoding="utf-8")
        records = [json.loads(line) for line in logged.strip().splitlines()]
        self.assertEqual(len(records), len(sequence))
        seen = [
            (record["config_dir"], record["windows"]["five_hour"]["utilization_percent"])
            for record in records
        ]
        self.assertEqual(
            seen,
            [(".claude", 10.0), (".claude-arcade", 30.0), (".claude", 30.0),
             (".claude-arcade", 10.0)],
        )

    def test_statusline_label_falls_back_to_first_resolved_root(self) -> None:
        # With no $CLAUDE_CONFIG_DIR, the fallback label must come from the
        # first resolved Claude root, not a hardcoded ".claude" - here the
        # only configured root is renamed, so the record must carry that
        # root's own basename.
        renamed = self.home / ".claude-only"
        renamed.mkdir(parents=True, exist_ok=True)
        config_path = self.root / "config" / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            '[claude]\nroots = ["%s"]\n' % str(renamed).replace("\\", "\\\\"),
            encoding="utf-8",
        )
        result = self.run_tool("snapshot", "--stdin", stdin=self._statusline_payload(12.0))
        self.assertEqual(result.returncode, 0)
        logged = (self.root / "state" / "snapshots.jsonl").read_text(encoding="utf-8")
        record = json.loads(logged.strip().splitlines()[-1])
        self.assertEqual(record["config_dir"], ".claude-only")

    def test_compact_drops_repeats_and_stale_entries(self) -> None:
        path = self.root / "state" / "snapshots.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        records = [
            {"source": "oauth", "config_dir": ".claude", "ts": now - 90 * 86400,
             "windows": {"five_hour": {"utilization_percent": 1.0}}},
            {"source": "oauth", "config_dir": ".claude", "ts": now - 3600,
             "windows": {"five_hour": {"utilization_percent": 5.0}}},
            {"source": "oauth", "config_dir": ".claude", "ts": now - 1800,
             "windows": {"five_hour": {"utilization_percent": 5.0}}},
            {"source": "oauth", "config_dir": ".claude", "ts": now - 900,
             "windows": {"five_hour": {"utilization_percent": 7.0}}},
        ]
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        result = self.run_tool("snapshot", "--compact")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"kept 2", result.stdout)
        kept = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").strip().splitlines()
        ]
        self.assertEqual(
            [record["windows"]["five_hour"]["utilization_percent"] for record in kept],
            [5.0, 7.0],
        )

    def test_version_comes_from_a_parsed_field(self) -> None:
        now = time.time() - 600
        self.write_claude(
            "ver.jsonl",
            [
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": "eeee0003-1111-2222-3333-444444444444",
                        "timestamp": iso(now),
                        "message": {"role": "user",
                                    "content": 'pasted text with "version":"9.9.9" inside'},
                    }
                ),
                claude_assistant_line(
                    now + 1, "eeee0003-1111-2222-3333-444444444444", "msg_v",
                    output_tokens=10
                ),
            ],
        )
        with self.env_applied():
            version = QD.detect_claude_version()
        # The assistant fixture declares 2.1.273; the pasted string must lose.
        self.assertEqual(version, "2.1.273")

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

    def test_reductions_json_carries_the_full_session_id(self) -> None:
        self.codex_run(
            "codex-red-0003",
            [100_000, 200_000, 300_000, 40_000, 45_000, 50_000, 55_000],
            compact_after=3,
        )
        payload = self.run_json("reductions", "--harness", "codex", "--json")
        row = payload["reductions"][0]
        self.assertEqual(row["session_id"], "codex-red-0003")
        self.assertEqual(row["short_id"], QD.short_id("codex-red-0003"))

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
               status: Optional[int] = None, redirect_to: Optional[str] = None
               ) -> List[Dict[str, Any]]:
        import io
        import urllib.error
        import urllib.request

        captured = {}
        harness = self

        class FakeResponse(io.BytesIO):
            def close(self_inner):
                io.BytesIO.close(self_inner)

        class FakeOpener(object):
            """Stands in for the private opener the sampler builds."""

            def __init__(self_inner, *handlers):
                self_inner.handlers = handlers

            def open(self_inner, request, timeout=None):
                captured["url"] = request.full_url
                captured["headers"] = dict(request.header_items())
                captured["timeout"] = timeout
                captured["handlers"] = self_inner.handlers
                if redirect_to is not None:
                    # Exercise the installed handler rather than asserting on
                    # its presence: a redirect must never be followed while
                    # the request carries a bearer token.
                    for handler in self_inner.handlers:
                        if isinstance(handler, urllib.request.HTTPRedirectHandler):
                            handler.redirect_request(
                                request, None, 302, "Found",
                                {"Location": redirect_to}, redirect_to
                            )
                    raise AssertionError("redirect was not refused")
                if status is not None:
                    raise urllib.error.HTTPError(
                        request.full_url, status, "rate limited", {}, None
                    )
                body = json.dumps(response if response is not None else harness.RESPONSE)
                return FakeResponse(body.encode("utf-8"))

        original = QD.build_oauth_opener
        QD.build_oauth_opener = lambda: FakeOpener(*QD.build_oauth_opener_handlers())
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            with self.env_applied():
                QD.main(list(args_list))
        finally:
            self.messages = sys.stderr.getvalue()
            sys.stdout = stdout
            sys.stderr = stderr
            QD.build_oauth_opener = original
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

    def test_redirects_are_refused(self) -> None:
        self.write_credentials()
        rows = self.sample(["snapshot", "--oauth"], redirect_to="https://evil.example/usage")
        self.assertEqual(rows, [])
        self.assertIn("usage request failed", self.messages)

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
        # Fresh and cached input vary independently; a fixture where they
        # always sum to the same total is collinear and correctly refused.
        mixes = [
            (900_000, 100_000), (900_000, 900_000), (100_000, 100_000),
            (100_000, 900_000), (500_000, 200_000), (200_000, 500_000),
            (800_000, 400_000), (400_000, 800_000), (600_000, 600_000),
            (300_000, 100_000), (100_000, 300_000), (700_000, 900_000),
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
        self.assertTrue(payload["windows"]["five_hour"]["usable"])
        fitted = payload["windows"]["five_hour"]["models"]["claude-opus-5"]
        self.assertLess(abs(fitted["input"] - input_rate) / input_rate, 0.05)
        self.assertLess(abs(fitted["cache_read"] - cache_rate) / cache_rate, 0.05)
        self.assertGreater(payload["windows"]["five_hour"]["r_squared"], 0.99)

    def test_iso_resets_at_jitter_is_not_a_rollover(self) -> None:
        # Claude stamps resets_at with fresh microseconds on every poll; a raw
        # equality test reads each one as a new window and reports the whole
        # utilisation as drain.
        now = time.time() - 3600
        path = self.root / "state" / "snapshots.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for index, (offset, percent, micro) in enumerate(
                ((0, 18.0, "104305"), (600, 20.0, "310581"), (1200, 23.0, "887001"))
            ):
                handle.write(
                    json.dumps(
                        {
                            "source": "oauth",
                            "config_dir": ".claude",
                            "ts": now + offset,
                            "windows": {
                                "five_hour": {
                                    "utilization_percent": percent,
                                    "resets_at": "2026-09-16T21:30:00.%s+00:00" % micro,
                                }
                            },
                        }
                    )
                    + "\n"
                )
        with self.env_applied():
            rows = QD.load_claude_snapshots("five_hour")
        intervals = QD.build_intervals(rows, QD.ANY_WINDOW)
        self.assertEqual(len(intervals), 2)
        self.assertEqual([interval.rollover for interval in intervals], [False, False])
        self.assertEqual([interval.drain for interval in intervals], [2.0, 3.0])

    def write_claude_snapshots(self, rows: Sequence[Tuple[float, float, str]]) -> None:
        path = self.root / "state" / "snapshots.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for ts, percent, resets_at in rows:
                handle.write(
                    json.dumps(
                        {
                            "source": "oauth",
                            "config_dir": ".claude",
                            "ts": ts,
                            "windows": {
                                "five_hour": {
                                    "utilization_percent": percent,
                                    "resets_at": resets_at,
                                }
                            },
                        }
                    )
                    + "\n"
                )

    def test_claude_rollover_with_a_rise_stays_a_rollover(self) -> None:
        # build_intervals also serves the Claude path (calibrate_claude, over
        # load_claude_snapshots rows with ANY_WINDOW): a rise across a bucket
        # change must still be classified as a real rollover (rule a) - the
        # narrowing of rule (b) to "no drain evidence" must not swallow this
        # case, the same shape as test_rollover_when_resets_at_changes for
        # Codex.
        now = time.time() - 4 * 3600
        window_a = now + 200
        window_b = window_a + 18000  # a genuinely new five-hour window
        self.write_claude_snapshots(
            [
                (now, 10.0, iso(window_a)),
                (now + 600, 10.0, iso(window_a)),  # same window, unchanged: no interval
                (now + 1200, 14.0, iso(window_b)),  # new window, a rise: rollover
            ]
        )
        with self.env_applied():
            rows = QD.load_claude_snapshots("five_hour")
        intervals = QD.build_intervals(rows, QD.ANY_WINDOW)
        self.assertEqual(len(intervals), 1)
        self.assertTrue(intervals[0].rollover)
        self.assertAlmostEqual(intervals[0].drain, 14.0, places=6)

    def test_claude_same_bucket_decrease_then_rise(self) -> None:
        # Same shape as the Codex decrease test: the interval starts at the
        # low reading, not at 0 and not measured back from the high reading.
        now = time.time() - 4 * 3600
        window_a = now + 200
        self.write_claude_snapshots(
            [
                (now, 40.0, iso(window_a)),
                (now + 600, 30.0, iso(window_a)),
                (now + 1200, 35.0, iso(window_a)),
            ]
        )
        with self.env_applied():
            rows = QD.load_claude_snapshots("five_hour")
        intervals = QD.build_intervals(rows, QD.ANY_WINDOW)
        self.assertEqual(len(intervals), 1)
        self.assertFalse(intervals[0].rollover)
        self.assertAlmostEqual(intervals[0].drain, 5.0, places=6)
        self.assertEqual(intervals[0].start, now + 600)

    def test_collinear_models_are_flagged_unidentified(self) -> None:
        # Two models that only ever run together cannot be told apart, and a
        # confident zero for either would price it as free.
        now = time.time() - 6 * 3600
        session = "abcd5555-aaaa-2222-3333-444444444444"
        lines = [claude_user_prompt_line(now, session)]
        samples = [(now, 0.0)]
        used = 0.0
        for index in range(12):
            stamp = now + (index + 1) * 120
            volume = 100_000 * (index + 1)
            lines.append(
                claude_assistant_line(
                    stamp - 70, session, "msg_a%d" % index, model="claude-opus-5",
                    input_tokens=volume, output_tokens=0
                )
            )
            lines.append(
                claude_assistant_line(
                    stamp - 60, session, "msg_b%d" % index, model="claude-sonnet-5",
                    input_tokens=volume, output_tokens=0
                )
            )
            used += volume * 8.0 / 1_000_000.0
            samples.append((stamp, used))
        self.write_claude("collinear.jsonl", lines)
        self.write_snapshots(samples)
        payload = self.run_json(
            "calibrate", "--harness", "claude", "--json", "--calibrate-bucket-hours", "0.01"
        )
        result = payload["windows"]["five_hour"]
        for model in ("claude-opus-5", "claude-sonnet-5"):
            note = result["diagnostics"][model]["input"]
            self.assertTrue(note["unidentified"], note)
            self.assertIn("collinear", note["reasons"])
            self.assertIsNone(result["models"][model]["input"])

    def test_a_null_weight_falls_back_to_the_list_price(self) -> None:
        # A calibrated table stores null for a coefficient the fit could not
        # identify; pricing it as zero would make that kind free.
        weights = QD.Weights(
            {"claude": {"models": {"claude-opus-5": {"input": 3.0, "cache_read": None}}}},
            ["test"],
        )
        units = weights.claude_units(
            "claude-opus-5", {"input": 1_000_000, "cache_read": 1_000_000}, None
        )
        self.assertAlmostEqual(units, 3.0 + 0.5)

    def test_calibrate_without_snapshots_explains_itself(self) -> None:
        result = self.run_tool("calibrate", "--harness", "claude")
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"snapshot --oauth", result.stderr)


class CorpusDedup(Harness):
    def test_same_message_id_under_two_sessions_counts_once(self) -> None:
        now = time.time() - 3600
        original = "aaaa0001-1111-2222-3333-444444444444"
        fork = "bbbb0002-1111-2222-3333-444444444444"
        shared = [
            claude_assistant_line(now + index, original, "msg_shared%d" % index,
                                  output_tokens=100)
            for index in range(3)
        ]
        first = self.write_claude("original.jsonl", shared)
        # A forked session replays the original's assistant lines verbatim
        # under a new sessionId, then continues with its own.
        replay = [line.replace(original, fork) for line in shared]
        replay.append(claude_assistant_line(now + 10, fork, "msg_own", output_tokens=70))
        second = self.write_claude("fork.jsonl", replay)
        os.utime(first, (now, now))
        os.utime(second, (now + 60, now + 60))

        payload = self.run_json("sessions", "--harness", "claude", "--json")
        rows = dict((row["short_id"], row) for row in payload["sessions"])
        self.assertEqual(rows["aaaa0001"]["tokens"]["output"], 300)
        self.assertEqual(rows["bbbb0002"]["tokens"]["output"], 70)
        self.assertEqual(rows["bbbb0002"]["fork_of"], "aaaa0001")
        self.assertEqual(rows["bbbb0002"]["duplicate_turns"], 3)
        self.assertEqual(rows["aaaa0001"]["fork_of"], "")

    def test_claude_share_uses_the_deduped_total(self) -> None:
        now = time.time() - 3600
        original = "aaaa0003-1111-2222-3333-444444444444"
        fork = "bbbb0004-1111-2222-3333-444444444444"
        shared = [
            claude_assistant_line(now, original, "msg_s1", output_tokens=1000)
        ]
        first = self.write_claude("orig2.jsonl", shared)
        second = self.write_claude(
            "fork2.jsonl", [line.replace(original, fork) for line in shared]
        )
        os.utime(first, (now, now))
        os.utime(second, (now + 60, now + 60))
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        # The replay contributes nothing, so the original holds the whole share.
        rows = dict((row["short_id"], row) for row in payload["sessions"])
        self.assertEqual(rows["aaaa0003"]["share_of_window_percent"], 100.0)
        self.assertNotIn("bbbb0004", rows)

    def test_codex_replayed_response_ids_count_once(self) -> None:
        now = time.time() - 3600
        session = "codex-dedup-0001"
        calls = [
            codex_usage_record_line(
                now + 10, session, input_tokens=50_000, cached_input_tokens=0,
                output_tokens=100, turn_id="t0"
            )
        ]
        first = self.write_codex(
            "rollout-orig.jsonl",
            [codex_session_meta_line(now, session, "/home/agent/dd")] + calls,
        )
        second = self.write_codex(
            "rollout-resume.jsonl",
            [codex_session_meta_line(now + 1, session, "/home/agent/dd")] + calls,
        )
        os.utime(first, (now, now))
        os.utime(second, (now + 60, now + 60))
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        self.assertEqual(len(payload["sessions"]), 1)
        self.assertEqual(payload["sessions"][0]["tokens"]["input"], 50_000)


class RangeWindowing(Harness):
    def build_two_event_session(self) -> str:
        session = "cccc0001-1111-2222-3333-444444444444"
        old = time.time() - 30 * 86400
        recent = time.time() - 600
        self.write_claude(
            "span.jsonl",
            [
                claude_user_prompt_line(old, session),
                claude_assistant_line(old, session, "msg_old", output_tokens=1_000_000),
                claude_user_prompt_line(recent, session),
                claude_assistant_line(recent, session, "msg_new", output_tokens=1_000),
            ],
        )
        return session

    def test_sessions_totals_are_windowed(self) -> None:
        self.build_two_event_session()
        payload = self.run_json("sessions", "--json", "--since", "3d")
        row = payload["sessions"][0]
        self.assertEqual(row["tokens"]["output"], 1_000)
        self.assertEqual(row["requests"], 1)

    def test_whole_session_opts_back_in(self) -> None:
        self.build_two_event_session()
        payload = self.run_json("sessions", "--json", "--since", "3d", "--whole-session")
        row = payload["sessions"][0]
        self.assertEqual(row["tokens"]["output"], 1_001_000)
        # The fan-out columns must describe the same span as the tokens.
        self.assertEqual(row["prompts"], 2)
        self.assertEqual(row["api_turns"], 2)

    def test_prompts_and_fanout_are_windowed(self) -> None:
        self.build_two_event_session()
        prompts = self.run_json("prompts", "--session", "cccc0001", "--json", "--since", "3d")
        self.assertEqual(len(prompts["prompts"]), 1)
        fanout = self.run_json("fanout", "--json", "--since", "3d")
        self.assertEqual(fanout["prompts"], 1)

    def test_timeline_and_sessions_agree_on_the_range(self) -> None:
        self.build_two_event_session()
        sessions = self.run_json("sessions", "--json", "--since", "3d")
        timeline = self.run_json("timeline", "--json", "--since", "3d")
        session_units = sum(row["weighted_units"] for row in sessions["sessions"])
        timeline_units = sum(
            bucket["claude"] + bucket["codex"] for bucket in timeline["buckets"]
        )
        self.assertAlmostEqual(session_units, timeline_units, places=6)


class SessionSelection(Harness):
    PREFIX = "dddd000"

    def build_two_matching_sessions(self) -> Tuple[str, str]:
        # `second` starts with the whole of `first`, so the full id of `first`
        # is both an exact match and a prefix of another session.
        now = time.time() - 1800
        first = "dddd0001-1111-2222"
        second = "dddd0001-1111-2222-3333-444444444444"
        for name, session, output in (
            ("busy.jsonl", first, 100_000), ("quiet.jsonl", second, 10),
        ):
            self.write_claude(
                name,
                [
                    claude_user_prompt_line(now, session),
                    claude_assistant_line(now + 1, session, "msg_" + name,
                                          output_tokens=output),
                ],
            )
        return first, second

    def test_ambiguous_session_prefix_fails_with_the_candidates(self) -> None:
        first, second = self.build_two_matching_sessions()
        result = self.run_tool("prompts", "--session", self.PREFIX, "--json")
        self.assertEqual(result.returncode, 1)
        stderr = result.stderr.decode("utf-8")
        self.assertIn("matches 2 sessions", stderr)
        self.assertIn(QD.short_id(first), stderr)
        self.assertIn(QD.short_id(second), stderr)
        self.assertEqual(result.stdout.decode("utf-8"), "")

    def test_first_opts_back_into_the_busiest_match(self) -> None:
        first, _second = self.build_two_matching_sessions()
        payload = self.run_json("prompts", "--session", self.PREFIX, "--first", "--json")
        self.assertEqual(payload["session_id"], first)

    def test_full_id_beats_a_shared_prefix(self) -> None:
        first, _second = self.build_two_matching_sessions()
        payload = self.run_json("prompts", "--session", first, "--json")
        self.assertEqual(payload["session_id"], first)

    def test_de_dashed_full_id_still_takes_the_exact_path(self) -> None:
        first, _second = self.build_two_matching_sessions()
        payload = self.run_json("prompts", "--session", first.replace("-", ""), "--json")
        self.assertEqual(payload["session_id"], first)

    def test_one_id_under_two_harnesses_is_still_ambiguous(self) -> None:
        now = time.time() - 1800
        session = "ffff0001-1111-2222-3333-444444444444"
        self.write_claude(
            "shared.jsonl",
            [
                claude_user_prompt_line(now, session),
                claude_assistant_line(now + 1, session, "msg_shared", output_tokens=100),
            ],
        )
        self.write_codex(
            "rollout-shared.jsonl",
            [
                codex_session_meta_line(now, session, "/shared"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_usage_record_line(now + 2, session, input_tokens=100,
                                        cached_input_tokens=0, output_tokens=10,
                                        turn_id="turn-0"),
            ],
            day=now,
        )
        result = self.run_tool("prompts", "--session", session, "--json")
        self.assertEqual(result.returncode, 1)
        self.assertIn("matches 2 sessions", result.stderr.decode("utf-8"))

    def test_prompts_json_carries_the_full_session_id(self) -> None:
        _first, second = self.build_two_matching_sessions()
        payload = self.run_json("prompts", "--session", second, "--json")
        self.assertEqual(payload["session_id"], second)
        self.assertEqual(payload["short_id"], QD.short_id(second))

    def test_no_match_warning_names_the_current_command(self) -> None:
        self.build_two_matching_sessions()
        result = self.run_tool("prompts", "--session", "eeee9999", "--json")
        self.assertEqual(result.returncode, 1)
        stderr = result.stderr.decode("utf-8")
        self.assertIn("run `nenpi sessions` for ids", stderr)
        self.assertNotIn("quota-drain", stderr)


class VerifyRange(Harness):
    def build_two_sessions(self) -> Tuple[str, str]:
        old_session = "eeee0001-1111-2222-3333-444444444444"
        new_session = "eeee0002-1111-2222-3333-444444444444"
        for session, epoch in ((old_session, time.time() - 30 * 86400),
                               (new_session, time.time() - 600)):
            self.write_claude(
                "verify-%s.jsonl" % session[:8],
                [
                    claude_assistant_line(epoch, session, "msg_" + session[:8],
                                          input_tokens=100, output_tokens=200),
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
        return old_session, new_session

    def test_verify_applies_until(self) -> None:
        old_session, new_session = self.build_two_sessions()
        everything = self.run_json("verify", "--harness", "claude", "--json")
        self.assertEqual(
            sorted(row["short_id"] for row in everything["rows"]),
            sorted([QD.short_id(old_session), QD.short_id(new_session)]),
        )
        windowed = self.run_json("verify", "--harness", "claude", "--json", "--until", "3d")
        self.assertEqual([row["short_id"] for row in windowed["rows"]],
                         [QD.short_id(old_session)])
        self.assertEqual(windowed["rows"][0]["session_id"], old_session)


class CacheIntegrity(Harness):
    def test_in_place_rewrite_that_grows_is_reparsed(self) -> None:
        now = time.time() - 600
        session = "dddd0001-1111-2222-3333-444444444444"
        path = self.claude_projects / "proj" / "rewrite.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            claude_assistant_line(now, session, "msg_v1", output_tokens=100) + "\n",
            encoding="utf-8",
        )
        first = self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(first["sessions"][0]["tokens"]["output"], 100)
        # Rewritten in place and longer: size and mtime both move, but the
        # stored offset no longer names a record boundary.
        other = "dddd0002-1111-2222-3333-444444444444"
        path.write_text(
            claude_assistant_line(now + 1, other, "msg_v2_longer_id", output_tokens=250)
            + "\n"
            + claude_assistant_line(now + 2, other, "msg_v3", output_tokens=250)
            + "\n",
            encoding="utf-8",
        )
        os.utime(path, (time.time(), time.time()))
        second = self.run_json("sessions", "--harness", "claude", "--json")
        rows = dict((row["short_id"], row) for row in second["sessions"])
        self.assertEqual(rows["dddd0002"]["tokens"]["output"], 500)
        self.assertNotIn("dddd0001", rows)

    def test_shard_for_a_deleted_transcript_is_pruned(self) -> None:
        now = time.time() - 600
        session = "dddd0003-1111-2222-3333-444444444444"
        path = self.write_claude(
            "gone.jsonl", [claude_assistant_line(now, session, "msg_g", output_tokens=10)]
        )
        self.run_json("sessions", "--json")
        shards = list((self.root / "cache").rglob("*.json"))
        self.assertEqual(len(shards), 1)
        path.unlink()
        self.run_json("sessions", "--json")
        self.assertEqual(list((self.root / "cache").rglob("*.json")), [])

    def test_single_harness_run_keeps_the_other_harness_shards(self) -> None:
        now = time.time() - 600
        self.write_claude(
            "keep.jsonl",
            [
                claude_assistant_line(
                    now, "dddd0005-1111-2222-3333-444444444444", "msg_k", output_tokens=10
                )
            ],
        )
        self.write_codex(
            "rollout-keep.jsonl",
            [
                codex_session_meta_line(now, "codex-keep-0001", "/home/agent/keep"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 1, "codex-keep-0001", input_tokens=1000, cached_input_tokens=0,
                    output_tokens=10
                ),
            ],
        )
        self.run_json("sessions", "--json")
        self.assertEqual(len(list((self.root / "cache").rglob("*.json"))), 2)
        # A harness-scoped run never looks at the other harness's transcripts,
        # so it must not conclude they are gone.
        self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(len(list((self.root / "cache").rglob("*.json"))), 2)
        self.run_json("sessions", "--json", "--since", "3d", "--rebuild-cache")
        self.assertEqual(len(list((self.root / "cache").rglob("*.json"))), 2)

    def test_small_file_resumes_across_an_append(self) -> None:
        now = time.time() - 600
        session = "dddd0006-1111-2222-3333-444444444444"
        path = self.write_claude(
            "small.jsonl",
            [claude_assistant_line(now, session, "msg_s1", output_tokens=100)],
        )
        self.assertLess(path.stat().st_size, 4096)
        first = self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(first["files_parsed"], 1)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                claude_assistant_line(now + 1, session, "msg_s2", output_tokens=200) + "\n"
            )
        os.utime(path, (time.time(), time.time()))
        second = self.run_json("sessions", "--harness", "claude", "--json")
        # Resumed from the stored offset rather than reparsed: the fingerprint
        # must cover only the bytes already read.
        self.assertEqual(second["sessions"][0]["tokens"]["output"], 300)
        self.assertEqual(second["sessions"][0]["requests"], 2)

    def test_superseded_schema_directory_is_removed(self) -> None:
        stale = self.root / "cache" / "v1" / "ab"
        stale.mkdir(parents=True)
        (stale / "old.json").write_text("{}", encoding="utf-8")
        now = time.time() - 600
        self.write_claude(
            "fresh.jsonl",
            [
                claude_assistant_line(
                    now, "dddd0004-1111-2222-3333-444444444444", "msg_f", output_tokens=10
                )
            ],
        )
        self.run_json("sessions", "--harness", "claude", "--json")
        self.assertFalse((self.root / "cache" / "v1").exists())


class WindowSelection(Harness):
    def write_snapshots(self, session: str, readings: Sequence[Tuple[float, float, int, int]]
                        ) -> None:
        now = time.time() - 3600
        lines = [codex_session_meta_line(now, session, "/home/agent/win"),
                 codex_turn_context_line(now, "gpt-5.6-sol")]
        for offset, percent, resets_at, minutes in readings:
            lines.append(
                codex_token_count_line(
                    now + offset, rate_limits=rate_limits(percent, resets_at, minutes)
                )
            )
        lines.insert(
            3,
            codex_usage_record_line(
                now + 12, session, input_tokens=100_000, cached_input_tokens=0,
                output_tokens=0
            ),
        )
        self.write_codex("rollout-win.jsonl", lines)

    def test_drain_is_never_summed_across_two_windows(self) -> None:
        resets_at = int(time.time()) + 7200
        self.write_snapshots(
            "codex-win-0001",
            [
                (5, 10.0, resets_at, 300),
                (6, 2.0, resets_at + 500_000, 10080),
                (30, 13.0, resets_at, 300),
                (31, 8.0, resets_at + 500_000, 10080),
            ],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        # 3% on the five-hour window and 6% on the weekly one; reporting 9%
        # would be adding two different denominators together.
        self.assertAlmostEqual(payload["sessions"][0]["drain_percent"], 3.0, places=6)
        self.assertIn("300 min", payload["codex_window"])

    def test_unknown_window_minutes_still_picks_one_window(self) -> None:
        resets_at = int(time.time()) + 7200
        self.write_snapshots(
            "codex-win-0002",
            [(5, 10.0, resets_at, 4321), (30, 14.0, resets_at, 4321)],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        self.assertAlmostEqual(payload["sessions"][0]["drain_percent"], 4.0, places=6)
        self.assertIn("4321", payload["codex_window"])

    def test_window_accepts_an_integer_minute_count(self) -> None:
        resets_at = int(time.time()) + 7200
        self.write_snapshots(
            "codex-win-0005",
            [
                (5, 10.0, resets_at, 300),
                (6, 2.0, resets_at + 500_000, 10080),
                (30, 13.0, resets_at, 300),
                (31, 8.0, resets_at + 500_000, 10080),
            ],
        )
        payload = self.run_json(
            "sessions", "--harness", "codex", "--json", "--window", "10080"
        )
        self.assertAlmostEqual(payload["sessions"][0]["drain_percent"], 6.0, places=6)
        self.assertIn("10080", payload["codex_window"])

    def test_null_resets_at_still_yields_intervals(self) -> None:
        self.write_snapshots(
            "codex-win-0003",
            [(5, 10.0, None, 300), (30, 15.0, None, 300)],
        )
        payload = self.run_json("sessions", "--harness", "codex", "--json")
        self.assertAlmostEqual(payload["sessions"][0]["drain_percent"], 5.0, places=6)

    def test_snapshots_without_intervals_warn(self) -> None:
        self.write_snapshots("codex-win-0004", [(5, 10.0, int(time.time()) + 7200, 300)])
        result = self.run_tool("sessions", "--harness", "codex", "--no-color")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"none produced a measurable interval", result.stderr)


class TtyStringIO(io.StringIO):
    """Stands in for a real UTF-8 terminal so the painter turns colors on."""

    encoding = "utf-8"

    def isatty(self) -> bool:  # noqa: D102 - stdlib override
        return True


class PromptsBarLegendColor(Harness):
    """Issue #12: the legend must match the bar cells and the harness color."""

    def render_prompts(self, *arguments: str) -> str:
        stdout = sys.stdout
        sys.stdout = TtyStringIO()
        try:
            with self.env_applied():
                QD.main(["prompts", "--width", "120"] + list(arguments))
            return sys.stdout.getvalue()
        finally:
            sys.stdout = stdout

    def test_legend_glyphs_match_bar_glyphs(self) -> None:
        now = time.time() - 1800
        session = "cccc9999-aaaa-2222-3333-444444444444"
        lines = [claude_user_prompt_line(now, session)]
        lines.append(
            claude_assistant_line(
                now + 1, session, "msg_p0", input_tokens=1000, cache_read=500,
                output_tokens=50
            )
        )
        self.write_claude("legend.jsonl", lines)
        text = self.render_prompts("--session", "cccc9999")

        legend_line = next(
            line for line in text.splitlines() if "input tokens sent per prompt" in line
        )
        bar_line = next(
            line for line in text.splitlines() if line.strip().startswith("1 ") and "█" in line
        )

        legend_filled = QD.ANSI["claude"] + "█" + QD.ANSI["reset"]
        legend_cached = QD.ANSI["dim"] + "▒" + QD.ANSI["reset"]
        self.assertIn(legend_filled, legend_line)
        self.assertIn(legend_cached, legend_line)

        # The legend glyphs must carry the same escape sequence prefix as the
        # bar's glyphs: same color for the filled cell, "dim" for cached.
        self.assertIn(QD.ANSI["claude"] + "█", bar_line)
        self.assertIn(QD.ANSI["dim"] + "▒", bar_line)

    def test_claude_session_uses_claude_color(self) -> None:
        now = time.time() - 1800
        session = "dddd9999-aaaa-2222-3333-444444444444"
        lines = [claude_user_prompt_line(now, session)]
        lines.append(
            claude_assistant_line(
                now + 1, session, "msg_p0", input_tokens=1000, output_tokens=50
            )
        )
        self.write_claude("claude_color.jsonl", lines)
        text = self.render_prompts("--session", "dddd9999")
        self.assertIn(QD.ANSI["claude"] + "█", text)
        self.assertNotIn(QD.ANSI["codex"] + "█", text)


class RootsAndConfig(Harness):
    """`config.toml`, root precedence, `nenpi config`, and the dir rename."""

    def config_path(self) -> Path:
        return self.root / "config" / "config.toml"

    def write_config(self, text: str) -> Path:
        path = self.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def make_extra_claude_root(self, name: str = ".claude-extra") -> Path:
        root = self.home / name
        (root / "projects" / "proj").mkdir(parents=True, exist_ok=True)
        return root

    def make_extra_codex_root(self, name: str = ".codex-extra") -> Path:
        root = self.home / name
        (root / "sessions").mkdir(parents=True, exist_ok=True)
        return root

    def test_default_roots_scan_only_dot_claude_and_dot_codex(self) -> None:
        extra = self.make_extra_claude_root()
        now = time.time() - 600
        self.write_claude(
            "default.jsonl",
            [claude_assistant_line(now, "aaaa0001-1111-2222-3333-444444444444", "msg_d",
                                   output_tokens=10)],
        )
        (extra / "projects" / "proj" / "extra.jsonl").write_text(
            claude_assistant_line(
                now, "bbbb0002-1111-2222-3333-444444444444", "msg_e", output_tokens=10
            )
            + "\n",
            encoding="utf-8",
        )
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        ids = {row["short_id"] for row in payload["sessions"]}
        self.assertIn("aaaa0001", ids)
        self.assertNotIn("bbbb0002", ids)

    def test_config_toml_adds_a_root(self) -> None:
        extra = self.make_extra_claude_root()
        now = time.time() - 600
        (extra / "projects" / "proj" / "extra.jsonl").write_text(
            claude_assistant_line(
                now, "bbbb0002-1111-2222-3333-444444444444", "msg_e", output_tokens=10
            )
            + "\n",
            encoding="utf-8",
        )
        self.write_config(
            '[claude]\nroots = ["%s", "%s"]\n' % (self.home / ".claude", extra)
        )
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        ids = {row["short_id"] for row in payload["sessions"]}
        self.assertIn("bbbb0002", ids)

    def test_flags_replace_config_and_defaults(self) -> None:
        extra = self.make_extra_claude_root()
        now = time.time() - 600
        self.write_claude(
            "default.jsonl",
            [claude_assistant_line(now, "aaaa0001-1111-2222-3333-444444444444", "msg_d",
                                   output_tokens=10)],
        )
        (extra / "projects" / "proj" / "extra.jsonl").write_text(
            claude_assistant_line(
                now, "bbbb0002-1111-2222-3333-444444444444", "msg_e", output_tokens=10
            )
            + "\n",
            encoding="utf-8",
        )
        self.write_config('[claude]\nroots = ["%s"]\n' % (self.home / ".claude"))
        payload = self.run_json(
            "sessions", "--harness", "claude", "--json", "--claude-root", str(extra)
        )
        ids = {row["short_id"] for row in payload["sessions"]}
        self.assertIn("bbbb0002", ids)
        self.assertNotIn("aaaa0001", ids)

    def test_claude_root_flag_accepts_the_old_projects_leaf(self) -> None:
        extra = self.make_extra_claude_root()
        now = time.time() - 600
        (extra / "projects" / "proj" / "extra.jsonl").write_text(
            claude_assistant_line(
                now, "bbbb0002-1111-2222-3333-444444444444", "msg_e", output_tokens=10
            )
            + "\n",
            encoding="utf-8",
        )
        result = self.run_tool(
            "sessions", "--harness", "claude", "--claude-root", str(extra / "projects")
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"deprecated", result.stderr)
        self.assertIn(b"bbbb0002", result.stdout)

    def test_malformed_toml_errors(self) -> None:
        self.write_config("not [ valid toml")
        result = self.run_tool("config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"nenpi: config", result.stderr)

    def test_unknown_key_warns_but_does_not_fail(self) -> None:
        self.write_config('[claude]\nroots = ["%s"]\nbogus = true\n' % (self.home / ".claude"))
        result = self.run_tool("config")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"unknown key", result.stderr)

    def test_config_command_shows_roots_and_account(self) -> None:
        (self.home / ".codex" / "auth.json").write_text(
            json.dumps({"tokens": {"account_id": "acct-codex-123", "access_token": "must-not-print"}}),
            encoding="utf-8",
        )
        payload = self.run_json("config", "--json")
        codex_rows = [row for row in payload["roots"] if row["harness"] == "codex"]
        self.assertTrue(codex_rows)
        self.assertEqual(codex_rows[0]["account_key"], "acct-codex-123")
        self.assertNotIn("must-not-print", json.dumps(payload))

    def test_config_command_falls_back_to_label_without_auth_json(self) -> None:
        payload = self.run_json("config", "--json")
        claude_rows = [row for row in payload["roots"] if row["harness"] == "claude"]
        self.assertTrue(claude_rows)
        self.assertEqual(claude_rows[0]["account_label"], ".claude")
        self.assertEqual(claude_rows[0]["account_key"], ".claude")

    def test_config_init_writes_starter_and_refuses_overwrite(self) -> None:
        first = self.run_tool("config", "--init")
        self.assertEqual(first.returncode, 0)
        text = self.config_path().read_text(encoding="utf-8")
        self.assertIn("[claude]", text)
        self.assertIn("[codex]", text)
        second = self.run_tool("config", "--init")
        self.assertNotEqual(second.returncode, 0)
        third = self.run_tool("config", "--init", "--force")
        self.assertEqual(third.returncode, 0)

    def test_legacy_quota_drain_env_still_honoured(self) -> None:
        environment = dict(self.environment)
        for name in ("HOME_DIR", "CACHE_DIR", "STATE_DIR", "CONFIG_DIR"):
            value = environment.pop("NENPI_" + name)
            environment["QUOTA_DRAIN_" + name] = value
        result = subprocess.run(
            DRAIN_COMMAND + ["config"],
            check=False,
            capture_output=True,
            env=environment,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"deprecated", result.stderr)

    def test_migration_moves_old_dir_to_new_name(self) -> None:
        environment = dict(self.environment)
        environment.pop("NENPI_CACHE_DIR")
        old_cache = self.home / ".cache" / "quota-drain"
        old_cache.mkdir(parents=True)
        (old_cache / "marker.txt").write_text("old", encoding="utf-8")
        result = subprocess.run(
            DRAIN_COMMAND + ["config"],
            check=False,
            capture_output=True,
            env=environment,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"moved", result.stderr)
        new_cache = self.home / ".cache" / "nenpi"
        self.assertTrue((new_cache / "marker.txt").is_file())
        self.assertFalse(old_cache.exists())

    def test_migration_leaves_both_when_new_dir_already_exists(self) -> None:
        environment = dict(self.environment)
        environment.pop("NENPI_CACHE_DIR")
        old_cache = self.home / ".cache" / "quota-drain"
        old_cache.mkdir(parents=True)
        (old_cache / "marker.txt").write_text("old", encoding="utf-8")
        new_cache = self.home / ".cache" / "nenpi"
        new_cache.mkdir(parents=True)
        (new_cache / "marker.txt").write_text("new", encoding="utf-8")
        result = subprocess.run(
            DRAIN_COMMAND + ["config"],
            check=False,
            capture_output=True,
            env=environment,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual((new_cache / "marker.txt").read_text(encoding="utf-8"), "new")
        self.assertEqual((old_cache / "marker.txt").read_text(encoding="utf-8"), "old")

    def test_resolve_roots_in_process_precedence(self) -> None:
        """Direct unit check of the precedence rule, independent of the CLI."""
        with self.env_applied():
            default_config = QC.Config()
            claude_default = QC.resolve_roots("claude", [], default_config)
            self.assertEqual(claude_default, [self.home / ".claude"])

            configured = QC.Config(claude_roots=[str(self.home / ".claude-extra")])
            (self.home / ".claude-extra").mkdir()
            claude_configured = QC.resolve_roots("claude", [], configured)
            self.assertEqual(claude_configured, [self.home / ".claude-extra"])

            flagged = QC.resolve_roots("claude", [str(self.home / ".claude")], configured)
            self.assertEqual(flagged, [self.home / ".claude"])

    def test_codex_only_harness_emits_no_claude_roots_warning(self) -> None:
        shutil.rmtree(self.home / ".claude")
        now = time.time() - 600
        session = "codex-only-0001"
        self.write_codex(
            "rollout-1-aaa.jsonl",
            [
                codex_session_meta_line(now, session, "/home/agent/repo"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 10, session, input_tokens=1_000, cached_input_tokens=0,
                    output_tokens=100
                ),
            ],
        )
        result = self.run_tool("sessions", "--harness", "codex")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertNotIn(b"no claude roots found", result.stderr)

    def test_config_command_shows_missing_root(self) -> None:
        missing = self.home / ".claude-typo"
        self.write_config('[claude]\nroots = ["%s"]\n' % missing)
        result = self.run_tool("config")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"exists=no", result.stdout)
        payload = self.run_json("config", "--json")
        claude_rows = [row for row in payload["roots"] if row["harness"] == "claude"]
        self.assertTrue(
            any(row["path"] == str(missing) and row["exists"] is False for row in claude_rows)
        )

    def test_plan_config_labels_the_header_when_nothing_measured(self) -> None:
        self.write_config('[plan]\nclaude = "max_20x"\n')
        now = time.time() - 600
        self.write_claude(
            "s.jsonl",
            [claude_assistant_line(now, "aaaa0001-1111-2222-3333-444444444444", "msg",
                                   output_tokens=10)],
        )
        result = self.run_tool("sessions", "--harness", "claude")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"claude=max_20x", result.stdout)

    def test_plan_config_never_overrides_a_measured_tier(self) -> None:
        self.write_config('[plan]\nclaude = "max_20x"\n')
        (self.home / ".claude" / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"organizationRateLimitTier": "pro_5x"}}),
            encoding="utf-8",
        )
        now = time.time() - 600
        self.write_claude(
            "s.jsonl",
            [claude_assistant_line(now, "aaaa0001-1111-2222-3333-444444444444", "msg",
                                   output_tokens=10)],
        )
        result = self.run_tool("sessions", "--harness", "claude")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"claude[.claude]=pro_5x", result.stdout)
        self.assertNotIn(b"max_20x", result.stdout)

    def test_config_init_escapes_special_characters_in_root_path(self) -> None:
        tricky = self.home / '.claude-weird"name\\dir'
        (tricky / "projects").mkdir(parents=True)
        result = self.run_tool("config", "--init")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        text = self.config_path().read_text(encoding="utf-8")
        self.assertIn('\\"', text)
        self.assertIn('\\\\', text)
        # A malformed escape would make this an invalid TOML string, or would
        # not round-trip to the same path; loading it back must recover it.
        with self.env_applied():
            loaded = QC.load_config()
        self.assertIn(str(tricky), loaded.claude_roots)

    def test_config_init_refuses_when_target_is_a_directory(self) -> None:
        directory_path = self.config_path()
        directory_path.mkdir(parents=True)
        result = self.run_tool("config", "--init")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"directory", result.stderr)

    def test_discover_candidate_roots_includes_env_var_outside_home(self) -> None:
        outside = self.root / "outside-codex"
        (outside / "sessions").mkdir(parents=True)
        environment = dict(self.environment)
        environment["CODEX_HOME"] = str(outside)
        result = subprocess.run(
            DRAIN_COMMAND + ["config", "--init"],
            check=False,
            capture_output=True,
            env=environment,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        text = self.config_path().read_text(encoding="utf-8")
        self.assertIn(str(outside), text)

    def test_legacy_config_file_env_has_no_fallback(self) -> None:
        environment = dict(self.environment)
        legacy_path = self.root / "legacy-config.toml"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text('[claude]\nroots = ["%s"]\n' % (self.home / ".claude"),
                               encoding="utf-8")
        environment["QUOTA_DRAIN_CONFIG_FILE"] = str(legacy_path)
        result = subprocess.run(
            DRAIN_COMMAND + ["config"],
            check=False,
            capture_output=True,
            env=environment,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn(b"deprecated", result.stderr)
        self.assertNotEqual(
            self.config_path(), legacy_path, "legacy CONFIG_FILE must not relocate config_path"
        )

    def test_warn_once_registry_is_shared_between_drain_and_config(self) -> None:
        self.assertIs(QD.warn_once, QC.warn_once)
        self.assertIs(QD.warn, QC.warn)


class AccountPools(Harness):
    """Per-account quota pools (#9).

    Two roots on different accounts must never be read as one alternating
    timeline (each gets its own pool, its own drain, its own window); two
    roots on the SAME account must still merge into one pool.
    """

    def write_codex_root(self, root_name: str, filename: str, lines: Sequence[str],
                         day: Optional[float] = None) -> Path:
        stamp = datetime.fromtimestamp(day or time.time(), timezone.utc)
        path = self.home / root_name / "sessions" / stamp.strftime("%Y/%m/%d") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
        return path

    def write_auth(self, root_name: str, account_id: str) -> None:
        root = self.home / root_name
        root.mkdir(parents=True, exist_ok=True)
        (root / "auth.json").write_text(
            json.dumps(
                {"tokens": {"account_id": account_id, "access_token": "must-not-print"}}
            ),
            encoding="utf-8",
        )

    def test_two_accounts_no_rollover_and_separate_drain(self) -> None:
        # Reproduces #9: two Codex roots on different accounts, same
        # limit_id/plan_type/window_minutes, readings interleaved every few
        # seconds, and the Arcade pool's resets_at is hours after the
        # personal pool's - exactly the shape that used to be read as one
        # timeline rolling over.
        self.write_auth(".codex", "acct-personal")
        self.write_auth(".codex-arcade", "acct-arcade")
        now = time.time() - 3600
        resets_a = int(now) + 7 * 86400
        resets_b = resets_a + 4 * 3600

        session_a = "aaaaaaaa-personal-2222-3333-444444444444"
        lines_a = [
            codex_session_meta_line(now, session_a, "/home/agent/personal"),
            codex_turn_context_line(now, "gpt-5.6-sol"),
            codex_token_count_line(now + 5, rate_limits=rate_limits(27.0, resets_a, 10080)),
            codex_usage_record_line(
                now + 20, session_a, input_tokens=100_000, cached_input_tokens=0,
                output_tokens=0,
            ),
            codex_token_count_line(now + 35, rate_limits=rate_limits(42.0, resets_a, 10080)),
        ]
        self.write_codex_root(".codex", "rollout-personal.jsonl", lines_a)

        session_b = "bbbbbbbb-arcade-2222-3333-444444444444"
        lines_b = [
            codex_session_meta_line(now, session_b, "/home/agent/arcade"),
            codex_turn_context_line(now, "gpt-5.6-sol"),
            codex_token_count_line(now + 10, rate_limits=rate_limits(33.0, resets_b, 10080)),
            codex_usage_record_line(
                now + 25, session_b, input_tokens=100_000, cached_input_tokens=0,
                output_tokens=0,
            ),
            codex_token_count_line(now + 40, rate_limits=rate_limits(90.0, resets_b, 10080)),
        ]
        self.write_codex_root(".codex-arcade", "rollout-arcade.jsonl", lines_b)

        root_args = [
            "--codex-root", str(self.home / ".codex"),
            "--codex-root", str(self.home / ".codex-arcade"),
        ]
        payload = self.run_json(
            "sessions", "--harness", "codex", "--json", "--window", "weekly", *root_args
        )
        row_a = self.session_by_short_id(payload, "aaaaaaaa")
        row_b = self.session_by_short_id(payload, "bbbbbbbb")
        # 15 = 42 - 27 (personal pool's own delta); 57 = 90 - 33 (Arcade's).
        # A merged timeline would instead charge one session the other
        # pool's whole used_percent as a bogus rollover.
        self.assertAlmostEqual(row_a["drain_percent"], 15.0, places=6)
        self.assertAlmostEqual(row_b["drain_percent"], 57.0, places=6)
        self.assertEqual(row_a["account_label"], ".codex")
        self.assertEqual(row_b["account_label"], ".codex-arcade")
        self.assertEqual(row_a["account"], "acct-personal")
        self.assertEqual(row_b["account"], "acct-arcade")

        windows_payload = self.run_json(
            "windows", "--harness", "codex", "--json", "--window", "weekly", *root_args
        )
        by_label = dict(
            (entry["account_label"], entry) for entry in windows_payload["windows"]
        )
        self.assertEqual(sorted(by_label), [".codex", ".codex-arcade"])
        self.assertAlmostEqual(by_label[".codex"]["peak_used_percent"], 42.0, places=6)
        self.assertAlmostEqual(by_label[".codex-arcade"]["peak_used_percent"], 90.0, places=6)

    def test_same_account_two_roots_merge_into_one_pool(self) -> None:
        # Two roots that happen to share the same account (a re-pointed
        # $CODEX_HOME, say) must still read as one continuous pool.
        self.write_auth(".codex", "acct-shared")
        self.write_auth(".codex-mirror", "acct-shared")
        now = time.time() - 3600
        resets_at = int(now) + 7 * 86400

        session_a = "cccccccc-shared-2222-3333-444444444444"
        self.write_codex_root(
            ".codex", "rollout-a.jsonl",
            [
                codex_session_meta_line(now, session_a, "/home/agent/shared"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_token_count_line(
                    now + 5, rate_limits=rate_limits(10.0, resets_at, 10080)
                ),
                codex_usage_record_line(
                    now + 20, session_a, input_tokens=50_000, cached_input_tokens=0,
                    output_tokens=0,
                ),
            ],
        )
        session_b = "dddddddd-shared-2222-3333-444444444444"
        self.write_codex_root(
            ".codex-mirror", "rollout-b.jsonl",
            [
                codex_session_meta_line(now, session_b, "/home/agent/shared"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_token_count_line(
                    now + 30, rate_limits=rate_limits(18.0, resets_at, 10080)
                ),
                codex_usage_record_line(
                    now + 25, session_b, input_tokens=50_000, cached_input_tokens=0,
                    output_tokens=0,
                ),
            ],
        )
        root_args = [
            "--codex-root", str(self.home / ".codex"),
            "--codex-root", str(self.home / ".codex-mirror"),
        ]
        windows_payload = self.run_json(
            "windows", "--harness", "codex", "--json", "--window", "weekly", *root_args
        )
        # One merged pool, not two: a single 10% -> 18% timeline split
        # between whichever session had events since the previous reading.
        self.assertEqual(len(windows_payload["windows"]), 1)
        self.assertAlmostEqual(
            windows_payload["windows"][0]["peak_used_percent"], 18.0, places=6
        )

    def test_cached_shard_keeps_the_right_account_after_reresolve(self) -> None:
        # A shard cached under a root on one run must still stamp the right
        # account on a later run that resolves the same root again - account
        # identity is never part of the per-file cache payload.
        self.write_auth(".codex", "acct-personal")
        now = time.time() - 3600
        session = "eeeeeeee-cache-2222-3333-444444444444"
        self.write_codex_root(
            ".codex", "rollout-cache.jsonl",
            [
                codex_session_meta_line(now, session, "/home/agent/personal"),
                codex_turn_context_line(now, "gpt-5.6-sol"),
                codex_usage_record_line(
                    now + 5, session, input_tokens=10_000, cached_input_tokens=0,
                    output_tokens=0,
                ),
            ],
        )
        first = self.run_json("sessions", "--harness", "codex", "--json")
        row = self.session_by_short_id(first, "eeeeeeee")
        self.assertEqual(row["account_label"], ".codex")
        self.assertEqual(row["account"], "acct-personal")
        # Second run reads the same file from cache (mtime/size unchanged);
        # the account still has to come from the root, not a stale copy.
        second = self.run_json("sessions", "--harness", "codex", "--json")
        row = self.session_by_short_id(second, "eeeeeeee")
        self.assertEqual(row["account_label"], ".codex")
        self.assertEqual(row["account"], "acct-personal")

    def test_account_filter_limits_the_report_to_one_pool(self) -> None:
        self.write_auth(".codex", "acct-personal")
        self.write_auth(".codex-arcade", "acct-arcade")
        now = time.time() - 3600
        resets_at = int(now) + 7 * 86400
        session_a = "ffffffff-filt-a222-3333-444444444444"
        session_b = "11111111-filt-b222-3333-444444444444"
        # Distinct offsets per root: the usage record's epoch feeds a
        # synthetic call id (`resp-%f`), and two roots sharing one would
        # collide in the corpus-wide dedup and silently drop one session.
        for root_name, session_id, offset in (
            (".codex", session_a, 5), (".codex-arcade", session_b, 7)
        ):
            self.write_codex_root(
                root_name, "rollout-r.jsonl",
                [
                    codex_session_meta_line(now, session_id, "/home/agent/x"),
                    codex_turn_context_line(now, "gpt-5.6-sol"),
                    codex_usage_record_line(
                        now + offset, session_id, input_tokens=10_000,
                        cached_input_tokens=0, output_tokens=0,
                    ),
                    codex_token_count_line(
                        now + 10, rate_limits=rate_limits(10.0, resets_at, 10080)
                    ),
                ],
                day=now,
            )
        payload = self.run_json(
            "sessions", "--harness", "codex", "--json",
            "--codex-root", str(self.home / ".codex"),
            "--codex-root", str(self.home / ".codex-arcade"),
            "--account", ".codex-arcade",
        )
        short_ids = [row["short_id"] for row in payload["sessions"]]
        self.assertEqual(short_ids, ["11111111"])

    def test_account_filter_by_either_label_returns_the_shared_key_pool(self) -> None:
        # Two roots sharing one account key (a re-pointed root, or the same
        # login copied to a second config dir) must merge into one pool
        # (`test_same_account_two_roots_merge_into_one_pool` above) - and
        # `--account` on EITHER root's label must resolve to that shared key
        # and return sessions from BOTH roots, not just the labelled one.
        self.write_auth(".codex", "acct-shared")
        self.write_auth(".codex-mirror", "acct-shared")
        now = time.time() - 3600
        resets_at = int(now) + 7 * 86400
        session_a = "22222222-share-a222-3333-444444444444"
        session_b = "33333333-share-b222-3333-444444444444"
        for root_name, session_id, offset in (
            (".codex", session_a, 5), (".codex-mirror", session_b, 7)
        ):
            self.write_codex_root(
                root_name, "rollout-shared.jsonl",
                [
                    codex_session_meta_line(now, session_id, "/home/agent/x"),
                    codex_turn_context_line(now, "gpt-5.6-sol"),
                    codex_usage_record_line(
                        now + offset, session_id, input_tokens=10_000,
                        cached_input_tokens=0, output_tokens=0,
                    ),
                    codex_token_count_line(
                        now + 10, rate_limits=rate_limits(10.0, resets_at, 10080)
                    ),
                ],
                day=now,
            )
        root_args = [
            "--codex-root", str(self.home / ".codex"),
            "--codex-root", str(self.home / ".codex-mirror"),
        ]
        for label in (".codex", ".codex-mirror"):
            payload = self.run_json(
                "sessions", "--harness", "codex", "--json", "--account", label, *root_args
            )
            short_ids = sorted(row["short_id"] for row in payload["sessions"])
            self.assertEqual(short_ids, ["22222222", "33333333"])


class ClaudeAccountPools(Harness):
    """Two Claude `config_dir`s map to two accounts (#9, Claude side)."""

    def write_claude_config(self, root_name: str, org_uuid: str) -> None:
        root = self.home / root_name
        root.mkdir(parents=True, exist_ok=True)
        (root / ".claude.json").write_text(
            json.dumps(
                {
                    "oauthAccount": {
                        "organizationUuid": org_uuid,
                        "organizationRateLimitTier": "max_20x",
                    }
                }
            ),
            encoding="utf-8",
        )

    def write_snapshots(self, records: Sequence[Dict[str, Any]]) -> None:
        path = self.root / "state" / "snapshots.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def test_two_config_dirs_map_to_two_accounts(self) -> None:
        self.write_claude_config(".claude", "org-personal")
        self.write_claude_config(".claude-arcade", "org-arcade")
        now = time.time() - 3600
        self.write_snapshots(
            [
                {
                    "source": "oauth",
                    "config_dir": ".claude",
                    "ts": now,
                    "windows": {
                        "five_hour": {
                            "utilization_percent": 10.0,
                            "resets_at": "2026-09-16T21:30:00+00:00",
                        }
                    },
                },
                {
                    "source": "oauth",
                    "config_dir": ".claude-arcade",
                    "ts": now + 60,
                    "windows": {
                        "five_hour": {
                            "utilization_percent": 40.0,
                            "resets_at": "2026-09-16T21:30:00+00:00",
                        }
                    },
                },
            ]
        )
        with self.env_applied():
            roots = [self.home / ".claude", self.home / ".claude-arcade"]
            rows = QD.load_claude_snapshots("five_hour", claude_roots=roots)
        by_label = dict((row["account_label"], row) for row in rows)
        self.assertEqual(by_label[".claude"]["account"], "org-personal")
        self.assertEqual(by_label[".claude-arcade"]["account"], "org-arcade")
        self.assertEqual(by_label[".claude"]["plan_type"], "max_20x")
        self.assertEqual(by_label[".claude-arcade"]["plan_type"], "max_20x")
        # The account key (an org uuid, resolved in memory from `.claude.json`)
        # is never written back to the snapshot record itself - only
        # `config_dir` (the label) is, per `oauth_snapshot_record`'s privacy
        # note. This asserts the file on disk still holds no org uuid.
        raw = (self.root / "state" / "snapshots.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("org-personal", raw)
        self.assertNotIn("org-arcade", raw)


class _StdinWithBuffer:
    def __init__(self, payload: bytes) -> None:
        self.buffer = io.BytesIO(payload)


class _StdoutWithBuffer:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def flush(self) -> None:
        pass


class UnifiedConfigStore(RootsAndConfig):
    """One store for the CLI and the Sources screen (issue #23)."""

    def test_toml_writer_round_trips_awkward_strings(self) -> None:
        import tomllib

        config = QC.Config(
            claude_roots=['/tmp/we"ird', "/tmp/back\\slash", "/tmp/plain"],
            codex_roots=[],
            claude_disabled=["/tmp/off"],
            codex_ignored=["/tmp/gone"],
            plan_claude="max_20x",
            ignore_unconfigured=True,
        )
        data = tomllib.loads(QC.dump_config(config))
        self.assertEqual(data["claude"]["roots"], config.claude_roots)
        self.assertEqual(data["claude"]["disabled"], ["/tmp/off"])
        self.assertEqual(data["codex"]["roots"], [])
        self.assertEqual(data["codex"]["ignored"], ["/tmp/gone"])
        self.assertEqual(data["plan"]["claude"], "max_20x")
        self.assertTrue(data["general"]["ignore_unconfigured"])

    def test_save_config_then_load_config_is_identity(self) -> None:
        with self.env_applied():
            written = QC.Config(
                claude_roots=[str(self.home / ".claude")],
                codex_roots=[str(self.home / ".codex")],
                codex_disabled=[str(self.home / ".codex-off")],
                plan_codex="pro",
            )
            QC.save_config(written, self.config_path())
            loaded = QC.load_config(self.config_path())
        self.assertEqual(loaded.claude_roots, written.claude_roots)
        self.assertEqual(loaded.codex_roots, written.codex_roots)
        self.assertEqual(loaded.codex_disabled, written.codex_disabled)
        self.assertEqual(loaded.plan_codex, "pro")

    def test_source_added_in_the_ui_is_visible_to_the_cli(self) -> None:
        extra = self.make_extra_codex_root(".codex-extra")
        with self.env_applied():
            settings = SourceSettings.load(self.config_path(), self.home)
            settings.add(extra, "codex")
            settings.save()
        payload = self.run_json("config", "--json")
        codex_roots = [row["path"] for row in payload["roots"]
                       if row["harness"] == "codex"]
        self.assertIn(str(extra), codex_roots)
        self.assertTrue(payload["config_present"])

    def test_config_json_is_migrated_with_the_harness_corrected(self) -> None:
        import tomllib

        claude_extra = self.make_extra_claude_root(".claude-extra")
        codex_extra = self.make_extra_codex_root(".codex-extra")
        legacy = self.root / "config" / "config.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text(json.dumps({
            "version": 1,
            "sources": [
                {"id": "a", "name": ".claude-extra", "harness": "claude",
                 "path": str(claude_extra), "enabled": True},
                # The Sources form saved this one with the wrong harness.
                {"id": "b", "name": ".codex-extra", "harness": "claude",
                 "path": str(codex_extra), "enabled": True},
                {"id": "c", "name": ".claude", "harness": "claude",
                 "path": str(self.home / ".claude"), "enabled": False},
            ],
            "ignored_discovered": [
                {"harness": "codex", "path": str(self.home / ".codex")}
            ],
        }), encoding="utf-8")
        with self.env_applied():
            SourceSettings.load(self.config_path(), self.home)
        data = tomllib.loads(self.config_path().read_text(encoding="utf-8"))
        self.assertEqual(data["claude"]["roots"], [str(claude_extra)])
        self.assertEqual(data["claude"]["disabled"], [str(self.home / ".claude")])
        self.assertEqual(data["codex"]["roots"], [str(codex_extra)])
        self.assertEqual(data["codex"]["ignored"], [str(self.home / ".codex")])
        self.assertFalse(legacy.exists())
        self.assertTrue(legacy.with_name("config.json.migrated").is_file())
        payload = self.run_json("config", "--json")
        paths = [row["path"] for row in payload["roots"]]
        self.assertIn(str(codex_extra), paths)
        self.assertNotIn(str(self.home / ".claude"), paths)

    def test_disabled_roots_do_not_fall_back_to_the_defaults(self) -> None:
        """An empty table the user wrote means "scan nothing" (review #1)."""

        self.write_config(
            '[claude]\nroots = []\ndisabled = ["%s"]\n' % (self.home / ".claude")
        )
        payload = self.run_json("config", "--json")
        self.assertEqual(
            [row["path"] for row in payload["roots"] if row["harness"] == "claude"], []
        )
        # The absent [codex] table still means "no opinion" -> defaults.
        self.assertEqual(
            [row["path"] for row in payload["roots"] if row["harness"] == "codex"],
            [str(self.home / ".codex")],
        )
        with self.env_applied():
            config = QC.load_config(self.config_path())
            self.assertEqual(QC.resolve_roots("claude", [], config, quiet=True), [])

    def test_source_disabled_in_the_ui_disappears_from_the_cli(self) -> None:
        now = time.time() - 600
        self.write_claude(
            "default.jsonl",
            [claude_assistant_line(now, "aaaa0001-1111-2222-3333-444444444444",
                                   "msg_d", output_tokens=10)],
        )
        with self.env_applied():
            settings = SourceSettings.load(self.config_path(), self.home)
            for source in settings.sources:
                settings.set_enabled(source.id, False)
            settings.save()
        payload = self.run_json("sessions", "--harness", "claude", "--json")
        self.assertEqual(payload["sessions"], [])

    def test_unknown_tables_and_keys_survive_a_save(self) -> None:
        import tomllib

        extra_root = self.make_extra_codex_root(".codex-extra")
        self.write_config(
            '[ui]\ntheme = "dark"\nrefresh = 30\n\n'
            '[ui.colors]\naccent = "teal"\n\n'
            '[claude]\nroots = ["%s"]\nfuture_key = true\n'
            % (self.home / ".claude")
        )
        with self.env_applied():
            settings = SourceSettings.load(self.config_path(), self.home)
            settings.add(extra_root, "codex")
            settings.save()
        data = tomllib.loads(self.config_path().read_text(encoding="utf-8"))
        self.assertEqual(data["ui"]["theme"], "dark")
        self.assertEqual(data["ui"]["refresh"], 30)
        self.assertEqual(data["ui"]["colors"]["accent"], "teal")
        self.assertTrue(data["claude"]["future_key"])
        self.assertIn(str(extra_root), data["codex"]["roots"])

    def test_config_init_force_keeps_unknown_tables_and_ui_state(self) -> None:
        import tomllib

        disabled = self.make_extra_claude_root(".claude-off")
        self.write_config(
            '[ui]\ntheme = "dark"\n\n'
            '[claude]\nroots = ["%s"]\ndisabled = ["%s"]\n'
            % (self.home / ".claude", disabled)
        )
        result = self.run_tool("config", "--init", "--force")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        data = tomllib.loads(self.config_path().read_text(encoding="utf-8"))
        self.assertEqual(data["ui"]["theme"], "dark")
        self.assertEqual(data["claude"]["disabled"], [str(disabled)])
        # A disabled root is not reseeded as an enabled one.
        self.assertNotIn(str(disabled), data["claude"]["roots"])
        self.assertIn(str(self.home / ".claude"), data["claude"]["roots"])

    def test_cli_alone_migrates_config_json(self) -> None:
        extra = self.make_extra_codex_root(".codex-extra")
        legacy = self.root / "config" / "config.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text(json.dumps({"version": 1, "sources": [
            {"id": "a", "name": ".codex-extra", "harness": "codex",
             "path": str(extra), "enabled": True}]}), encoding="utf-8")
        result = self.run_tool("config")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn("imported", result.stderr.decode("utf-8", "replace"))
        self.assertIn(str(extra), result.stdout.decode("utf-8", "replace"))
        self.assertTrue(self.config_path().is_file())
        self.assertTrue(legacy.with_name("config.json.migrated").is_file())

    def test_migration_is_skipped_when_config_toml_exists(self) -> None:
        self.write_config('[claude]\nroots = ["%s"]\n' % (self.home / ".claude"))
        legacy = self.root / "config" / "config.json"
        legacy.write_text('{"version": 1, "sources": []}', encoding="utf-8")
        with self.env_applied():
            self.assertIsNone(migrate_json_store(self.config_path()))
        self.assertTrue(legacy.is_file())


class UnconfiguredSiblingNotice(RootsAndConfig):
    """One stderr note about harness dirs nothing will scan (issue #24)."""

    NOTE = "unconfigured harness dirs"

    def stderr_of(self, *arguments: str) -> str:
        result = self.run_tool(*arguments)
        self.assertEqual(result.returncode, 0,
                         result.stderr.decode("utf-8", "replace"))
        return result.stderr.decode("utf-8", "replace")

    def test_sibling_dirs_are_reported_once_on_stderr(self) -> None:
        self.make_extra_claude_root(".claude-extra")
        self.make_extra_codex_root(".codex-extra")
        stderr = self.stderr_of("sessions", "--harness", "all")
        self.assertIn(self.NOTE, stderr)
        self.assertIn("~/.claude-extra", stderr)
        self.assertIn("~/.codex-extra", stderr)
        self.assertEqual(stderr.count(self.NOTE), 1)

    def test_no_note_when_every_sibling_is_configured(self) -> None:
        extra = self.make_extra_claude_root(".claude-extra")
        self.write_config(
            '[claude]\nroots = ["%s", "%s"]\n' % (self.home / ".claude", extra)
        )
        self.assertNotIn(self.NOTE, self.stderr_of("sessions", "--harness", "claude"))

    def test_json_output_carries_no_note(self) -> None:
        self.make_extra_claude_root(".claude-extra")
        result = self.run_tool("sessions", "--harness", "claude", "--json")
        self.assertEqual(result.returncode, 0)
        self.assertNotIn(self.NOTE, result.stderr.decode("utf-8", "replace"))

    def test_ignore_unconfigured_silences_the_note(self) -> None:
        self.make_extra_claude_root(".claude-extra")
        self.write_config("[general]\nignore_unconfigured = true\n")
        self.assertNotIn(self.NOTE, self.stderr_of("sessions", "--harness", "claude"))

    def test_disabled_and_ignored_roots_are_not_unconfigured(self) -> None:
        extra = self.make_extra_claude_root(".claude-extra")
        other = self.make_extra_codex_root(".codex-extra")
        self.write_config(
            '[claude]\nroots = ["%s"]\ndisabled = ["%s"]\n\n'
            '[codex]\nroots = ["%s"]\nignored = ["%s"]\n'
            % (self.home / ".claude", extra, self.home / ".codex", other)
        )
        self.assertNotIn(self.NOTE, self.stderr_of("sessions", "--harness", "all"))

    def test_snapshot_stdin_never_globs_or_warns(self) -> None:
        """The statusline path resolves its own root quietly and globs never.

        It does read the default Claude root to label the snapshot, but that
        lookup is quiet; the sibling glob must not run on a path the
        statusline takes on every prompt.
        """

        self.make_extra_claude_root(".claude-extra")
        payload = json.dumps({"rate_limits": {"five_hour": {
            "used_percentage": 10.0, "resets_at": "2026-01-01T00:00:00Z"}}})

        def forbidden(*arguments: Any, **keywords: Any) -> None:
            raise AssertionError("snapshot --stdin globbed for sibling roots")

        errors = io.StringIO()
        saved_in, saved_out, saved_err = sys.stdin, sys.stdout, sys.stderr
        saved_discover = QC.discover_candidate_roots
        sys.stdin = _StdinWithBuffer(payload.encode("utf-8"))
        sys.stdout = _StdoutWithBuffer()
        sys.stderr = errors
        QC.discover_candidate_roots = forbidden
        try:
            with self.env_applied():
                self.assertEqual(QD.main(["snapshot", "--stdin"]), 0)
        finally:
            sys.stdin, sys.stdout, sys.stderr = saved_in, saved_out, saved_err
            QC.discover_candidate_roots = saved_discover
        self.assertEqual(errors.getvalue(), "")
        self.assertTrue((self.root / "state" / "snapshots.jsonl").is_file())

    def test_config_lists_unconfigured_roots(self) -> None:
        extra = self.make_extra_claude_root(".claude-extra")
        payload = self.run_json("config", "--json")
        self.assertEqual(
            [row["path"] for row in payload["unconfigured_roots"]], [str(extra)]
        )
        result = self.run_tool("config")
        self.assertIn("unconfigured: ~/.claude-extra",
                      result.stdout.decode("utf-8", "replace"))


def synthetic_event(
    session_id: str,
    model: str,
    epoch: float,
    tokens: Sequence[int],
    long_context: bool = False,
    subagent: bool = False,
    call_id: str = "",
) -> List[Any]:
    """One synthetic event row in the layout `absorb`/`build_prompt` expect.

    Never built from a transcript: the numbers are made up so the arithmetic
    under test is the only thing the assertions depend on.
    """
    row = [session_id, model, epoch] + [0] * QD.EVENT_KIND_SLOTS
    for offset, value in enumerate(tokens):
        row[QD.EVENT_KINDS + offset] = value
    row.extend([long_context, subagent, None, None, call_id])
    return row


class CountingMap(dict):
    """A dict that records how many key lookups went through it.

    Every `Weights` memo test that only compares return values passes with
    the memo removed, so the tests below count the lookups the memo is
    supposed to prevent instead.
    """

    def __init__(self, *arguments: Any, **keywords: Any) -> None:
        super().__init__(*arguments, **keywords)
        self.lookups = 0

    def get(self, key: Any, default: Any = None) -> Any:
        self.lookups += 1
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        self.lookups += 1
        return super().__getitem__(key)


class WeightMemoization(unittest.TestCase):
    """`Weights` memoizes pure lookups; the memo must not change an answer."""

    def setUp(self) -> None:
        self.weights = QD.load_weights(False)
        # A second Weights over instrumented maps, so a lookup that reaches
        # the table can be counted rather than inferred.
        self.claude_entry = CountingMap(
            input=2.0, cache_read=0.2, cache_write_5m=2.5,
            cache_write_1h=4.0, output=10.0,
        )
        self.claude_models = CountingMap({"model-a": self.claude_entry})
        self.codex_models = CountingMap(
            {"model-b": CountingMap(input=100.0, cached_input=10.0, output=500.0)}
        )
        self.counted = QD.Weights(
            {
                "claude": {"unit": "usd_per_mtok", "models": self.claude_models},
                "codex": {"unit": "credit_units_per_mtok", "models": self.codex_models},
            },
            ["test"],
        )

    def test_repeat_model_entry_does_no_table_lookup(self) -> None:
        self.assertIsNotNone(self.counted.model_entry("claude", "model-a"))
        after_first = self.claude_models.lookups
        self.assertEqual(after_first, 1)
        for _ in range(5):
            self.assertIsNotNone(self.counted.model_entry("claude", "model-a"))
        self.assertEqual(self.claude_models.lookups, after_first)

    def test_a_repeated_miss_is_remembered_as_a_miss(self) -> None:
        self.assertIsNone(self.counted.model_entry("claude", "no-such-model"))
        after_first = self.claude_models.lookups
        self.assertEqual(after_first, 1)
        for _ in range(5):
            self.assertIsNone(self.counted.model_entry("claude", "no-such-model"))
        self.assertEqual(self.claude_models.lookups, after_first)

    def test_repeat_pricing_does_not_reread_the_entry(self) -> None:
        tokens = {"input": 1_000_000, "output": 1_000_000}
        self.assertEqual(self.counted.claude_units("model-a", tokens, None), 12.0)
        after_first = self.claude_entry.lookups
        self.assertGreater(after_first, 0)
        for _ in range(5):
            self.assertEqual(self.counted.claude_units("model-a", tokens, None), 12.0)
            self.counted.event_vector("claude", "model-a", None)
            self.counted.event_vector("claude", "model-a", None, True)
        self.assertEqual(self.claude_entry.lookups, after_first)

    def test_memo_survives_a_miss_without_poisoning_a_hit(self) -> None:
        self.assertIsNone(self.counted.model_entry("claude", "model-b"))
        self.assertIsNotNone(self.counted.model_entry("codex", "model-b"))
        self.assertIsNone(self.counted.model_entry("claude", "model-b"))

    def test_model_entry_returns_the_table_entry(self) -> None:
        entry = self.weights.model_entry("claude", "claude-sonnet-5")
        self.assertIsNotNone(entry)
        self.assertEqual(
            self.weights.table["claude"]["models"]["claude-sonnet-5"], entry
        )

    def test_memo_does_not_leak_between_harnesses(self) -> None:
        # "gpt-5.5" is a Codex model and has no Claude entry; a memo keyed on
        # the model alone would hand the Codex entry back for Claude.
        self.assertIsNotNone(self.weights.model_entry("codex", "gpt-5.5"))
        self.assertIsNone(self.weights.model_entry("claude", "gpt-5.5"))
        self.assertIsNotNone(self.weights.model_entry("codex", "gpt-5.5"))

    def test_invalidate_picks_up_a_table_edit(self) -> None:
        self.assertIsNone(self.weights.model_entry("codex", "made-up-model"))
        self.weights.table["codex"]["models"]["made-up-model"] = {
            "input": 1.0, "cached_input": 0.5, "output": 2.0
        }
        self.assertIsNone(self.weights.model_entry("codex", "made-up-model"))
        self.weights.invalidate()
        self.assertIsNotNone(self.weights.model_entry("codex", "made-up-model"))
        self.assertEqual(
            self.weights.codex_units("made-up-model", {"input": 1_000_000}), 1.0
        )

    def test_cache_read_weight_is_part_of_the_key(self) -> None:
        tokens = {"cache_read": 1_000_000}
        plain = self.weights.claude_units("claude-sonnet-5", tokens, None)
        weighted = self.weights.claude_units("claude-sonnet-5", tokens, 1.0)
        self.assertNotEqual(plain, weighted)
        # Re-asking in the other order must not serve the other row's memo.
        self.assertEqual(self.weights.claude_units("claude-sonnet-5", tokens, 1.0), weighted)
        self.assertEqual(self.weights.claude_units("claude-sonnet-5", tokens, None), plain)


class EventVectorEquivalence(unittest.TestCase):
    """`event_vector` + `vector_units` must be bit-identical to the dict path."""

    def setUp(self) -> None:
        self.weights = QD.load_weights(False)
        self.args = argparse.Namespace(
            claude_cache_read_weight=None, long_context_multiplier=1.0
        )

    def check(self, harness: str, model: str, tokens: Sequence[int]) -> None:
        kinds = QD.CLAUDE_KINDS if harness == "claude" else QD.CODEX_KINDS
        row = synthetic_event("s1", model, 1_700_000_000.0, tokens)
        as_dict = QD.event_tokens(row, kinds)
        for cache_read_weight in (None, 0.0, 0.1, 0.5):
            vector = self.weights.event_vector(harness, model, cache_read_weight)
            expected = QD.weighted_units(
                harness, model, as_dict, self.weights,
                argparse.Namespace(
                    claude_cache_read_weight=cache_read_weight,
                    long_context_multiplier=1.0,
                ),
            )
            actual = 0.0 if vector is None else QD.vector_units(vector, row)
            self.assertEqual(actual, expected, (harness, model, cache_read_weight))
            input_vector = self.weights.event_vector(
                harness, model, cache_read_weight, True
            )
            expected_input = QD.input_side_units(
                harness, model, as_dict, self.weights, cache_read_weight
            )
            actual_input = 0.0 if input_vector is None else QD.vector_units(input_vector, row)
            self.assertEqual(actual_input, expected_input, (harness, model, cache_read_weight))

    def test_claude_models(self) -> None:
        for model in ("claude-sonnet-5", "claude-opus-5", "unweighted-nonsense"):
            self.check("claude", model, (11_111, 222_222, 3_333, 444, 55_555))

    def test_codex_models(self) -> None:
        for model in ("gpt-5.5", "gpt-5.4", "unweighted-nonsense"):
            self.check("codex", model, (98_765, 4_321_000, 777, 12_345))

    def test_output_is_priced_out_of_the_input_side_vector(self) -> None:
        row = synthetic_event("s1", "claude-sonnet-5", 1.0, (0, 0, 0, 0, 1_000_000))
        vector = self.weights.event_vector("claude", "claude-sonnet-5", None, True)
        self.assertEqual(QD.vector_units(vector, row), 0.0)
        full = self.weights.event_vector("claude", "claude-sonnet-5", None, False)
        self.assertGreater(QD.vector_units(full, row), 0.0)

    def test_unweighted_model_has_no_vector(self) -> None:
        self.assertIsNone(self.weights.event_vector("codex", "not-a-real-model"))
        self.assertIsNone(self.weights.event_vector("claude", "not-a-real-model"))


class ProfileFlag(Harness):
    def test_profile_prints_phase_timings_to_stderr(self) -> None:
        now = time.time() - 600
        session = "77777777-aaaa-2222-3333-444444444444"
        self.write_claude(
            "profiled.jsonl",
            [claude_assistant_line(now, session, "msg_1", output_tokens=100)],
        )
        result = self.run_tool("sessions", "--harness", "claude", "--profile")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        report = result.stderr.decode("utf-8")
        for phase in ("scan", "totals", "prompts", "intervals", "attribute", "total"):
            self.assertRegex(report, r"nenpi: %s +\d+\.\d+s" % phase)
        # The report is stderr-only, so piping stdout to jq or a table stays clean.
        self.assertNotIn("nenpi: total", result.stdout.decode("utf-8"))

    def test_profile_is_silent_by_default(self) -> None:
        now = time.time() - 600
        session = "88888888-aaaa-2222-3333-444444444444"
        self.write_claude(
            "quiet.jsonl",
            [claude_assistant_line(now, session, "msg_1", output_tokens=100)],
        )
        result = self.run_tool("sessions", "--harness", "claude")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertNotIn("nenpi: total", result.stderr.decode("utf-8"))

    def test_nenpi_profile_env_var_turns_it_on(self) -> None:
        now = time.time() - 600
        session = "99999999-aaaa-2222-3333-444444444444"
        self.write_claude(
            "env.jsonl",
            [claude_assistant_line(now, session, "msg_1", output_tokens=100)],
        )
        result = self.run_tool(
            "sessions", "--harness", "claude", extra_env={"NENPI_PROFILE": "1"}
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn("nenpi: total", result.stderr.decode("utf-8"))


class ToolAttribution(Harness):
    """Tool names and result SIZES only; never a byte of tool content."""

    def claude_session(self, name: str, session: str) -> float:
        now = time.time() - 3600
        self.write_claude(
            name,
            [
                claude_user_prompt_line(now, session),
                claude_tool_use_line(
                    now + 1, session, "msg_1",
                    [("toolu_a", "Bash"), ("toolu_b", "mcp__github__list_issues")],
                    input_tokens=100, cache_read=1000, output_tokens=20,
                ),
                claude_tool_output_line(now + 2, session, "toolu_a", "x" * 400),
                claude_tool_output_line(
                    now + 3, session, "toolu_b",
                    [{"type": "text", "text": "y" * 800}],
                ),
                claude_assistant_line(
                    now + 4, session, "msg_2",
                    input_tokens=100, cache_read=2200, output_tokens=30,
                ),
            ],
        )
        return now

    def two_matching_tool_sessions(self) -> Tuple[str, str]:
        """Two sessions with tool calls, one id a strict prefix of the other."""
        busy = "20000000-1111-2222"
        quiet = "20000000-1111-2222-3333-444444444444"
        now = time.time() - 3600
        for tag, session, output in (("busy", busy, 100_000), ("quiet", quiet, 10)):
            self.write_claude(
                "tools-%s.jsonl" % tag,
                [
                    claude_user_prompt_line(now, session),
                    claude_tool_use_line(
                        now + 1, session, "msg_%s_1" % tag, [("toolu_%s" % tag, "Bash")],
                        input_tokens=100, cache_read=1000, output_tokens=output,
                    ),
                    claude_tool_output_line(now + 2, session, "toolu_%s" % tag, "x" * 400),
                    claude_assistant_line(
                        now + 3, session, "msg_%s_2" % tag,
                        input_tokens=100, cache_read=2200, output_tokens=30,
                    ),
                ],
            )
        return busy, quiet

    def test_tools_ambiguous_session_prefix_fails(self) -> None:
        busy, _quiet = self.two_matching_tool_sessions()
        result = self.run_tool("tools", "--harness", "claude", "--session", "2000000", "--json")
        self.assertEqual(result.returncode, 1)
        stderr = result.stderr.decode("utf-8")
        self.assertIn("matches 2 sessions", stderr)
        self.assertEqual(result.stdout.decode("utf-8"), "")
        # The full id still resolves, even though it prefixes the other one.
        payload = self.run_json("tools", "--harness", "claude", "--session", busy, "--json")
        self.assertEqual(payload["session_id"], busy)
        self.assertEqual(payload["session"], QD.short_id(busy))

    def test_tools_first_picks_the_busiest_match(self) -> None:
        busy, _quiet = self.two_matching_tool_sessions()
        payload = self.run_json(
            "tools", "--harness", "claude", "--session", "2000000", "--first", "--json"
        )
        self.assertEqual(payload["session_id"], busy)

    def test_claude_tool_names_and_sizes_are_parsed(self) -> None:
        session = "10000000-1111-2222-3333-444444444444"
        self.claude_session("tools.jsonl", session)
        payload = self.run_json("tools", "--harness", "claude", "--json")
        names = dict((row["tool"], row) for row in payload["tools"])
        self.assertEqual(sorted(names), ["Bash", "mcp__github__list_issues"])
        self.assertEqual(names["Bash"]["result_chars"], 400)
        self.assertEqual(names["Bash"]["est_tokens"], 100.0)
        # MCP names survive whole, server and tool.
        self.assertEqual(names["mcp__github__list_issues"]["result_chars"], 800)
        self.assertEqual(payload["tool_calls"], 2)

    def test_measured_growth_splits_over_the_turn(self) -> None:
        session = "11000000-1111-2222-3333-444444444444"
        self.claude_session("measured.jsonl", session)
        payload = self.run_json("tools", "--harness", "claude", "--json")
        names = dict((row["tool"], row) for row in payload["tools"])
        # Context grew 1100 -> 2300 tokens across the two calls; the split is
        # proportional to the 400/800 result sizes.
        self.assertAlmostEqual(names["Bash"]["measured_tokens"], 400.0, places=6)
        self.assertAlmostEqual(
            names["mcp__github__list_issues"]["measured_tokens"], 800.0, places=6
        )
        self.assertEqual(payload["context_growth_tokens"], 1200)

    def test_unmatched_tool_result_is_kept_as_unknown(self) -> None:
        session = "12000000-1111-2222-3333-444444444444"
        now = time.time() - 3600
        self.write_claude(
            "orphan.jsonl",
            [
                claude_user_prompt_line(now, session),
                claude_assistant_line(now + 1, session, "msg_1", output_tokens=5),
                claude_tool_output_line(now + 2, session, "toolu_missing", "z" * 40),
            ],
        )
        payload = self.run_json("tools", "--harness", "claude", "--json")
        self.assertEqual([row["tool"] for row in payload["tools"]], ["unknown"])
        self.assertEqual(payload["tools"][0]["result_chars"], 40)

    def test_claude_subagent_spawn_is_flagged(self) -> None:
        session = "13000000-1111-2222-3333-444444444444"
        now = time.time() - 3600
        self.write_claude(
            "spawn.jsonl",
            [
                claude_user_prompt_line(now, session),
                claude_tool_use_line(
                    now + 1, session, "msg_1", [("toolu_t", "Task")], output_tokens=5
                ),
                claude_tool_output_line(now + 2, session, "toolu_t", "s" * 20),
            ],
        )
        payload = self.run_json("tools", "--harness", "claude", "--json")
        self.assertEqual(payload["tools"][0]["spawns"], 1)
        self.assertTrue(payload["largest_results"][0]["spawned_subagent"])

    def codex_session(self, name: str, session: str) -> float:
        now = time.time() - 3600
        self.write_codex(
            name,
            [
                codex_session_meta_line(now, session, "/home/agent/project"),
                codex_turn_context_line(now + 1, "gpt-5-codex"),
                codex_task_started_line(now + 1),
                codex_tool_call_line(now + 2, "call_1", "exec", item="custom_tool_call"),
                codex_tool_output_line(
                    now + 3, "call_1",
                    [{"type": "text", "text": "a" * 600}],
                    item="custom_tool_call_output",
                ),
                codex_tool_call_line(
                    now + 4, "call_2", "search", namespace="mcp__docs"
                ),
                codex_tool_output_line(now + 5, "call_2", "b" * 200),
                codex_usage_record_line(
                    now + 6, session, input_tokens=5000,
                    cached_input_tokens=1000, output_tokens=50, turn_id="turn-1",
                ),
            ],
            day=now,
        )
        return now

    def test_codex_function_and_custom_tool_calls_are_parsed(self) -> None:
        session = "20000000-1111-2222-3333-444444444444"
        self.codex_session("rollout-tools.jsonl", session)
        payload = self.run_json("tools", "--harness", "codex", "--json")
        names = dict((row["tool"], row) for row in payload["tools"])
        self.assertEqual(sorted(names), ["exec", "mcp__docs.search"])
        self.assertEqual(names["exec"]["result_chars"], 600)
        self.assertEqual(names["mcp__docs.search"]["result_chars"], 200)

    def test_session_prefix_scopes_the_report(self) -> None:
        first = "30000000-1111-2222-3333-444444444444"
        second = "40000000-1111-2222-3333-444444444444"
        self.claude_session("one.jsonl", first)
        now = time.time() - 3600
        self.write_claude(
            "two.jsonl",
            [
                claude_user_prompt_line(now, second),
                claude_tool_use_line(
                    now + 1, second, "msg_9", [("toolu_z", "Read")], output_tokens=5
                ),
                claude_tool_output_line(now + 2, second, "toolu_z", "q" * 120),
            ],
        )
        payload = self.run_json(
            "tools", "--harness", "claude", "--session", "40000000", "--json"
        )
        self.assertEqual([row["tool"] for row in payload["tools"]], ["Read"])
        self.assertEqual(payload["session"], "40000000")

    def test_text_output_lists_tools_without_content(self) -> None:
        session = "50000000-1111-2222-3333-444444444444"
        self.claude_session("text.jsonl", session)
        result = self.run_tool("tools", "--harness", "claude", "--no-color")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        text = result.stdout.decode("utf-8")
        self.assertIn("Bash", text)
        self.assertIn("mcp__github__list_issues", text)
        self.assertIn("largest single results", text)
        self.assertNotIn("x" * 20, text)
        self.assertNotIn("y" * 20, text)

    def test_sort_by_calls(self) -> None:
        session = "51000000-1111-2222-3333-444444444444"
        now = time.time() - 3600
        lines = [claude_user_prompt_line(now, session)]
        for step in range(3):
            lines.append(
                claude_tool_use_line(
                    now + 1 + step, session, "msg_%d" % step,
                    [("toolu_s%d" % step, "Read")], output_tokens=5,
                )
            )
            lines.append(
                claude_tool_output_line(now + 1.5 + step, session, "toolu_s%d" % step, "r" * 10)
            )
        lines.append(
            claude_tool_use_line(now + 8, session, "msg_big", [("toolu_big", "Bash")],
                                 output_tokens=5)
        )
        lines.append(claude_tool_output_line(now + 9, session, "toolu_big", "B" * 5000))
        self.write_claude("sorted.jsonl", lines)
        by_context = self.run_json("tools", "--harness", "claude", "--json")
        self.assertEqual(by_context["tools"][0]["tool"], "Bash")
        by_calls = self.run_json(
            "tools", "--harness", "claude", "--sort", "calls", "--json"
        )
        self.assertEqual(by_calls["tools"][0]["tool"], "Read")

    def test_prompts_tools_columns_are_opt_in(self) -> None:
        session = "52000000-1111-2222-3333-444444444444"
        self.claude_session("prompt-tools.jsonl", session)
        plain = self.run_tool(
            "prompts", "--session", "52000000", "--harness", "claude", "--no-color"
        )
        self.assertEqual(plain.returncode, 0, plain.stderr.decode("utf-8", "replace"))
        self.assertNotIn("largest tool", plain.stdout.decode("utf-8"))
        with_tools = self.run_tool(
            "prompts", "--session", "52000000", "--harness", "claude",
            "--no-color", "--tools",
        )
        self.assertEqual(with_tools.returncode, 0,
                         with_tools.stderr.decode("utf-8", "replace"))
        text = with_tools.stdout.decode("utf-8")
        self.assertIn("largest tool", text)
        self.assertIn("mcp__github__li", text)
        payload = self.run_json(
            "prompts", "--session", "52000000", "--harness", "claude", "--json"
        )
        prompt = payload["prompts"][0]
        self.assertEqual(prompt["tool_calls"], 2)
        self.assertEqual(prompt["tool_result_chars"], 1200)
        self.assertEqual(prompt["largest_tool"], "mcp__github__list_issues")

    def test_shard_round_trip_stores_sizes_only(self) -> None:
        session = "60000000-1111-2222-3333-444444444444"
        self.claude_session("shard.jsonl", session)
        first = self.run_json("tools", "--harness", "claude", "--json")
        shards = list((self.root / "cache").rglob("*.json"))
        self.assertTrue(shards)
        blob = "\n".join(path.read_text(encoding="utf-8") for path in shards)
        self.assertIn("Bash", blob)
        self.assertNotIn("x" * 20, blob)
        self.assertNotIn("y" * 20, blob)
        self.assertNotIn("do the thing", blob)
        # Second run is served from the shard and must agree.
        second = self.run_json("tools", "--harness", "claude", "--json")
        self.assertEqual(first["tools"], second["tools"])

    def test_old_schema_shards_are_rebuilt(self) -> None:
        session = "61000000-1111-2222-3333-444444444444"
        self.claude_session("schema.jsonl", session)
        self.run_json("sessions", "--harness", "claude", "--json")
        current = self.root / "cache" / ("v%d" % QD.CACHE_SCHEMA)
        self.assertTrue(current.is_dir())
        stale = self.root / "cache" / ("v%d" % (QD.CACHE_SCHEMA - 1))
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "old.json").write_text("{}", encoding="utf-8")
        payload = self.run_json("tools", "--harness", "claude", "--json")
        self.assertFalse(stale.exists())
        self.assertEqual(payload["tool_calls"], 2)

    def test_replayed_tool_calls_count_once(self) -> None:
        session = "62000000-1111-2222-3333-444444444444"
        resumed = "63000000-1111-2222-3333-444444444444"
        now = time.time() - 3600
        lines = [
            claude_user_prompt_line(now, session),
            claude_tool_use_line(
                now + 1, session, "msg_1", [("toolu_r", "Grep")], output_tokens=5
            ),
            claude_tool_output_line(now + 2, session, "toolu_r", "g" * 100),
        ]
        self.write_claude("origin.jsonl", lines)
        # A resumed transcript replays the same call under a new session id.
        self.write_claude(
            "resumed.jsonl",
            [line.replace(session, resumed) for line in lines],
        )
        payload = self.run_json("tools", "--harness", "claude", "--json")
        self.assertEqual(payload["tool_calls"], 1)


if __name__ == "__main__":
    unittest.main()
