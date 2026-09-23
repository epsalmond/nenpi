"""Focused fixtures for Codex prompt drilldown."""

from __future__ import annotations

import json
import time
import unittest
from typing import Any, Dict

from tests.test_drain import (
    Harness,
    codex_session_meta_line,
    codex_subagent_meta_line,
    codex_task_started_line,
    codex_tool_call_line,
    codex_tool_output_line,
    codex_turn_context_line,
    iso,
)


def codex_usage_line(
    epoch: float,
    session_id: str,
    thread_id: str,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    reasoning_output_tokens: int = 0,
    turn_id: str | None = None,
) -> str:
    usage = {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_input_tokens": 0,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": input_tokens + cached_input_tokens + output_tokens,
    }
    payload: Dict[str, Any] = {
        "thread_id": thread_id,
        "session_id": session_id,
        "response_id": "drill-resp-%f" % epoch,
        "usage": usage,
        "turn_token_usage": usage,
        "thread_token_usage": usage,
    }
    if turn_id is not None:
        payload["turn_id"] = turn_id
        payload["root_turn_id"] = turn_id
    return json.dumps({"type": "token_usage_record", "timestamp": iso(epoch), "payload": payload})


def nested_meta_line(epoch: float, thread_id: str, session_id: str, parent: str, depth: int) -> str:
    payload = json.loads(codex_subagent_meta_line(epoch, thread_id, session_id, "/home/agent/project"))
    payload["payload"]["parent_thread_id"] = parent
    payload["payload"]["source"]["subagent"]["thread_spawn"]["parent_thread_id"] = parent
    payload["payload"]["source"]["subagent"]["thread_spawn"]["depth"] = depth
    return json.dumps(payload)


def unknown_meta_line(epoch: float, thread_id: str, session_id: str) -> str:
    return json.dumps({
        "type": "session_meta",
        "timestamp": iso(epoch),
        "payload": {
            "id": thread_id,
            "session_id": session_id,
            "timestamp": iso(epoch),
            "cwd": "/home/agent/project",
            "source": {"subagent": {"thread_spawn": {"depth": 1}}},
        },
    })


class TestCodexPromptDrilldown(Harness):
    def test_drilldown_reconciles_lineage_tools_messages_and_waits(self) -> None:
        base = time.time() - 900
        session = "drill-session-0001"
        root = session
        child = "drill-child-0001"
        lines = [
            codex_session_meta_line(base, root, "/home/agent/project"),
            codex_task_started_line(base + 1),
            codex_turn_context_line(base + 1.1, "gpt-5.5"),
            codex_tool_call_line(base + 2, "wait-1", "wait_agent", namespace="collaboration"),
            codex_usage_line(base + 2.1, session, root, input_tokens=100, cached_input_tokens=20,
                             output_tokens=10, turn_id="turn-1"),
            codex_usage_line(base + 2.1, session, root, input_tokens=100, cached_input_tokens=20,
                             output_tokens=10, turn_id="turn-1"),
            codex_tool_call_line(base + 3, "wait-2", "wait_agent", namespace="collaboration"),
            codex_usage_line(base + 3.1, session, root, input_tokens=40, cached_input_tokens=30,
                             output_tokens=8, turn_id="turn-1"),
            codex_tool_call_line(
                base + 3.2, "mixed-1", "send_message", namespace="collaboration",
                command=json.dumps({"target_thread_id": child, "message": "mixed action"}),
            ),
            codex_usage_line(base + 3.3, session, root, input_tokens=6, cached_input_tokens=2,
                             output_tokens=2, turn_id="turn-1"),
            codex_subagent_meta_line(base + 1.5, child, session, "/home/agent/project"),
            codex_tool_call_line(
                base + 2.5,
                "message-1",
                "send_message",
                namespace="collaboration",
                command=json.dumps({"target_thread_id": child, "message": "opaque body"}),
            ),
            codex_tool_call_line(
                base + 2.5,
                "message-1",
                "send_message",
                namespace="collaboration",
                command=json.dumps({"target_thread_id": child, "message": "opaque body"}),
            ),
            codex_tool_output_line(base + 2.6, "message-1", "opaque result"),
            codex_usage_line(base + 2.7, session, child, input_tokens=12, cached_input_tokens=3,
                             output_tokens=7, reasoning_output_tokens=4),
        ]
        self.write_codex("rollout-drill.jsonl", lines, day=base)
        grandchild = "drill-grandchild-0001"
        self.write_codex("rollout-grandchild.jsonl", [
            nested_meta_line(base + 1.6, grandchild, session, child, 2),
            codex_usage_line(base + 2.8, session, grandchild, input_tokens=5, cached_input_tokens=1, output_tokens=2),
        ], day=base)
        unknown = "drill-unknown-0001"
        self.write_codex("rollout-unknown.jsonl", [
            unknown_meta_line(base + 1.7, unknown, session),
            codex_usage_line(base + 2.9, session, unknown, input_tokens=3, cached_input_tokens=0, output_tokens=1),
        ], day=base)
        self.write_codex("rollout-next-prompt.jsonl", [
            codex_session_meta_line(base + 3.9, root, "/home/agent/project"),
            codex_task_started_line(base + 4),
            codex_turn_context_line(base + 4.1, "gpt-5.5"),
            codex_usage_line(base + 4.2, session, root, input_tokens=8, cached_input_tokens=0, output_tokens=2, turn_id="turn-2"),
        ], day=base)

        result = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        payload = json.loads(result.stdout.decode("utf-8"))
        drilldown = payload["drilldown"]
        self.assertTrue(drilldown["reconciliation"]["matches_prompt"])
        self.assertEqual(drilldown["root"]["api_calls"], 3)
        self.assertEqual(len(drilldown["descendants"]), 2)
        self.assertEqual(drilldown["descendants"][0]["reasoning_output_tokens"], 4)
        self.assertEqual(drilldown["descendants"][0]["tool_family_rankings"][0]["tool"],
                         "collaboration.send_message")
        self.assertEqual(drilldown["descendants"][1]["depth"], 2)
        self.assertEqual(drilldown["unknown"]["api_calls"], 1)
        self.assertEqual(drilldown["agent_rankings"][0]["rank"], 1)
        self.assertIn("explicit_lineage", drilldown["attribution"])
        self.assertEqual(drilldown["descendants"][0]["messages"]["known_payload_bytes"],
                         drilldown["descendants"][0]["messages"]["payload_bytes"])
        self.assertEqual(drilldown["root"]["wait_streaks"][0]["length"], 2)
        self.assertEqual(drilldown["root"]["action_counts"]["collaboration.send_message"], 1)
        self.assertNotIn("opaque body", result.stdout.decode("utf-8", "replace"))
        for path in (self.root / "cache").rglob("*"):
            if path.is_file():
                self.assertNotIn("opaque body", path.read_text(encoding="utf-8"))
        warm = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--json"
        )
        self.assertEqual(json.loads(warm.stdout.decode())["drilldown"], drilldown)
        text = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--no-color"
        )
        rendered = text.stdout.decode("utf-8", "replace")
        for phrase in ("agent rankings", "tool", "messages:", "wait streak:", "reasoning_output", "combined totals", "caveats"):
            self.assertIn(phrase, rendered)
        second_prompt = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "2", "--drilldown", "--json"
        )
        self.assertEqual(json.loads(second_prompt.stdout.decode())["drilldown"]["root"]["api_calls"], 1)

    def test_drilldown_flags_validate_session_and_prompt(self) -> None:
        for arguments in (
            ("prompts", "--prompt", "1", "--json"),
            ("prompts", "--drilldown", "--json"),
        ):
            result = self.run_tool(*arguments)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires", result.stderr.decode("utf-8", "replace"))
