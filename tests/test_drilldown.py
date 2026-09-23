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
from nenpi.drain import codex_message_metadata


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
    def test_broken_cyclic_and_missing_parent_chains_are_unknown(self) -> None:
        base = time.time() - 900
        session = "lineage-session-0001"
        missing = "lineage-missing-0001"
        broken_parent = "lineage-broken-parent-0001"
        broken = "lineage-broken-0001"
        cycle_a = "lineage-cycle-a-0001"
        cycle_b = "lineage-cycle-b-0001"
        self.write_codex("rollout-root.jsonl", [
            codex_session_meta_line(base, session, "/home/agent/project"),
            codex_task_started_line(base + 1),
            codex_turn_context_line(base + 1.1, "gpt-5.5"),
            codex_usage_line(base + 2, session, session, input_tokens=1, cached_input_tokens=0, output_tokens=1),
        ], day=base)
        self.write_codex("rollout-missing.jsonl", [
            nested_meta_line(base + 1.2, missing, session, "absent-parent-0001", 1),
            codex_usage_line(base + 2.2, session, missing, input_tokens=2, cached_input_tokens=0, output_tokens=1),
        ], day=base)
        self.write_codex("rollout-broken.jsonl", [
            nested_meta_line(base + 1.3, broken_parent, session, "absent-parent-0002", 1),
            nested_meta_line(base + 1.4, broken, session, broken_parent, 2),
            codex_usage_line(base + 2.25, session, broken_parent, input_tokens=2, cached_input_tokens=0, output_tokens=1),
            codex_usage_line(base + 2.3, session, broken, input_tokens=3, cached_input_tokens=0, output_tokens=1),
        ], day=base)
        self.write_codex("rollout-cycle.jsonl", [
            nested_meta_line(base + 1.5, cycle_a, session, cycle_b, 1),
            nested_meta_line(base + 1.6, cycle_b, session, cycle_a, 1),
            codex_usage_line(base + 2.4, session, cycle_a, input_tokens=4, cached_input_tokens=0, output_tokens=1),
            codex_usage_line(base + 2.5, session, cycle_b, input_tokens=5, cached_input_tokens=0, output_tokens=1),
        ], day=base)

        result = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        drilldown = json.loads(result.stdout.decode("utf-8"))["drilldown"]
        unknown_threads = {item["thread_id"] for item in drilldown["unknown_threads"]}
        self.assertEqual(unknown_threads, {missing, broken_parent, broken, cycle_a, cycle_b})
        self.assertEqual(len(drilldown["descendants"]), 0)

    def test_message_payload_sizes_use_utf8_bytes_and_missing_is_unknown(self) -> None:
        raw = '{"target_thread_id":"child","message":"café"}'
        target, payload_size = codex_message_metadata({"arguments": raw})
        self.assertEqual(target, "child")
        self.assertEqual(payload_size, len(raw.encode("utf-8")))

        structured = {"target_thread_id": "child", "message": "café"}
        _, structured_size = codex_message_metadata({"input": structured})
        expected = json.dumps(structured, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.assertEqual(structured_size, len(expected))
        self.assertEqual(codex_message_metadata({}), ("unknown", None))

    def test_mixed_message_payload_aggregates_are_partial_not_zero(self) -> None:
        base = time.time() - 900
        session = "mixed-payload-session-0001"
        unknown_thread = "mixed-payload-unknown-0001"
        missing_payload = json.dumps({
            "type": "response_item",
            "timestamp": iso(base + 2.5),
            "payload": {
                "type": "function_call",
                "call_id": "missing-payload",
                "name": "send_message",
                "namespace": "collaboration",
            },
        })
        self.write_codex("rollout-root.jsonl", [
            codex_session_meta_line(base, session, "/home/agent/project"),
            codex_task_started_line(base + 1),
            codex_turn_context_line(base + 1.1, "gpt-5.5"),
            codex_tool_call_line(
                base + 2, "known-payload", "send_message", namespace="collaboration",
                command=json.dumps({"target_thread_id": unknown_thread, "message": "known"}),
            ),
            codex_tool_output_line(base + 2.1, "known-payload", "ok"),
            codex_usage_line(base + 2.2, session, session, input_tokens=10, cached_input_tokens=0, output_tokens=1),
        ], day=base)
        self.write_codex("rollout-unknown.jsonl", [
            nested_meta_line(base + 1.2, unknown_thread, session, "missing-parent-0001", 1),
            missing_payload,
            codex_tool_output_line(base + 2.6, "missing-payload", "ok"),
            codex_usage_line(base + 2.7, session, unknown_thread, input_tokens=11, cached_input_tokens=0, output_tokens=1),
        ], day=base)

        result = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        drilldown = json.loads(result.stdout.decode("utf-8"))["drilldown"]
        root_messages = drilldown["root"]["messages"]
        unknown_messages = drilldown["unknown"]["messages"]
        combined_messages = drilldown["combined"]["messages"]
        self.assertEqual(root_messages["calls"], 1)
        self.assertEqual(root_messages["payload_bytes"], root_messages["known_payload_bytes"])
        self.assertGreater(root_messages["known_payload_bytes"], 0)
        self.assertIsNone(unknown_messages["payload_bytes"])
        self.assertEqual(unknown_messages["known_payload_bytes"], 0)
        self.assertEqual(unknown_messages["unknown_payload_calls"], 1)
        self.assertIsNone(combined_messages["payload_bytes"])
        self.assertEqual(combined_messages["known_payload_bytes"], root_messages["known_payload_bytes"])
        routes = {str(route["target"]): route for route in unknown_messages["routes"]}
        self.assertIsNone(routes["unknown"]["payload_bytes"])
        self.assertEqual(routes["unknown"]["unknown_payload_calls"], 1)
        self.assertEqual(routes["unknown"]["known_payload_bytes"], 0)

        text_result = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--no-color",
        )
        text = text_result.stdout.decode("utf-8")
        self.assertIn("payload_bytes=unknown/partial", text)
        self.assertIn("known_payload_bytes=0", text)
        self.assertIn("unknown_payloads=1", text)

    def test_assistant_message_breaks_pure_wait_streak(self) -> None:
        base = time.time() - 900
        session = "wait-break-session-0001"
        assistant_message = json.dumps({
            "type": "response_item",
            "timestamp": iso(base + 2.5),
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        })
        self.write_codex("rollout-wait-break.jsonl", [
            codex_session_meta_line(base, session, "/home/agent/project"),
            codex_task_started_line(base + 1),
            codex_turn_context_line(base + 1.1, "gpt-5.5"),
            codex_tool_call_line(base + 2, "wait-1", "wait_agent", namespace="collaboration"),
            codex_usage_line(base + 2.1, session, session, input_tokens=10, cached_input_tokens=1, output_tokens=1),
            assistant_message,
            codex_tool_call_line(base + 3, "wait-2", "wait_agent", namespace="collaboration"),
            codex_usage_line(base + 3.1, session, session, input_tokens=12, cached_input_tokens=2, output_tokens=1),
        ], day=base)

        result = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        drilldown = json.loads(result.stdout.decode("utf-8"))["drilldown"]
        self.assertEqual([streak["length"] for streak in drilldown["root"]["wait_streaks"]], [1])
        text_result = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--no-color",
        )
        text = text_result.stdout.decode("utf-8")
        for label in ("first_context=", "last_context=", "start=", "end="):
            self.assertIn(label, text)

    def test_unknown_thread_keeps_auxiliary_data_before_aggregate(self) -> None:
        base = time.time() - 900
        session = "unknown-data-session-0001"
        thread = "unknown-data-thread-0001"
        self.write_codex("rollout-root.jsonl", [
            codex_session_meta_line(base, session, "/home/agent/project"),
            codex_task_started_line(base + 1),
            codex_turn_context_line(base + 1.1, "gpt-5.5"),
            codex_usage_line(base + 2, session, session, input_tokens=1, cached_input_tokens=0, output_tokens=1),
        ], day=base)
        self.write_codex("rollout-unknown-data.jsonl", [
            nested_meta_line(base + 1.2, thread, session, "missing-parent-0001", 1),
            codex_tool_call_line(
                base + 2.1, "message-unknown", "send_message", namespace="collaboration",
                command=json.dumps({"target_thread_id": "other-thread", "message": "opaque"}),
            ),
            codex_tool_output_line(base + 2.2, "message-unknown", "result"),
            codex_usage_line(base + 2.3, session, thread, input_tokens=7, cached_input_tokens=0, output_tokens=1),
            codex_tool_call_line(base + 3.1, "wait-unknown", "wait_agent", namespace="collaboration"),
            codex_usage_line(base + 3.2, session, thread, input_tokens=8, cached_input_tokens=1, output_tokens=1),
        ], day=base)

        result = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        drilldown = json.loads(result.stdout.decode("utf-8"))["drilldown"]
        self.assertEqual([item["thread_id"] for item in drilldown["unknown_threads"]], [thread])
        self.assertEqual(drilldown["unknown_threads"][0]["messages"]["calls"], 1)
        self.assertEqual(drilldown["unknown_threads"][0]["tool_family_rankings"][0]["tool"], "collaboration.send_message")
        self.assertEqual(len(drilldown["unknown_threads"][0]["wait_streaks"]), 1)
        self.assertEqual(drilldown["unknown"]["messages"]["calls"], 1)

    def test_inherited_root_metadata_does_not_reparent_child_usage(self) -> None:
        base = time.time() - 900
        session = "inherited-session-0001"
        child = "inherited-child-0001"
        self.write_codex("rollout-root.jsonl", [
            codex_session_meta_line(base, session, "/home/agent/project"),
            codex_task_started_line(base + 1),
            codex_turn_context_line(base + 1.1, "gpt-5.5"),
            codex_usage_line(base + 2, session, session, input_tokens=10, cached_input_tokens=1, output_tokens=2, turn_id="turn-1"),
        ], day=base)
        self.write_codex("rollout-child.jsonl", [
            codex_subagent_meta_line(base + 1.5, child, session, "/home/agent/project"),
            codex_usage_line(base + 2.5, session, child, input_tokens=20, cached_input_tokens=2, output_tokens=3),
            codex_session_meta_line(base + 3, session, "/home/agent/project"),
            codex_usage_line(base + 3.5, session, child, input_tokens=30, cached_input_tokens=3, output_tokens=4),
        ], day=base)

        result = self.run_tool(
            "prompts", "--harness", "codex", "--codex-root", str(self.home / ".codex"),
            "--session", session, "--prompt", "1", "--drilldown", "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        drilldown = json.loads(result.stdout.decode("utf-8"))["drilldown"]
        self.assertEqual(drilldown["root"]["api_calls"], 1)
        self.assertEqual(len(drilldown["descendants"]), 1)
        self.assertEqual(drilldown["descendants"][0]["thread_id"], child)
        self.assertEqual(drilldown["descendants"][0]["api_calls"], 2)

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
