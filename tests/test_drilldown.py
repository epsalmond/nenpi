"""Focused fixtures for Codex prompt drilldown."""

from __future__ import annotations

import json
import os
import time
import unittest
from typing import Any, Dict

from tests.test_drain import (
    Harness,
    claude_assistant_line,
    claude_user_prompt_line,
    codex_session_meta_line,
    codex_subagent_meta_line,
    codex_task_started_line,
    codex_tool_call_line,
    codex_tool_output_line,
    codex_turn_context_line,
    iso,
)
from nenpi.drain import (
    claude_cache_write_tokens,
    codex_message_metadata,
    load_weights,
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


class TestClaudePromptDrilldown(Harness):
    def test_linked_agent_drilldown_reconciles_disjoint_usage(self) -> None:
        base = time.time() - 900
        session = "claude-drilldown-session-0001"
        parent_call = "claude-agent-call-0001"
        child = "claude-child-0001"

        parent = json.loads(
            claude_assistant_line(
                base + 2, session, "claude-root-message-0001",
                input_tokens=10, cache_read=20, cache_write_5m=3,
                cache_write_1h=4, output_tokens=8,
            )
        )
        parent["message"]["content"] = [{
            "type": "tool_use", "id": parent_call, "name": "Agent",
            "input": {
                "subagent_type": "Explore",
                "description": "synthetic fixture",
                "prompt": "SYNTHETIC PRIVATE ARGUMENT MUST NOT BE CACHED",
            },
        }]
        parent["message"]["usage"]["output_tokens_details"] = {"thinking_tokens": 2}
        first_parent = json.loads(
            claude_assistant_line(
                base + 2, session, "claude-root-message-0001",
                input_tokens=10, cache_read=20, cache_write_5m=3,
                cache_write_1h=4, output_tokens=8,
            )
        )
        first_parent["message"]["usage"]["output_tokens_details"] = {"thinking_tokens": 2}
        child_call = "claude-bash-call-0001"
        child_event = json.loads(
            claude_assistant_line(
                base + 3, session, "claude-child-message-0001",
                model="claude-sonnet-5", input_tokens=7, cache_read=3,
                cache_write_5m=2, cache_write_1h=1, output_tokens=6,
                sidechain=True,
            )
        )
        child_event["message"]["content"] = [{
            "type": "tool_use", "id": child_call, "name": "Bash",
            "input": {"command": "SYNTHETIC CHILD TOOL ARGUMENT MUST NOT BE CACHED"},
        }]
        # Thinking is absent from this usage record and must remain unknown.
        child_event["message"]["usage"].pop("output_tokens_details")
        child_result = {
            "type": "user", "sessionId": session,
            "timestamp": iso(base + 3.1), "isSidechain": True,
            "message": {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": child_call,
                "content": "synthetic result only contributes a size",
            }]},
        }
        self.write_claude(session + ".jsonl", [
            claude_user_prompt_line(base + 1, session, text="synthetic prompt"),
            # Repeated streamed records keep the existing usage dedup, while
            # the later fragment contributes its tool action.
            json.dumps(first_parent),
            json.dumps(parent),
            json.dumps({
                "type": "user", "sessionId": session,
                "timestamp": iso(base + 2.2), "isSidechain": False,
                "message": {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": parent_call,
                    "content": "synthetic agent result",
                }]},
            }),
        ])
        child_path = self.write_claude(
            "subagents/agent-%s.jsonl" % child, [
                json.dumps(child_event), json.dumps(child_result),
            ],
        )
        child_path.with_suffix(".meta.json").write_text(json.dumps({
            "toolUseId": parent_call,
            "spawnDepth": 1,
            "agentType": "Explore",
            "model": "claude-sonnet-5",
        }), encoding="utf-8")

        result = self.run_tool(
            "prompts", "--harness", "all", "--claude-root", str(self.home / ".claude"),
            "--session", session, "--prompt", "1", "--drilldown", "--json", "--quiet",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        response = json.loads(result.stdout.decode("utf-8"))
        drilldown = response["drilldown"]
        self.assertTrue(drilldown["reconciliation"]["matches_prompt"])
        self.assertEqual(drilldown["root"]["api_calls"], 1)
        self.assertEqual(len(drilldown["descendants"]), 1)
        self.assertEqual(drilldown["descendants"][0]["thread_id"], child)
        self.assertEqual(drilldown["root"]["action_counts"]["Agent"], 1)
        self.assertEqual(drilldown["descendants"][0]["reasoning_unknown_calls"], 1)
        self.assertIsNone(drilldown["descendants"][0]["reasoning_output_tokens"])
        self.assertEqual(drilldown["descendants"][0]["known_reasoning_output_tokens"], 0)
        self.assertEqual(drilldown["root"]["reasoning_output_tokens"], 2)

        cached = ""
        for path in (self.root / "cache").rglob("*.json"):
            cached += path.read_text(encoding="utf-8")
        self.assertNotIn("SYNTHETIC PRIVATE ARGUMENT", cached)
        self.assertNotIn("SYNTHETIC CHILD TOOL ARGUMENT", cached)

    def test_nested_missing_and_unmatched_agent_lineage(self) -> None:
        base = time.time() - 900
        session = "claude-lineage-session-0001"
        root_call = "root-agent-call-0001"
        nested_call = "nested-agent-call-0001"
        child = "claude-child-0002"
        grandchild = "claude-grandchild-0002"
        missing = "claude-missing-meta-0002"
        unmatched = "claude-unmatched-meta-0002"

        def usage(epoch: float, message_id: str, actions: list[dict[str, Any]] | None = None) -> str:
            record = json.loads(
                claude_assistant_line(epoch, session, message_id, output_tokens=1)
            )
            record["message"]["content"] = actions or []
            return json.dumps(record)

        self.write_claude(session + ".jsonl", [
            claude_user_prompt_line(base + 1, session, text="synthetic lineage prompt"),
            usage(base + 2, "lineage-root-message", [
                {"type": "tool_use", "id": root_call, "name": "Agent", "input": {}},
            ]),
        ])
        child_path = self.write_claude(
            "subagents/agent-%s.jsonl" % child, [
                usage(base + 3, "lineage-child-message", [
                    {"type": "tool_use", "id": nested_call, "name": "Agent", "input": {}},
                ]),
            ],
        )
        child_path.with_suffix(".meta.json").write_text(json.dumps({
            "toolUseId": root_call, "spawnDepth": 1,
            "agentType": "Explore", "model": "claude-sonnet-5",
        }), encoding="utf-8")
        grandchild_path = self.write_claude(
            "subagents/agent-%s.jsonl" % grandchild,
            [usage(base + 4, "lineage-grandchild-message")],
        )
        grandchild_path.with_suffix(".meta.json").write_text(json.dumps({
            "toolUseId": nested_call, "spawnDepth": 2,
            "agentType": "Explore", "model": "claude-haiku-4-5",
        }), encoding="utf-8")
        self.write_claude(
            "subagents/agent-%s.jsonl" % missing,
            [usage(base + 4.2, "lineage-missing-message")],
        )
        unmatched_path = self.write_claude(
            "subagents/agent-%s.jsonl" % unmatched,
            [usage(base + 4.3, "lineage-unmatched-message")],
        )
        unmatched_path.with_suffix(".meta.json").write_text(json.dumps({
            "toolUseId": "no-matching-parent-call", "spawnDepth": 1,
        }), encoding="utf-8")

        response = self.run_json(
            "prompts", "--harness", "claude", "--claude-root", str(self.home / ".claude"),
            "--session", session, "--prompt", "1", "--drilldown", "--json", "--quiet",
        )
        drilldown = response["drilldown"]
        descendants = {item["thread_id"]: item for item in drilldown["descendants"]}
        self.assertEqual(set(descendants), {child, grandchild})
        self.assertEqual(descendants[grandchild]["parent_thread_id"], child)
        self.assertEqual(descendants[grandchild]["depth"], 2)
        unknown = {item["thread_id"] for item in drilldown["unknown_threads"]}
        self.assertEqual(unknown, {missing, unmatched})
        self.assertTrue(drilldown["reconciliation"]["matches_prompt"])

    def test_claude_wait_streaks_keep_thread_provenance(self) -> None:
        base = time.time() - 900
        session = "claude-wait-lineage-session-0001"
        child = "claude-wait-child-0001"

        def event(epoch: float, message_id: str, sidechain: bool,
                  action: str | None) -> str:
            row = json.loads(
                claude_assistant_line(
                    epoch, session, message_id, input_tokens=int(epoch - base),
                    output_tokens=1, sidechain=sidechain,
                )
            )
            row["message"]["content"] = (
                [{"type": "tool_use", "id": "wait-" + message_id,
                  "name": "TaskOutput", "input": {"block": True}}]
                if action == "wait" else
                [{"type": "tool_use", "id": "run-" + message_id,
                  "name": "Bash", "input": {}}]
                if action == "other" else []
            )
            return json.dumps(row)

        self.write_claude(session + ".jsonl", [
            claude_user_prompt_line(base + 1, session),
            event(base + 2, "wait-root-first", False, "wait"),
            event(base + 5, "wait-root-last", False, "wait"),
        ])
        self.write_claude(
            "subagents/agent-%s.jsonl" % child,
            [
                event(base + 3, "wait-child-first", True, "wait"),
                event(base + 4, "wait-child-nonwait", True, "other"),
            ],
        )

        drilldown = self.run_json(
            "prompts", "--harness", "claude", "--claude-root", str(self.home / ".claude"),
            "--session", session, "--prompt", "1", "--drilldown", "--json", "--quiet",
        )["drilldown"]
        streaks = drilldown["combined"]["wait_streaks"]
        by_thread = {item["thread_id"]: item for item in streaks}
        self.assertEqual(set(by_thread), {session, child})
        self.assertEqual(by_thread[session]["length"], 2)
        self.assertEqual(by_thread[child]["length"], 1)

    def test_streamed_duplicate_updates_wait_and_reasoning_evidence(self) -> None:
        base = time.time() - 900
        session = "claude-stream-merge-session-0001"

        def event(message_id: str, epoch: float, output: int,
                  actions: list[dict[str, Any]], thinking: int | None) -> str:
            row = json.loads(
                claude_assistant_line(epoch, session, message_id, output_tokens=output)
            )
            row["message"]["content"] = actions
            if thinking is None:
                row["message"]["usage"].pop("output_tokens_details")
            else:
                row["message"]["usage"]["output_tokens_details"] = {
                    "thinking_tokens": thinking,
                }
            return json.dumps(row)

        wait_action = {
            "type": "tool_use", "id": "streamed-wait-call", "name": "TaskOutput",
            "input": {"block": True},
        }
        extra_action = {
            "type": "tool_use", "id": "streamed-extra-call", "name": "Bash",
            "input": {},
        }
        self.write_claude(session + ".jsonl", [
            claude_user_prompt_line(base + 1, session),
            event("streamed-wait-message", base + 2, 5, [], None),
            event("streamed-wait-message", base + 2, 5, [wait_action], None),
            event("streamed-mixed-message", base + 3, 1, [wait_action], None),
            event("streamed-mixed-message", base + 3, 10, [extra_action], 5),
        ])

        drilldown = self.run_json(
            "prompts", "--harness", "claude", "--claude-root", str(self.home / ".claude"),
            "--session", session, "--prompt", "1", "--drilldown", "--json", "--quiet",
        )["drilldown"]
        self.assertEqual([item["length"] for item in drilldown["root"]["wait_streaks"]], [1])
        self.assertAlmostEqual(
            drilldown["root"]["wait_streaks"][0]["start"], base + 2, places=2
        )
        self.assertEqual(drilldown["root"]["api_calls"], 2)
        self.assertEqual(drilldown["root"]["output_tokens"], 6)
        self.assertIsNone(drilldown["root"]["reasoning_output_tokens"])
        self.assertEqual(drilldown["root"]["known_reasoning_output_tokens"], 1)

    def test_sidecar_arrival_and_edits_reparse_unchanged_transcript(self) -> None:
        base = time.time() - 900
        session = "claude-sidecar-cache-session-0001"
        call_id = "sidecar-parent-call-0001"
        child = "claude-sidecar-child-0001"
        root = json.loads(
            claude_assistant_line(base + 2, session, "sidecar-root-message", output_tokens=1)
        )
        root["message"]["content"] = [{
            "type": "tool_use", "id": call_id, "name": "Agent", "input": {},
        }]
        self.write_claude(session + ".jsonl", [
            claude_user_prompt_line(base + 1, session), json.dumps(root),
        ])
        child_path = self.write_claude(
            "subagents/agent-%s.jsonl" % child,
            [claude_assistant_line(base + 3, session, "sidecar-child-message", output_tokens=1,
                                   sidechain=True)],
        )
        command = (
            "sessions", "--harness", "claude", "--claude-root",
            str(self.home / ".claude"), "--json", "--quiet",
        )
        first = self.run_json(*command)
        self.assertEqual(first["files_parsed"], 2)
        warm = self.run_json(*command)
        self.assertEqual(warm["files_parsed"], 0)

        sidecar = child_path.with_suffix(".meta.json")
        sidecar.write_text(json.dumps({
            "toolUseId": call_id, "spawnDepth": 1, "agentType": "Explore",
        }), encoding="utf-8")
        arrived = self.run_json(*command)
        self.assertEqual(arrived["files_parsed"], 1)

        updated = self.run_json(
            "prompts", "--harness", "claude", "--claude-root", str(self.home / ".claude"),
            "--session", session, "--prompt", "1", "--drilldown", "--json", "--quiet",
        )["drilldown"]
        self.assertEqual(len(updated["descendants"]), 1)
        self.assertEqual(updated["descendants"][0]["thread_id"], child)

        sidecar.write_text(json.dumps({
            "toolUseId": "changed-to-unmatched-call", "spawnDepth": 1,
            "agentType": "Explore",
        }), encoding="utf-8")
        changed = self.run_json(*command)
        self.assertEqual(changed["files_parsed"], 1)
        drilldown = self.run_json(
            "prompts", "--harness", "claude", "--claude-root", str(self.home / ".claude"),
            "--session", session, "--prompt", "1", "--drilldown", "--json", "--quiet",
        )["drilldown"]
        self.assertEqual(drilldown["descendants"], [])
        self.assertEqual([row["thread_id"] for row in drilldown["unknown_threads"]], [child])

    def test_resumed_file_merges_safe_actions_onto_first_deduplicated_call(self) -> None:
        base = time.time() - 900
        session = "claude-resume-lineage-session-0001"
        call_id = "resume-agent-call-0001"
        child = "claude-resume-child-0001"
        original = self.write_claude(session + "-original.jsonl", [
            claude_user_prompt_line(base + 1, session),
            claude_assistant_line(base + 2, session, "resumed-message-id", output_tokens=3),
        ])
        resumed = json.loads(
            claude_assistant_line(base + 2.1, session, "resumed-message-id", output_tokens=99)
        )
        resumed["message"]["content"] = [{
            "type": "tool_use", "id": call_id, "name": "Agent",
            "input": {"prompt": "SYNTHETIC RESUMED ARGUMENT"},
        }]
        resumed_path = self.write_claude(session + "-resumed.jsonl", [json.dumps(resumed)])
        os.utime(original, (base + 10, base + 10))
        os.utime(resumed_path, (base + 20, base + 20))
        child_path = self.write_claude(
            "subagents/agent-%s.jsonl" % child,
            [claude_assistant_line(base + 3, session, "resumed-child-message",
                                   output_tokens=5, sidechain=True)],
        )
        child_path.with_suffix(".meta.json").write_text(json.dumps({
            "toolUseId": call_id, "spawnDepth": 1, "agentType": "Explore",
        }), encoding="utf-8")

        drilldown = self.run_json(
            "prompts", "--harness", "claude", "--claude-root", str(self.home / ".claude"),
            "--session", session, "--prompt", "1", "--drilldown", "--json", "--quiet",
        )["drilldown"]
        self.assertTrue(drilldown["reconciliation"]["matches_prompt"])
        self.assertEqual(drilldown["root"]["api_calls"], 1)
        self.assertEqual(drilldown["root"]["output_tokens"], 3)
        self.assertEqual(drilldown["root"]["action_counts"]["Agent"], 1)
        self.assertEqual([row["thread_id"] for row in drilldown["descendants"]], [child])
        cached = ""
        for path in (self.root / "cache").rglob("*.json"):
            cached += path.read_text(encoding="utf-8")
        self.assertNotIn("SYNTHETIC RESUMED ARGUMENT", cached)

    def test_cache_write_breakdown_keeps_unknown_tokens_and_legacy_price(self) -> None:
        self.assertEqual(
            claude_cache_write_tokens({
                "cache_creation_input_tokens": 100,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 20,
                    "ephemeral_1h_input_tokens": 30,
                },
            }),
            (20, 30, 50),
        )
        self.assertEqual(
            claude_cache_write_tokens({
                "cache_creation_input_tokens": 100,
                "cache_creation": {"ephemeral_5m_input_tokens": 20},
            }),
            (20, 0, 80),
        )
        self.assertEqual(
            claude_cache_write_tokens({
                "cache_creation_input_tokens": 100,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 80,
                    "ephemeral_1h_input_tokens": 50,
                },
            }),
            (0, 0, 100),
        )
        self.assertEqual(
            claude_cache_write_tokens({"cache_creation_input_tokens": 100}),
            (0, 0, 100),
        )
        weights = load_weights(False)
        self.assertEqual(
            weights.claude_units("claude-opus-5", {
                "cache_write_unknown": 1_000_000,
            }, None),
            weights.claude_units("claude-opus-5", {
                "cache_write_5m": 1_000_000,
            }, None),
        )

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
