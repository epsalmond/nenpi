"""Synthetic accounting, delivery, and privacy contracts for the native exporter."""
import json
import unittest
import time
import os
import sqlite3
from pathlib import Path

from tests.test_drain import Harness, claude_assistant_line, claude_user_prompt_line


class Export(Harness):
    def export(self, *flags):
        mapping = self.root / "identities.json"
        mapping.write_text(json.dumps({"sources": [{"harness": "claude",
            "source_root": str(self.claude_projects.parent), "provider": "anthropic", "account_alias": "personal"}]}))
        result = self.run_tool("export", "--since", "1970-01-01", "--identity-map", str(mapping), "--export-state", str(self.root / "export.sqlite"), *flags)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()]

    def seed(self):
        self.write_claude("export.jsonl", [claude_user_prompt_line(100, "session", text="SECRET PROMPT"),
            claude_assistant_line(101, "session", "msg", input_tokens=100, cache_read=200,
                cache_write_5m=300, cache_write_1h=400, output_tokens=50)])

    def test_pending_retry_ack_and_unchanged_repeat(self):
        self.seed()
        batch = self.export()
        self.assertEqual(batch, self.export())
        self.export("--ack", batch[-1]["batch_id"])
        self.assertEqual([r["event"] for r in self.export()], ["analytics_export_health"])

    def test_native_tokens_and_content_free(self):
        self.seed()
        batch = self.export()
        prompt = next(r for r in batch if r["event"] == "analytics_prompt")
        self.assertEqual(prompt["tokens"]["cache_write_5m"], 300)
        self.assertEqual(prompt["tokens"]["cache_write_1h"], 400)
        self.assertEqual(prompt["turns"], 1)
        self.assertEqual(prompt["root_turns"], 1)
        self.assertNotIn("SECRET", json.dumps(batch))
        self.assertNotIn(str(self.root), json.dumps(batch))

    def test_append_revises_same_record_without_idle_finalization(self):
        self.seed()
        first = self.export()
        old = next(r for r in first if r["event"] == "analytics_prompt")
        self.export("--ack", first[-1]["batch_id"])
        with (self.claude_projects / "proj" / "export.jsonl").open("a") as stream:
            stream.write(claude_assistant_line(10000, "session", "late", input_tokens=10, output_tokens=5) + "\n")
        second = self.export()
        new = next(r for r in second if r["event"] == "analytics_prompt")
        self.assertEqual(new["record_id"], old["record_id"])
        self.assertEqual(new["revision"], old["revision"] + 1)
        self.assertEqual(new["turns"], 2)

    def test_bounded_parse_resumes_cached_offsets(self):
        self.seed()
        first = self.export("--max-scan-bytes", "600")
        self.assertEqual(first[0]["coverage"], "incomplete")
        second = self.export("--max-scan-bytes", "600")
        self.assertTrue(any(r["event"] == "analytics_prompt" for r in second))
        self.assertEqual(next(r for r in second if r["event"] == "analytics_prompt")["turns"], 1)

    def test_pages_are_durable_until_each_ack(self):
        self.seed()
        page = self.export("--max-records", "2")
        self.assertEqual(len(page), 2)
        kinds = []
        for _ in range(4):
            kinds.append(page[0]["event"])
            self.assertEqual(page, self.export("--max-records", "2"))
            self.export("--ack", page[-1]["batch_id"])
            page = self.export("--max-records", "2")
        self.assertEqual(set(kinds), {"analytics_prompt","analytics_partition","analytics_session","analytics_account"})
        self.assertEqual(page[0]["event"], "analytics_export_health")

    def test_unknown_prompt_is_excluded_from_human_denominator(self):
        self.write_claude("unknown.jsonl", [claude_assistant_line(101, "session", "msg", input_tokens=100)])
        batch = self.export()
        prompt = next(r for r in batch if r["event"] == "analytics_prompt")
        self.assertEqual(prompt["prompt_membership"], "unknown")
        self.assertFalse(prompt["human_prompt"])

    def test_invalid_identity_label_is_rejected(self):
        from nenpi.export import identity_sources
        mapping = self.root / "bad-map.json"
        mapping.write_text(json.dumps({"sources": [{"harness": "claude", "source_root": str(self.home),
            "provider": "anthropic", "account_alias": "person@example.com"}]}))
        with self.assertRaises(ValueError):
            identity_sources(mapping)

    def test_longest_root_and_ambiguous_outcomes(self):
        from nenpi.export import resolve_identity
        sources = [dict(harness="codex", root=Path("/home/a"), provider="openai", account_alias="one"),
            dict(harness="codex", root=Path("/home/a/project"), provider="openai", account_alias="two")]
        self.assertEqual(resolve_identity(Path("/home/a/project/file"), "codex", sources)["account_alias"], "two")
        sources.append(dict(sources[-1], account_alias="three"))
        self.assertEqual(resolve_identity(Path("/home/a/project/file"), "codex", sources)["identity_status"], "ambiguous")
        self.assertEqual(resolve_identity(Path("/elsewhere"), "codex", sources)["identity_status"], "unknown")

    def test_replayed_response_is_not_counted_twice(self):
        self.seed()
        self.write_claude("replayed.jsonl", [claude_assistant_line(101, "session", "msg", input_tokens=100,
            cache_read=200, cache_write_5m=300, cache_write_1h=400, output_tokens=50)])
        batch = self.export()
        prompt = next(r for r in batch if r["event"] == "analytics_prompt")
        self.assertEqual(prompt["turns"], 1)

    def test_late_descendant_revises_parent_and_uses_native_type(self):
        self.seed()
        first = self.export()
        self.export("--ack", first[-1]["batch_id"])
        child = self.write_claude("session/subagents/agent-child.jsonl", [claude_assistant_line(105, "session", "child-msg",
            input_tokens=20, output_tokens=3, sidechain=True)])
        child.with_suffix(".meta.json").write_text(json.dumps({"agentType": "implementation"}))
        batch = self.export()
        prompt = next(r for r in batch if r["event"] == "analytics_prompt")
        self.assertEqual(prompt["turns"], 2)
        parts = [r for r in batch if r["event"] == "analytics_partition"]
        self.assertTrue(any(p["agent_type"] == "implementation" for p in parts))

    def test_late_child_preserves_parent_outside_active_discovery_window(self):
        self.seed()
        first = self.export()
        original = next(r for r in first if r["event"] == "analytics_prompt")
        self.export("--ack", first[-1]["batch_id"])
        os.utime(self.claude_projects / "proj/export.jsonl", (1, 1))
        self.write_claude("session/subagents/agent-late.jsonl", [claude_assistant_line(105, "session", "late-child",
            input_tokens=20, output_tokens=3, sidechain=True)])
        batch = self.export()
        prompt = next(r for r in batch if r["event"] == "analytics_prompt" and r["record_id"] == original["record_id"])
        self.assertFalse(prompt.get("deleted", False))
        self.assertEqual((prompt["turns"], prompt["output_tokens"]), (2, 53))
        self.assertTrue(prompt["human_prompt"])

    def test_resumed_parent_preserves_child_outside_active_discovery_window(self):
        self.seed()
        child = self.write_claude("session/subagents/agent-old.jsonl", [claude_assistant_line(105, "session", "old-child",
            input_tokens=20, output_tokens=3, sidechain=True)])
        first = self.export()
        original = next(r for r in first if r["event"] == "analytics_prompt")
        self.export("--ack", first[-1]["batch_id"])
        os.utime(child, (1, 1))
        self.write_claude("export.jsonl", [claude_assistant_line(110, "session", "resume", output_tokens=7)])
        batch = self.export()
        prompt = next(r for r in batch if r["event"] == "analytics_prompt" and r["record_id"] == original["record_id"])
        self.assertEqual((prompt["turns"], prompt["output_tokens"]), (3, 60))

    def test_historical_closure_expansion_respects_file_budget_without_tombstones(self):
        self.seed()
        first = self.export()
        self.export("--ack", first[-1]["batch_id"])
        os.utime(self.claude_projects / "proj/export.jsonl", (1, 1))
        self.write_claude("session/subagents/agent-late.jsonl", [claude_assistant_line(105, "session", "late-child",
            input_tokens=20, output_tokens=3, sidechain=True)])
        batch = self.export("--max-files", "1")
        self.assertEqual(batch[-1]["coverage"], "incomplete")
        self.assertFalse(any(r.get("deleted") for r in batch))

    def test_classification_limit_does_not_starve_another_cached_account(self):
        sources = []
        for alias,output in (("aaa",50),("bbb",7)):
            root = self.home / (".claude-" + alias)
            path = root / "projects/proj/session.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(claude_user_prompt_line(100,"session",text="private") + "\n" +
                claude_assistant_line(101,"session","message",input_tokens=10,output_tokens=output) + "\n")
            sources.append(dict(harness="claude",source_root=str(root),provider="anthropic",account_alias=alias))
        mapping = self.root / "mapped.json"
        mapping.write_text(json.dumps(dict(sources=sources)))
        command = ["export","--identity-map",str(mapping),"--export-state",str(self.root / "export.sqlite")]
        first = self.run_tool(*command)
        self.assertEqual(first.returncode,0,first.stderr)
        batch = [json.loads(r) for r in first.stdout.splitlines()]
        self.assertEqual(self.run_tool(*command,"--ack",batch[-1]["batch_id"]).returncode,0)
        first_path = Path(sources[0]["source_root"]) / "projects/proj/session.jsonl"
        first_path.write_text(first_path.read_text().replace('"output_tokens": 50','"output_tokens": 55'))
        # Native data fits; a classification reread requires twice the budget.
        result = self.run_tool(*command,"--threshold","4","--max-scan-bytes",str(first_path.stat().st_size))
        self.assertEqual(result.returncode,0,result.stderr)
        batch = [json.loads(r) for r in result.stdout.splitlines()]
        self.assertEqual(batch[-1]["coverage"],"incomplete")
        self.assertEqual({r["account_alias"] for r in batch if r["event"] == "analytics_prompt"},{"aaa","bbb"})

    def test_classifier_budget_limit_marks_batch_incomplete_and_retries(self):
        self.seed()
        path = self.claude_projects / "proj/export.jsonl"
        first = self.export("--max-scan-bytes", str(path.stat().st_size))
        self.assertEqual(first[-1]["coverage"], "incomplete")
        self.export("--ack", first[-1]["batch_id"])
        later = self.export("--max-scan-bytes", str(path.stat().st_size * 4))
        self.assertEqual(later[-1]["coverage"], "complete")

    def test_combined_batch_counts_output_once_and_cached_family_is_stable(self):
        line = json.loads(claude_assistant_line(101, "session", "msg", input_tokens=100, output_tokens=50))
        line["message"]["content"] = [{"type": "tool_use", "id": "one", "name": "Bash", "input": {"command": "cat SECRET_PATH"}},
            {"type": "tool_use", "id": "two", "name": "Read", "input": {"file_path": "SECRET_PATH"}}]
        self.write_claude("batch.jsonl", [claude_user_prompt_line(100, "session", text="secret"), json.dumps(line)])
        first = self.export()
        part = next(r for r in first if r["event"] == "analytics_partition")
        self.assertEqual(part["tool_association"], "combined_batch")
        self.assertEqual(part["tool_batch"], "read+shell")
        self.assertEqual(part["tokens"]["output"], 50)
        self.assertNotIn("SECRET_PATH", json.dumps(first))
        self.export("--ack", first[-1]["batch_id"])
        self.write_claude("batch.jsonl", [claude_assistant_line(102, "session", "next", input_tokens=1)])
        second = self.export()
        self.assertTrue(any(p.get("tool_batch") == "read+shell" for p in second))

    def test_wrong_ack_preserves_pending(self):
        self.seed()
        batch = self.export()
        mapping = self.root / "identities.json"
        result = self.run_tool("export", "--identity-map", str(mapping), "--export-state", str(self.root / "export.sqlite"), "--ack", "wrong")
        self.assertEqual(result.returncode, 1)
        self.assertEqual(batch, self.export())

    def test_native_cache_write_zero_presence_differs_from_missing(self):
        line = json.loads(claude_assistant_line(101, "session", "msg", input_tokens=100))
        line["message"]["usage"].pop("cache_creation_input_tokens")
        line["message"]["usage"].pop("cache_creation")
        self.write_claude("missing.jsonl", [json.dumps(line)])
        prompt = next(r for r in self.export() if r["event"] == "analytics_prompt")
        self.assertEqual(prompt["tokens"]["cache_write_5m"], 0)
        self.assertFalse(prompt["token_kinds_known"]["cache_write_5m"])

    def test_same_size_rewrite_revises_native_usage(self):
        self.seed()
        first = self.export()
        self.export("--ack", first[-1]["batch_id"])
        path = self.claude_projects / "proj" / "export.jsonl"
        before = path.read_text()
        path.write_text(before.replace('"output_tokens": 50', '"output_tokens": 55'))
        self.assertEqual(path.stat().st_size, len(before.encode()))
        batch = self.export()
        prompt = next(r for r in batch if r["event"] == "analytics_prompt")
        self.assertEqual(prompt["output_tokens"], 55)

    def test_root_human_prompt_without_usage_is_present(self):
        self.write_claude("no-response.jsonl", [claude_user_prompt_line(100,"session",text="pending")])
        batch = self.export()
        prompt = next(r for r in batch if r["event"] == "analytics_prompt")
        self.assertTrue(prompt["human_prompt"])
        self.assertEqual(prompt["turns"], 0)

    def test_removed_source_emits_zero_tombstones(self):
        self.seed()
        first = self.export()
        self.export("--ack",first[-1]["batch_id"])
        (self.claude_projects / "proj" / "export.jsonl").unlink()
        batch = self.export()
        records = [r for r in batch if r["event"] != "analytics_export_batch"]
        self.assertTrue(records)
        self.assertTrue(all(r["deleted"] and r["turns"] == 0 and r["output_tokens"] == 0 for r in records))

    def test_accounts_and_providers_with_same_native_ids_stay_separate(self):
        sources = []
        for alias,provider,tokens in (("one","anthropic",100),("two","anthropic",200),("three","other",300)):
            root = self.home / (".claude-" + alias)
            path = root / "projects/proj/session.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(claude_user_prompt_line(100,"same-session",text="private") + "\n" +
                claude_assistant_line(101,"same-session","same-message",input_tokens=tokens) + "\n")
            sources.append(dict(harness="claude",source_root=str(root),provider=provider,account_alias=alias))
        mapping = self.root / "identity-map.json"
        mapping.write_text(json.dumps(dict(sources=sources)))
        result = self.run_tool("export","--identity-map",str(mapping),"--export-state",str(self.root / "export.sqlite"))
        self.assertEqual(result.returncode,0,result.stderr)
        prompts = [json.loads(line) for line in result.stdout.splitlines() if json.loads(line)["event"] == "analytics_prompt"]
        self.assertEqual({(p["provider"],p["account_alias"],p["tokens"]["input"]) for p in prompts},
            {("anthropic","one",100),("anthropic","two",200),("other","three",300)})

    def test_reclassification_tombstones_old_unknown_partition(self):
        self.seed()
        first = self.export()
        old = next(r for r in first if r["event"] == "analytics_partition")
        self.export("--ack",first[-1]["batch_id"])
        record = json.loads(claude_assistant_line(101,"session","msg",input_tokens=100,cache_read=200,
            cache_write_5m=300,cache_write_1h=400,output_tokens=50))
        record["message"]["content"] = [dict(type="tool_use",id="call",name="Read",input=dict(file_path="PRIVATE"))]
        self.write_claude("export.jsonl",[json.dumps(record)])
        batch = self.export()
        parts = [r for r in batch if r["event"] == "analytics_partition"]
        self.assertTrue(any(p["record_id"] == old["record_id"] and p["deleted"] and p["output_tokens"] == 0 for p in parts))
        self.assertEqual(sum(p["output_tokens"] for p in parts),50)

    def test_missing_lineage_metadata_is_unknown_and_bad_native_ids_do_not_collide(self):
        from nenpi.export import native_id,native_lineage
        from types import SimpleNamespace
        analysis = SimpleNamespace(scan=SimpleNamespace(thread_metadata={}))
        self.assertEqual(native_lineage(analysis,"codex","session","session"),"unknown")
        self.assertNotEqual(native_id("private native id one"),native_id("private native id two"))
        self.assertNotIn("private",native_id("private native id one"))

    def test_projection_upgrade_revisits_unchanged_sources(self):
        import contextlib
        import io
        from unittest.mock import patch
        from nenpi import export as exporter
        from nenpi import drain
        self.seed()
        first = self.export()
        self.export("--ack",first[-1]["batch_id"])
        output = io.StringIO()
        with self.env_applied(),patch.object(exporter,"PROJECTION_VERSION",exporter.PROJECTION_VERSION + 1), \
                patch.object(exporter,"native_lineage",return_value="unknown"),contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
            result = drain.main(["export","--identity-map",str(self.root / "identities.json"),"--export-state",str(self.root / "export.sqlite")])
        self.assertEqual(result,0)
        prompt = next(json.loads(line) for line in output.getvalue().splitlines() if json.loads(line)["event"] == "analytics_prompt")
        self.assertEqual(prompt["revision"],2)
        self.assertEqual(prompt["unknown_turns"],1)

    def test_native_counter_survives_fork_ownership_reassignment(self):
        self.seed()
        first = self.export()
        self.export("--ack",first[-1]["batch_id"])
        fork = self.write_claude("fork.jsonl",[claude_assistant_line(101,"fork","msg",input_tokens=100,
            cache_read=200,cache_write_5m=300,cache_write_1h=400,output_tokens=50)])
        now = time.time()
        os.utime(fork,(now - 10,now - 10))
        os.utime(self.claude_projects / "proj/export.jsonl",(now + 10,now + 10))
        second = self.export()
        self.assertTrue(any(r.get("session_id") == "fork" for r in second))
        with sqlite3.connect(self.root / "export.sqlite") as database:
            self.assertEqual(database.execute("SELECT SUM(value) FROM account_counters WHERE kind='cache_write'").fetchone()[0],700)
            self.assertEqual(database.execute("SELECT COUNT(*) FROM response_usage").fetchone()[0],1)

    def test_counter_write_ttl_reclassification_does_not_add_tokens(self):
        line = json.loads(claude_assistant_line(101,"session","msg",input_tokens=100,cache_write_5m=300))
        line["message"]["usage"].pop("cache_creation")
        path = self.write_claude("ttl.jsonl",[json.dumps(line)])
        first = self.export()
        self.export("--ack",first[-1]["batch_id"])
        line["message"]["usage"]["cache_creation"] = dict(ephemeral_5m_input_tokens=300,ephemeral_1h_input_tokens=0)
        path.write_text(json.dumps(line) + "\n")
        self.export()
        with sqlite3.connect(self.root / "export.sqlite") as database:
            self.assertEqual(database.execute("SELECT SUM(value) FROM account_counters WHERE kind='cache_write'").fetchone()[0],300)

    def test_counter_identity_capacity_is_explicit(self):
        self.seed()
        first = self.export("--max-response-ids","1")
        self.export("--ack",first[-1]["batch_id"])
        self.write_claude("export.jsonl",[claude_assistant_line(102,"session","next",input_tokens=10)])
        second = self.export("--max-response-ids","1")
        counter = next(r for r in second if r["event"] == "analytics_account")
        self.assertFalse(counter["counter_identity_budget_complete"])
        self.assertEqual(counter["native_responses"],1)

    def test_old_sources_do_not_consume_active_file_budget(self):
        from nenpi.export import discover
        from types import SimpleNamespace
        self.seed()
        old = self.write_claude("old.jsonl",[])
        os.utime(old,(1,1))
        sources = [dict(harness="claude",root=self.claude_projects.parent,provider="anthropic",account_alias="personal")]
        files = discover(sources,SimpleNamespace(max_files=1,scan_seconds=10),time.time() - 86400)
        self.assertEqual(len(files),1)

    def test_source_window_expiration_preserves_historical_snapshots(self):
        self.seed()
        first = self.export()
        self.export("--ack",first[-1]["batch_id"])
        path = self.claude_projects / "proj/export.jsonl"
        os.utime(path,(1,1))
        batch = self.export()
        self.assertFalse(any(r.get("deleted") for r in batch))


if __name__ == "__main__":
    unittest.main()
