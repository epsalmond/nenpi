"""Synthetic accounting, delivery, and privacy contracts for the native exporter."""
import json
import unittest
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


if __name__ == "__main__":
    unittest.main()
