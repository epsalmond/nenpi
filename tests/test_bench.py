#!/usr/bin/env python3
"""Contract tests for quota-bench.

Nothing here spends quota or touches the network: `claude` is a fake
executable on PATH whose usage comes from a fixture, and the sampler is a fake
quota-drain that derives utilisation from the fake calls with known weights.
The fixture is sized so every call costs exactly half a percent, which makes
tick bracketing exact and lets the fit be checked against the weights that
produced it.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
QUOTA_BENCH = SCRIPT_DIR / "quota-bench"
QUOTA_DRAIN = SCRIPT_DIR / "quota-drain"


def load(name: str, path: Path) -> Any:
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QD = load("quota_drain_bench_contract", QUOTA_DRAIN)
QB = load("quota_bench_contract", QUOTA_BENCH)
QB._QD = QD


# Percent per million tokens the fake sampler bills at; the fit has to find
# these back from the calls alone.
KNOWN_WEIGHTS = {"input": 1.0, "cache_read": 0.1, "cache_write_5m": 1.25, "output": 5.0}

# Each entry costs exactly 0.5% under KNOWN_WEIGHTS.
USAGE_FIXTURE = {
    "fresh|short": {"input": 400_000, "cache_read": 0, "cache_write": 0, "output": 20_000},
    "warm|short": {"input": 50_000, "cache_read": 4_000_000, "cache_write": 0, "output": 10_000},
    "fresh|long": {"input": 100_000, "cache_read": 1_000_000, "cache_write": 0, "output": 60_000},
    "warm|long": {"input": 100_000, "cache_read": 1_000_000, "cache_write": 0, "output": 60_000},
    "default": {"input": 500_000, "cache_read": 0, "cache_write": 0, "output": 0},
}

FAKE_CLAUDE = '''#!/usr/bin/env python3
"""Stand-in for `claude -p --output-format json`. Spends nothing."""

import hashlib
import json
import os
import sys


def main() -> int:
    argv = sys.argv[1:]
    model = "claude-haiku-4-5-20251001"
    for index, item in enumerate(argv):
        if item == "--model" and index + 1 < len(argv):
            model = argv[index + 1]
    prompt = sys.stdin.read()
    system_file = None
    for index, item in enumerate(argv):
        if item == "--system-prompt-file" and index + 1 < len(argv):
            system_file = argv[index + 1]
    prefix = prompt.rsplit("\\n\\n", 1)[0]
    if system_file and os.path.exists(system_file):
        with open(system_file, encoding="utf-8") as handle:
            prefix = handle.read()
    digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    seen_path = os.environ["BENCH_FAKE_SEEN"]
    seen = set()
    if os.path.exists(seen_path):
        with open(seen_path, encoding="utf-8") as handle:
            seen = set(handle.read().split())
    state = "warm" if digest in seen else "fresh"
    if state == "fresh":
        with open(seen_path, "a", encoding="utf-8") as handle:
            handle.write(digest + "\\n")
    mode = "long" if "2000-word essay" in prompt else "short"
    with open(os.environ["BENCH_FAKE_USAGE"], encoding="utf-8") as handle:
        fixture = json.load(handle)
    entry = fixture.get("%s|%s" % (state, mode)) or fixture["default"]
    tokens = {
        "input": entry["input"],
        "cache_read": entry["cache_read"],
        "cache_write_5m": entry["cache_write"],
        "cache_write_1h": 0,
        "output": entry["output"],
    }
    with open(os.environ["BENCH_FAKE_LOG"], "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"model": model, "tokens": tokens, "argv": argv,
                                 "prompt_bytes": len(prompt)}) + "\\n")
    if os.environ.get("BENCH_FAKE_FAIL"):
        sys.stderr.write("fake failure\\n")
        return 1
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 10,
        "duration_api_ms": 8,
        "num_turns": 1,
        "result": "ok",
        "session_id": os.environ.get("BENCH_FAKE_SESSION", "bench-session"),
        "total_cost_usd": 0.001,
        "usage": {
            "input_tokens": tokens["input"],
            "cache_read_input_tokens": tokens["cache_read"],
            "cache_creation_input_tokens": tokens["cache_write_5m"],
            "output_tokens": tokens["output"],
        },
        "modelUsage": {
            model: {
                "inputTokens": tokens["input"],
                "cacheReadInputTokens": tokens["cache_read"],
                "cacheCreationInputTokens": tokens["cache_write_5m"],
                "outputTokens": tokens["output"],
                "webSearchRequests": 0,
                "costUSD": 0.001,
                "contextWindow": 200000,
                "maxOutputTokens": 8192,
            }
        },
    }
    helper = os.environ.get("BENCH_FAKE_HELPER_MODEL")
    if helper:
        # A second model the run touched on its own, as a real session does
        # when a subagent or an internal helper call is made.
        result["modelUsage"][helper] = {
            "inputTokens": 1000,
            "cacheReadInputTokens": 0,
            "cacheCreationInputTokens": 0,
            "outputTokens": 100,
            "webSearchRequests": 0,
            "costUSD": 0.0001,
            "contextWindow": 200000,
            "maxOutputTokens": 8192,
        }
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

FAKE_QUOTA_DRAIN = '''#!/usr/bin/env python3
"""Stand-in for quota-drain: real module surface, fake sampler and sessions.

quota-bench imports this file for the snapshot format and the NNLS fit, and
runs it as a subprocess to sample. The import re-exports the real module so
only the two subcommands are faked; the OAuth path is never reached.
"""

import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import time

_REAL_PATH = os.environ["BENCH_REAL_QUOTA_DRAIN"]
_LOADER = importlib.machinery.SourceFileLoader("quota_drain_under_fake", _REAL_PATH)
_SPEC = importlib.util.spec_from_loader(_LOADER.name, _LOADER)
_REAL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _REAL
_SPEC.loader.exec_module(_REAL)
for _name, _value in vars(_REAL).items():
    if _name not in globals():
        globals()[_name] = _value


def _fake_snapshot() -> int:
    if os.environ.get("BENCH_SAMPLER_FAIL_AFTER"):
        attempts = 0
        marker = os.environ["BENCH_FAKE_LOG"] + ".sampler-attempts"
        if os.path.exists(marker):
            with open(marker, encoding="utf-8") as handle:
                attempts = int(handle.read().strip() or 0)
        attempts += 1
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(str(attempts))
        if attempts > int(os.environ["BENCH_SAMPLER_FAIL_AFTER"]):
            sys.stderr.write("fake sampler failure\\n")
            return 1
    if os.environ.get("BENCH_SAMPLER_STALE_AFTER"):
        attempts = 0
        marker = os.environ["BENCH_FAKE_LOG"] + ".stale-attempts"
        if os.path.exists(marker):
            with open(marker, encoding="utf-8") as handle:
                attempts = int(handle.read().strip() or 0)
        attempts += 1
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(str(attempts))
        if attempts > int(os.environ["BENCH_SAMPLER_STALE_AFTER"]):
            # Exits clean and logs nothing: the snapshot stops advancing.
            return 0
    weights = json.loads(os.environ["BENCH_FAKE_WEIGHTS"])
    total = 0.0
    log = os.environ["BENCH_FAKE_LOG"]
    if os.path.exists(log):
        with open(log, encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                for kind, value in entry["tokens"].items():
                    total += value / 1_000_000.0 * weights.get(kind, 0.0)
    percent = float(math.floor(round(total, 6)))
    record = {
        "source": "oauth",
        "config_dir": ".claude",
        "ts": time.time(),
        "windows": {
            "five_hour": {"utilization_percent": percent,
                          "resets_at": "2026-09-16T21:30:00.000000+00:00"},
            "seven_day": {"utilization_percent": float(math.floor(percent / 10.0)),
                          "resets_at": "2026-09-22T21:30:00.000000+00:00"},
        },
    }
    _REAL.append_snapshot(_REAL.state_dir() / "snapshots.jsonl", record)
    return 0


def _fake_sessions() -> int:
    path = os.environ.get("BENCH_FAKE_SESSIONS")
    payload = {"schema": 1, "command": "sessions", "sessions": []}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    _ARGV = sys.argv[1:]
    if _ARGV[:1] == ["snapshot"]:
        sys.exit(_fake_snapshot())
    if _ARGV[:1] == ["sessions"]:
        sys.exit(_fake_sessions())
    sys.exit(_REAL.main(_ARGV))
'''


class Harness(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.call_log = self.root / "calls.jsonl"
        self.seen = self.root / "seen.txt"
        self.usage_fixture = self.root / "usage.json"
        self.usage_fixture.write_text(json.dumps(USAGE_FIXTURE), encoding="utf-8")
        self.sessions_fixture = self.root / "sessions.json"

        self.fake_claude = self.bin / "claude"
        self.fake_claude.write_text(FAKE_CLAUDE, encoding="utf-8")
        self.fake_claude.chmod(0o755)
        self.fake_drain = self.root / "fake-quota-drain"
        self.fake_drain.write_text(FAKE_QUOTA_DRAIN, encoding="utf-8")
        self.fake_drain.chmod(0o755)

        self.environment = dict(os.environ)
        self.environment.update(
            {
                "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
                "QUOTA_DRAIN_HOME_DIR": str(self.root / "home"),
                "QUOTA_DRAIN_CACHE_DIR": str(self.root / "cache"),
                "QUOTA_DRAIN_STATE_DIR": str(self.root / "state"),
                "QUOTA_DRAIN_CONFIG_DIR": str(self.root / "config"),
                "BENCH_REAL_QUOTA_DRAIN": str(QUOTA_DRAIN),
                "BENCH_FAKE_LOG": str(self.call_log),
                "BENCH_FAKE_SEEN": str(self.seen),
                "BENCH_FAKE_USAGE": str(self.usage_fixture),
                "BENCH_FAKE_WEIGHTS": json.dumps(KNOWN_WEIGHTS),
                "TZ": "UTC",
            }
        )

    def bench(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(QUOTA_BENCH)] + list(arguments),
            check=False,
            capture_output=True,
            env=self.environment,
            timeout=300,
        )

    def bench_json(self, *arguments: str) -> Dict[str, Any]:
        result = self.bench(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        return json.loads(result.stdout.decode("utf-8"))

    def run_scenarios(self, *arguments: str) -> subprocess.CompletedProcess:
        return self.bench(
            "run", "--yes", "--quota-drain", str(self.fake_drain),
            "--sample-interval", "0", "--ticks", "3", "--max-percent", "50",
            "--max-percent-weekly", "50", "--json", *arguments
        )

    def fake_calls(self) -> List[Dict[str, Any]]:
        if not self.call_log.is_file():
            return []
        return [json.loads(line) for line in self.call_log.read_text(encoding="utf-8").splitlines()]

    def latest_report(self) -> Dict[str, Any]:
        runs = sorted((self.root / "state" / "bench").iterdir())
        return json.loads((runs[-1] / "report.json").read_text(encoding="utf-8"))

    def scenario_records(self) -> List[Dict[str, Any]]:
        runs = sorted((self.root / "state" / "bench").iterdir())
        path = runs[-1] / "scenarios.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def write_sessions(self, sessions: Sequence[Dict[str, Any]]) -> None:
        self.sessions_fixture.write_text(
            json.dumps({"schema": 1, "command": "sessions", "sessions": list(sessions)}),
            encoding="utf-8",
        )
        self.environment["BENCH_FAKE_SESSIONS"] = str(self.sessions_fixture)


class Planning(Harness):
    def test_plan_spends_nothing(self) -> None:
        payload = self.bench_json("plan", "--models", "claude-haiku-4-5-20251001",
                                  "--contexts", "10k", "--cache", "warm",
                                  "--max-percent", "1", "--json")
        self.assertEqual(len(payload["scenarios"]), 1)
        self.assertEqual(payload["scenarios"][0]["key"], "claude-haiku-4-5/10k/warm/short")
        self.assertGreater(payload["scenarios"][0]["projected_calls"], 0)
        self.assertEqual(self.fake_calls(), [])

    def test_plan_orders_the_cheapest_scenario_first(self) -> None:
        payload = self.bench_json("plan", "--models", "claude-opus-5,claude-haiku-4-5",
                                  "--contexts", "10k,150k", "--cache", "warm,cold", "--json")
        keys = [row["key"] for row in payload["scenarios"]]
        self.assertEqual(keys[0], "claude-haiku-4-5/10k/warm/short")
        self.assertLess(keys.index("claude-haiku-4-5/150k/cold/short"),
                        keys.index("claude-opus-5/10k/warm/short"))

    def test_plan_names_a_scenario_it_cannot_afford_to_finish(self) -> None:
        result = self.bench("plan", "--models", "claude-haiku-4-5", "--contexts", "10k",
                            "--cache", "warm", "--max-calls", "10")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"but --max-calls stops it at 10", result.stdout)

    def test_the_first_call_of_a_warm_scenario_is_priced_as_cache_creation(self) -> None:
        payload = self.bench_json("plan", "--models", "claude-haiku-4-5", "--contexts", "150k",
                                  "--cache", "warm", "--json")
        row = payload["scenarios"][0]
        # 150K tokens written at 1.25x costs far more than reading them at 0.1x.
        self.assertGreater(row["projected_percent_first_call"],
                           row["projected_percent_per_call"] * 5)
        self.assertAlmostEqual(
            row["projected_percent"],
            row["projected_percent_first_call"]
            + row["projected_percent_per_call"] * (row["projected_calls"] - 1),
            places=6,
        )

    def test_cold_is_projected_as_cache_creation_by_default(self) -> None:
        conservative = self.bench_json("plan", "--models", "claude-haiku-4-5",
                                       "--contexts", "60k", "--cache", "cold", "--json")
        plain = self.bench_json("plan", "--models", "claude-haiku-4-5", "--contexts", "60k",
                                "--cache", "cold", "--no-cold-as-creation", "--json")
        self.assertAlmostEqual(
            conservative["scenarios"][0]["projected_percent_per_call"]
            / plain["scenarios"][0]["projected_percent_per_call"],
            1.25,
            places=2,
        )

    def test_the_weekly_cap_is_not_projected_in_five_hour_percent(self) -> None:
        result = self.bench("plan", "--models", "claude-haiku-4-5", "--contexts", "10k")
        self.assertNotIn(b"projected weekly spend", result.stdout)
        self.assertIn(b"enforced from live samples", result.stdout)

    def test_run_refuses_a_projection_over_the_cap(self) -> None:
        result = self.bench(
            "run", "--yes", "--quota-drain", str(self.fake_drain), "--sample-interval", "0",
            "--models", "claude-haiku-4-5", "--contexts", "150k", "--cache", "cold",
            "--max-percent", "0.01",
        )
        self.assertEqual(result.returncode, QB.EXIT_USAGE)
        self.assertIn(b"already exceeds --max-percent", result.stderr)
        self.assertEqual(self.fake_calls(), [])
        forced = self.bench(
            "run", "--yes", "--force-projection", "--quota-drain", str(self.fake_drain),
            "--sample-interval", "0", "--models", "claude-haiku-4-5", "--contexts", "150k",
            "--cache", "cold", "--max-percent", "0.01",
        )
        self.assertEqual(forced.returncode, QB.EXIT_BUDGET)

    def test_run_refuses_without_yes(self) -> None:
        result = self.bench("run", "--quota-drain", str(self.fake_drain),
                            "--models", "claude-haiku-4-5", "--contexts", "10k")
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"--yes", result.stderr)
        self.assertEqual(self.fake_calls(), [])

    def test_unknown_context_is_rejected(self) -> None:
        result = self.bench("plan", "--contexts", "42k")
        self.assertEqual(result.returncode, 1)
        self.assertIn(b"unknown context size", result.stderr)


class Bracketing(Harness):
    def test_tick_bracketing_discards_the_partial_percents(self) -> None:
        result = self.run_scenarios("--models", "claude-haiku-4-5-20251001",
                                    "--contexts", "10k", "--cache", "warm")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        record = self.scenario_records()[0]
        self.assertEqual(record["stop_reason"], "ticks")
        self.assertEqual(len(record["ticks"]), 3)
        # Two calls buy one percent, so six calls produce ticks at 1, 2 and 3.
        self.assertEqual(record["calls"], 6)

        report = self.latest_report()
        scenario = report["scenarios"][0]
        self.assertTrue(scenario["bracketed"])
        self.assertEqual(scenario["percent"], 2.0)
        # The bracket holds the four calls between the first and the last tick,
        # not the two that bought the discarded partial percent.
        per_percent = scenario["tokens_per_percent"]
        self.assertAlmostEqual(per_percent["cache_read"], 2 * 4_000_000, delta=1.0)
        self.assertAlmostEqual(per_percent["input"], 2 * 50_000, delta=1.0)
        self.assertAlmostEqual(per_percent["output"], 2 * 10_000, delta=1.0)

    def test_one_tick_estimate_is_reported_unbracketed(self) -> None:
        result = self.bench(
            "run", "--yes", "--quota-drain", str(self.fake_drain), "--sample-interval", "0",
            "--ticks", "1", "--max-percent", "50", "--max-percent-weekly", "50", "--json",
            "--models", "claude-haiku-4-5", "--contexts", "10k", "--cache", "warm",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        report = self.latest_report()
        scenario = report["scenarios"][0]
        self.assertFalse(scenario["bracketed"])
        self.assertTrue(scenario["upper_bound"])
        self.assertEqual(scenario["ticks"], 1)
        # An upper bound is not a measurement, so it stays out of the design
        # matrix unless the caller asks for it.
        self.assertIsNone(report["fit"])
        self.assertIn(b"fit excludes", result.stderr)
        opted_in = self.bench_json("report", "--ticks", "1", "--include-unbracketed",
                                   "--quota-drain", str(self.fake_drain), "--json")
        self.assertIsNotNone(opted_in["fit"])

    def test_estimate_needs_two_ticks(self) -> None:
        record = {
            "key": "m/10k/warm/short", "model": "m", "context": "10k", "cache": "warm",
            "output": "short", "calls": 1, "start_percent": 0.0,
            "ticks": [{"ts": 1.0, "percent": 1.0, "calls": 1, "tokens": {"input": 10}, "models": {}}],
        }
        self.assertIsNone(QB.estimate_scenario(record, required_ticks=2))
        self.assertIsNotNone(QB.estimate_scenario(record, required_ticks=1))

    def test_window_rollover_drops_the_bracket(self) -> None:
        budget = QB.Budget({"five_hour": 3.0})
        budget.observe(QB.Sample(1.0, {"five_hour": 90.0}))
        budget.observe(QB.Sample(2.0, {"five_hour": 92.0}))
        self.assertEqual(budget.spent["five_hour"], 2.0)
        self.assertEqual(budget.observe(QB.Sample(3.0, {"five_hour": 1.0})), ["five_hour"])
        self.assertEqual(budget.spent["five_hour"], 2.0)


class Budgets(Harness):
    def test_run_aborts_at_max_percent(self) -> None:
        result = self.bench(
            "run", "--yes", "--quota-drain", str(self.fake_drain), "--sample-interval", "0",
            "--ticks", "9", "--max-percent", "2", "--max-percent-weekly", "50",
            "--models", "claude-haiku-4-5", "--contexts", "10k", "--cache", "warm",
        )
        self.assertEqual(result.returncode, QB.EXIT_BUDGET)
        self.assertIn(b"budget reached", result.stderr)
        record = self.scenario_records()[0]
        self.assertEqual(record["stop_reason"], "budget:five_hour")
        # Four calls is two percent; the fifth would have overrun the budget.
        self.assertEqual(record["calls"], 4)

    def test_the_five_hour_cap_applies_when_measuring_the_weekly_window(self) -> None:
        result = self.bench(
            "run", "--yes", "--quota-drain", str(self.fake_drain), "--sample-interval", "0",
            "--window", "seven_day", "--ticks", "9", "--max-percent", "2",
            "--max-percent-weekly", "50", "--models", "claude-haiku-4-5",
            "--contexts", "10k", "--cache", "warm",
        )
        self.assertEqual(result.returncode, QB.EXIT_BUDGET)
        self.assertEqual(self.scenario_records()[0]["stop_reason"], "budget:five_hour")
        runs = sorted((self.root / "state" / "bench").iterdir())
        meta = json.loads((runs[-1] / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(meta["budget"]["limits"]), ["five_hour", "seven_day"])

    def test_max_calls_stops_a_scenario(self) -> None:
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm", "--max-calls", "3")
        self.assertEqual(result.returncode, QB.EXIT_MAX_CALLS,
                         result.stderr.decode("utf-8", "replace"))
        self.assertEqual(self.scenario_records()[0]["stop_reason"], "max-calls")
        self.assertIn(b"fewer than 3 ticks", result.stderr)


class SamplerGuard(Harness):
    def test_a_failing_sampler_stops_the_run(self) -> None:
        # The baseline sample succeeds, then every later attempt exits 1. A run
        # that cannot see utilisation cannot see what it is spending.
        self.environment["BENCH_SAMPLER_FAIL_AFTER"] = "1"
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        self.assertEqual(result.returncode, QB.EXIT_GUARD)
        self.assertIn(b"sampler exited 1", result.stderr)
        # One call was already launched against the last good sample; the
        # failure is caught before a second one, so a dead sampler costs at
        # most a single call rather than the whole budget.
        self.assertEqual(len(self.fake_calls()), 1)
        self.assertTrue(self.scenario_records()[0]["stop_reason"].startswith("sampler:"))

    def test_a_sampler_that_never_works_spends_nothing(self) -> None:
        self.environment["BENCH_SAMPLER_FAIL_AFTER"] = "0"
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        self.assertEqual(result.returncode, QB.EXIT_DEPENDENCY)
        self.assertIn(b"no utilisation sample", result.stderr)
        self.assertEqual(self.fake_calls(), [])

    def test_a_stale_sample_stops_the_run(self) -> None:
        # The sampler keeps exiting clean but stops logging new observations.
        self.environment["BENCH_SAMPLER_STALE_AFTER"] = "3"
        result = self.bench(
            "run", "--yes", "--quota-drain", str(self.fake_drain), "--sample-interval", "0",
            "--ticks", "9", "--max-percent", "50", "--max-percent-weekly", "50",
            "--models", "claude-haiku-4-5", "--contexts", "10k", "--cache", "warm",
        )
        self.assertEqual(result.returncode, QB.EXIT_GUARD)
        self.assertIn(b"no new observation", result.stderr)
        # Two calls ran while sampling still worked; nothing after that.
        self.assertLessEqual(len(self.fake_calls()), 3)

    def test_sampler_freshness_is_checked_before_each_call(self) -> None:
        sampler = QB.Sampler(QD, self.fake_drain, [], interval=0.0, max_age=90.0)
        self.assertEqual(sampler.blocker(), "no utilisation sample yet")
        sampler.latest = QB.Sample(time.time(), {"five_hour": 1.0})
        self.assertIsNone(sampler.blocker())
        sampler.latest = QB.Sample(time.time() - 600.0, {"five_hour": 1.0})
        self.assertIn("600s old", sampler.blocker())
        sampler.latest = QB.Sample(time.time(), {"five_hour": 1.0})
        sampler.failure = "sampler exited 7"
        self.assertEqual(sampler.blocker(), "sampler exited 7")


class ManyCallsPerSample(Harness):
    def test_the_budget_holds_when_calls_outrun_the_sampler(self) -> None:
        # The endpoint cannot be polled more than once a minute, so a run is
        # blind between samples. With a one-second interval and instant fake
        # calls, many calls land inside one blind window; the budget still has
        # to stop the run rather than let it spend on.
        result = self.bench(
            "run", "--yes", "--quota-drain", str(self.fake_drain), "--sample-interval", "1",
            "--ticks", "9", "--max-percent", "5", "--max-percent-weekly", "50",
            "--models", "claude-haiku-4-5", "--contexts", "10k", "--cache", "warm",
        )
        self.assertEqual(result.returncode, QB.EXIT_BUDGET,
                         result.stderr.decode("utf-8", "replace"))
        runs = sorted((self.root / "state" / "bench").iterdir())
        lines = [json.loads(line) for line in
                 (runs[-1] / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertGreater(len(lines), 5)
        per_sample = {}  # type: Dict[float, int]
        for line in lines:
            stamp = line["sample"]["ts"]
            per_sample[stamp] = per_sample.get(stamp, 0) + 1
        self.assertGreater(max(per_sample.values()), 1)
        meta = json.loads((runs[-1] / "meta.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(meta["budget"]["spent"]["five_hour"], 5.0)
        # Each call is half a percent, so the blind window is what the overshoot
        # is bounded by, not the cap.
        self.assertLessEqual(meta["budget"]["spent"]["five_hour"], 0.5 * len(lines))
        self.assertEqual(self.scenario_records()[0]["stop_reason"], "budget:five_hour")


class Contamination(Harness):
    def test_a_foreign_session_flags_the_scenario(self) -> None:
        now = time.time()
        self.write_sessions([{"harness": "claude", "session_id": "someone-else",
                              "start": now - 60, "end": now + 600}])
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        record = self.scenario_records()[0]
        self.assertTrue(record["contaminated"])
        self.assertEqual(record["contaminated_by"], ["someone-else"])

    def test_our_own_session_is_not_contamination(self) -> None:
        now = time.time()
        self.write_sessions([{"harness": "claude", "session_id": "bench-session",
                              "start": now - 60, "end": now + 600},
                             {"harness": "codex", "session_id": "codex-one",
                              "start": now - 60, "end": now + 600}])
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertFalse(self.scenario_records()[0]["contaminated"])

    def test_a_failed_check_contaminates_the_scenario(self) -> None:
        # An unreadable sessions payload is not evidence of a clean window.
        self.sessions_fixture.write_text("not json", encoding="utf-8")
        self.environment["BENCH_FAKE_SESSIONS"] = str(self.sessions_fixture)
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        record = self.scenario_records()[0]
        self.assertTrue(record["contaminated"])
        self.assertIn("check_failed", record["contamination_reasons"])

    def test_require_idle_refuses_when_the_check_fails(self) -> None:
        self.sessions_fixture.write_text("not json", encoding="utf-8")
        self.environment["BENCH_FAKE_SESSIONS"] = str(self.sessions_fixture)
        result = self.bench(
            "run", "--yes", "--require-idle", "--quota-drain", str(self.fake_drain),
            "--sample-interval", "0", "--models", "claude-haiku-4-5", "--contexts", "10k",
        )
        self.assertEqual(result.returncode, QB.EXIT_GUARD)
        self.assertIn(b"did not run", result.stderr)
        self.assertEqual(self.fake_calls(), [])

    def test_a_session_active_before_the_scenario_contaminates_it(self) -> None:
        now = time.time()
        self.write_sessions([{"harness": "claude", "session_id": "earlier-one",
                              "start": now - 200, "end": now - 100}])
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        record = self.scenario_records()[0]
        self.assertTrue(record["contaminated"])
        self.assertEqual(record["contaminated_by"], ["earlier-one"])

    def test_require_idle_refuses_to_start(self) -> None:
        now = time.time()
        self.write_sessions([{"harness": "claude", "session_id": "someone-else",
                              "start": now - 60, "end": now}])
        result = self.bench(
            "run", "--yes", "--require-idle", "--quota-drain", str(self.fake_drain),
            "--sample-interval", "0", "--models", "claude-haiku-4-5", "--contexts", "10k",
        )
        self.assertEqual(result.returncode, QB.EXIT_GUARD)
        self.assertIn(b"--require-idle", result.stderr)
        self.assertEqual(self.fake_calls(), [])


class Fitting(Harness):
    def run_three_scenarios(self) -> Dict[str, Any]:
        result = self.run_scenarios("--models", "claude-haiku-4-5-20251001",
                                    "--contexts", "10k", "--cache", "warm,cold",
                                    "--outputs", "short,long")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        return self.latest_report()

    def test_fit_recovers_the_known_weights(self) -> None:
        report = self.run_three_scenarios()
        fitted = report["fit"]["models"]["claude-haiku-4-5"]
        for kind in ("input", "cache_read", "output"):
            self.assertLess(
                abs(fitted[kind] - KNOWN_WEIGHTS[kind]) / KNOWN_WEIGHTS[kind],
                0.05,
                "%s fitted at %.4f, expected %.4f" % (kind, fitted[kind], KNOWN_WEIGHTS[kind]),
            )
        # Every scenario stops at the same tick count, so the targets carry no
        # variance and R^2 is undefined; the residual is what grades the fit.
        self.assertIsNone(report["fit"]["r_squared"])
        self.assertLess(report["fit"]["residual_relative"], 0.01)
        self.assertTrue(report["fit"]["usable"])
        self.assertFalse(report["fit"]["underdetermined"])
        self.assertEqual(report["fit"]["identified"], 3)
        self.assertGreater(report["fit"]["fallback_scale"], 0.0)
        for kind in ("input", "cache_read", "output"):
            diagnostic = report["fit"]["diagnostics"]["claude-haiku-4-5"][kind]
            self.assertFalse(diagnostic["unidentified"], diagnostic["reasons"])

    def test_an_unidentifiable_coefficient_is_null_not_zero(self) -> None:
        # One scenario cannot separate three kinds; a confident zero there would
        # price a kind as free, so the fit stores null instead.
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        fit = self.latest_report()["fit"]
        self.assertTrue(fit["underdetermined"])
        self.assertFalse(fit["usable"])
        self.assertEqual(fit["identified"], 0)
        entry = fit["models"]["claude-haiku-4-5"]
        self.assertIsNone(entry["cache_read"])
        self.assertIn("too few scenarios",
                      fit["diagnostics"]["claude-haiku-4-5"]["cache_read"]["reasons"])
        text = self.bench("report", "--ticks", "3",
                          "--quota-drain", str(self.fake_drain)).stdout.decode("utf-8")
        self.assertIn("unidentified (too few scenarios)", text)

    def test_weights_file_carries_the_quota_drain_shape(self) -> None:
        self.run_three_scenarios()
        payload = json.loads(
            (self.root / "state" / "claude-weights.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["harness"], "claude")
        self.assertEqual(payload["unit"], "percent_per_mtok")
        self.assertEqual(payload["claude"]["unit"], "percent_per_mtok")
        self.assertIn("claude-haiku-4-5", payload["claude"]["models"])
        self.assertIn("claude-haiku-4-5", payload["windows"]["five_hour"]["models"])
        merged = QD.builtin_weights()
        QD.merge_weights(merged, {"claude": payload["claude"]})
        self.assertAlmostEqual(
            merged["claude"]["models"]["claude-haiku-4-5"]["input"],
            payload["claude"]["models"]["claude-haiku-4-5"]["input"],
        )

    def test_an_unidentified_kind_is_left_out_of_the_price_table(self) -> None:
        # input and cache_read move together across every scenario, so neither
        # can be separated from the other; output can.
        model = "claude-opus-5"
        fit = QB.fit_weights([
            {"key": "a", "percent": 1.0, "contaminated": False, "bracketed": True,
             "models": {model: {"input": 1_000_000.0, "cache_read": 1_000_000.0,
                                "output": 100_000.0}}},
            {"key": "b", "percent": 2.0, "contaminated": False, "bracketed": True,
             "models": {model: {"input": 2_000_000.0, "cache_read": 2_000_000.0,
                                "output": 500_000.0}}},
            {"key": "c", "percent": 3.0, "contaminated": False, "bracketed": True,
             "models": {model: {"input": 3_000_000.0, "cache_read": 3_000_000.0,
                                "output": 200_000.0}}},
        ])
        payload = QB.weights_payload(fit, "five_hour", "run")
        self.assertIsNone(payload["fit"]["models"][model]["cache_read"])
        self.assertIn("collinear", fit["diagnostics"][model]["cache_read"]["reasons"])
        self.assertNotIn("cache_read", payload["claude"]["models"][model])
        # The section quota-drain merges must price without error, and an
        # omitted kind keeps its built-in price rather than becoming free.
        merged = QD.builtin_weights()
        QD.merge_weights(merged, {"claude": payload["claude"]})
        units = QD.Weights(merged, ["test"]).claude_units(
            model, {"cache_read": 1_000_000}, None
        )
        self.assertAlmostEqual(units, 0.5)

    def test_a_fit_without_a_scale_is_not_used_for_planning(self) -> None:
        state = self.root / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / "claude-weights.json").write_text(
            json.dumps({
                "usable": True,
                "fallback_scale": 0.0,
                "claude": {"unit": "percent_per_mtok",
                           "models": {"claude-haiku-4-5": {"input": 0.9}}},
            }),
            encoding="utf-8",
        )
        result = self.bench("plan", "--models", "claude-haiku-4-5", "--contexts", "10k",
                            "--cache", "warm")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn(b"no list-price scale", result.stderr)
        self.assertIn(b"weights: list price", result.stdout)

    def test_max_sample_age_cannot_go_below_the_polling_floor(self) -> None:
        result = self.bench(
            "run", "--yes", "--quota-drain", str(self.fake_drain), "--sample-interval", "0",
            "--max-sample-age", "5", "--ticks", "3", "--max-percent", "50",
            "--max-percent-weekly", "50", "--models", "claude-haiku-4-5",
            "--contexts", "10k", "--cache", "warm",
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertIn(b"below the 60s the usage endpoint allows", result.stderr)
        runs = sorted((self.root / "state" / "bench").iterdir())
        meta = json.loads((runs[-1] / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["max_sample_age_s"], 60.0)

    def test_an_unusable_fit_is_named_in_one_sentence(self) -> None:
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        message = [line for line in result.stderr.decode("utf-8").splitlines()
                   if "not usable" in line]
        self.assertEqual(len(message), 1)
        self.assertIn("claude-weights.unusable.json", message[0])
        self.assertIn("claude-weights.json was left alone", message[0])

    def test_an_unusable_fit_does_not_become_the_weights_file(self) -> None:
        self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                           "--cache", "warm")
        state = self.root / "state"
        self.assertFalse((state / "claude-weights.json").exists())
        self.assertTrue((state / "claude-weights.unusable.json").exists())
        payload = json.loads((state / "claude-weights.unusable.json").read_text())
        self.assertFalse(payload["usable"])

    def test_an_exact_interpolation_is_not_usable(self) -> None:
        # Three rows and three columns pass through every point whatever the
        # weights are, so the fit has nothing to be wrong about.
        rows = [
            {"key": "a", "percent": 1.0, "contaminated": False, "bracketed": True,
             "models": {"m": {"input": 1_000_000.0}}},
            {"key": "b", "percent": 2.0, "contaminated": False, "bracketed": True,
             "models": {"m": {"input": 2_000_000.0}}},
        ]
        fit = QB.fit_weights(rows[:1])
        self.assertEqual(fit["degrees_of_freedom"], 0)
        self.assertFalse(fit["usable"])
        self.assertTrue(QB.fit_weights(rows)["usable"])

    def test_report_replays_the_stored_run(self) -> None:
        first = self.run_three_scenarios()
        run_id = first["run_id"]
        payload = self.bench_json("report", "--run-id", run_id, "--ticks", "3",
                                  "--quota-drain", str(self.fake_drain), "--json")
        self.assertEqual(payload["run_id"], run_id)
        self.assertEqual(
            payload["fit"]["models"]["claude-haiku-4-5"],
            first["fit"]["models"]["claude-haiku-4-5"],
        )

    def test_markdown_report_names_the_measured_ratios(self) -> None:
        self.run_three_scenarios()
        result = self.bench("report", "--ticks", "3", "--quota-drain", str(self.fake_drain))
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        text = result.stdout.decode("utf-8")
        self.assertIn("## Tokens per percent", text)
        self.assertIn("## Fitted weights", text)
        self.assertIn("claude-haiku-4-5 | cache_read", text)

    def test_varying_targets_score_r_squared(self) -> None:
        estimates = [
            {"key": "a", "percent": 1.0, "contaminated": False,
             "models": {"m": {"input": 1_000_000.0}}},
            {"key": "b", "percent": 3.0, "contaminated": False,
             "models": {"m": {"input": 3_000_000.0}}},
        ]
        fit = QB.fit_weights(estimates)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=6)
        self.assertTrue(fit["usable"])

    def test_contaminated_scenarios_stay_out_of_the_fit(self) -> None:
        estimates = [
            {"key": "a", "percent": 1.0, "contaminated": False,
             "models": {"m": {"input": 1_000_000.0}}},
            {"key": "b", "percent": 99.0, "contaminated": True,
             "models": {"m": {"input": 1_000_000.0}}},
        ]
        fit = QB.fit_weights(estimates)
        self.assertEqual(fit["samples"], 1)
        self.assertAlmostEqual(fit["models"]["m"]["input"], 1.0, places=3)


class Hygiene(Harness):
    def test_run_logs_hold_no_prompt_text(self) -> None:
        self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                           "--cache", "warm")
        runs = sorted((self.root / "state" / "bench").iterdir())
        for name in ("calls.jsonl", "scenarios.jsonl", "meta.json", "report.json"):
            text = (runs[-1] / name).read_text(encoding="utf-8")
            for needle in ("zephyr", "juniper", "one word", "accessToken", "Bearer"):
                self.assertNotIn(needle, text, "%s leaked %r" % (name, needle))

    def test_call_log_carries_usage_and_the_nearest_sample(self) -> None:
        self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                           "--cache", "warm")
        runs = sorted((self.root / "state" / "bench").iterdir())
        lines = (runs[-1] / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        self.assertEqual(first["scenario"], "claude-haiku-4-5/10k/warm/short")
        self.assertEqual(set(first["tokens"]), set(QD.CLAUDE_KINDS))
        self.assertIn("five_hour", first["sample"])
        self.assertLess(abs(first["sample_age_s"]), 60.0)

    def test_measured_context_comes_from_the_first_call(self) -> None:
        self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                           "--cache", "warm")
        record = self.scenario_records()[0]
        fixture = USAGE_FIXTURE["fresh|short"]
        self.assertEqual(record["measured_context_tokens"],
                         fixture["input"] + fixture["cache_read"] + fixture["cache_write"])

    def test_cold_mode_sends_fresh_filler_every_call(self) -> None:
        self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                           "--cache", "cold")
        calls = self.fake_calls()
        self.assertGreater(len(calls), 1)
        # Every cold call is a cache miss, so the fake never bills a cache read.
        self.assertTrue(all(call["tokens"]["cache_read"] == 0 for call in calls))

    def test_calls_run_with_the_built_in_tools_off(self) -> None:
        self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                           "--cache", "warm")
        argv = self.fake_calls()[0]["argv"]
        self.assertIn("--tools", argv)
        self.assertEqual(argv[argv.index("--tools") + 1], "")
        self.assertIn("--strict-mcp-config", argv)
        runs = sorted((self.root / "state" / "bench").iterdir())
        meta = json.loads((runs[-1] / "meta.json").read_text(encoding="utf-8"))
        self.assertIn("--tools", meta["claude_flags"])
        self.assertEqual(meta["claude_extra_args"], 0)
        self.assertFalse(meta["claude_extra_args_redacted"])

    def test_run_ids_do_not_collide_within_a_second(self) -> None:
        for _ in range(2):
            self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                               "--cache", "warm", "--max-calls", "1")
        self.assertEqual(len(list((self.root / "state" / "bench").iterdir())), 2)

    def test_a_warm_call_past_the_cache_ttl_is_marked_and_dropped(self) -> None:
        estimate = QB.estimate_scenario(
            {
                "key": "m/10k/warm/short", "model": "m", "context": "10k", "cache": "warm",
                "output": "short", "calls": 4, "start_percent": 0.0, "warm_drift": True,
                "ticks": [
                    {"ts": 1.0, "percent": 1.0, "calls": 2, "tokens": {"input": 10},
                     "models": {"m": {"input": 10}}},
                    {"ts": 2.0, "percent": 2.0, "calls": 4, "tokens": {"input": 20},
                     "models": {"m": {"input": 20}}},
                ],
            },
            required_ticks=2,
        )
        self.assertTrue(estimate["warm_drift"])
        self.assertIsNone(QB.fit_weights([estimate]))
        self.assertIsNotNone(QB.fit_weights([estimate], include_drifted=True))

    def test_system_prompt_file_channel_passes_the_flag(self) -> None:
        self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                           "--cache", "warm", "--filler-channel", "system-prompt-file")
        call = self.fake_calls()[0]
        self.assertIn("--system-prompt-file", call["argv"])
        self.assertLess(call["prompt_bytes"], 200)

    def test_failed_call_stops_the_scenario(self) -> None:
        self.environment["BENCH_FAKE_FAIL"] = "1"
        result = self.run_scenarios("--models", "claude-haiku-4-5", "--contexts", "10k",
                                    "--cache", "warm")
        self.assertEqual(result.returncode, QB.EXIT_DEPENDENCY)
        self.assertEqual(self.scenario_records()[0]["stop_reason"], "call-failed")

    def test_an_ignored_signal_stays_ignored(self) -> None:
        # A nohup'd run ignores SIGHUP on purpose; installing a handler over
        # SIG_IGN would kill it when the terminal closes.
        import signal as signal_module

        previous = signal_module.signal(signal_module.SIGHUP, signal_module.SIG_IGN)
        self.addCleanup(signal_module.signal, signal_module.SIGHUP, previous)
        workdir = self.root / "scratch"
        workdir.mkdir()
        QB.install_workdir_cleanup(workdir)
        self.assertEqual(signal_module.getsignal(signal_module.SIGHUP),
                         signal_module.SIG_IGN)
        # The other signals still get the handler.
        self.assertTrue(callable(signal_module.getsignal(signal_module.SIGTERM)))
        signal_module.signal(signal_module.SIGTERM, signal_module.SIG_DFL)

    def test_a_warm_call_past_the_ttl_is_flagged_while_running(self) -> None:
        # The gap check is driven by the module constant, so the driver patches
        # it to zero: every call then counts as past the cache TTL.
        driver = self.root / "drive-quota-bench.py"
        driver.write_text(
            "import importlib.machinery, importlib.util, sys\n"
            "loader = importlib.machinery.SourceFileLoader('qb', %r)\n"
            "spec = importlib.util.spec_from_loader(loader.name, loader)\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "sys.modules['qb'] = module\n"
            "spec.loader.exec_module(module)\n"
            "module.WARM_MAX_GAP_SECONDS = 0.0\n"
            "sys.exit(module.main(sys.argv[1:]))\n" % str(QUOTA_BENCH),
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, str(driver), "run", "--yes", "--quota-drain",
             str(self.fake_drain), "--sample-interval", "0", "--ticks", "3",
             "--max-percent", "50", "--max-percent-weekly", "50",
             "--models", "claude-haiku-4-5", "--contexts", "10k", "--cache", "warm"],
            check=False, capture_output=True, env=self.environment, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        record = self.scenario_records()[0]
        self.assertTrue(record["warm_drift"])
        runs = sorted((self.root / "state" / "bench").iterdir())
        lines = [json.loads(line) for line in
                 (runs[-1] / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        # The first call has no predecessor to drift from; the rest do.
        self.assertFalse(lines[0]["cache_ttl_risk"])
        self.assertTrue(all(line["cache_ttl_risk"] for line in lines[1:]))
        self.assertIn(b"warm_drift", result.stderr)

    def test_missing_quota_drain_is_named(self) -> None:
        result = self.bench("plan", "--quota-drain", str(self.root / "absent"))
        self.assertEqual(result.returncode, QB.EXIT_DEPENDENCY)
        self.assertIn(b"no quota-drain at", result.stderr)


class ResultParsing(Harness):
    def test_a_second_model_gets_its_own_row(self) -> None:
        self.environment["BENCH_FAKE_HELPER_MODEL"] = "claude-opus-5"
        result = self.run_scenarios("--models", "claude-haiku-4-5-20251001",
                                    "--contexts", "10k", "--cache", "warm")
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        record = self.scenario_records()[0]
        self.assertEqual(sorted(record["models"]), ["claude-haiku-4-5", "claude-opus-5"])
        self.assertEqual(record["models"]["claude-opus-5"]["input"], 1000 * record["calls"])
        # The helper's tokens are not part of the scenario's context.
        fixture = USAGE_FIXTURE["fresh|short"]
        self.assertEqual(record["measured_context_tokens"],
                         fixture["input"] + fixture["cache_read"] + fixture["cache_write"])
        estimate = self.latest_report()["scenarios"][0]
        self.assertIn("claude-opus-5", estimate["models"])
        self.assertIn("claude-haiku-4-5", estimate["models"])

    def test_two_models_are_not_blended(self) -> None:
        call = QB.parse_result(
            {
                "type": "result", "is_error": False, "session_id": "s",
                "modelUsage": {
                    "claude-haiku-4-5-20251001": {"inputTokens": 10, "outputTokens": 1},
                    "claude-opus-5": {"inputTokens": 500, "outputTokens": 50},
                },
            },
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(call.requested, "claude-haiku-4-5")
        self.assertEqual(call.models["claude-haiku-4-5"]["input"], 10)
        self.assertEqual(call.models["claude-opus-5"]["input"], 500)
        self.assertEqual(call.tokens["input"], 510)
        self.assertEqual(call.requested_tokens["input"], 10)

    def test_zeroed_model_usage_falls_back_to_usage(self) -> None:
        call = QB.parse_result(
            {
                "type": "result", "is_error": False, "session_id": "s",
                "modelUsage": {"claude-opus-5": {"inputTokens": 0, "outputTokens": 0}},
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
            "claude-opus-5",
        )
        self.assertEqual(call.models["claude-opus-5"]["input"], 7)

    def test_model_usage_is_preferred(self) -> None:
        call = QB.parse_result(
            {
                "type": "result",
                "is_error": False,
                "session_id": "s",
                "total_cost_usd": 0.5,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "modelUsage": {
                    "claude-haiku-4-5-20251001": {
                        "inputTokens": 10, "cacheReadInputTokens": 20,
                        "cacheCreationInputTokens": 30, "outputTokens": 40,
                    }
                },
            },
            "claude-sonnet-5",
        )
        self.assertEqual(call.requested, "claude-sonnet-5")
        self.assertEqual(sorted(call.models), ["claude-haiku-4-5"])
        self.assertEqual(call.tokens["input"], 10)
        self.assertEqual(call.tokens["cache_read"], 20)
        self.assertEqual(call.tokens["cache_write_5m"], 30)
        self.assertEqual(call.tokens["output"], 40)
        self.assertEqual(call.cost, 0.5)

    def test_usage_is_the_fallback(self) -> None:
        call = QB.parse_result(
            {
                "type": "result",
                "is_error": False,
                "session_id": "s",
                "usage": {
                    "input_tokens": 5,
                    "cache_read_input_tokens": 6,
                    "cache_creation_input_tokens": 7,
                    "cache_creation": {"ephemeral_5m_input_tokens": 3,
                                       "ephemeral_1h_input_tokens": 4},
                    "output_tokens": 8,
                },
            },
            "claude-opus-5",
        )
        self.assertEqual(call.requested, "claude-opus-5")
        self.assertEqual(call.tokens["cache_write_5m"], 3)
        self.assertEqual(call.tokens["cache_write_1h"], 4)
        self.assertIsNone(call.cost)

    def test_error_results_are_refused(self) -> None:
        self.assertIsNone(QB.parse_result({"type": "result", "is_error": True}, "m"))
        self.assertIsNone(QB.parse_result({"type": "assistant"}, "m"))


if __name__ == "__main__":
    unittest.main()
