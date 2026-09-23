"""Measure what one percent of a Claude plan window costs, per token kind.

Fires controlled `claude -p` calls, watches live utilisation tick over through
`nenpi snapshot --oauth`, and brackets the tokens spent between ticks.
The output is a per-scenario tokens-per-percent table plus a non-negative
least squares fit of percent per million tokens per model and kind, stored in
the shape `nenpi calibrate --harness claude` reports.

Every percent this spends is real quota: `run` refuses to start without
`--yes` and aborts at `--max-percent`. Prompt text, filler text and the OAuth
token are never printed, logged or stored; token handling stays inside
nenpi-bench, which owns the credential read.

Test path overrides use the same ``NENPI_*`` environment variables nenpi
reads: ``HOME_DIR``, ``CACHE_DIR``, ``STATE_DIR``, ``CONFIG_DIR`` (the old
``QUOTA_DRAIN_*`` names still work, with a deprecation warning).
"""

from __future__ import annotations

import argparse
import atexit
import importlib.machinery
import importlib.util
import json
import math
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from nenpi.config import migrate_dirs


DEFAULT_MAX_PERCENT = 3.0
DEFAULT_MAX_PERCENT_WEEKLY = 1.0
DEFAULT_TICKS = 2
DEFAULT_MAX_CALLS = 400
DEFAULT_SAMPLE_INTERVAL = 60.0
# A call may only be launched while the newest utilisation sample is younger
# than this; older than it, the budget check is reading a stale number.
MIN_SAMPLE_AGE = 90.0
# nenpi will not poll the usage endpoint more than once a minute, so a
# shorter limit than this can only ever fire against its own floor.
SAMPLE_AGE_FLOOR = 60.0
DEFAULT_CALL_TIMEOUT = 900.0
DEFAULT_CONTAMINATION_WINDOW = "6h"

# The 5-minute prompt cache: a warm scenario must stay inside it, and a
# write-heavy scenario must fall outside it so every call pays cache creation.
WARM_MAX_GAP_SECONDS = 240.0
WRITE_HEAVY_GAP_SECONDS = 360.0

# No dollar ceiling is published for any plan and every account seen reports
# `limit_dollars` null, so planning falls back to this prior when no fit and no
# dollar field is available. It only scales the projected call counts `plan`
# prints; nothing measured depends on it.
PLANNING_USD_PER_PERCENT = 1.40

CONTEXT_SIZES = {"10k": 10_000, "60k": 60_000, "150k": 150_000}
CACHE_MODES = ("cold", "warm", "write-heavy")
OUTPUT_MODES = ("short", "long")

# Cheapest first: the early scenarios buy the tokens-per-percent estimate that
# tells the operator what the expensive ones will cost.
MODEL_COST_ORDER = (
    "claude-haiku-4-5",
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-fable-5",
    "claude-fable-5-1",
)
CACHE_COST_ORDER = ("warm", "cold", "write-heavy")
OUTPUT_COST_ORDER = ("short", "long")

SHORT_TASK = "Answer with exactly one word: ok. Do not use any tools."
LONG_TASK = (
    "Write a 2000-word essay on the history of mechanical clocks. "
    "Do not use any tools and do not ask questions."
)
SHORT_OUTPUT_TOKENS = 10
LONG_OUTPUT_TOKENS = 2700

# Deterministic filler. Short ASCII words keep the words-to-tokens ratio stable
# across sizes, which is what makes the same label mean the same context.
FILLER_WORDS = (
    "alpha", "bravo", "cinder", "delta", "ember", "forge", "gravel", "harbor",
    "indigo", "juniper", "kelvin", "lantern", "marble", "nimbus", "onyx",
    "pewter", "quarry", "ribbon", "slate", "timber", "umber", "velvet",
    "willow", "xenon", "yarrow", "zephyr",
)
WORDS_PER_TOKEN = 0.75

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_DEPENDENCY = 2
# A guard stopped the run: the budget was reached, or utilisation stopped being
# observable, or another session was in the window. Distinct codes so a caller
# can tell a completed measurement from one that was cut short.
EXIT_BUDGET = 3
EXIT_GUARD = 4
EXIT_MAX_CALLS = 5


def warn(message: str) -> None:
    sys.stderr.write("nenpi-bench: " + message + "\n")


def load_quota_drain(path: Path) -> Any:
    """Import quota-drain for its snapshot format, NNLS fit and state paths."""
    if not path.is_file():
        warn("no quota-drain at %s; pass --quota-drain PATH" % path)
        raise SystemExit(EXIT_DEPENDENCY)
    loader = importlib.machinery.SourceFileLoader("quota_drain_for_bench", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None or spec.loader is None:
        warn("cannot import %s" % path)
        raise SystemExit(EXIT_DEPENDENCY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# scenarios


class Scenario:
    def __init__(self, model: str, context: str, cache: str, output: str):
        self.model = model
        self.context = context
        self.cache = cache
        self.output = output

    @property
    def normalized_model(self) -> str:
        return normalize_model(self.model)

    @property
    def key(self) -> str:
        return "%s/%s/%s/%s" % (self.normalized_model, self.context, self.cache, self.output)

    @property
    def context_tokens(self) -> int:
        return CONTEXT_SIZES[self.context]

    @property
    def sort_key(self) -> Tuple[int, int, int, int]:
        def rank(order: Sequence[str], value: str) -> int:
            return order.index(value) if value in order else len(order)

        return (
            rank(MODEL_COST_ORDER, self.normalized_model),
            rank(CACHE_COST_ORDER, self.cache),
            sorted(CONTEXT_SIZES, key=lambda name: CONTEXT_SIZES[name]).index(self.context),
            rank(OUTPUT_COST_ORDER, self.output),
        )


_QD = None  # type: Any


def normalize_model(model: str) -> str:
    if _QD is not None:
        return _QD.normalize_claude_model(model)
    return str(model)


def split_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_scenarios(args: argparse.Namespace) -> List[Scenario]:
    contexts = split_list(args.contexts)
    caches = split_list(args.cache)
    outputs = split_list(args.outputs)
    for name in contexts:
        if name not in CONTEXT_SIZES:
            warn("unknown context size %r; choose from %s" % (name, ",".join(CONTEXT_SIZES)))
            raise SystemExit(EXIT_USAGE)
    for name in caches:
        if name not in CACHE_MODES:
            warn("unknown cache mode %r; choose from %s" % (name, ",".join(CACHE_MODES)))
            raise SystemExit(EXIT_USAGE)
    for name in outputs:
        if name not in OUTPUT_MODES:
            warn("unknown output mode %r; choose from %s" % (name, ",".join(OUTPUT_MODES)))
            raise SystemExit(EXIT_USAGE)
    scenarios = [
        Scenario(model, context, cache, output)
        for model in split_list(args.models)
        for context in contexts
        for cache in caches
        for output in outputs
    ]
    if not scenarios:
        warn("no scenarios selected")
        raise SystemExit(EXIT_USAGE)
    scenarios.sort(key=lambda item: item.sort_key)
    return scenarios


# --------------------------------------------------------------------------
# filler and prompts


def filler_text(target_tokens: int, seed: int) -> str:
    """Deterministic filler of roughly `target_tokens` tokens."""
    rng = random.Random(seed)
    words = int(target_tokens * (1.0 / WORDS_PER_TOKEN))
    chosen = [rng.choice(FILLER_WORDS) for _ in range(max(1, words))]
    lines = []
    for start in range(0, len(chosen), 20):
        lines.append(" ".join(chosen[start:start + 20]))
    return "\n".join(lines)


def scenario_task(scenario: Scenario) -> str:
    return LONG_TASK if scenario.output == "long" else SHORT_TASK


def expected_tokens(scenario: Scenario, first_call: bool,
                    cold_as_creation: bool = True) -> Dict[str, float]:
    """Per-call token mix a scenario is expected to produce, for planning only.

    Claude Code sets cache breakpoints on its own prefix, so a call that sends
    a prefix the cache has not seen is expected to pay cache creation at 1.25x
    rather than plain input. That is the conservative reading and it is
    unverified; `--cold-as-creation` turns it off.
    """
    context = float(scenario.context_tokens)
    output = float(LONG_OUTPUT_TOKENS if scenario.output == "long" else SHORT_OUTPUT_TOKENS)
    tokens = {"input": 0.0, "cache_read": 0.0, "cache_write_5m": 0.0,
              "cache_write_1h": 0.0, "output": output,
              "cache_write_unknown": 0.0}
    fresh_prefix = scenario.cache != "warm" or first_call
    if not fresh_prefix:
        tokens["cache_read"] = context
        tokens["input"] = 200.0
    elif scenario.cache == "cold" and not cold_as_creation:
        tokens["input"] = context
    else:
        tokens["cache_write_5m"] = context
    return tokens


# --------------------------------------------------------------------------
# sampling


class Sample:
    def __init__(self, ts: float, windows: Mapping[str, float]):
        self.ts = ts
        self.windows = dict(windows)

    def percent(self, window: str) -> Optional[float]:
        value = self.windows.get(window)
        return None if value is None else float(value)

    def as_json(self) -> Dict[str, Any]:
        record = {"ts": self.ts}
        record.update(self.windows)
        return record


class Sampler:
    """Runs `nenpi snapshot --oauth` and reads back what it logged.

    The OAuth token is read, used and discarded inside quota-drain; nothing
    about it crosses this boundary.
    """

    def __init__(self, qd: Any, drain_command: Sequence[str], config_dirs: Sequence[str],
                 interval: float, max_age: float):
        self.qd = qd
        self.drain_command = drain_command
        self.config_dirs = list(config_dirs)
        self.interval = interval
        self.max_age = max_age
        self.last_attempt = 0.0
        self.latest = None  # type: Optional[Sample]
        self.failure = None  # type: Optional[str]

    def due(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) - self.last_attempt >= self.interval

    def sample(self, force: bool = False) -> Optional[Sample]:
        """Sample if due. Any failure is recorded; it is never swallowed.

        A run that cannot see utilisation cannot see what it is spending, so
        every failure mode - the sampler exiting non-zero, crashing, or logging
        no new observation - has to reach the caller.
        """
        now = time.time()
        if not force and not self.due(now):
            return self.latest
        self.last_attempt = now
        command = list(self.drain_command) + ["snapshot", "--oauth"]
        for config in self.config_dirs:
            command += ["--config-dir", config]
        try:
            result = subprocess.run(command, check=False, capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as error:
            self.failure = "sampler failed (%s)" % type(error).__name__
            return self.latest
        if result.returncode != 0:
            self.failure = "sampler exited %d" % result.returncode
            return self.latest
        observed = self.read_latest()
        if observed is None:
            self.failure = "sampler logged no readable observation"
            return self.latest
        if self.latest is not None and observed.ts <= self.latest.ts:
            self.failure = "sampler logged no new observation"
            return self.latest
        self.latest = observed
        self.failure = None
        return self.latest

    def stale_for(self, now: Optional[float] = None) -> Optional[float]:
        """Age of the newest observation when it is older than `max_age`."""
        if self.latest is None:
            return float("inf")
        age = (now or time.time()) - self.latest.ts
        return age if age > self.max_age else None

    def blocker(self) -> Optional[str]:
        """Why a call must not be launched right now, if anything."""
        if self.failure is not None:
            return self.failure
        age = self.stale_for()
        if age is None:
            return None
        if age == float("inf"):
            return "no utilisation sample yet"
        return "newest utilisation sample is %.0fs old (limit %.0fs)" % (age, self.max_age)

    def read_latest(self) -> Optional[Sample]:
        # The snapshot log holds two months of observations and a run only ever
        # wants the newest, so each read is bounded to the last hour. The bound
        # is computed per read: a long run would outlive one fixed at startup.
        since = time.time() - 3600.0
        windows = {}
        stamp = None
        for window in ("five_hour", "seven_day"):
            rows = self.qd.load_claude_snapshots(window, since=since)
            if not rows:
                continue
            row = rows[-1]
            windows[window] = float(row["used_percent"])
            stamp = max(stamp or 0.0, float(row["ts"]))
        if not windows or stamp is None:
            return None
        return Sample(stamp, windows)


def foreign_sessions(
    drain_command: Sequence[str], since: str, start: float, end: float, own: Sequence[str]
) -> Tuple[List[str], bool]:
    """Claude sessions other than this run's that had API turns in the interval.

    Returns the sessions found and whether the check actually ran. A check that
    could not run is not evidence of an idle window, so its failure is reported
    rather than read as a clean result.
    """
    command = list(drain_command) + ["sessions", "--harness", "claude",
               "--since", since, "--json", "--top", "200"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as error:
        warn("contamination check failed (%s)" % type(error).__name__)
        return [], False
    if result.returncode != 0:
        warn("contamination check exited %d" % result.returncode)
        return [], False
    try:
        payload = json.loads(result.stdout.decode("utf-8", "replace"))
    except ValueError:
        warn("contamination check returned no JSON")
        return [], False
    if not isinstance(payload, Mapping) or not isinstance(payload.get("sessions"), list):
        warn("contamination check returned no session list")
        return [], False
    mine = set(own)
    found = []
    for row in payload.get("sessions") or []:
        if row.get("harness") != "claude":
            continue
        session_id = str(row.get("session_id") or "")
        if session_id in mine:
            continue
        row_start = row.get("start")
        row_end = row.get("end")
        if not isinstance(row_start, (int, float)) or not isinstance(row_end, (int, float)):
            continue
        if float(row_end) >= start and float(row_start) <= end:
            found.append(session_id)
    return sorted(set(found)), True


# --------------------------------------------------------------------------
# one claude -p call


class CallResult:
    """One call's usage, kept per model.

    A run can touch more than one model - a subagent, a compaction, an internal
    helper - and blending those into one dict would price another model's
    tokens at the requested model's weight. Each model keeps its own row.
    """

    def __init__(self, models: Mapping[str, Mapping[str, int]], requested: str,
                 cost: Optional[float], session_id: str, started: float, ended: float):
        self.models = dict((name, dict(counts)) for name, counts in models.items())
        self.requested = requested
        self.cost = cost
        self.session_id = session_id
        self.started = started
        self.ended = ended

    @property
    def tokens(self) -> Dict[str, int]:
        """Every model's tokens summed, for the call log and the budget."""
        total = {}  # type: Dict[str, int]
        for counts in self.models.values():
            for kind, value in counts.items():
                total[kind] = total.get(kind, 0) + value
        return total

    @property
    def requested_tokens(self) -> Dict[str, int]:
        return dict(self.models.get(self.requested) or {})


def usage_tokens(usage: Mapping[str, Any]) -> Dict[str, int]:
    """Map one `usage` block onto quota-drain's Claude token kinds."""
    def count(name: str) -> int:
        value = usage.get(name)
        return int(value) if isinstance(value, (int, float)) else 0

    write_5m, write_1h, write_unknown = _QD.claude_cache_write_tokens(usage)
    return {
        "input": count("input_tokens"),
        "cache_read": count("cache_read_input_tokens"),
        "cache_write_5m": write_5m,
        "cache_write_1h": write_1h,
        "output": count("output_tokens"),
        "cache_write_unknown": write_unknown,
    }


def model_usage_tokens(entry: Mapping[str, Any]) -> Dict[str, int]:
    """Map one `modelUsage` entry (camelCase) onto the same token kinds."""
    def count(name: str) -> int:
        value = entry.get(name)
        return int(value) if isinstance(value, (int, float)) else 0

    return {
        "input": count("inputTokens"),
        "cache_read": count("cacheReadInputTokens"),
        "cache_write_5m": 0,
        "cache_write_1h": 0,
        "output": count("outputTokens"),
        "cache_write_unknown": count("cacheCreationInputTokens"),
    }


def parse_result(payload: Mapping[str, Any], fallback_model: str) -> Optional[CallResult]:
    """Read the `--output-format json` result envelope.

    `modelUsage` is the field the CLI documents for token accounting: it covers
    every model call the run made, keyed by model id. `usage` is the main loop
    only, and is the fallback when a build reports no `modelUsage`.
    """
    if payload.get("type") != "result" or payload.get("is_error"):
        return None
    model_usage = payload.get("modelUsage")
    models = {}  # type: Dict[str, Dict[str, int]]
    requested = normalize_model(fallback_model)
    if isinstance(model_usage, Mapping):
        for name, entry in model_usage.items():
            if not isinstance(entry, Mapping):
                continue
            counted = model_usage_tokens(entry)
            if not sum(counted.values()):
                continue
            key = normalize_model(str(name))
            booked = models.setdefault(key, dict((kind, 0) for kind in counted))
            for kind, value in counted.items():
                booked[kind] = booked.get(kind, 0) + value
    if not models:
        # No modelUsage, or every entry zeroed: fall back to the main-loop
        # block, which carries no model of its own.
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            return None
        counted = usage_tokens(usage)
        if not sum(counted.values()):
            return None
        models = {requested: counted}
    cost = payload.get("total_cost_usd")
    return CallResult(
        models,
        requested,
        float(cost) if isinstance(cost, (int, float)) else None,
        str(payload.get("session_id") or ""),
        0.0,
        0.0,
    )


def call_flags(args: argparse.Namespace) -> List[str]:
    """The flags every benchmark call carries, recorded in meta.json.

    A tool call would add turns and tokens this scenario did not ask for, so
    the built-in tools are switched off. `--tools ""` is the documented way to
    do that on 2.1.273; that build's `--help` lists no turn limit, and with no
    tools there is nothing for a second turn to do.
    """
    flags = ["--output-format", "json", "--strict-mcp-config"]
    if not args.allow_tools:
        flags += ["--tools", ""]
    return flags


def run_call(scenario: Scenario, prompt: str, system_file: Optional[Path],
             args: argparse.Namespace, workdir: Path) -> Optional[CallResult]:
    command = [args.claude, "-p", "--model", scenario.model] + call_flags(args)
    if system_file is not None:
        command += ["--system-prompt-file", str(system_file)]
    command += list(args.claude_arg)
    started = time.time()
    try:
        result = subprocess.run(
            command,
            check=False,
            input=prompt.encode("utf-8"),
            capture_output=True,
            cwd=str(workdir),
            timeout=args.call_timeout,
        )
    except subprocess.TimeoutExpired:
        warn("%s: call timed out after %.0fs" % (scenario.key, args.call_timeout))
        return None
    except OSError as error:
        warn("%s: cannot run %s (%s)" % (scenario.key, args.claude, error.strerror))
        return None
    ended = time.time()
    if result.returncode != 0:
        # stderr can echo the prompt back; only the status is reportable.
        warn("%s: claude exited %d" % (scenario.key, result.returncode))
        return None
    try:
        payload = json.loads(result.stdout.decode("utf-8", "replace"))
    except ValueError:
        warn("%s: claude returned no JSON result" % scenario.key)
        return None
    call = parse_result(payload, scenario.model)
    if call is None:
        warn("%s: result carried no usage" % scenario.key)
        return None
    call.started = started
    call.ended = ended
    return call


# --------------------------------------------------------------------------
# the run


class ScenarioRun:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.calls = 0
        self.started = time.time()
        self.ended = self.started
        self.cumulative = {kind: 0 for kind in _QD.CLAUDE_KINDS}
        self.ticks = []  # type: List[Dict[str, Any]]
        self.stop_reason = "incomplete"
        self.contaminated_by = []  # type: List[str]
        self.contamination_reasons = []  # type: List[str]
        self.rollover = False
        self.warm_drift = False
        self.start_percent = None  # type: Optional[float]
        self.measured_context = None  # type: Optional[int]
        self.models = {}  # type: Dict[str, Dict[str, int]]
        self.cost_usd = 0.0

    def absorb(self, call: CallResult) -> None:
        self.calls += 1
        self.ended = call.ended
        for name, counts in call.models.items():
            booked = self.models.setdefault(
                name, dict((kind, 0) for kind in _QD.CLAUDE_KINDS)
            )
            for kind, value in counts.items():
                self.cumulative[kind] = self.cumulative.get(kind, 0) + value
                booked[kind] = booked.get(kind, 0) + value
        if call.cost:
            self.cost_usd += call.cost
        if self.measured_context is None:
            # The context the scenario set up is the requested model's input
            # side; another model's tokens are not this scenario's context.
            requested = call.requested_tokens
            self.measured_context = (
                requested.get("input", 0)
                + requested.get("cache_read", 0)
                + requested.get("cache_write_5m", 0)
                + requested.get("cache_write_1h", 0)
                + requested.get("cache_write_unknown", 0)
            )

    def record_tick(self, sample: Sample, percent: float) -> None:
        self.ticks.append(
            {
                "ts": sample.ts,
                "percent": percent,
                "calls": self.calls,
                "tokens": dict(self.cumulative),
                "models": dict((name, dict(counts)) for name, counts in self.models.items()),
            }
        )

    def as_json(self) -> Dict[str, Any]:
        scenario = self.scenario
        return {
            "key": scenario.key,
            "model": scenario.model,
            "normalized_model": scenario.normalized_model,
            "context": scenario.context,
            "context_target_tokens": scenario.context_tokens,
            "measured_context_tokens": self.measured_context,
            "cache": scenario.cache,
            "output": scenario.output,
            "calls": self.calls,
            "started": self.started,
            "ended": self.ended,
            "start_percent": self.start_percent,
            "tokens": dict(self.cumulative),
            "models": dict((name, dict(counts)) for name, counts in self.models.items()),
            "cost_usd": self.cost_usd,
            "ticks": self.ticks,
            "stop_reason": self.stop_reason,
            "contaminated": bool(self.contaminated_by or self.contamination_reasons),
            "contaminated_by": list(self.contaminated_by),
            "contamination_reasons": list(self.contamination_reasons),
            "warm_drift": self.warm_drift,
            "window_rollover": self.rollover,
        }


class Budget:
    """Percent spent since the run started, per window, across rollovers."""

    def __init__(self, limits: Mapping[str, float]):
        self.limits = dict(limits)
        self.spent = dict((window, 0.0) for window in limits)
        self.previous = {}  # type: Dict[str, float]

    def observe(self, sample: Sample) -> List[str]:
        """Return the windows that rolled over at this sample."""
        rolled = []
        for window in self.limits:
            percent = sample.percent(window)
            if percent is None:
                continue
            last = self.previous.get(window)
            if last is None:
                self.previous[window] = percent
                continue
            if percent < last:
                rolled.append(window)
            else:
                self.spent[window] += percent - last
            self.previous[window] = percent
        return rolled

    def exceeded(self) -> Optional[str]:
        for window, limit in sorted(self.limits.items()):
            if limit > 0 and self.spent.get(window, 0.0) >= limit:
                return window
        return None

    def as_json(self) -> Dict[str, Any]:
        return {"limits": dict(self.limits), "spent": dict(self.spent)}


def run_scenario(scenario: Scenario, args: argparse.Namespace, sampler: Sampler,
                 budget: Budget, workdir: Path, calls_log, own_sessions: List[str]) -> ScenarioRun:
    state = ScenarioRun(scenario)
    warm_filler = filler_text(scenario.context_tokens, seed=hash_seed(scenario.key))
    last_started = None  # type: Optional[float]
    last_finished = None  # type: Optional[float]
    seen = sampler.latest
    last_percent = seen.percent(args.window) if seen is not None else None
    state.start_percent = last_percent

    def poll(force: bool = False) -> None:
        """Sample if due, then fold the observation into ticks and the budget."""
        nonlocal last_percent
        if not force and not sampler.due():
            return
        previous = sampler.latest
        sample = sampler.sample(force=force)
        if sample is None or (previous is not None and sample.ts <= previous.ts):
            return
        for rolled in budget.observe(sample):
            if rolled == args.window:
                # The bracket spans a window boundary and measures nothing.
                state.rollover = True
                state.ticks = []
                last_percent = None
        percent = sample.percent(args.window)
        if percent is None:
            return
        if last_percent is not None and percent > last_percent:
            state.record_tick(sample, percent)
        if state.start_percent is None:
            state.start_percent = percent
        last_percent = percent

    def wait_for_cache_mode() -> None:
        """Hold a write-heavy scenario outside the cache TTL, sampling meanwhile."""
        if scenario.cache != "write-heavy" or last_started is None:
            return
        while True:
            remaining = args.write_heavy_gap - (time.time() - last_started)
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5.0))
            poll()

    while True:
        if state.calls >= args.max_calls:
            state.stop_reason = "max-calls"
            break
        window = budget.exceeded()
        if window is not None:
            state.stop_reason = "budget:" + window
            break
        if len(state.ticks) >= args.ticks:
            state.stop_reason = "ticks"
            break

        wait_for_cache_mode()
        # A call may only be launched while utilisation is observable. Without
        # a fresh sample the budget check above is measuring a stale number,
        # and the run would spend past its cap without noticing.
        if sampler.blocker() is not None:
            poll(force=True)
        blocker = sampler.blocker()
        if blocker is not None:
            warn("%s: %s; stopping before the next call" % (scenario.key, blocker))
            state.stop_reason = "sampler:" + blocker
            break
        if scenario.cache == "cold":
            filler = filler_text(scenario.context_tokens,
                                 seed=hash_seed("%s#%d" % (scenario.key, state.calls)))
        else:
            filler = warm_filler
        system_file = None  # type: Optional[Path]
        if args.filler_channel == "system-prompt-file":
            system_file = workdir / "filler.txt"
            system_file.write_text(filler, encoding="utf-8")
            prompt = scenario_task(scenario)
        else:
            prompt = filler + "\n\n" + scenario_task(scenario)

        last_started = time.time()
        # A warm scenario means "the prefix is still cached". Past the TTL it
        # is not, whatever the label says, so the drift is recorded and the
        # scenario stays out of the fit.
        cache_ttl_risk = (
            scenario.cache == "warm"
            and last_finished is not None
            and last_started - last_finished > WARM_MAX_GAP_SECONDS
        )
        if cache_ttl_risk:
            state.warm_drift = True
        call = run_call(scenario, prompt, system_file, args, workdir)
        if call is None:
            state.stop_reason = "call-failed"
            break
        last_finished = call.ended
        state.absorb(call)
        if call.session_id and call.session_id not in own_sessions:
            own_sessions.append(call.session_id)

        poll()
        write_call_line(calls_log, scenario, call, sampler.latest, cache_ttl_risk)

    state.ended = time.time()
    return state


def hash_seed(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) & 0xFFFFFFFF
    return value


def write_call_line(handle, scenario: Scenario, call: CallResult,
                    sample: Optional[Sample], cache_ttl_risk: bool = False) -> None:
    """One line per call. Usage and timing only; never prompt text."""
    record = {
        "scenario": scenario.key,
        "model": call.requested,
        "started": call.started,
        "ended": call.ended,
        "duration_s": call.ended - call.started,
        "tokens": dict(call.tokens),
        "models": dict((name, dict(counts)) for name, counts in call.models.items()),
        "cost_usd": call.cost,
        "session_id": call.session_id,
        "cache_ttl_risk": cache_ttl_risk,
    }
    if sample is not None:
        record["sample"] = sample.as_json()
        record["sample_age_s"] = call.ended - sample.ts
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


# --------------------------------------------------------------------------
# estimates and fit


def estimate_scenario(record: Mapping[str, Any], required_ticks: int) -> Optional[Dict[str, Any]]:
    """Tokens per percent between the first and last observed tick.

    Everything before the first tick and after the last is a partial percent of
    unknown size and is discarded. With `--ticks 1` there is no fully bracketed
    percent; the estimate then spans the scenario start to the single tick and
    is reported unbracketed.
    """
    ticks = list(record.get("ticks") or [])
    if len(ticks) >= 2:
        first, last = ticks[0], ticks[-1]
        percent = float(last["percent"]) - float(first["percent"])
        base = first["tokens"]
        base_models = first.get("models") or {}
        bracketed = True
    elif len(ticks) == 1 and required_ticks <= 1:
        last = ticks[0]
        start = record.get("start_percent")
        percent = (float(last["percent"]) - float(start)) if isinstance(start, (int, float)) else 1.0
        percent = max(1.0, percent)
        base = dict((kind, 0) for kind in last["tokens"])
        base_models = {}
        bracketed = False
    else:
        return None
    if percent <= 0:
        return None
    tokens = dict(
        (kind, float(last["tokens"].get(kind, 0)) - float(base.get(kind, 0)))
        for kind in last["tokens"]
    )
    models = {}  # type: Dict[str, Dict[str, float]]
    for name, counts in (last.get("models") or {}).items():
        previous = base_models.get(name) or {}
        models[name] = dict(
            (kind, float(value) - float(previous.get(kind, 0))) for kind, value in counts.items()
        )
    return {
        "key": record["key"],
        "model": record.get("normalized_model") or record["model"],
        "context": record["context"],
        "cache": record["cache"],
        "output": record["output"],
        "percent": percent,
        "bracketed": bracketed,
        "ticks": len(ticks),
        "calls": int(last.get("calls") or record.get("calls") or 0),
        "tokens": tokens,
        "models": models,
        "tokens_per_percent": dict((kind, value / percent) for kind, value in tokens.items()),
        "upper_bound": not bracketed,
        "warm_drift": bool(record.get("warm_drift")),
        "contaminated": bool(record.get("contaminated")),
        "window_rollover": bool(record.get("window_rollover")),
        "measured_context_tokens": record.get("measured_context_tokens"),
    }


def fit_weights(estimates: Sequence[Mapping[str, Any]], include_unbracketed: bool = False,
                include_drifted: bool = False) -> Optional[Dict[str, Any]]:
    """Non-negative least squares of measured percent against token features.

    Only fully bracketed rows are fitted. An unbracketed row divides everything
    spent since the scenario started by an assumed single percent, including
    the partial percent that was already burnt before the first tick, so it is
    an upper bound rather than a measurement.

    Coefficients the scenarios cannot separate are stored as null rather than
    as a confident zero, the same way `quota-drain calibrate` does, because a
    zero weight prices that model and kind as free. The identifiability rules
    differ from quota-drain's: its rows are time buckets of whatever ran, while
    these rows are designed scenarios, so a kind that appears in one scenario
    is still identified as long as the system is not underdetermined.
    """
    rows = []
    excluded = []
    for item in estimates:
        if item["contaminated"] or item["percent"] <= 0:
            excluded.append((item["key"], "contaminated"))
            continue
        if not item.get("bracketed", True) and not include_unbracketed:
            excluded.append((item["key"], "unbracketed"))
            continue
        if item.get("warm_drift") and not include_drifted:
            excluded.append((item["key"], "warm_drift"))
            continue
        rows.append(item)
    for key, reason in excluded:
        warn("fit excludes %s (%s)" % (key, reason))
    if not rows:
        return None
    features = set()
    for row in rows:
        for model, counts in row["models"].items():
            for kind, value in counts.items():
                if value > 0:
                    features.add((model, kind))
    ordered = sorted(features)
    if not ordered:
        return None
    matrix = [
        [(row["models"].get(model) or {}).get(kind, 0.0) / 1_000_000.0
         for model, kind in ordered]
        for row in rows
    ]
    target = [row["percent"] for row in rows]
    coefficients = _QD.nnls(matrix, target)
    mean = sum(target) / len(target)
    spread = sum((value - mean) ** 2 for value in target)
    # Scenarios deliberately stop at the same tick count, so every target can be
    # the same number of percents. R^2 is undefined against zero variance; the
    # relative residual is what says whether the fit explains the rows.
    score = _QD.r_squared(matrix, target, coefficients) if spread > 0 else None
    underdetermined = len(rows) < len(ordered)
    mass = [sum(row[column] for row in matrix) for column in range(len(ordered))]
    total_mass = sum(mass) or 1.0
    fitted = {}  # type: Dict[str, Dict[str, Optional[float]]]
    diagnostics = {}  # type: Dict[str, Dict[str, Any]]
    identified = 0
    for column, (model, kind) in enumerate(ordered):
        scenarios = sum(1 for row in matrix if row[column] > 0)
        share = mass[column] / total_mass
        correlation = 0.0
        for other in range(len(ordered)):
            if other != column:
                correlation = max(correlation, _QD.column_correlation(matrix, column, other))
        at_boundary = coefficients[column] <= 0.0
        reasons = []
        if correlation > 0.95:
            reasons.append("collinear")
        if at_boundary and share < 0.05:
            reasons.append("at zero with little token mass")
        if underdetermined and scenarios < 2:
            reasons.append("too few scenarios")
        diagnostics.setdefault(model, {})[kind] = {
            "scenarios": scenarios,
            "token_share": share,
            "max_correlation": correlation,
            "at_boundary": at_boundary,
            "unidentified": bool(reasons),
            "reasons": reasons,
        }
        fitted.setdefault(model, {})[kind] = None if reasons else coefficients[column]
        if not reasons:
            identified += 1
    residuals = []
    squared = 0.0
    for index, row in enumerate(rows):
        predicted = sum(
            matrix[index][column] * coefficients[column] for column in range(len(coefficients))
        )
        squared += (row["percent"] - predicted) ** 2
        residuals.append(
            {"key": row["key"], "measured_percent": row["percent"],
             "predicted_percent": predicted, "residual": row["percent"] - predicted}
        )
    relative = math.sqrt(squared / len(rows)) / mean if mean > 0 else float("inf")
    explained = score >= 0.5 if score is not None else relative < 0.05
    scale = _QD.fit_fallback_scale(
        [(row["percent"], dict(((model, kind), (row["models"].get(model) or {}).get(kind, 0.0)
                                / 1_000_000.0)
                               for model, kind in ordered))
         for row in rows],
        ordered,
        target,
        (_QD.builtin_weights().get("claude") or {}),
    )
    # An exact interpolation explains nothing: with one row per column the fit
    # passes through every point whatever the weights are, so a usable fit
    # needs at least one row the solver could have failed on.
    freedom = len(rows) - len(ordered)
    return {
        "samples": len(rows),
        "intervals": len(rows),
        "columns": len(ordered),
        "degrees_of_freedom": freedom,
        "underdetermined": underdetermined,
        "r_squared": score,
        "residual_relative": relative,
        "identified": identified,
        "fallback_scale": scale,
        "usable": explained and identified > 0 and freedom >= 1,
        "models": fitted,
        "diagnostics": diagnostics,
        "residuals": residuals,
        "excluded": [{"key": key, "reason": reason} for key, reason in excluded],
    }


def weights_payload(fit: Mapping[str, Any], window: str, run_id: str) -> Dict[str, Any]:
    """The shape `quota-drain calibrate --harness claude` reports.

    The top-level `claude` section is what `merge_weights` consumes, so it
    carries measured numbers only: a kind the fit could not identify is left
    out entirely and keeps its built-in price. The nulls and their reasons live
    in the `fit` section, which is a record of the run rather than a price
    table.
    """
    models = {}  # type: Dict[str, Dict[str, float]]
    for name, entry in fit["models"].items():
        measured = dict((kind, float(value)) for kind, value in entry.items()
                        if isinstance(value, (int, float)))
        if measured:
            models[name] = measured
    fit_models = dict((name, dict(entry)) for name, entry in fit["models"].items())
    window_fit = {
        "samples": fit["samples"],
        "intervals": fit["intervals"],
        "r_squared": fit["r_squared"],
        "residual_relative": fit["residual_relative"],
        "identified": fit["identified"],
        "fallback_scale": fit["fallback_scale"],
        "usable": fit["usable"],
        "models": fit_models,
        "diagnostics": fit["diagnostics"],
    }
    return {
        "schema": _QD.JSON_SCHEMA,
        "version": 1,
        "command": "bench",
        "harness": "claude",
        "run_id": run_id,
        "fitted_at": time.time(),
        "unit": "percent_per_mtok",
        "window": window,
        "samples": fit["samples"],
        "intervals": fit["intervals"],
        "r_squared": fit["r_squared"],
        "residual_relative": fit["residual_relative"],
        "identified": fit["identified"],
        "fallback_scale": fit["fallback_scale"],
        "degrees_of_freedom": fit["degrees_of_freedom"],
        "usable": fit["usable"],
        "claude": {"unit": "percent_per_mtok", "models": models},
        "fit": {"unit": "percent_per_mtok", "models": fit_models,
                "diagnostics": fit["diagnostics"]},
        "windows": {window: window_fit},
    }


# --------------------------------------------------------------------------
# projection


def list_price_weights(usd_per_percent: float) -> Dict[str, Dict[str, float]]:
    models = {}
    for name, price in _QD.CLAUDE_PRICES.items():
        models[name] = {
            "input": price[0] / usd_per_percent,
            "cache_read": price[1] / usd_per_percent,
            "cache_write_5m": price[2] / usd_per_percent,
            "cache_write_1h": price[3] / usd_per_percent,
            "output": price[4] / usd_per_percent,
            "cache_write_unknown": price[2] / usd_per_percent,
        }
    return models


def planning_weights(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, float]], str]:
    """Percent per million tokens per model, best available source.

    A stored fit carries null for every coefficient it could not identify.
    Those fall back to the list price scaled onto the fitted unit, the way
    quota-drain prices an unidentified Codex model, so the whole table stays on
    one scale.
    """
    fit_file = _QD.state_dir() / "claude-weights.json"
    if fit_file.is_file():
        try:
            payload = json.loads(fit_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            warn("ignoring %s: %s" % (fit_file, error))
        else:
            section = (payload.get("claude") or {}) if isinstance(payload, Mapping) else {}
            models = section.get("models") if isinstance(section, Mapping) else None
            if isinstance(payload, Mapping) and not payload.get("usable"):
                warn("ignoring %s: the stored fit is not usable" % fit_file)
                models = None
            if isinstance(models, Mapping) and models:
                scale = float(payload.get("fallback_scale") or 0.0)
                if scale <= 0:
                    # Without a scale there is no way to price the kinds the fit
                    # left out, and pricing them at zero would make them free.
                    warn("ignoring %s: the stored fit carries no list-price scale"
                         % fit_file)
                    models = None
            if isinstance(models, Mapping) and models:
                scaled = list_price_weights(1.0 / scale)
                table = {}
                for name, entry in models.items():
                    priced = dict(scaled.get(str(name)) or {})
                    for kind, value in entry.items():
                        if isinstance(value, (int, float)):
                            priced[kind] = float(value)
                    if priced:
                        table[str(name)] = priced
                for name, priced in scaled.items():
                    table.setdefault(name, priced)
                if table:
                    return table, "fit at %s" % fit_file
    usd = args.usd_per_percent or _QD.claude_dollars_per_percent() or PLANNING_USD_PER_PERCENT
    return list_price_weights(usd), "list price at %.2f USD per percent" % usd


def projected_percent(scenario: Scenario, weights: Mapping[str, Mapping[str, float]],
                      first_call: bool, cold_as_creation: bool = True) -> float:
    entry = weights.get(scenario.normalized_model) or {}
    tokens = expected_tokens(scenario, first_call, cold_as_creation)
    return sum(value / 1_000_000.0 * float(entry.get(kind, 0.0)) for kind, value in tokens.items())


def project(scenarios: Sequence[Scenario], args: argparse.Namespace
            ) -> Tuple[List[Dict[str, Any]], str]:
    weights, source = planning_weights(args)
    cold_as_creation = not args.no_cold_as_creation
    projections = []
    for scenario in scenarios:
        per_call = projected_percent(scenario, weights, False, cold_as_creation)
        # The first call of a warm or write-heavy scenario pays cache creation
        # at 1.25x rather than a cache read, so it is priced on its own.
        first = projected_percent(scenario, weights, True, cold_as_creation)
        # One percent beyond the requested ticks: the partial percent before the
        # first tick is discarded, so it still has to be paid for.
        wanted = float(args.ticks) + 1.0
        if per_call > 0:
            needed = 1 + int(math.ceil(max(0.0, wanted - first) / per_call))
        else:
            needed = args.max_calls
        needed = max(1, needed)
        calls = min(needed, args.max_calls)
        projections.append(
            {
                "scenario": scenario,
                "percent_per_call": per_call,
                "percent_first_call": first,
                "calls": calls,
                "calls_needed": needed,
                "capped": needed > args.max_calls,
                "percent": first + per_call * (calls - 1),
                "tokens_per_call": expected_tokens(scenario, False, cold_as_creation),
            }
        )
    return projections, source


def render_plan(projections: Sequence[Mapping[str, Any]], source: str,
                args: argparse.Namespace) -> List[str]:
    lines = ["scenario projection (dry run, nothing is spent)",
             "weights: %s" % source,
             ""]
    lines.append("%-40s %7s %7s %12s %10s %9s" % ("scenario", "calls", "needed",
                                                  "tokens/call", "%/call", "% total"))
    total = 0.0
    capped = []
    needed_calls = {}
    for item in projections:
        scenario = item["scenario"]
        tokens = sum(item["tokens_per_call"].values())
        total += item["percent"]
        if item["capped"]:
            capped.append(scenario.key)
            needed_calls[scenario.key] = item["calls_needed"]
        lines.append(
            "%-40s %7d %7d %12s %10.4f %9.3f"
            % (scenario.key, item["calls"], item["calls_needed"], _QD.format_tokens(tokens),
               item["percent_per_call"], item["percent"])
        )
    lines.append("")
    for key in capped:
        lines.append("%s needs %d calls to reach %d tick(s) but --max-calls stops it at "
                     "%d; raise --max-calls, use a larger context or a cache mode that "
                     "spends faster, or lower --ticks"
                     % (key, needed_calls[key], args.ticks, args.max_calls))
    if capped:
        lines.append("")
    lines.append("projected five_hour spend %.2f%% against --max-percent %.2f%%"
                 % (total, args.max_percent))
    # The weekly cap is in percent of the seven-day window, which the same
    # tokens move by a different and unknown amount. It is enforced from
    # measured samples during the run, not projected here.
    lines.append("weekly spend is not projected; --max-percent-weekly %.2f%% is enforced "
                 "from live samples" % args.max_percent_weekly)
    if total > args.max_percent:
        lines.append("OVER BUDGET: raise --max-percent, cut scenarios, or lower --ticks")
    return lines


# --------------------------------------------------------------------------
# report rendering


def render_report(meta: Mapping[str, Any], estimates: Sequence[Mapping[str, Any]],
                  fit: Optional[Mapping[str, Any]]) -> List[str]:
    lines = ["# quota-bench run %s" % meta.get("run_id"),
             "",
             "- window: %s" % meta.get("window"),
             "- started: %s" % _QD.local_label(float(meta.get("started") or 0.0)),
             "- scenarios: %d" % len(estimates),
             ""]
    lines.append("## Tokens per percent")
    lines.append("")
    header = ["scenario", "calls", "%", "input", "cache read", "cache write", "output", "flags"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for item in estimates:
        per = item["tokens_per_percent"]
        flags = []
        if not item["bracketed"]:
            flags.append("unbracketed")
        if item["contaminated"]:
            flags.append("contaminated")
        if item["window_rollover"]:
            flags.append("rollover")
        lines.append(
            "| %s | %d | %.0f | %s | %s | %s | %s | %s |"
            % (
                item["key"],
                item["calls"],
                item["percent"],
                _QD.format_tokens(per.get("input", 0.0)),
                _QD.format_tokens(per.get("cache_read", 0.0)),
                _QD.format_tokens(
                    per.get("cache_write_5m", 0.0)
                    + per.get("cache_write_1h", 0.0)
                    + per.get("cache_write_unknown", 0.0)
                ),
                _QD.format_tokens(per.get("output", 0.0)),
                ", ".join(flags) or "-",
            )
        )
    lines.append("")
    if fit is None:
        lines.append("No fit: no bracketed, uncontaminated scenario.")
        return lines
    lines.append("## Fitted weights (percent per million tokens)")
    lines.append("")
    quality = ("R^2 %.4f" % fit["r_squared"] if fit["r_squared"] is not None
               else "R^2 undefined (every scenario bracketed the same percent)")
    lines.append("%s, relative residual %.4f over %d scenarios, %d of %d columns identified%s"
                 % (quality, fit["residual_relative"], fit["samples"], fit["identified"],
                    fit["columns"], "  (UNDERDETERMINED)" if fit["underdetermined"] else ""))
    if fit["fallback_scale"] > 0:
        lines.append("")
        lines.append("list-price scale: %.4f percent per USD (%.2f USD per percent)"
                     % (fit["fallback_scale"], 1.0 / fit["fallback_scale"]))
    lines.append("")
    lines.append("| model | kind | %/Mtok | ratio to input | list ratio |")
    lines.append("| --- | --- | --- | --- | --- |")
    list_ratio = {"input": 1.0, "cache_read": 0.1, "cache_write_5m": 1.25,
                  "cache_write_1h": 2.0, "output": 5.0,
                  "cache_write_unknown": 1.25}
    for model in sorted(fit["models"]):
        entry = fit["models"][model]
        base = entry.get("input")
        for kind in _QD.CLAUDE_KINDS:
            if kind not in entry:
                continue
            value = entry[kind]
            if value is None:
                reasons = ((fit["diagnostics"].get(model) or {}).get(kind) or {}).get("reasons")
                lines.append("| %s | %s | unidentified (%s) | - | %.3f |"
                             % (model, kind, ", ".join(reasons or ["unidentified"]),
                                list_ratio.get(kind, float("nan"))))
                continue
            ratio = "%.3f" % (value / base) if base else "-"
            lines.append(
                "| %s | %s | %.4f | %s | %.3f |"
                % (model, kind, value, ratio, list_ratio.get(kind, float("nan")))
            )
    lines.append("")
    lines.append("## Residuals")
    lines.append("")
    lines.append("| scenario | measured % | predicted % | residual |")
    lines.append("| --- | --- | --- | --- |")
    for item in fit["residuals"]:
        lines.append("| %s | %.2f | %.2f | %+.3f |"
                     % (item["key"], item["measured_percent"], item["predicted_percent"],
                        item["residual"]))
    return lines


# --------------------------------------------------------------------------
# storage


def bench_root() -> Path:
    return _QD.state_dir() / "bench"


def run_dir(run_id: str) -> Path:
    return bench_root() / run_id


def latest_run_id() -> Optional[str]:
    root = bench_root()
    if not root.is_dir():
        return None
    candidates = sorted(item.name for item in root.iterdir() if (item / "meta.json").is_file())
    return candidates[-1] if candidates else None


def load_run(run_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    directory = run_dir(run_id)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    records = []
    path = directory / "scenarios.jsonl"
    if path.is_file():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    return meta, records


def write_report(run_id: str, meta: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
                 args: argparse.Namespace) -> int:
    required = int(meta.get("ticks") or args.ticks)
    estimates = []
    for record in records:
        estimate = estimate_scenario(record, required)
        if estimate is None:
            warn("%s: fewer than %d ticks; no estimate" % (record.get("key"), required))
            continue
        estimates.append(estimate)
    if not estimates:
        warn("no scenario reached its tick target; nothing to report")
        return EXIT_USAGE
    fit = fit_weights(estimates, include_unbracketed=args.include_unbracketed,
                      include_drifted=args.include_drifted)
    window = str(meta.get("window") or "five_hour")
    payload = {
        "schema": _QD.JSON_SCHEMA,
        "command": "bench-report",
        "run_id": run_id,
        "window": window,
        "scenarios": estimates,
        "fit": fit,
    }
    directory = run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if fit is not None and not args.no_write_weights:
        # Only a usable fit becomes the weights file readers price with. An
        # unusable one is still worth keeping, under a name nothing loads.
        name = "claude-weights.json" if fit["usable"] else "claude-weights.unusable.json"
        destination = _QD.state_dir() / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(weights_payload(fit, window, run_id), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload["weights_file"] = str(destination)
        if not fit["usable"]:
            warn("the fit is not usable (%d of %d columns identified, %d degrees of "
                 "freedom), so it was written to %s and the calibration file "
                 "claude-weights.json was left alone."
                 % (fit["identified"], fit["columns"], fit["degrees_of_freedom"],
                    destination))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return EXIT_OK
    for line in render_report(meta, estimates, fit):
        print(line)
    if "weights_file" in payload:
        print("")
        print("weights written to %s" % payload["weights_file"])
    return EXIT_OK


# --------------------------------------------------------------------------
# subcommands


def command_plan(args: argparse.Namespace) -> int:
    scenarios = build_scenarios(args)
    projections, source = project(scenarios, args)
    if args.json:
        print(json.dumps(
            {
                "schema": _QD.JSON_SCHEMA,
                "command": "plan",
                "weight_source": source,
                "max_percent": args.max_percent,
                "max_percent_weekly": args.max_percent_weekly,
                "ticks": args.ticks,
                "scenarios": [
                    {
                        "key": item["scenario"].key,
                        "model": item["scenario"].model,
                        "context": item["scenario"].context,
                        "cache": item["scenario"].cache,
                        "output": item["scenario"].output,
                        "projected_calls": item["calls"],
                        "projected_percent_per_call": item["percent_per_call"],
                        "projected_percent_first_call": item["percent_first_call"],
                        "projected_percent": item["percent"],
                        "expected_tokens_per_call": item["tokens_per_call"],
                    }
                    for item in projections
                ],
                "projected_percent_total": sum(item["percent"] for item in projections),
            },
            indent=2,
            sort_keys=True,
        ))
        return EXIT_OK
    for line in render_plan(projections, source, args):
        print(line)
    return EXIT_OK


def install_workdir_cleanup(workdir: Path) -> None:
    """Remove the filler scratch directory even when the run is killed."""
    def remove(*_: Any) -> None:
        shutil.rmtree(workdir, ignore_errors=True)

    atexit.register(remove)
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        previous = signal.getsignal(number)
        if previous == signal.SIG_IGN:
            # A nohup'd run ignores SIGHUP on purpose and must survive the
            # terminal closing; atexit still removes the directory.
            continue

        def handler(signum: int, frame: Any, previous=previous) -> None:
            remove()
            if callable(previous):
                previous(signum, frame)
            else:
                raise SystemExit(128 + signum)

        try:
            signal.signal(number, handler)
        except (OSError, ValueError):  # not the main thread, or unsupported
            continue


def command_run(args: argparse.Namespace) -> int:
    scenarios = build_scenarios(args)
    projections, source = project(scenarios, args)
    for line in render_plan(projections, source, args):
        print(line)
    print("")
    if not args.yes:
        warn("every percent above is real quota; re-run with --yes to spend it")
        return EXIT_USAGE
    projected = sum(item["percent"] for item in projections)
    if projected > args.max_percent and not args.force_projection:
        warn("the projection (%.2f%%) already exceeds --max-percent (%.2f%%); the run "
             "would abort part way through. Cut scenarios, lower --ticks, raise the cap, "
             "or pass --force-projection to start anyway"
             % (projected, args.max_percent))
        return EXIT_USAGE
    if shutil.which(args.claude) is None and not Path(args.claude).is_file():
        warn("no claude executable at %r" % args.claude)
        return EXIT_DEPENDENCY

    max_sample_age = args.max_sample_age
    if max_sample_age is None:
        max_sample_age = max(MIN_SAMPLE_AGE, 2.0 * args.sample_interval)
    elif max_sample_age < SAMPLE_AGE_FLOOR:
        # The endpoint refuses more than one poll a minute, so anything under a
        # minute would stop every run on its own sampling floor.
        warn("--max-sample-age %.0fs is below the %.0fs the usage endpoint allows "
             "between polls; using %.0fs"
             % (max_sample_age, SAMPLE_AGE_FLOOR, SAMPLE_AGE_FLOOR))
        max_sample_age = SAMPLE_AGE_FLOOR
    sampler = Sampler(_QD, args.drain_command, args.config_dir, args.sample_interval,
                      max_sample_age)
    baseline = sampler.sample(force=True)
    if baseline is None:
        warn("no utilisation sample (%s); check `nenpi snapshot --oauth`"
             % (sampler.failure or "no observation logged"))
        return EXIT_DEPENDENCY
    # Both caps always apply: --window only selects which window the ticks are
    # read from, and collapsing the two would drop the five-hour cap whenever
    # the run measured against the weekly window.
    budget = Budget({"five_hour": args.max_percent, "seven_day": args.max_percent_weekly})
    budget.observe(baseline)

    own_sessions = []  # type: List[str]
    started = time.time()
    if args.require_idle:
        foreign, checked = foreign_sessions(args.drain_command, args.contamination_since,
                                            started - 600.0, started, own_sessions)
        if foreign:
            warn("--require-idle: %d other Claude session(s) active in the last 10 minutes"
                 % len(foreign))
            return EXIT_GUARD
        if not checked:
            warn("--require-idle: the contamination check did not run, so an idle "
                 "window cannot be confirmed; nothing was spent")
            return EXIT_GUARD

    # Two runs started in the same second must not share a directory.
    run_id = "%s-%04x" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started)),
                          random.getrandbits(16))
    directory = run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "started": started,
        "window": args.window,
        "ticks": args.ticks,
        "max_percent": args.max_percent,
        "max_percent_weekly": args.max_percent_weekly,
        "sample_interval_s": args.sample_interval,
        "max_sample_age_s": max_sample_age,
        "filler_channel": args.filler_channel,
        "claude_flags": call_flags(args),
        # The values may carry anything the operator passed; only the shape of
        # the extra arguments is recorded.
        "claude_extra_args": len(args.claude_arg),
        "claude_extra_args_redacted": bool(args.claude_arg),
        "cold_as_creation": not args.no_cold_as_creation,
        "scenarios": [scenario.key for scenario in scenarios],
        "baseline": baseline.as_json(),
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    workdir = Path(tempfile.mkdtemp(prefix="quota-bench-"))
    install_workdir_cleanup(workdir)
    records = []  # type: List[Dict[str, Any]]
    status = EXIT_OK
    try:
        with open(directory / "calls.jsonl", "a", encoding="utf-8") as calls_log, \
                open(directory / "scenarios.jsonl", "a", encoding="utf-8") as scenario_log:
            for scenario in scenarios:
                print("running %s" % scenario.key)
                scenario_start = time.time()
                before, before_checked = foreign_sessions(
                    args.drain_command, args.contamination_since,
                    scenario_start - 300.0, scenario_start, own_sessions)
                state = run_scenario(scenario, args, sampler, budget, workdir,
                                     calls_log, own_sessions)
                after, after_checked = foreign_sessions(
                    args.drain_command, args.contamination_since,
                    scenario_start, state.ended, own_sessions)
                # The pre-check runs before this scenario's own session ids are
                # known, so its findings are re-filtered once they are.
                state.contaminated_by = sorted(
                    (set(before) | set(after)) - set(own_sessions)
                )
                if not (before_checked and after_checked):
                    # A check that did not run cannot clear the window.
                    state.contamination_reasons.append("check_failed")
                if state.contaminated_by:
                    state.contamination_reasons.append("foreign_session")
                foreign = state.contaminated_by or state.contamination_reasons
                record = state.as_json()
                records.append(record)
                scenario_log.write(json.dumps(record, sort_keys=True) + "\n")
                scenario_log.flush()
                print("  %d calls, %d tick(s), stop: %s%s"
                      % (state.calls, len(state.ticks), state.stop_reason,
                         "  CONTAMINATED" if foreign else ""))
                if state.stop_reason.startswith("budget:"):
                    warn("budget reached on %s; stopping the run"
                         % state.stop_reason.split(":", 1)[1])
                    status = EXIT_BUDGET
                    break
                if state.stop_reason.startswith("sampler:"):
                    warn("utilisation is no longer observable; stopping the run")
                    status = EXIT_GUARD
                    break
                if state.stop_reason == "max-calls":
                    status = EXIT_MAX_CALLS
                    break
                if state.stop_reason == "call-failed":
                    status = EXIT_DEPENDENCY
                    break
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    meta["ended"] = time.time()
    meta["budget"] = budget.as_json()
    (directory / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("")
    print("spent %s" % ", ".join(
        "%s %.2f%%" % (window, budget.spent.get(window, 0.0)) for window in sorted(budget.limits)
    ))
    print("run logs in %s" % directory)
    print("")
    reported = write_report(run_id, meta, records, args)
    return status if status != EXIT_OK else reported


def command_report(args: argparse.Namespace) -> int:
    run_id = args.run_id or latest_run_id()
    if run_id is None:
        warn("no stored runs under %s" % bench_root())
        return EXIT_USAGE
    try:
        meta, records = load_run(run_id)
    except (OSError, ValueError) as error:
        warn("cannot read run %s: %s" % (run_id, error))
        return EXIT_USAGE
    if not records:
        warn("run %s has no scenario records" % run_id)
        return EXIT_USAGE
    return write_report(run_id, meta, records, args)


# --------------------------------------------------------------------------
# CLI


def add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models", default="claude-haiku-4-5-20251001", metavar="LIST",
        help="comma-separated Claude model IDs (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--contexts", default="10k", metavar="LIST",
        help="comma-separated target filler sizes in tokens: 10k, 60k, or 150k "
             "(default: 10k)",
    )
    parser.add_argument(
        "--cache", default="warm", metavar="LIST",
        help="comma-separated modes: cold, warm (reuse a prefix), or write-heavy "
             "(wait for cache expiry); default: warm",
    )
    parser.add_argument(
        "--outputs", default="short", metavar="LIST",
        help="comma-separated output modes: short (one word) or long (2000-word task) "
             "(default: short)",
    )
    parser.add_argument(
        "--ticks", type=int, default=DEFAULT_TICKS, metavar="N",
        help="utilisation percentage ticks to bracket per scenario; 2 gives one full "
             "bracket (default: 2)",
    )
    parser.add_argument(
        "--max-percent", type=float, default=DEFAULT_MAX_PERCENT, metavar="PERCENT",
        help="five-hour quota cap for the run and projection, checked before every call "
             "(default: 3)",
    )
    parser.add_argument(
        "--max-percent-weekly", type=float,
        default=DEFAULT_MAX_PERCENT_WEEKLY, metavar="PERCENT",
        help="seven-day quota cap for the run; plan reports it but does not project it "
             "(default: 1)",
    )
    parser.add_argument(
        "--max-calls", type=int, default=DEFAULT_MAX_CALLS, metavar="N",
        help="maximum Claude calls per scenario (default: 400)",
    )
    parser.add_argument(
        "--usd-per-percent", type=float, default=None, metavar="USD",
        help="dollars-per-five-hour-percent assumption for planning only "
             "(default: stored/live value or $1.40 prior)",
    )
    parser.add_argument("--no-cold-as-creation", action="store_true",
                        help="project cold calls as uncached input rather than cache "
                             "creation (the default is the conservative 1.25x)")
    parser.add_argument(
        "--json", action="store_true",
        help="write plan/report data as indented JSON; run still prints its preflight plan",
    )


def add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--include-unbracketed", action="store_true",
                        help="fit rows with fewer than two ticks; their percent is an "
                             "upper bound, not a measurement")
    parser.add_argument("--include-drifted", action="store_true",
                        help="include warm scenarios whose calls fell outside the cache TTL")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quota-drain", type=Path, default=None,
                        metavar="PATH",
                        help="quota-drain used for sampling and contamination checks "
                             "(default: the bundled nenpi.drain module)")
    parser.add_argument(
        "--window", choices=("five_hour", "seven_day"), default="five_hour",
        help="window used to count ticks and project spend (default: five_hour); "
             "both budget caps still apply",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nenpi-bench",
        description=(
            "Measure how many Claude tokens consume one percentage point of plan utilisation. "
            "plan is a dry run; run spends real quota; report fits stored run logs."
        ),
        epilog=(
            "Examples:\n"
            "  nenpi-bench plan --contexts 10k,60k --cache cold,warm\n"
            "  nenpi-bench run --models claude-haiku-4-5 --ticks 2 --yes\n"
            "  nenpi-bench report --run-id 20260917T120000Z-abcd --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    plan = sub.add_parser(
        "plan", help="list scenarios and project quota; spends nothing",
        description=(
            "Expand the scenario cross-product and estimate calls and quota before a run. "
            "No Claude process or snapshot is started."
        ),
        epilog=(
            "Examples:\n"
            "  nenpi-bench plan --models claude-haiku-4-5 --contexts 10k,60k\n"
            "  nenpi-bench plan --cache cold,warm --outputs short,long --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_scenario_arguments(plan)
    add_common_arguments(plan)
    plan.set_defaults(handler=command_plan)

    run = sub.add_parser(
        "run", help="spend quota to measure tokens per percent",
        description=(
            "Run each scenario with claude -p, sample live utilisation, and bracket the "
            "tokens between percentage ticks. Every percentage spent is real quota."
        ),
        epilog=(
            "Examples:\n"
            "  nenpi-bench run --models claude-haiku-4-5 --contexts 10k --yes\n"
            "  nenpi-bench run --require-idle --sample-interval 90 --yes"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_scenario_arguments(run)
    add_common_arguments(run)
    run.add_argument(
        "--yes", action="store_true",
        help="confirm that the projected calls may spend real quota; required to start",
    )
    run.add_argument("--require-idle", action="store_true",
                     help="refuse to start unless no other Claude session overlaps each "
                          "contamination check")
    run.add_argument(
        "--claude", default="claude", metavar="PATH",
        help="Claude executable to invoke (default: claude)",
    )
    run.add_argument("--allow-tools", action="store_true",
                     help="leave built-in tools enabled (default: disabled; tools add turns "
                          "and tokens)")
    run.add_argument("--force-projection", action="store_true",
                     help="start even when the projected five-hour spend exceeds "
                          "--max-percent")
    run.add_argument("--claude-arg", action="append", default=[], metavar="ARG",
                     help="extra argument for every claude call; repeatable")
    run.add_argument("--config-dir", action="append", default=[], metavar="PATH",
                     help="Claude config directory passed to snapshot --oauth; repeatable")
    run.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL,
                     metavar="SECONDS",
                     help="minimum seconds between utilisation polls (default: 60; endpoint "
                          "also enforces 60)")
    run.add_argument("--max-sample-age", type=float, default=None, metavar="SECONDS",
                     help="stop before the next call when the newest sample is older "
                          "(default: twice --sample-interval, at least 90s)")
    run.add_argument("--call-timeout", type=float, default=DEFAULT_CALL_TIMEOUT,
                     metavar="SECONDS",
                     help="abort an individual claude call after this many seconds "
                          "(default: 900)")
    run.add_argument("--write-heavy-gap", type=float, default=WRITE_HEAVY_GAP_SECONDS,
                     metavar="SECONDS",
                     help="seconds between write-heavy calls so the 5-minute cache expires "
                          "(default: 360)")
    run.add_argument("--filler-channel", choices=("prompt", "system-prompt-file"),
                     default="prompt",
                     help="send filler on stdin or via --system-prompt-file (default: prompt)")
    run.add_argument("--contamination-since", default=DEFAULT_CONTAMINATION_WINDOW,
                     metavar="DURATION",
                     help="look back this far for other Claude sessions during contamination "
                          "checks (default: 6h)")
    run.add_argument(
        "--no-write-weights", action="store_true",
        help="do not save a usable fitted Claude weight table after the run",
    )
    add_fit_arguments(run)
    run.set_defaults(handler=command_run)

    report = sub.add_parser(
        "report", help="fit and report from stored run logs",
        description=(
            "Read the latest stored run, or the run named by --run-id, and fit weights "
            "from eligible scenario records. This command spends no quota."
        ),
        epilog=(
            "Examples:\n"
            "  nenpi-bench report\n"
            "  nenpi-bench report --run-id 20260917T120000Z-abcd --include-unbracketed --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    report.add_argument(
        "--run-id", default=None, metavar="ID",
        help="stored run ID (default: most recent run under the state directory)",
    )
    report.add_argument(
        "--ticks", type=int, default=DEFAULT_TICKS, metavar="N",
        help="minimum ticks required for a bracketed estimate (default: 2; normally read "
             "from the run metadata)",
    )
    report.add_argument(
        "--no-write-weights", action="store_true",
        help="do not save a usable fitted Claude weight table",
    )
    add_fit_arguments(report)
    report.add_argument(
        "--json", action="store_true",
        help="write the report payload as indented JSON",
    )
    add_common_arguments(report)
    report.set_defaults(handler=command_report)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _QD
    migrate_dirs()
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if not getattr(args, "handler", None):
        parser.print_help()
        return EXIT_USAGE
    if args.quota_drain is None:
        _QD = importlib.import_module("nenpi.drain")
        args.drain_command = [sys.executable, "-m", "nenpi.drain"]
    else:
        path = Path(args.quota_drain).expanduser()
        _QD = load_quota_drain(path)
        args.drain_command = [sys.executable, str(path)]
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
