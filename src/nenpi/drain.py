#!/usr/bin/env python3
"""Report which Claude Code and Codex CLI sessions drained subscription quota.

Codex quota is measured from the `rate_limits` snapshots the CLI writes into
its own rollouts. Claude writes no quota data at all, so Claude sessions are
modelled as API list-price dollars and reported as a share of the observed
total; see docs/quota-drain.md for what is official, community-sourced, and
unknown.

Test path overrides use the ``QUOTA_DRAIN_*`` environment variables:
``HOME_DIR``, ``CACHE_DIR``, ``STATE_DIR``, and ``CONFIG_DIR``.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PREFIX = "QUOTA_DRAIN_"
CACHE_SCHEMA = 2
JSON_SCHEMA = 1
LONG_CONTEXT_THRESHOLD = 200_000
CONTEXT_REDUCTION_FRACTION = 0.30
MIN_REDUCTION_CONTEXT = 20_000
REDUCTION_PERSISTENCE_TURNS = 3
BOUNDARY_DEDUP_SECONDS = 5.0
MAX_CONFIG_JSON_BYTES = 128 * 1024 * 1024
DEDUP_CARRY_IDS = 64
PROGRESS_INTERVAL_SECONDS = 0.5

CLAUDE_KINDS = ("input", "cache_read", "cache_write_5m", "cache_write_1h", "output")
CODEX_KINDS = ("input", "cached_input", "cache_write", "output")
CODEX_FIT_KINDS = ("input", "cached_input", "output")

UNWEIGHTED = "unweighted"

# Anthropic list prices, USD per million tokens (claude-api skill, 2026-09-16).
# Cache reads are 0.1x base input everywhere except Claude Fable 5.1 (0.025x);
# cache writes are 1.25x for the 5-minute TTL and 2x for the 1-hour TTL.
CLAUDE_PRICES = {
    "claude-opus-5": (5.0, 0.5, 6.25, 10.0, 25.0),
    "claude-opus-4-8": (5.0, 0.5, 6.25, 10.0, 25.0),
    "claude-fable-5-1": (10.0, 0.25, 12.5, 20.0, 50.0),
    "claude-fable-5": (10.0, 1.0, 12.5, 20.0, 50.0),
    "claude-sonnet-5": (2.0, 0.2, 2.5, 4.0, 10.0),
    "claude-haiku-4-5": (1.0, 0.1, 1.25, 2.0, 5.0),
}

# OpenAI credit rate card (learn.chatgpt.com/docs/pricing), relative units per
# million tokens. Models released before the 5.6 line have no published rate;
# they are guessed at the sol rate and flagged.
CODEX_CREDITS = {
    "gpt-6-astra": (250.0, 25.0, 1250.0, False),
    "gpt-5.6-sol": (100.0, 10.0, 500.0, False),
    "gpt-5.6-terra": (50.0, 5.0, 300.0, False),
    "gpt-5.6-luna": (5.0, 0.5, 30.0, False),
    "gpt-5.5": (100.0, 10.0, 500.0, True),
    "gpt-5.4": (100.0, 10.0, 500.0, True),
    "gpt-5.4-mini": (100.0, 10.0, 500.0, True),
    "gpt-5.3-codex": (100.0, 10.0, 500.0, True),
    "gpt-5.3-codex-spark": (100.0, 10.0, 500.0, True),
}

CODEX_MODEL_ALIASES = {
    "luna": "gpt-5.6-luna",
    "gpt-luna": "gpt-5.6-luna",
    "sol": "gpt-5.6-sol",
    "terra": "gpt-5.6-terra",
    "astra": "gpt-6-astra",
}

CLAUDE_DATE_SUFFIX = re.compile(r"-20\d{6}$")
DURATION_ARG = re.compile(r"^(\d+(?:\.\d+)?)([hdwm])$")
CODEX_DAY_DIR = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/[^/]+$")

CODEX_LINE_MARKERS = (
    b'"session_meta"',
    b'"turn_context"',
    b'"token_usage_record"',
    b'"token_count"',
    b'"task_started"',
    b'"compacted"',
)

# A Claude user line that carries a tool result is a fan-out step, not a new
# prompt, and its payload is the bulk of a transcript's bytes. Screening those
# out before json.loads keeps the scan cheap.
CLAUDE_USER_MARKERS = (b'"type":"user"', b'"type": "user"')
CLAUDE_NOT_A_PROMPT = (b'"toolUseResult"', b'"tool_result"')

ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "claude": "\033[36m",
    "codex": "\033[35m",
    "warn": "\033[33m",
}

BAR_GLYPHS = "▏▎▍▌▋▊▉█"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(PREFIX + name, str(default))).expanduser()


def home_dir() -> Path:
    return env_path("HOME_DIR", Path.home())


def cache_dir() -> Path:
    return env_path("CACHE_DIR", home_dir() / ".cache" / "quota-drain")


def state_dir() -> Path:
    return env_path("STATE_DIR", home_dir() / ".local" / "state" / "quota-drain")


def config_dir() -> Path:
    return env_path("CONFIG_DIR", home_dir() / ".config" / "quota-drain")


# --------------------------------------------------------------------------
# time helpers


_TS_CACHE: Dict[str, float] = {}


def parse_timestamp(value: Any) -> Optional[float]:
    """Parse an ISO-8601 transcript timestamp into a UTC epoch float."""
    if not isinstance(value, str) or not value:
        return None
    cached = _TS_CACHE.get(value)
    if cached is not None:
        return cached
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    epoch = parsed.timestamp()
    if len(_TS_CACHE) < 200_000:
        _TS_CACHE[value] = epoch
    return epoch


def parse_since(value: Optional[str], now: float) -> Optional[float]:
    if not value:
        return None
    match = DURATION_ARG.match(value.strip())
    if match:
        amount = float(match.group(1))
        unit = match.group(2)
        seconds = {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
        return now - amount * seconds
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise SystemExit("quota-drain: cannot parse time %r (use 7d, 12h, or 2026-09-10)" % value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.timestamp()


def local_label(epoch: float, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.fromtimestamp(epoch).astimezone().strftime(fmt)


def local_zone_name() -> str:
    return datetime.now().astimezone().strftime("%Z") or "local"


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    if seconds < 60:
        return "%ds" % int(seconds)
    if seconds < 3600:
        return "%dm" % int(seconds // 60)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return "%dh%02dm" % (hours, minutes)


def format_tokens(count: float) -> str:
    value = float(count)
    if value >= 1_000_000_000:
        return "%.1fG" % (value / 1_000_000_000)
    if value >= 1_000_000:
        return "%.1fM" % (value / 1_000_000)
    if value >= 1_000:
        return "%.1fK" % (value / 1_000)
    return "%d" % int(value)


# --------------------------------------------------------------------------
# weights


def empty_tokens(kinds: Sequence[str]) -> Dict[str, int]:
    return dict((kind, 0) for kind in kinds)


def copy_token_map(payload: Any) -> Dict[str, Dict[str, int]]:
    """Deep-copy a model->tokens map so scan totals never alias cache entries."""
    if not isinstance(payload, Mapping):
        return {}
    copied = {}
    for model, tokens in payload.items():
        if isinstance(tokens, Mapping):
            copied[str(model)] = dict((str(k), int(v)) for k, v in tokens.items())
    return copied


def normalize_claude_model(model: Any) -> str:
    if not isinstance(model, str) or not model.strip():
        return UNWEIGHTED
    text = model.strip()
    if text.startswith("<") or text.endswith(">"):
        return UNWEIGHTED
    return CLAUDE_DATE_SUFFIX.sub("", text)


def normalize_codex_model(model: Any) -> str:
    if not isinstance(model, str) or not model.strip():
        return UNWEIGHTED
    text = model.strip().lower()
    return CODEX_MODEL_ALIASES.get(text, text)


def builtin_weights() -> Dict[str, Any]:
    claude_models = {}
    for name, price in CLAUDE_PRICES.items():
        claude_models[name] = {
            "input": price[0],
            "cache_read": price[1],
            "cache_write_5m": price[2],
            "cache_write_1h": price[3],
            "output": price[4],
        }
    codex_models = {}
    for name, credit in CODEX_CREDITS.items():
        entry = {"input": credit[0], "cached_input": credit[1], "output": credit[2]}
        if credit[3]:
            entry["guessed"] = True
        codex_models[name] = entry
    return {
        "version": 1,
        "claude": {"unit": "usd_per_mtok", "models": claude_models},
        "codex": {"unit": "credit_units_per_mtok", "models": codex_models},
    }


def merge_weights(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for harness in ("claude", "codex"):
        section = override.get(harness)
        if not isinstance(section, Mapping):
            continue
        target = base.setdefault(harness, {"models": {}})
        if isinstance(section.get("unit"), str):
            target["unit"] = section["unit"]
        models = section.get("models")
        if isinstance(models, Mapping):
            for name, entry in models.items():
                if isinstance(entry, Mapping):
                    target.setdefault("models", {})[str(name)] = dict(entry)
    return base


class Weights:
    """Per-model price tables plus the provenance shown in the report header."""

    def __init__(self, table: Dict[str, Any], sources: Sequence[str]):
        self.table = table
        self.sources = list(sources)

    @property
    def source_label(self) -> str:
        return " + ".join(self.sources) if self.sources else "default"

    def model_entry(self, harness: str, model: str) -> Optional[Mapping[str, Any]]:
        section = self.table.get(harness) or {}
        models = section.get("models") or {}
        entry = models.get(model)
        return entry if isinstance(entry, Mapping) else None

    def guessed_models(self, harness: str, used: Iterable[str]) -> List[str]:
        found = []
        for model in used:
            entry = self.model_entry(harness, model)
            if entry and entry.get("guessed"):
                found.append(model)
        return sorted(set(found))

    def claude_units(
        self, model: str, tokens: Mapping[str, int], cache_read_weight: Optional[float]
    ) -> float:
        entry = self.model_entry("claude", model)
        if entry is None:
            return 0.0
        cache_read_price = entry.get("cache_read", 0.0)
        if cache_read_weight is not None:
            cache_read_price = float(entry.get("input", 0.0)) * cache_read_weight
        total = 0.0
        for kind in CLAUDE_KINDS:
            price = cache_read_price if kind == "cache_read" else float(entry.get(kind, 0.0))
            total += tokens.get(kind, 0) / 1_000_000.0 * price
        return total

    def codex_units(self, model: str, tokens: Mapping[str, int]) -> float:
        entry = self.model_entry("codex", model)
        if entry is None:
            return 0.0
        total = 0.0
        for kind in CODEX_FIT_KINDS:
            total += tokens.get(kind, 0) / 1_000_000.0 * float(entry.get(kind, 0.0))
        return total


def load_weights(use_calibrated: bool) -> Weights:
    table = builtin_weights()
    sources = ["default"]
    config_file = config_dir() / "weights.json"
    if config_file.is_file():
        try:
            override = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            warn("ignoring %s: %s" % (config_file, error))
        else:
            if isinstance(override, Mapping):
                merge_weights(table, override)
                sources.append("config")
    if use_calibrated:
        fit_file = state_dir() / "codex-weights.json"
        if fit_file.is_file():
            try:
                fit = json.loads(fit_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                warn("ignoring %s: %s" % (fit_file, error))
            else:
                if isinstance(fit, Mapping):
                    merge_weights(table, {"codex": fit.get("codex", fit)})
                    sources.append("calibrated")
        else:
            warn("--use-calibrated: no fit at %s; run `quota-drain calibrate`" % fit_file)
    return Weights(table, sources)


def warn(message: str) -> None:
    sys.stderr.write("quota-drain: %s\n" % message)


# --------------------------------------------------------------------------
# transcript discovery


def discover_roots(home: Path, kind: str) -> List[Path]:
    leaf = "projects" if kind == "claude" else "sessions"
    roots = []
    try:
        candidates = sorted(home.glob(".%s*" % kind))
    except OSError:
        candidates = []
    for candidate in candidates:
        root = candidate / leaf
        if root.is_dir():
            roots.append(root)
    return roots


def claude_transcripts(roots: Sequence[Path]) -> Iterator[Path]:
    for root in roots:
        for path in sorted(root.rglob("*.jsonl")):
            if path.is_file():
                yield path


def codex_day_epoch(path: Path) -> Optional[float]:
    match = CODEX_DAY_DIR.search(str(path))
    if not match:
        return None
    try:
        day = datetime(
            int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return day.timestamp()


def codex_transcripts(roots: Sequence[Path], since: Optional[float]) -> Iterator[Path]:
    for root in roots:
        for path in sorted(root.rglob("rollout-*.jsonl")):
            if not path.is_file():
                continue
            if since is not None:
                day = codex_day_epoch(path)
                # A rollout directory is named for the day it opened; allow one
                # day of slack so a session that spans midnight is not pruned.
                if day is not None and day < since - 86400:
                    continue
            yield path


def read_lines_from(path: Path, offset: int) -> Iterator[Tuple[int, bytes]]:
    """Yield (offset-after-line, raw line) for every complete line from offset."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        position = offset
        for raw in handle:
            if not raw.endswith(b"\n"):
                # A transcript being appended to right now; stop before the
                # partial line so the next run re-reads it whole.
                break
            position += len(raw)
            yield position, raw


# --------------------------------------------------------------------------
# parsed records


class SessionSummary:
    """Per-session token totals, kept small enough to cache as JSON."""

    def __init__(self, harness: str, session_id: str):
        self.harness = harness
        self.session_id = session_id
        self.cwd = ""
        self.version = ""
        self.start = None  # type: Optional[float]
        self.end = None  # type: Optional[float]
        self.models = {}  # type: Dict[str, Dict[str, int]]
        self.models_lc = {}  # type: Dict[str, Dict[str, int]]
        self.sub_models = {}  # type: Dict[str, Dict[str, int]]
        self.requests = 0
        self.sub_requests = 0
        self.cost_state = None  # type: Optional[Dict[str, Any]]
        self.thread_usage = None  # type: Optional[Dict[str, int]]
        self.originator = ""

    @property
    def kinds(self) -> Sequence[str]:
        return CLAUDE_KINDS if self.harness == "claude" else CODEX_KINDS

    def touch(self, epoch: Optional[float]) -> None:
        if epoch is None:
            return
        if self.start is None or epoch < self.start:
            self.start = epoch
        if self.end is None or epoch > self.end:
            self.end = epoch

    def add(
        self,
        model: str,
        tokens: Mapping[str, int],
        *,
        sidechain: bool = False,
        long_context: bool = False,
    ) -> None:
        bucket = self.sub_models if sidechain else self.models
        target = bucket.setdefault(model, empty_tokens(self.kinds))
        for kind, value in tokens.items():
            target[kind] = target.get(kind, 0) + int(value)
        if not sidechain:
            self.requests += 1
        else:
            self.sub_requests += 1
        # Subagent usage is also part of the parent session's totals.
        if sidechain:
            rollup = self.models.setdefault(model, empty_tokens(self.kinds))
            for kind, value in tokens.items():
                rollup[kind] = rollup.get(kind, 0) + int(value)
        if long_context:
            lc = self.models_lc.setdefault(model, empty_tokens(self.kinds))
            for kind, value in tokens.items():
                lc[kind] = lc.get(kind, 0) + int(value)

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "harness": self.harness,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "version": self.version,
            "start": self.start,
            "end": self.end,
            "models": self.models,
            "models_lc": self.models_lc,
            "sub_models": self.sub_models,
            "requests": self.requests,
            "sub_requests": self.sub_requests,
        }
        if self.cost_state is not None:
            payload["cost_state"] = self.cost_state
        if self.thread_usage is not None:
            payload["thread_usage"] = self.thread_usage
        if self.originator:
            payload["originator"] = self.originator
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "SessionSummary":
        summary = cls(str(payload.get("harness", "claude")), str(payload.get("session_id", "")))
        summary.cwd = str(payload.get("cwd", ""))
        summary.version = str(payload.get("version", ""))
        summary.start = payload.get("start")
        summary.end = payload.get("end")
        summary.models = copy_token_map(payload.get("models"))
        summary.models_lc = copy_token_map(payload.get("models_lc"))
        summary.sub_models = copy_token_map(payload.get("sub_models"))
        summary.requests = int(payload.get("requests") or 0)
        summary.sub_requests = int(payload.get("sub_requests") or 0)
        summary.cost_state = payload.get("cost_state")
        summary.thread_usage = payload.get("thread_usage")
        summary.originator = str(payload.get("originator", ""))
        return summary

    def merge(self, other: "SessionSummary") -> None:
        self.cwd = self.cwd or other.cwd
        self.version = other.version or self.version
        self.originator = self.originator or other.originator
        self.touch(other.start)
        self.touch(other.end)
        for attribute in ("models", "models_lc", "sub_models"):
            source = getattr(other, attribute)
            target = getattr(self, attribute)
            for model, tokens in source.items():
                slot = target.setdefault(model, empty_tokens(self.kinds))
                for kind, value in tokens.items():
                    slot[kind] = slot.get(kind, 0) + int(value)
        self.requests += other.requests
        self.sub_requests += other.sub_requests
        if other.cost_state is not None:
            self.cost_state = other.cost_state
        if other.thread_usage is not None:
            self.thread_usage = other.thread_usage


class FileIndex:
    """Cached parse state for one transcript file."""

    def __init__(self, harness: str):
        self.harness = harness
        self.size = 0
        self.mtime = 0.0
        self.offset = 0
        self.sessions = {}  # type: Dict[str, SessionSummary]
        self.events = []  # type: List[List[Any]]
        self.snapshots = []  # type: List[Dict[str, Any]]
        self.carry_ids = []  # type: List[str]
        self.has_usage_records = False
        self.cumulative = None  # type: Optional[Dict[str, int]]
        self.last_session = ""
        self.last_model = UNWEIGHTED
        self.boundaries = []  # type: List[List[Any]]
        self.compactions = []  # type: List[List[Any]]
        self.thread_id = ""
        self.is_subagent = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "harness": self.harness,
            "size": self.size,
            "mtime": self.mtime,
            "offset": self.offset,
            "sessions": dict((key, value.to_json()) for key, value in self.sessions.items()),
            "events": self.events,
            "snapshots": self.snapshots,
            "carry_ids": self.carry_ids,
            "has_usage_records": self.has_usage_records,
            "cumulative": self.cumulative,
            "last_session": self.last_session,
            "last_model": self.last_model,
            "boundaries": self.boundaries,
            "compactions": self.compactions,
            "thread_id": self.thread_id,
            "is_subagent": self.is_subagent,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "FileIndex":
        index = cls(str(payload.get("harness", "claude")))
        index.size = int(payload.get("size") or 0)
        index.mtime = float(payload.get("mtime") or 0.0)
        index.offset = int(payload.get("offset") or 0)
        for key, value in (payload.get("sessions") or {}).items():
            index.sessions[str(key)] = SessionSummary.from_json(value)
        index.events = list(payload.get("events") or [])
        index.snapshots = list(payload.get("snapshots") or [])
        index.carry_ids = list(payload.get("carry_ids") or [])
        index.has_usage_records = bool(payload.get("has_usage_records"))
        index.cumulative = payload.get("cumulative")
        index.last_session = str(payload.get("last_session", ""))
        index.last_model = str(payload.get("last_model", UNWEIGHTED))
        index.boundaries = list(payload.get("boundaries") or [])
        index.compactions = list(payload.get("compactions") or [])
        index.thread_id = str(payload.get("thread_id", ""))
        index.is_subagent = bool(payload.get("is_subagent"))
        return index


# Event layout, one row per deduped API call:
# [session_id, model, timestamp, k0, k1, k2, k3, long_context, subagent, turn_id]
# assemble_prompts appends the owning prompt index as EVENT_PROMPT.
EVENT_SESSION, EVENT_MODEL, EVENT_TS = 0, 1, 2
EVENT_KINDS = 3
EVENT_LONG = 7
EVENT_SUB = 8
EVENT_TURN = 9
EVENT_THREAD = 10
EVENT_PROMPT = 11
EVENT_STORED_WIDTH = 11


def event_tokens(event: Sequence[Any], kinds: Sequence[str]) -> Dict[str, int]:
    return dict((kind, int(event[EVENT_KINDS + offset])) for offset, kind in enumerate(kinds))


def event_context(event: Sequence[Any], harness: str) -> int:
    """Tokens the harness re-sent to the API for this one call."""
    if harness == "claude":
        # input + cache_read + both cache-write TTLs
        return int(event[3]) + int(event[4]) + int(event[5]) + int(event[6])
    # Codex splits input into uncached and cached; together they are the context.
    return int(event[3]) + int(event[4])


# --------------------------------------------------------------------------
# Claude parsing


def is_claude_prompt(record: Mapping[str, Any], is_subagent_file: bool) -> bool:
    """True for a line that represents a person typing, not a fan-out step."""
    if record.get("isSidechain") or is_subagent_file or record.get("isMeta"):
        return False
    if "toolUseResult" in record:
        return False
    message = record.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            return False
    return any(
        isinstance(block, dict) and block.get("type") == "text" for block in content
    )


def parse_claude_file(path: Path, index: FileIndex) -> FileIndex:
    is_subagent_file = "/subagents/" in str(path)
    seen_ids = set(index.carry_ids)
    recent = list(index.carry_ids)
    offset = index.offset
    for position, raw in read_lines_from(path, index.offset):
        offset = position
        wants_usage = b'"usage"' in raw or b'"cost-state"' in raw
        wants_prompt = (
            any(marker in raw for marker in CLAUDE_USER_MARKERS)
            and not any(marker in raw for marker in CLAUDE_NOT_A_PROMPT)
        )
        if not wants_usage and not wants_prompt:
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        kind = record.get("type")
        session_id = record.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            continue
        summary = index.sessions.get(session_id)
        if summary is None:
            summary = SessionSummary("claude", session_id)
            index.sessions[session_id] = summary
        if kind == "cost-state":
            index.sessions[session_id].cost_state = {
                "totalCostUSD": record.get("totalCostUSD"),
                "modelUsage": record.get("modelUsage") or {},
                "totalDuration": record.get("totalDuration"),
                "totalAPIDuration": record.get("totalAPIDuration"),
            }
            continue
        if kind == "user":
            if is_claude_prompt(record, is_subagent_file):
                epoch = parse_timestamp(record.get("timestamp"))
                if epoch is not None:
                    index.boundaries.append([session_id, epoch])
                    summary.touch(epoch)
            continue
        if kind != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        message_id = message.get("id") or record.get("requestId")
        if isinstance(message_id, str) and message_id:
            if message_id in seen_ids:
                continue
            seen_ids.add(message_id)
            recent.append(message_id)
            if len(recent) > DEDUP_CARRY_IDS * 8:
                recent = recent[-DEDUP_CARRY_IDS:]
        model = normalize_claude_model(message.get("model"))
        epoch = parse_timestamp(record.get("timestamp"))
        cwd = record.get("cwd")
        if isinstance(cwd, str) and cwd and not summary.cwd:
            summary.cwd = cwd
        version = record.get("version")
        if isinstance(version, str) and version:
            summary.version = version
        summary.touch(epoch)

        creation = usage.get("cache_creation")
        write_5m = 0
        write_1h = 0
        if isinstance(creation, dict):
            write_5m = int(creation.get("ephemeral_5m_input_tokens") or 0)
            write_1h = int(creation.get("ephemeral_1h_input_tokens") or 0)
        total_write = int(usage.get("cache_creation_input_tokens") or 0)
        if write_5m + write_1h == 0:
            write_5m = total_write
        tokens = {
            "input": int(usage.get("input_tokens") or 0),
            "cache_read": int(usage.get("cache_read_input_tokens") or 0),
            "cache_write_5m": write_5m,
            "cache_write_1h": write_1h,
            "output": int(usage.get("output_tokens") or 0),
        }
        sidechain = bool(record.get("isSidechain")) or is_subagent_file
        context_size = tokens["input"] + tokens["cache_read"] + write_5m + write_1h
        long_context = context_size > LONG_CONTEXT_THRESHOLD
        summary.add(model, tokens, sidechain=sidechain, long_context=long_context)
        if epoch is not None:
            index.events.append(
                [
                    session_id,
                    model,
                    epoch,
                    tokens["input"],
                    tokens["cache_read"],
                    write_5m,
                    write_1h,
                    1 if long_context else 0,
                    1 if sidechain else 0,
                    "",
                    "",
                ]
            )
    index.offset = offset
    index.carry_ids = recent[-DEDUP_CARRY_IDS:]
    return index


# --------------------------------------------------------------------------
# Codex parsing


def codex_usage_tokens(usage: Mapping[str, Any]) -> Dict[str, int]:
    """Split a Codex TokenUsage into disjoint buckets.

    `cached_input_tokens` is a subset of `input_tokens` and
    `reasoning_output_tokens` a subset of `output_tokens`
    (codex-rs/protocol/src/protocol.rs `TokenUsage::non_cached_input` and
    `blended_total`), so uncached input is the difference and output is used
    whole. `cache_write_input_tokens` is reported for display only; it is part
    of the input count and OpenAI publishes no separate cache-write rate.
    """
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    if cached < 0:
        cached = 0
    if cached > input_tokens:
        cached = input_tokens
    return {
        "input": input_tokens - cached,
        "cached_input": cached,
        "cache_write": int(usage.get("cache_write_input_tokens") or 0),
        "output": int(usage.get("output_tokens") or 0),
    }


def diff_cumulative(current: Mapping[str, Any], previous: Optional[Mapping[str, int]]) -> Dict[str, int]:
    tokens = codex_usage_tokens(current)
    if previous is None:
        return tokens
    delta = {}
    for kind in CODEX_KINDS:
        value = tokens.get(kind, 0) - int(previous.get(kind, 0))
        delta[kind] = value if value > 0 else 0
    return delta


def parse_codex_file(path: Path, index: FileIndex) -> FileIndex:
    fallback_events = []  # type: List[List[Any]]
    session_id = index.last_session
    model = index.last_model
    cumulative = index.cumulative
    offset = index.offset
    for position, raw in read_lines_from(path, index.offset):
        offset = position
        if not any(marker in raw for marker in CODEX_LINE_MARKERS):
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        kind = record.get("type")
        epoch = parse_timestamp(record.get("timestamp"))

        if kind == "compacted":
            # Fields sit directly on the record. Only the timestamp and the
            # window id are read; the history fields alongside them
            # (`message`, `replacement_history`, `guardian_history`,
            # `retained_context`) hold prompt text and are never touched.
            if epoch is not None and session_id:
                index.compactions.append(
                    [session_id, epoch, str(record.get("window_id") or "")]
                )
            continue

        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue

        if kind == "session_meta":
            # A Codex session spans several rollouts: the root thread plus one
            # file per spawned subagent, all sharing `session_id`.
            thread = payload.get("id")
            if isinstance(thread, str) and thread:
                index.thread_id = thread
            umbrella = payload.get("session_id")
            if isinstance(umbrella, str) and umbrella:
                session_id = umbrella
            elif isinstance(thread, str) and thread:
                session_id = thread
            if not session_id:
                session_id = path.stem
            index.is_subagent = is_codex_subagent(payload)
            summary = index.sessions.setdefault(session_id, SessionSummary("codex", session_id))
            cwd = payload.get("cwd")
            if isinstance(cwd, str) and cwd:
                summary.cwd = cwd
            originator = payload.get("originator")
            if isinstance(originator, str):
                summary.originator = originator
            version = payload.get("cli_version")
            if isinstance(version, str):
                summary.version = version
            summary.touch(epoch or parse_timestamp(payload.get("timestamp")))
            continue

        if not session_id:
            session_id = path.stem
        summary = index.sessions.setdefault(session_id, SessionSummary("codex", session_id))

        if kind == "turn_context":
            if epoch is not None and not index.is_subagent:
                add_boundary(index.boundaries, session_id, epoch)
            candidate = payload.get("model")
            collaboration = payload.get("collaboration_mode")
            if isinstance(collaboration, dict):
                settings = collaboration.get("settings")
                if isinstance(settings, dict) and settings.get("model"):
                    candidate = settings.get("model")
            normalized = normalize_codex_model(candidate)
            if normalized != UNWEIGHTED:
                model = normalized
            summary.touch(epoch)
            continue

        if kind == "token_usage_record":
            usage = payload.get("usage")
            if not isinstance(usage, dict):
                continue
            index.has_usage_records = True
            tokens = codex_usage_tokens(usage)
            thread_usage = payload.get("thread_token_usage")
            if isinstance(thread_usage, dict):
                summary.thread_usage = codex_usage_tokens(thread_usage)
            turn_id = payload.get("turn_id")
            record_event(
                index.events,
                summary,
                session_id,
                model,
                epoch,
                tokens,
                CODEX_KINDS,
                turn=turn_id if isinstance(turn_id, str) else "",
                thread=index.thread_id or path.stem,
                sidechain=index.is_subagent,
            )
            continue

        if kind == "event_msg" and payload.get("type") == "task_started":
            if epoch is not None and not index.is_subagent:
                add_boundary(index.boundaries, session_id, epoch)
            continue

        if kind == "event_msg" and payload.get("type") == "token_count":
            info = payload.get("info")
            if isinstance(info, dict):
                total = info.get("total_token_usage")
                if isinstance(total, dict):
                    delta = diff_cumulative(total, cumulative)
                    cumulative = codex_usage_tokens(total)
                    if any(delta.values()):
                        fallback_events.append([session_id, model, epoch, delta])
            limits = payload.get("rate_limits")
            if isinstance(limits, dict) and epoch is not None:
                index.snapshots.extend(snapshot_rows(limits, epoch))
            summary.touch(epoch)
            continue
    index.offset = offset
    index.cumulative = cumulative
    index.last_session = session_id
    index.last_model = model
    if not index.has_usage_records and fallback_events:
        for session_key, event_model, event_epoch, delta in fallback_events:
            summary = index.sessions.setdefault(
                session_key, SessionSummary("codex", session_key)
            )
            record_event(
                index.events,
                summary,
                session_key,
                event_model,
                event_epoch,
                delta,
                CODEX_KINDS,
                thread=index.thread_id or path.stem,
                sidechain=index.is_subagent,
            )
    return index


def is_codex_subagent(payload: Mapping[str, Any]) -> bool:
    """True when a rollout is a spawned subagent thread rather than the root."""
    if payload.get("parent_thread_id"):
        return True
    if payload.get("thread_source") == "subagent":
        return True
    source = payload.get("source")
    if isinstance(source, str):
        return source == "subagent"
    if isinstance(source, Mapping):
        return "subagent" in source
    return False


def add_boundary(boundaries: List[List[Any]], session_id: str, epoch: float) -> None:
    """Record a prompt start, collapsing the task_started/turn_context pair.

    Codex writes both within a second or two of each other for the same user
    turn, so a naive append would double every prompt.
    """
    if boundaries:
        last_session, last_epoch = boundaries[-1][0], boundaries[-1][1]
        if last_session == session_id and abs(epoch - last_epoch) <= BOUNDARY_DEDUP_SECONDS:
            return
    boundaries.append([session_id, epoch])


def record_event(
    events: List[List[Any]],
    summary: SessionSummary,
    session_id: str,
    model: str,
    epoch: Optional[float],
    tokens: Mapping[str, int],
    kinds: Sequence[str],
    turn: str = "",
    thread: str = "",
    sidechain: bool = False,
) -> None:
    context_size = tokens.get("input", 0) + tokens.get("cached_input", 0)
    long_context = context_size > LONG_CONTEXT_THRESHOLD
    summary.touch(epoch)
    summary.add(model, tokens, sidechain=sidechain, long_context=long_context)
    if epoch is None:
        return
    row = [session_id, model, epoch]
    row.extend(int(tokens.get(kind, 0)) for kind in kinds)
    row.append(1 if long_context else 0)
    row.append(1 if sidechain else 0)
    row.append(turn)
    row.append(thread)
    events.append(row)


def snapshot_rows(limits: Mapping[str, Any], epoch: float) -> List[Dict[str, Any]]:
    rows = []
    limit_id = limits.get("limit_id")
    plan_type = limits.get("plan_type")
    for slot in ("primary", "secondary"):
        window = limits.get(slot)
        if not isinstance(window, dict):
            continue
        used = window.get("used_percent")
        if not isinstance(used, (int, float)):
            continue
        rows.append(
            {
                "ts": epoch,
                "limit_id": limit_id if isinstance(limit_id, str) else "codex",
                "plan_type": plan_type if isinstance(plan_type, str) else "unknown",
                "window_minutes": window.get("window_minutes"),
                "used_percent": float(used),
                "resets_at": window.get("resets_at"),
                "slot": slot,
            }
        )
    return rows


# --------------------------------------------------------------------------
# cache


class Cache:
    """One JSON shard per transcript, loaded only for files a run actually reads.

    A single index would have to be parsed whole on every invocation; with ~10
    GB of rollouts that index reaches nine figures of JSON and dominates the
    runtime of a narrow `--since` query.
    """

    def __init__(self, root: Path, rebuild: bool):
        self.root = root
        self.rebuild = rebuild
        self.entries = {}  # type: Dict[str, FileIndex]
        self.dirty = set()  # type: set

    def shard_path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.root / ("v%d" % CACHE_SCHEMA) / digest[:2] / (digest + ".json")

    def stored(self, key: str, harness: str) -> Optional[FileIndex]:
        if self.rebuild:
            return None
        cached = self.entries.get(key)
        if cached is not None:
            return cached
        try:
            payload = json.loads(self.shard_path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict) or payload.get("path") != key:
            return None
        try:
            entry = FileIndex.from_json(payload)
        except (TypeError, ValueError):
            return None
        if entry.harness != harness:
            return None
        self.entries[key] = entry
        return entry

    def entry_for(self, path: Path, harness: str, stat: os.stat_result) -> Tuple[FileIndex, bool]:
        key = str(path)
        entry = self.stored(key, harness)
        if entry is None:
            entry = FileIndex(harness)
            self.entries[key] = entry
            return entry, True
        if stat.st_size < entry.size or stat.st_mtime < entry.mtime - 1:
            # Truncated or rewritten behind our back; the stored offset is
            # meaningless, so start the file over.
            entry = FileIndex(harness)
            self.entries[key] = entry
            return entry, True
        if stat.st_size == entry.size and abs(stat.st_mtime - entry.mtime) <= 1:
            return entry, False
        return entry, True

    def mark(self, path: Path) -> None:
        self.dirty.add(str(path))

    def forget(self, path: Path) -> None:
        self.entries.pop(str(path), None)

    def flush(self) -> None:
        for key in sorted(self.dirty):
            entry = self.entries.get(key)
            if entry is None:
                continue
            payload = entry.to_json()
            payload["path"] = key
            destination = self.shard_path(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            os.replace(str(temporary), str(destination))
        self.dirty.clear()


class Progress:
    def __init__(self, enabled: bool, total: int, label: str):
        self.enabled = enabled and total > 0
        self.total = total
        self.label = label
        self.done = 0
        self.last = 0.0

    def step(self) -> None:
        self.done += 1
        if not self.enabled:
            return
        now = time.time()
        if now - self.last < PROGRESS_INTERVAL_SECONDS and self.done < self.total:
            return
        self.last = now
        sys.stderr.write(
            "\rquota-drain: %s %d/%d" % (self.label, self.done, self.total)
        )
        sys.stderr.flush()

    def finish(self) -> None:
        if self.enabled:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()


# --------------------------------------------------------------------------
# scan


class Scan:
    def __init__(self):
        self.sessions = {}  # type: Dict[Tuple[str, str], SessionSummary]
        self.events = {"claude": [], "codex": []}  # type: Dict[str, List[List[Any]]]
        self.snapshots = []  # type: List[Dict[str, Any]]
        self.boundaries = {}  # type: Dict[Tuple[str, str], List[float]]
        self.compactions = {}  # type: Dict[Tuple[str, str], List[float]]
        self.files_read = 0
        self.files_seen = 0
        self.bytes_read = 0


def collect(args: argparse.Namespace, since: Optional[float]) -> Scan:
    home = home_dir()
    harness = args.harness
    cache = Cache(cache_dir(), args.rebuild_cache)

    targets = []  # type: List[Tuple[Path, str]]
    if harness in ("claude", "all"):
        roots = [Path(p).expanduser() for p in (args.claude_root or [])]
        roots.extend(discover_roots(home, "claude"))
        for path in claude_transcripts(roots):
            targets.append((path, "claude"))
    if harness in ("codex", "all"):
        roots = [Path(p).expanduser() for p in (args.codex_root or [])]
        roots.extend(discover_roots(home, "codex"))
        for path in codex_transcripts(roots, since):
            targets.append((path, "codex"))

    scan = Scan()
    scan.files_seen = len(targets)
    progress = Progress(sys.stderr.isatty(), len(targets), "scanning")
    for path, kind in targets:
        progress.step()
        try:
            stat = path.stat()
        except OSError:
            continue
        # A transcript last written before the window cannot hold events
        # inside it, so its shard is never opened.
        if since is not None and stat.st_mtime < since:
            continue
        entry, stale = cache.entry_for(path, kind, stat)
        if stale:
            try:
                if kind == "claude":
                    parse_claude_file(path, entry)
                else:
                    parse_codex_file(path, entry)
            except OSError as error:
                warn("skipping %s: %s" % (path.name, error))
                continue
            scan.bytes_read += max(0, stat.st_size - entry.size)
            entry.size = stat.st_size
            entry.mtime = stat.st_mtime
            cache.mark(path)
            scan.files_read += 1
        absorb(scan, entry, kind)
        try:
            cache.flush()
        except OSError as error:
            warn("cache not written: %s" % error)
        cache.forget(path)
    progress.finish()
    return scan


def absorb(scan: Scan, entry: FileIndex, harness: str) -> None:
    for session_id, summary in entry.sessions.items():
        key = (harness, session_id)
        existing = scan.sessions.get(key)
        if existing is None:
            scan.sessions[key] = SessionSummary.from_json(summary.to_json())
        else:
            existing.merge(summary)
    scan.events[harness].extend(entry.events)
    scan.snapshots.extend(entry.snapshots)
    for row in entry.boundaries:
        scan.boundaries.setdefault((harness, str(row[0])), []).append(float(row[1]))
    for row in entry.compactions:
        scan.compactions.setdefault((harness, str(row[0])), []).append(float(row[1]))


# --------------------------------------------------------------------------
# Codex measured attribution


class Interval:
    def __init__(self, key: Tuple[str, str, Any], start: float, end: float, drain: float,
                 resets_at: Any, rollover: bool):
        self.key = key
        self.start = start
        self.end = end
        self.drain = drain
        self.resets_at = resets_at
        self.rollover = rollover
        self.sessions = {}  # type: Dict[str, float]
        self.prompts = {}  # type: Dict[Tuple[str, Any], float]
        self.features = {}  # type: Dict[Tuple[str, str], float]


def resets_bucket(value: Any) -> Any:
    """Collapse the +/- 1s jitter Codex writes into `resets_at`.

    Consecutive readings of the same window differ by a second or two, so a
    raw equality test reads every line as a fresh window rollover.
    """
    if isinstance(value, (int, float)):
        return int(round(float(value) / 60.0))
    return value


def snapshot_windows(
    snapshots: Sequence[Mapping[str, Any]]
) -> Dict[Tuple[str, str, Any], List[Mapping[str, Any]]]:
    grouped = {}  # type: Dict[Tuple[str, str, Any], List[Mapping[str, Any]]]
    for row in snapshots:
        key = (row.get("limit_id") or "codex", row.get("plan_type") or "unknown",
               row.get("window_minutes"))
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["ts"])
    return grouped


def choose_window(snapshots: Sequence[Mapping[str, Any]], wanted: Optional[str]) -> Optional[str]:
    """Pick one window size so drains are never summed across two denominators."""
    if wanted and wanted != "auto":
        return wanted
    counts = {}  # type: Dict[int, int]
    for row in snapshots:
        minutes = row.get("window_minutes")
        if isinstance(minutes, (int, float)):
            counts[int(minutes)] = counts.get(int(minutes), 0) + 1
    if not counts:
        return None
    best = max(counts, key=lambda key: counts[key])
    return {300: "five_hour", 10080: "weekly"}.get(best)


def window_matches(window_minutes: Any, wanted: str) -> bool:
    if not isinstance(window_minutes, (int, float)):
        return False
    if wanted == "five_hour":
        return int(window_minutes) == 300
    if wanted == "weekly":
        return int(window_minutes) == 10080
    return True


def build_intervals(
    snapshots: Sequence[Mapping[str, Any]], window_filter: Optional[str]
) -> List[Interval]:
    intervals = []
    for key, rows in snapshot_windows(snapshots).items():
        if window_filter and not window_matches(key[2], window_filter):
            continue
        current = None  # type: Any
        running = 0.0
        anchor_ts = None  # type: Optional[float]
        for row in rows:
            bucket = resets_bucket(row.get("resets_at"))
            used = row["used_percent"]
            if (
                current is not None
                and isinstance(bucket, int)
                and isinstance(current, int)
                and bucket < current
            ):
                # A reading from a window that has already rolled over; two
                # sessions polling concurrently can interleave them.
                continue
            if current is None:
                current, running, anchor_ts = bucket, used, row["ts"]
                continue
            if bucket != current:
                drain = min(100.0, max(0.0, used))
                if drain > 0 and anchor_ts is not None and row["ts"] > anchor_ts:
                    intervals.append(
                        Interval(key, anchor_ts, row["ts"], drain, row.get("resets_at"), True)
                    )
                current, running, anchor_ts = bucket, used, row["ts"]
                continue
            delta = used - running
            if delta <= 0:
                continue
            # `used_percent` is reported in whole percent, so a step covers
            # everything since the last change, not just the last poll.
            if anchor_ts is not None and row["ts"] > anchor_ts:
                intervals.append(
                    Interval(key, anchor_ts, row["ts"], delta, row.get("resets_at"), False)
                )
            running, anchor_ts = used, row["ts"]
    intervals.sort(key=lambda item: item.start)
    return intervals


def attribute(
    intervals: Sequence[Interval], events: Sequence[Sequence[Any]], weights: Weights
) -> None:
    """Split each interval's measured drain across the sessions active in it."""
    ordered = sorted((event for event in events if event[EVENT_TS] is not None),
                     key=lambda event: event[EVENT_TS])
    if not ordered:
        return
    stamps = [event[EVENT_TS] for event in ordered]
    for interval in intervals:
        low = bisect.bisect_right(stamps, interval.start)
        high = bisect.bisect_right(stamps, interval.end)
        total = 0.0
        shares = {}  # type: Dict[str, float]
        prompt_shares = {}  # type: Dict[Tuple[str, Any], float]
        for event in ordered[low:high]:
            tokens = event_tokens(event, CODEX_KINDS)
            units = weights.codex_units(event[EVENT_MODEL], tokens)
            if units <= 0:
                continue
            total += units
            shares[event[EVENT_SESSION]] = shares.get(event[EVENT_SESSION], 0.0) + units
            if len(event) > EVENT_PROMPT and event[EVENT_PROMPT] is not None:
                prompt_key = (event[EVENT_SESSION], event[EVENT_PROMPT])
                prompt_shares[prompt_key] = prompt_shares.get(prompt_key, 0.0) + units
            for kind in CODEX_FIT_KINDS:
                feature = (event[EVENT_MODEL], kind)
                interval.features[feature] = (
                    interval.features.get(feature, 0.0) + tokens.get(kind, 0) / 1_000_000.0
                )
        if total <= 0:
            continue
        for session_id, units in shares.items():
            interval.sessions[session_id] = interval.drain * units / total
        for prompt_key, units in prompt_shares.items():
            interval.prompts[prompt_key] = interval.drain * units / total


# --------------------------------------------------------------------------
# prompts, fan-out, and context reductions


class Prompt:
    """One user turn and the whole fan-out of API calls it triggered."""

    def __init__(self, harness: str, session_id: str, index: int):
        self.harness = harness
        self.session_id = session_id
        self.index = index
        self.start = None  # type: Optional[float]
        self.end = None  # type: Optional[float]
        self.turns = 0
        self.sub_turns = 0
        self.context_start = 0
        self.context_peak = 0
        self.tokens = empty_tokens(CLAUDE_KINDS if harness == "claude" else CODEX_KINDS)
        self.units = 0.0
        self.resent_units = 0.0
        self.input_tokens = 0
        self.drain_percent = None  # type: Optional[float]
        self.model = UNWEIGHTED
        self.reduction = ""

    @property
    def kinds(self) -> Sequence[str]:
        return CLAUDE_KINDS if self.harness == "claude" else CODEX_KINDS

    @property
    def wall_seconds(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)

    def to_json(self) -> Dict[str, Any]:
        return {
            "harness": self.harness,
            "session_id": self.session_id,
            "short_id": short_id(self.session_id),
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "wall_seconds": self.wall_seconds,
            "api_turns": self.turns,
            "subagent_turns": self.sub_turns,
            "context_start": self.context_start,
            "context_peak": self.context_peak,
            "tokens": self.tokens,
            "input_side_tokens": self.input_tokens,
            "weighted_units": self.units,
            "resent_units": self.resent_units,
            "drain_percent": self.drain_percent,
            "model": self.model,
            "reduction": self.reduction,
        }


def input_side_units(harness: str, model: str, tokens: Mapping[str, int], weights: Weights,
                     cache_read_weight: Optional[float]) -> float:
    """Weighted cost of context sent to the API, excluding generated output."""
    if harness == "claude":
        trimmed = dict((kind, tokens.get(kind, 0)) for kind in CLAUDE_KINDS if kind != "output")
        return weights.claude_units(model, trimmed, cache_read_weight)
    trimmed = dict((kind, tokens.get(kind, 0)) for kind in CODEX_FIT_KINDS if kind != "output")
    return weights.codex_units(model, trimmed)


def assemble_prompts(
    scan: Scan, weights: Weights, args: argparse.Namespace
) -> Dict[Tuple[str, str], List[Prompt]]:
    """Group API calls under the user prompt that triggered them.

    Codex numbers its own turns, so `turn_id` is authoritative for the root
    thread. Claude has no turn id, so calls fall under the most recent
    user-prompt line. Subagent calls belong to whichever prompt was running
    when they started, in both harnesses.
    """
    assembled = {}  # type: Dict[Tuple[str, str], List[Prompt]]
    for harness, events in scan.events.items():
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        for session_id, rows in group_events_by_session(events).items():
            bounds = sorted(scan.boundaries.get((harness, session_id), []))
            groups = group_rows_by_prompt(rows, bounds, harness)
            prompts = []
            for position, (start_ts, group) in enumerate(groups, start=1):
                prompts.append(
                    build_prompt(harness, session_id, position, start_ts, group, kinds,
                                 weights, args)
                )
            if prompts:
                assembled[(harness, session_id)] = prompts
    return assembled


def group_rows_by_prompt(
    rows: Sequence[List[Any]], bounds: Sequence[float], harness: str
) -> List[Tuple[float, List[List[Any]]]]:
    main = [row for row in rows if not row[EVENT_SUB]]
    spawned = [row for row in rows if row[EVENT_SUB]]
    if not main:
        main, spawned = list(rows), []
    groups = assign_prompt_keys(main, bounds, harness)
    starts = [start for start, _ in groups]
    for row in spawned:
        position = bisect.bisect_right(starts, row[EVENT_TS]) - 1
        groups[max(0, position)][1].append(row)
    for _, group in groups:
        group.sort(key=lambda row: row[EVENT_TS])
    return groups


def build_prompt(
    harness: str,
    session_id: str,
    position: int,
    start_ts: float,
    group: Sequence[List[Any]],
    kinds: Sequence[str],
    weights: Weights,
    args: argparse.Namespace,
) -> Prompt:
    prompt = Prompt(harness, session_id, position)
    prompt.start = start_ts
    baseline = 0.0
    seen_main = False
    for row in group:
        if len(row) <= EVENT_PROMPT:
            row.extend([None] * (EVENT_PROMPT + 1 - len(row)))
        row[EVENT_PROMPT] = position
        tokens = event_tokens(row, kinds)
        context = event_context(row, harness)
        prompt.end = row[EVENT_TS]
        prompt.turns += 1
        if row[EVENT_SUB]:
            prompt.sub_turns += 1
        else:
            prompt.context_peak = max(prompt.context_peak, context)
            if not seen_main:
                prompt.context_start = context
                prompt.model = row[EVENT_MODEL]
                baseline = input_side_units(
                    harness, row[EVENT_MODEL], tokens, weights,
                    args.claude_cache_read_weight
                )
                seen_main = True
        for kind in kinds:
            prompt.tokens[kind] += tokens[kind]
        prompt.input_tokens += context
        if harness == "claude":
            prompt.units += weights.claude_units(
                row[EVENT_MODEL], tokens, args.claude_cache_read_weight
            )
        else:
            prompt.units += weights.codex_units(row[EVENT_MODEL], tokens)
        prompt.resent_units += input_side_units(
            harness, row[EVENT_MODEL], tokens, weights, args.claude_cache_read_weight
        )
    prompt.resent_units = max(0.0, prompt.resent_units - baseline)
    if prompt.start is None:
        prompt.start = group[0][EVENT_TS] if group else None
    return prompt


def assign_prompt_keys(
    rows: Sequence[List[Any]], bounds: Sequence[float], harness: str
) -> List[Tuple[float, List[List[Any]]]]:
    """Return ordered (prompt start, rows) groups for one session's main thread."""
    groups = []  # type: List[Tuple[Any, List[List[Any]]]]
    order = {}  # type: Dict[Any, int]
    use_turns = harness == "codex" and all(row[EVENT_TURN] for row in rows)
    for row in rows:
        if use_turns:
            key = row[EVENT_TURN]
        else:
            position = bisect.bisect_right(bounds, row[EVENT_TS]) - 1
            key = bounds[position] if position >= 0 else None
        slot = order.get(key)
        if slot is None:
            order[key] = len(groups)
            groups.append((key, [row]))
        else:
            groups[slot][1].append(row)
    anchored = []
    for key, group in groups:
        if use_turns or key is None:
            # A turn id carries no time; anchor to the task_started line that
            # preceded its first call so wall time covers the user's wait.
            first_ts = group[0][EVENT_TS]
            position = bisect.bisect_right(bounds, first_ts) - 1
            anchored.append((bounds[position] if position >= 0 else first_ts, group))
        else:
            anchored.append((key, group))
    anchored.sort(key=lambda item: item[0])
    return anchored


class Reduction:
    def __init__(self, harness: str, session_id: str, epoch: float, kind: str,
                 before: int, after: int):
        self.harness = harness
        self.session_id = session_id
        self.epoch = epoch
        self.kind = kind
        self.before = before
        self.after = after
        self.turns_after = 0
        self.saved_units = 0.0
        self.saved_units_upper = 0.0
        self.model = UNWEIGHTED

    @property
    def removed(self) -> int:
        return max(0, self.before - self.after)

    def to_json(self) -> Dict[str, Any]:
        return {
            "harness": self.harness,
            "short_id": short_id(self.session_id),
            "time": self.epoch,
            "kind": self.kind,
            "before": self.before,
            "after": self.after,
            "removed_tokens": self.removed,
            "turns_after": self.turns_after,
            "model": self.model,
            "saved_units": self.saved_units,
            "saved_units_upper_bound": self.saved_units_upper,
        }


def detect_reductions(
    scan: Scan, weights: Weights, args: argparse.Namespace
) -> List[Reduction]:
    """Find points where one thread's context shrank sharply.

    Codex writes a `compacted` record for its own compaction. The ported
    /shake writes no marker at all, so any large unexplained drop is reported
    as `unmarked`. Comparisons stay inside one thread: a Codex session runs
    several concurrent subagent threads whose small contexts would otherwise
    read as drops against the root thread's.
    """
    found = []
    for harness, events in scan.events.items():
        for (session_id, thread), rows in group_events_by_thread(events).items():
            if len(rows) < REDUCTION_PERSISTENCE_TURNS + 2:
                continue
            marks = sorted(scan.compactions.get((harness, session_id), []))
            for position in range(1, len(rows) - REDUCTION_PERSISTENCE_TURNS):
                previous = rows[position - 1]
                current = rows[position]
                before = event_context(previous, harness)
                after = event_context(current, harness)
                if before < MIN_REDUCTION_CONTEXT:
                    continue
                ceiling = before * (1.0 - CONTEXT_REDUCTION_FRACTION)
                if after >= ceiling:
                    continue
                # A genuine reduction sticks. Codex interleaves a second,
                # smaller-context call stream into the same thread, so a drop
                # that bounces straight back is that stream, not a reduction.
                window = rows[position + 1 : position + 1 + REDUCTION_PERSISTENCE_TURNS]
                if any(event_context(row, harness) >= ceiling for row in window):
                    continue
                marked = any(
                    previous[EVENT_TS] < mark <= current[EVENT_TS] for mark in marks
                )
                reduction = Reduction(
                    harness,
                    session_id,
                    current[EVENT_TS],
                    "compact" if marked else "unmarked",
                    before,
                    after,
                )
                reduction.model = current[EVENT_MODEL]
                reduction.turns_after = len(rows) - position - 1
                estimate_savings(reduction, weights, args)
                found.append(reduction)
    found.sort(key=lambda item: item.saved_units, reverse=True)
    return found


def group_events_by_thread(
    events: Sequence[List[Any]]
) -> Dict[Tuple[str, str], List[List[Any]]]:
    grouped = {}  # type: Dict[Tuple[str, str], List[List[Any]]]
    for row in events:
        thread = row[EVENT_THREAD] if len(row) > EVENT_THREAD and row[EVENT_THREAD] else ""
        if not thread:
            # Claude has no thread id; subagent transcripts are the only
            # parallel context and they are already flagged.
            thread = "sub" if row[EVENT_SUB] else "main"
        grouped.setdefault((row[EVENT_SESSION], thread), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row[EVENT_TS])
    return grouped


def estimate_savings(
    reduction: Reduction, weights: Weights, args: argparse.Namespace
) -> None:
    """Price the context every later turn no longer had to re-send.

    Removed context would mostly have been served from cache, so the cache-read
    rate is the estimate and the uncached rate is the upper bound.
    """
    per_turn = reduction.removed / 1_000_000.0
    entry = weights.model_entry(reduction.harness, reduction.model)
    if entry is None:
        return
    if reduction.harness == "claude":
        cached_rate = float(entry.get("cache_read", 0.0))
        if args.claude_cache_read_weight is not None:
            cached_rate = float(entry.get("input", 0.0)) * args.claude_cache_read_weight
        uncached_rate = float(entry.get("input", 0.0))
    else:
        cached_rate = float(entry.get("cached_input", 0.0))
        uncached_rate = float(entry.get("input", 0.0))
    reduction.saved_units = per_turn * reduction.turns_after * cached_rate
    reduction.saved_units_upper = per_turn * reduction.turns_after * uncached_rate


def group_events_by_session(events: Sequence[Sequence[Any]]) -> Dict[str, List[Sequence[Any]]]:
    grouped = {}  # type: Dict[str, List[Sequence[Any]]]
    for row in events:
        grouped.setdefault(row[EVENT_SESSION], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row[EVENT_TS])
    return grouped


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[max(0, min(len(ordered) - 1, position))]


def solve_normal(matrix: Sequence[Sequence[float]], target: Sequence[float]) -> List[float]:
    """Ordinary least squares by Gaussian elimination on the normal equations."""
    rows = len(matrix)
    if rows == 0 or not matrix[0]:
        return []
    columns = len(matrix[0])
    gram = [[0.0] * (columns + 1) for _ in range(columns)]
    for row in range(rows):
        line = matrix[row]
        for i in range(columns):
            for j in range(columns):
                gram[i][j] += line[i] * line[j]
            gram[i][columns] += line[i] * target[row]
    for i in range(columns):
        pivot = max(range(i, columns), key=lambda r: abs(gram[r][i]))
        if abs(gram[pivot][i]) < 1e-12:
            return [0.0] * columns
        gram[i], gram[pivot] = gram[pivot], gram[i]
        divisor = gram[i][i]
        for j in range(i, columns + 1):
            gram[i][j] /= divisor
        for r in range(columns):
            if r == i or gram[r][i] == 0.0:
                continue
            factor = gram[r][i]
            for j in range(i, columns + 1):
                gram[r][j] -= factor * gram[i][j]
    return [gram[i][columns] for i in range(columns)]


def fit_growth(prompts: Sequence[Prompt]) -> Dict[str, Any]:
    """Fit per-prompt cost against prompt index, linearly and quadratically."""
    points = [(float(prompt.index), prompt.units) for prompt in prompts]
    if len(points) < 3:
        return {"samples": len(points), "better": "insufficient"}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    linear_design = [[x, 1.0] for x in xs]
    quadratic_design = [[x * x, x, 1.0] for x in xs]
    linear = solve_normal(linear_design, ys)
    quadratic = solve_normal(quadratic_design, ys)
    tail_start = int(len(prompts) * 0.8)
    tail = sum(prompt.units for prompt in prompts[tail_start:])
    total = sum(ys)
    return {
        "samples": len(points),
        "linear": {"slope": linear[0], "intercept": linear[1],
                   "r_squared": r_squared(linear_design, ys, linear)},
        "quadratic": {
            "square_term": quadratic[0],
            "linear_term": quadratic[1],
            "intercept": quadratic[2],
            "r_squared": r_squared(quadratic_design, ys, quadratic),
        },
        "better": (
            "quadratic"
            if r_squared(quadratic_design, ys, quadratic)
            > r_squared(linear_design, ys, linear) + 0.01
            else "linear"
        ),
        "last_20_percent_share": (tail / total) if total > 0 else 0.0,
    }


# --------------------------------------------------------------------------
# non-negative least squares


def nnls(matrix: Sequence[Sequence[float]], target: Sequence[float], iterations: int = 20000
         ) -> List[float]:
    """Projected-gradient NNLS over the normal equations; stdlib only.

    Working on Gram matrices keeps each iteration O(p^2) instead of O(n*p), so
    thousands of snapshot intervals stay cheap.
    """
    rows = len(matrix)
    if rows == 0 or not matrix[0]:
        return []
    columns = len(matrix[0])
    scale = []
    for column in range(columns):
        peak = max(abs(matrix[row][column]) for row in range(rows))
        scale.append(peak if peak > 0 else 1.0)
    scaled = [[matrix[row][column] / scale[column] for column in range(columns)]
              for row in range(rows)]
    gram = [[0.0] * columns for _ in range(columns)]
    projection = [0.0] * columns
    for row in range(rows):
        line = scaled[row]
        for i in range(columns):
            value = line[i]
            if value == 0.0:
                continue
            projection[i] += value * target[row]
            gram_row = gram[i]
            for j in range(columns):
                gram_row[j] += value * line[j]
    lipschitz = sum(gram[i][i] for i in range(columns))
    if lipschitz <= 0:
        return [0.0] * columns
    step = 1.0 / lipschitz
    weights = [0.0] * columns
    for _ in range(iterations):
        moved = 0.0
        for i in range(columns):
            gradient = -projection[i]
            gram_row = gram[i]
            for j in range(columns):
                if weights[j]:
                    gradient += gram_row[j] * weights[j]
            updated = weights[i] - step * gradient
            if updated < 0.0:
                updated = 0.0
            moved = max(moved, abs(updated - weights[i]))
            weights[i] = updated
        if moved < 1e-14:
            break
    return [weights[column] / scale[column] for column in range(columns)]


def r_squared(matrix: Sequence[Sequence[float]], target: Sequence[float],
              coefficients: Sequence[float]) -> float:
    if not target:
        return 0.0
    mean = sum(target) / len(target)
    residual = 0.0
    total = 0.0
    for row in range(len(target)):
        predicted = sum(matrix[row][column] * coefficients[column]
                        for column in range(len(coefficients)))
        residual += (target[row] - predicted) ** 2
        total += (target[row] - mean) ** 2
    if total <= 0:
        return 0.0
    return 1.0 - residual / total


# --------------------------------------------------------------------------
# session rows


class Row:
    def __init__(self, summary: SessionSummary):
        self.summary = summary
        self.tokens = {}  # type: Dict[str, int]
        self.sub_tokens = {}  # type: Dict[str, int]
        self.units = 0.0
        self.sub_units = 0.0
        self.unweighted_tokens = 0
        self.drain_percent = None  # type: Optional[float]
        self.share_of_window = None  # type: Optional[float]
        self.resets_at = None  # type: Optional[Any]
        self.primary_model = UNWEIGHTED
        self.estimated = False
        self.relative = 0.0
        self.prompt_count = 0
        self.api_turns = 0
        self.turns_p90 = 0.0
        self.peak_context = 0
        self.resent_units = 0.0


def session_rows(
    scan: Scan,
    weights: Weights,
    args: argparse.Namespace,
    since: Optional[float],
    until: Optional[float],
) -> List[Row]:
    rows = []
    for (harness, session_id), summary in scan.sessions.items():
        if summary.end is None:
            continue
        if since is not None and summary.end < since:
            continue
        if until is not None and summary.start is not None and summary.start > until:
            continue
        row = Row(summary)
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        row.tokens = empty_tokens(kinds)
        row.sub_tokens = empty_tokens(kinds)
        best = 0.0
        for model, tokens in summary.models.items():
            for kind in kinds:
                row.tokens[kind] += int(tokens.get(kind, 0))
            long_tokens = summary.models_lc.get(model) or {}
            if harness == "claude":
                units = weights.claude_units(model, tokens, args.claude_cache_read_weight)
                long_units = weights.claude_units(
                    model, long_tokens, args.claude_cache_read_weight
                )
            else:
                units = weights.codex_units(model, tokens)
                long_units = weights.codex_units(model, long_tokens)
            units += long_units * (args.long_context_multiplier - 1.0)
            if weights.model_entry(harness, model) is None:
                row.unweighted_tokens += sum(int(tokens.get(kind, 0)) for kind in kinds)
            row.units += units
            if units > best:
                best = units
                row.primary_model = model
        if row.primary_model == UNWEIGHTED and summary.models:
            row.primary_model = sorted(summary.models)[0]
        for model, tokens in summary.sub_models.items():
            for kind in kinds:
                row.sub_tokens[kind] += int(tokens.get(kind, 0))
            if harness == "claude":
                row.sub_units += weights.claude_units(
                    model, tokens, args.claude_cache_read_weight
                )
            else:
                row.sub_units += weights.codex_units(model, tokens)
        rows.append(row)
    return rows


def attach_fanout(rows: Sequence[Row], analysis: "Analysis") -> None:
    """Summarise each session's prompt fan-out onto its report row."""
    for row in rows:
        prompts = [
            prompt
            for prompt in analysis.prompts_for(row.summary.harness, row.summary.session_id)
            if analysis.in_range(prompt.start) or analysis.in_range(prompt.end)
        ]
        if not prompts:
            continue
        row.prompt_count = len(prompts)
        row.api_turns = sum(prompt.turns for prompt in prompts)
        row.turns_p90 = percentile([float(prompt.turns) for prompt in prompts], 0.9)
        row.peak_context = max(prompt.context_peak for prompt in prompts)
        row.resent_units = sum(prompt.resent_units for prompt in prompts)


def resent_share(row: Row) -> float:
    """Share of a session's weighted cost spent re-sending context it already sent."""
    if row.units <= 0:
        return 0.0
    return min(1.0, row.resent_units / row.units)


def apply_codex_drain(rows: Sequence[Row], intervals: Sequence[Interval]) -> None:
    drained = {}  # type: Dict[str, float]
    window_totals = {}  # type: Dict[Any, float]
    session_window = {}  # type: Dict[str, Dict[Any, float]]
    labels = {}  # type: Dict[Any, Any]
    for interval in intervals:
        bucket = resets_bucket(interval.resets_at)
        labels[bucket] = interval.resets_at
        window_totals[bucket] = window_totals.get(bucket, 0.0) + interval.drain
        for session_id, share in interval.sessions.items():
            drained[session_id] = drained.get(session_id, 0.0) + share
            per_window = session_window.setdefault(session_id, {})
            per_window[bucket] = per_window.get(bucket, 0.0) + share
    for row in rows:
        if row.summary.harness != "codex":
            continue
        value = drained.get(row.summary.session_id)
        if value is None:
            continue
        row.drain_percent = value
        windows = session_window.get(row.summary.session_id) or {}
        if windows:
            resets_at = max(windows, key=lambda key: windows[key])
            row.resets_at = labels.get(resets_at, resets_at)
            total = window_totals.get(resets_at) or 0.0
            if total > 0:
                row.share_of_window = 100.0 * windows[resets_at] / total


def claude_dollars_per_percent() -> Optional[float]:
    """Derive dollars-per-percent from logged Claude utilisation snapshots.

    Anthropic ships `limit_dollars` / `used_dollars` as null on every plan seen
    so far, so this normally returns None and Claude drain stays dollar-
    equivalent rather than a percent.
    """
    path = state_dir() / "snapshots.jsonl"
    if not path.is_file():
        return None
    observations = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                window = (record.get("windows") or {}).get("five_hour")
                if not isinstance(window, dict):
                    continue
                limit = window.get("limit_dollars")
                if isinstance(limit, (int, float)) and limit > 0:
                    observations.append(float(limit) / 100.0)
                    continue
                used = window.get("utilization_percent")
                spent = window.get("used_dollars")
                if isinstance(spent, (int, float)) and isinstance(used, (int, float)) and used > 0:
                    observations.append(float(spent) / float(used))
    except OSError:
        return None
    if not observations:
        return None
    observations.sort()
    return observations[len(observations) // 2]


def apply_claude_estimate(rows: Sequence[Row], dollars_per_percent: Optional[float]) -> None:
    total = sum(row.units for row in rows if row.summary.harness == "claude")
    for row in rows:
        if row.summary.harness != "claude":
            continue
        if dollars_per_percent and dollars_per_percent > 0:
            row.drain_percent = row.units / dollars_per_percent
            row.estimated = True
        if total > 0:
            row.share_of_window = 100.0 * row.units / total


# --------------------------------------------------------------------------
# rendering


class Painter:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        prefix = "".join(ANSI.get(style, "") for style in styles)
        return prefix + text + ANSI["reset"]


def make_painter(args: argparse.Namespace) -> Painter:
    return Painter(sys.stdout.isatty() and not args.no_color)


def terminal_width(args: argparse.Namespace) -> int:
    if args.width:
        return max(60, int(args.width))
    return max(60, shutil.get_terminal_size((120, 24)).columns)


def bar(value: float, peak: float, width: int) -> str:
    if width <= 0 or peak <= 0 or value <= 0:
        return ""
    filled = min(1.0, value / peak) * width
    whole = int(filled)
    remainder = filled - whole
    text = "█" * whole
    if whole < width and remainder > 1.0 / len(BAR_GLYPHS):
        text += BAR_GLYPHS[min(len(BAR_GLYPHS) - 1, int(remainder * len(BAR_GLYPHS)))]
    return text


def short_id(session_id: str) -> str:
    return session_id.replace("-", "")[:8] or "?"


def cwd_label(cwd: str) -> str:
    if not cwd:
        return "-"
    return Path(cwd).name or cwd


def header_lines(
    scan: Scan, rows: Sequence[Row], weights: Weights, args: argparse.Namespace,
    dollars_per_percent: Optional[float], window: Optional[str] = None
) -> List[str]:
    lines = []
    tiers = []
    claude_tier = read_claude_tier()
    if claude_tier:
        tiers.append("claude=%s" % claude_tier)
    codex_plan = latest_codex_plan(scan.snapshots)
    if codex_plan:
        tiers.append("codex=%s" % codex_plan)
    lines.append(
        "plan: %s | weights: %s | codex window: %s | zone: %s"
        % (", ".join(tiers) or "unknown", weights.source_label, window or "none",
           local_zone_name())
    )
    used_codex = set()
    used_claude = set()
    for row in rows:
        target = used_claude if row.summary.harness == "claude" else used_codex
        target.update(row.summary.models)
    guessed = weights.guessed_models("codex", used_codex)
    if guessed:
        lines.append("caveat: guessed Codex weights for %s (weights_guessed)" % ", ".join(guessed))
    if any(row.summary.harness == "claude" for row in rows):
        if args.claude_cache_read_weight is None:
            lines.append(
                "caveat: Claude cache reads priced at the API list rate (0.1x input, 0.025x on "
                "Fable 5.1); claude-code#24147 reports them draining quota harder - retry with "
                "--claude-cache-read-weight"
            )
        else:
            lines.append(
                "caveat: Claude cache reads forced to %.3fx input price"
                % args.claude_cache_read_weight
            )
        if dollars_per_percent:
            lines.append(
                "caveat: Claude percent is est, from %.4f $/%% of logged snapshots"
                % dollars_per_percent
            )
        else:
            lines.append(
                "caveat: Claude plan ceiling is UNKNOWN; drain shown as $eq and share of the "
                "listed Claude sessions"
            )
    if len({row.summary.harness for row in rows}) > 1:
        lines.append(
            "caveat: Codex drain is a measured % of the shared pool and Claude drain is a "
            "modelled $ equivalent; rank and bar are each row's share of its own harness peak"
        )
    unweighted = sum(row.unweighted_tokens for row in rows)
    if unweighted:
        lines.append("caveat: %s tokens on models with no weight (unweighted)"
                     % format_tokens(unweighted))
    if args.long_context_multiplier != 1.0:
        lines.append(
            "caveat: long-context multiplier %.2fx applied above %s tokens"
            % (args.long_context_multiplier, format_tokens(LONG_CONTEXT_THRESHOLD))
        )
    return lines


def read_claude_tier() -> str:
    tiers = []
    for config in sorted(home_dir().glob(".claude*/.claude.json")):
        payload = read_claude_config(config)
        if payload is None:
            continue
        account = payload.get("oauthAccount")
        if isinstance(account, dict):
            tier = account.get("organizationRateLimitTier")
            if isinstance(tier, str) and tier:
                tiers.append(tier)
    return "/".join(sorted(set(tiers)))


def read_claude_config(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.stat().st_size > MAX_CONFIG_JSON_BYTES:
            warn("%s is too large to parse; skipping" % path.name)
            return None
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def latest_codex_plan(snapshots: Sequence[Mapping[str, Any]]) -> str:
    latest = None
    for row in snapshots:
        if latest is None or row["ts"] > latest["ts"]:
            latest = row
    if latest is None:
        return ""
    return str(latest.get("plan_type") or "")


def render_sessions(rows: Sequence[Row], args: argparse.Namespace, paint: Painter,
                    width: int) -> List[str]:
    columns = [
        ("H", 1), ("session", 8), ("cwd", 18), ("model", 18), ("start", 16),
        ("dur", 6), ("in", 7), ("cached", 7), ("write", 7), ("out", 7),
        ("units", 9), ("drain", 10), ("prm", 4), ("resent", 6),
    ]
    fixed = sum(size for _, size in columns) + len(columns)
    bar_width = max(6, width - fixed - 1)
    header = (
        " ".join(name.ljust(size)[:size] for name, size in columns)
        + " "
        + "share".ljust(bar_width)[:bar_width]
    )
    lines = [paint(header, "bold")]

    for row in rows:
        summary = row.summary
        harness_glyph = "C" if summary.harness == "claude" else "X"
        if summary.harness == "claude":
            cached = row.tokens.get("cache_read", 0)
            write = row.tokens.get("cache_write_5m", 0) + row.tokens.get("cache_write_1h", 0)
        else:
            cached = row.tokens.get("cached_input", 0)
            write = row.tokens.get("cache_write", 0)
        drain_text = format_drain(row)
        cells = [
            harness_glyph,
            short_id(summary.session_id),
            cwd_label(summary.cwd),
            row.primary_model,
            local_label(summary.start or summary.end or 0),
            format_duration((summary.end or 0) - (summary.start or 0)),
            format_tokens(row.tokens.get("input", 0)),
            format_tokens(cached),
            format_tokens(write),
            format_tokens(row.tokens.get("output", 0)),
            "%.2f" % row.units,
            drain_text,
            "%d" % row.prompt_count if row.prompt_count else "-",
            "%.0f%%" % (100.0 * resent_share(row)) if row.resent_units else "-",
        ]
        body = " ".join(
            value.ljust(size)[:size] for value, (_, size) in zip(cells, columns)
        )
        glyphs = bar(row.relative, 1.0, bar_width)
        style = "claude" if summary.harness == "claude" else "codex"
        lines.append(body + " " + paint(glyphs, style))
        detail = []
        if row.api_turns:
            detail.append(
                "api turns %d, turns/prompt p90 %.0f, peak context %s"
                % (row.api_turns, row.turns_p90, format_tokens(row.peak_context))
            )
        if row.sub_units > 0:
            detail.append(
                "subagents %d requests / %.2f units" % (summary.sub_requests, row.sub_units)
            )
        if detail:
            lines.append(paint(" " * 10 + "; ".join(detail), "dim"))
    return lines


def format_drain(row: Row) -> str:
    if row.drain_percent is None:
        if row.summary.harness == "claude":
            return "$%.2f" % row.units
        return "%.0fu" % row.units
    if row.estimated:
        return "~%.2f%%" % row.drain_percent
    return "%.3f%%" % row.drain_percent


def score_rows(rows: Sequence[Row]) -> None:
    """Score each row against the top drainer of its own harness.

    Codex drain is a measured percent of a shared pool and Claude drain is a
    modelled dollar equivalent. The two cannot share a scale, so ranking and
    the bar both use a within-harness fraction.
    """
    peaks = {}  # type: Dict[str, float]
    for row in rows:
        value = row.drain_percent if row.drain_percent is not None else row.units
        peaks[row.summary.harness] = max(peaks.get(row.summary.harness, 0.0), value or 0.0)
    for row in rows:
        value = row.drain_percent if row.drain_percent is not None else row.units
        peak = peaks.get(row.summary.harness) or 0.0
        row.relative = (value or 0.0) / peak if peak > 0 else 0.0


def sort_rows(rows: List[Row], key: str) -> List[Row]:
    if key == "tokens":
        return sorted(rows, key=lambda row: sum(row.tokens.values()), reverse=True)
    if key == "start":
        return sorted(rows, key=lambda row: row.summary.start or 0.0, reverse=True)
    return sorted(rows, key=lambda row: row.relative, reverse=True)


def row_json(row: Row) -> Dict[str, Any]:
    summary = row.summary
    return {
        "harness": summary.harness,
        "session_id": summary.session_id,
        "short_id": short_id(summary.session_id),
        "cwd": summary.cwd,
        "primary_model": row.primary_model,
        "start": summary.start,
        "end": summary.end,
        "requests": summary.requests,
        "subagent_requests": summary.sub_requests,
        "tokens": row.tokens,
        "subagent_tokens": row.sub_tokens,
        "weighted_units": row.units,
        "subagent_weighted_units": row.sub_units,
        "unweighted_tokens": row.unweighted_tokens,
        "drain_percent": row.drain_percent,
        "drain_is_estimate": row.estimated,
        "relative_to_harness_peak": row.relative,
        "prompts": row.prompt_count,
        "api_turns": row.api_turns,
        "turns_per_prompt_p90": row.turns_p90,
        "peak_context_tokens": row.peak_context,
        "resent_units": row.resent_units,
        "resent_share": resent_share(row),
        "share_of_window_percent": row.share_of_window,
        "window_resets_at": row.resets_at,
    }


# --------------------------------------------------------------------------
# subcommands


class Analysis:
    def __init__(self, scan: Scan, weights: Weights, since: Optional[float],
                 until: Optional[float], args: argparse.Namespace):
        self.args = args
        self.scan = scan
        self.weights = weights
        self.since = since
        self.until = until
        self.window = None  # type: Optional[str]
        self.intervals = []  # type: List[Interval]
        self.prompts = {}  # type: Dict[Tuple[str, str], List[Prompt]]

    def prompts_for(self, harness: str, session_id: str) -> List[Prompt]:
        return self.prompts.get((harness, session_id), [])

    def in_range(self, epoch: Optional[float]) -> bool:
        if epoch is None:
            return False
        if self.since is not None and epoch < self.since:
            return False
        if self.until is not None and epoch > self.until:
            return False
        return True


def prepare(args: argparse.Namespace) -> Analysis:
    now = time.time()
    since = parse_since(args.since, now)
    until = parse_since(args.until, now) if getattr(args, "until", None) else None
    weights = load_weights(getattr(args, "use_calibrated", False))
    scan = collect(args, since)
    analysis = Analysis(scan, weights, since, until, args)
    # Prompt keys must exist before attribution so measured drain can be split
    # down to the prompt as well as the session.
    analysis.prompts = assemble_prompts(scan, weights, args)
    analysis.window = choose_window(scan.snapshots, args.window)
    analysis.intervals = build_intervals(scan.snapshots, analysis.window)
    attribute(analysis.intervals, scan.events["codex"], weights)
    apply_prompt_drain(analysis)
    return analysis


def apply_prompt_drain(analysis: Analysis) -> None:
    shares = {}  # type: Dict[Tuple[str, Any], float]
    for interval in analysis.intervals:
        for key, value in interval.prompts.items():
            shares[key] = shares.get(key, 0.0) + value
    for (harness, session_id), prompts in analysis.prompts.items():
        if harness != "codex":
            continue
        for prompt in prompts:
            value = shares.get((session_id, prompt.index))
            if value is not None:
                prompt.drain_percent = value


def command_sessions(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    scan, weights = analysis.scan, analysis.weights
    since, until = analysis.since, analysis.until
    window, intervals = analysis.window, analysis.intervals
    rows = session_rows(scan, weights, args, since, until)
    attach_fanout(rows, analysis)
    apply_codex_drain(rows, intervals)
    dollars_per_percent = claude_dollars_per_percent()
    apply_claude_estimate(rows, dollars_per_percent)
    score_rows(rows)
    rows = sort_rows(rows, args.sort)[: args.top]
    if args.json:
        payload = {
            "schema": JSON_SCHEMA,
            "command": "sessions",
            "generated_at": time.time(),
            "weight_source": weights.source_label,
            "claude_dollars_per_percent": dollars_per_percent,
            "codex_window": window,
            "files_scanned": scan.files_seen,
            "files_parsed": scan.files_read,
            "sessions": [row_json(row) for row in rows],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    for line in header_lines(scan, rows, weights, args, dollars_per_percent, window):
        print(paint(line, "dim"))
    if not rows:
        print("no sessions in range")
        return 0
    print("")
    for line in render_sessions(rows, args, paint, terminal_width(args)):
        print(line)
    return 0


def command_timeline(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    scan, weights = analysis.scan, analysis.weights
    since, until = analysis.since, analysis.until
    bucket_seconds = {"1h": 3600, "5h": 18000, "1d": 86400}[args.bucket]
    buckets = {}  # type: Dict[int, Dict[str, float]]
    for harness, events in scan.events.items():
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        for event in events:
            epoch = event[EVENT_TS]
            if since is not None and epoch < since:
                continue
            if until is not None and epoch > until:
                continue
            tokens = event_tokens(event, kinds)
            if harness == "claude":
                units = weights.claude_units(
                    event[EVENT_MODEL], tokens, args.claude_cache_read_weight
                )
            else:
                units = weights.codex_units(event[EVENT_MODEL], tokens)
            slot = int(epoch // bucket_seconds) * bucket_seconds
            entry = buckets.setdefault(slot, {"claude": 0.0, "codex": 0.0, "used_percent": 0.0})
            entry[harness] += units
    timeline_window = analysis.window
    for row in scan.snapshots:
        epoch = row["ts"]
        if since is not None and epoch < since:
            continue
        if until is not None and epoch > until:
            continue
        if timeline_window and not window_matches(row.get("window_minutes"), timeline_window):
            continue
        slot = int(epoch // bucket_seconds) * bucket_seconds
        entry = buckets.setdefault(slot, {"claude": 0.0, "codex": 0.0, "used_percent": 0.0})
        entry["used_percent"] = max(entry["used_percent"], row["used_percent"])
    if args.json:
        payload = {
            "schema": JSON_SCHEMA,
            "command": "timeline",
            "bucket_seconds": bucket_seconds,
            "buckets": [
                dict(start=slot, **buckets[slot]) for slot in sorted(buckets)
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    if not buckets:
        print("no activity in range")
        return 0
    width = terminal_width(args)
    bar_width = max(10, width - 44)
    peak = max(entry["claude"] + entry["codex"] for entry in buckets.values()) or 1.0
    print(paint("bucket             claude    codex  used%  claude ▄ / codex ▀", "bold"))
    for slot in sorted(buckets):
        entry = buckets[slot]
        claude_cells = int(round(entry["claude"] / peak * bar_width))
        codex_cells = int(round(entry["codex"] / peak * bar_width))
        glyphs = paint("▄" * claude_cells, "claude") + paint("▀" * codex_cells, "codex")
        used = "%5.1f" % entry["used_percent"] if entry["used_percent"] else "    -"
        print(
            "%-16s %9s %9s %s  %s"
            % (
                local_label(slot, "%m-%d %H:%M"),
                "%.1f" % entry["claude"],
                "%.1f" % entry["codex"],
                used,
                glyphs,
            )
        )
    return 0


def command_windows(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    scan, weights = analysis.scan, analysis.weights
    since, until = analysis.since, analysis.until
    window, intervals = analysis.window, analysis.intervals
    windows = {}  # type: Dict[Tuple[Any, Any], Dict[str, Any]]
    for row in scan.snapshots:
        if since is not None and row["ts"] < since:
            continue
        if window and not window_matches(row.get("window_minutes"), window):
            continue
        key = (row.get("window_minutes"), resets_bucket(row.get("resets_at")))
        entry = windows.setdefault(
            key,
            {
                "window_minutes": row.get("window_minutes"),
                "resets_at": row.get("resets_at"),
                "start": row["ts"],
                "peak_used_percent": 0.0,
                "sessions": {},
            },
        )
        entry["start"] = min(entry["start"], row["ts"])
        entry["peak_used_percent"] = max(entry["peak_used_percent"], row["used_percent"])
    for interval in intervals:
        key = (interval.key[2], resets_bucket(interval.resets_at))
        entry = windows.get(key)
        if entry is None:
            continue
        for session_id, share in interval.sessions.items():
            entry["sessions"][session_id] = entry["sessions"].get(session_id, 0.0) + share
    ordered = sorted(windows.values(), key=lambda item: item["start"], reverse=True)
    if args.json:
        payload = {
            "schema": JSON_SCHEMA,
            "command": "windows",
            "windows": [
                {
                    "window_minutes": entry["window_minutes"],
                    "resets_at": entry["resets_at"],
                    "start": entry["start"],
                    "peak_used_percent": entry["peak_used_percent"],
                    "top_sessions": top_session_list(entry["sessions"], args.top),
                }
                for entry in ordered
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    if not ordered:
        print("no Codex quota windows observed in range")
        return 0
    for entry in ordered:
        resets = entry["resets_at"]
        resets_text = local_label(float(resets)) if isinstance(resets, (int, float)) else "-"
        print(
            paint(
                "window %s min  start %s  resets %s  peak %.1f%%"
                % (
                    entry["window_minutes"],
                    local_label(entry["start"]),
                    resets_text,
                    entry["peak_used_percent"],
                ),
                "bold",
            )
        )
        for item in top_session_list(entry["sessions"], args.top):
            print("    %-10s %6.3f%%" % (item["short_id"], item["drain_percent"]))
    return 0


def top_session_list(sessions: Mapping[str, float], top: int) -> List[Dict[str, Any]]:
    ordered = sorted(sessions.items(), key=lambda item: item[1], reverse=True)[:top]
    return [{"short_id": short_id(key), "drain_percent": value} for key, value in ordered]


def command_calibrate(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    scan, weights = analysis.scan, analysis.weights
    since, until = analysis.since, analysis.until
    if args.harness == "claude":
        return calibrate_claude(args)
    window, intervals = analysis.window, analysis.intervals
    usable = [interval for interval in intervals if not interval.rollover and interval.features]
    features = sorted(set(key for interval in usable for key in interval.features))
    if len(usable) < len(features) or not features:
        warn("not enough snapshot intervals (%d) for %d coefficients" % (len(usable), len(features)))
        if not usable:
            return 1
    matrix = [[interval.features.get(feature, 0.0) for feature in features]
              for interval in usable]
    target = [interval.drain for interval in usable]
    coefficients = nnls(matrix, target)
    score = r_squared(matrix, target, coefficients)
    fitted = {}  # type: Dict[str, Dict[str, float]]
    for (model, kind), value in zip(features, coefficients):
        fitted.setdefault(model, {})[kind] = value
    payload = {
        "version": 1,
        "fitted_at": time.time(),
        "samples": len(usable),
        "r_squared": score,
        "unit": "percent_per_mtok",
        "codex": {"unit": "percent_per_mtok", "models": fitted},
    }
    destination = state_dir() / "codex-weights.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    print(paint("codex calibration: %d intervals, R^2 %.4f" % (len(usable), score), "bold"))
    print("%-22s %-14s %14s %14s" % ("model", "kind", "fit %/Mtok", "default units"))
    for model in sorted(fitted):
        entry = weights.model_entry("codex", model) or {}
        for kind in CODEX_FIT_KINDS:
            if kind not in fitted[model]:
                continue
            print(
                "%-22s %-14s %14.6f %14.2f"
                % (model, kind, fitted[model][kind], float(entry.get(kind, 0.0)))
            )
    print("saved to %s" % destination)
    return 0


def calibrate_claude(args: argparse.Namespace) -> int:
    dollars_per_percent = claude_dollars_per_percent()
    if dollars_per_percent is None:
        warn(
            "no Claude snapshots yet; run `quota-drain snapshot` or wire "
            "`quota-drain snapshot --stdin` into the statusline"
        )
        return 1
    payload = {
        "schema": JSON_SCHEMA,
        "command": "calibrate",
        "harness": "claude",
        "dollars_per_percent": dollars_per_percent,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("claude five_hour window: %.4f USD per percent (from logged snapshots)"
              % dollars_per_percent)
    return 0


def command_prompts(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    matches = [
        (key, prompts)
        for key, prompts in analysis.prompts.items()
        if key[1].replace("-", "").startswith(args.session.replace("-", ""))
        or key[1].startswith(args.session)
    ]
    if not matches:
        warn("no session matching %r; run `quota-drain sessions` for ids" % args.session)
        return 1
    if len(matches) > 1:
        warn(
            "%r matches %d sessions (%s); using the busiest"
            % (args.session, len(matches), ", ".join(short_id(key[1]) for key, _ in matches[:5]))
        )
    (harness, session_id), prompts = max(
        matches, key=lambda item: sum(prompt.units for prompt in item[1])
    )
    mark_reductions(prompts, analysis)
    growth = fit_growth(prompts)
    shown = prompts[-args.top:] if args.top and len(prompts) > args.top else prompts
    if args.json:
        print(
            json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "command": "prompts",
                    "harness": harness,
                    "short_id": short_id(session_id),
                    "growth": growth,
                    "prompts": [prompt.to_json() for prompt in shown],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    paint = make_painter(args)
    width = terminal_width(args)
    print(
        paint(
            "session %s (%s) - %d prompts, %d API calls"
            % (
                short_id(session_id),
                harness,
                len(prompts),
                sum(prompt.turns for prompt in prompts),
            ),
            "bold",
        )
    )
    print("")
    print(paint("%-4s %-16s %6s %6s %10s %10s %10s %9s %-8s" % (
        "#", "start", "wall", "turns", "ctx start", "ctx peak", "input sent", "units", "note"),
        "bold"))
    for prompt in shown:
        print(
            "%-4d %-16s %6s %6d %10s %10s %10s %9.2f %-8s"
            % (
                prompt.index,
                local_label(prompt.start or 0),
                format_duration(prompt.wall_seconds),
                prompt.turns,
                format_tokens(prompt.context_start),
                format_tokens(prompt.context_peak),
                format_tokens(prompt.input_tokens),
                prompt.units,
                prompt.reduction or "",
            )
        )
    print("")
    bar_width = max(10, width - 30)
    peak_input = max([prompt.input_tokens for prompt in shown] or [0])
    print(paint("input tokens sent per prompt (█ uncached, ▒ cached)", "bold"))
    for prompt in shown:
        cached = prompt.tokens.get("cache_read", 0) + prompt.tokens.get("cached_input", 0)
        uncached = max(0, prompt.input_tokens - cached)
        total_cells = int(round(prompt.input_tokens / peak_input * bar_width)) if peak_input else 0
        cached_cells = int(round(total_cells * cached / prompt.input_tokens)) if prompt.input_tokens else 0
        glyphs = paint("█" * (total_cells - cached_cells), "codex") + paint("▒" * cached_cells, "dim")
        print("%-4d %10s %s" % (prompt.index, format_tokens(prompt.input_tokens), glyphs))
    print("")
    peak_context = max([prompt.context_peak for prompt in shown] or [0])
    print(paint("context size at peak per prompt", "bold"))
    for prompt in shown:
        cells = int(round(prompt.context_peak / peak_context * bar_width)) if peak_context else 0
        print(
            "%-4d %10s %s"
            % (prompt.index, format_tokens(prompt.context_peak), paint("▁" * cells, "claude"))
        )
    print("")
    print(paint(describe_growth(growth), "bold"))
    return 0


def describe_growth(growth: Mapping[str, Any]) -> str:
    if growth.get("better") == "insufficient":
        return "growth: too few prompts to fit (%d)" % growth.get("samples", 0)
    linear = growth["linear"]
    quadratic = growth["quadratic"]
    return (
        "growth: %s fits better | linear slope %.3f u/prompt (R^2 %.3f) | "
        "quadratic x^2 term %.4f (R^2 %.3f) | last 20%% of prompts = %.0f%% of session cost"
        % (
            growth["better"],
            linear["slope"],
            linear["r_squared"],
            quadratic["square_term"],
            quadratic["r_squared"],
            100.0 * growth["last_20_percent_share"],
        )
    )


def mark_reductions(prompts: Sequence[Prompt], analysis: "Analysis") -> None:
    reductions = detect_reductions_cached(analysis)
    for prompt in prompts:
        for reduction in reductions:
            if reduction.session_id != prompt.session_id:
                continue
            if prompt.start is not None and prompt.end is not None:
                if prompt.start <= reduction.epoch <= prompt.end:
                    prompt.reduction = reduction.kind


_REDUCTION_CACHE = {}  # type: Dict[int, List[Reduction]]


def detect_reductions_cached(analysis: "Analysis") -> List[Reduction]:
    key = id(analysis)
    found = _REDUCTION_CACHE.get(key)
    if found is None:
        found = detect_reductions(analysis.scan, analysis.weights, analysis.args)
        _REDUCTION_CACHE[key] = found
    return found


def command_fanout(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    prompts = [
        prompt
        for prompts in analysis.prompts.values()
        for prompt in prompts
        if analysis.in_range(prompt.start) or analysis.in_range(prompt.end)
    ]
    if not prompts:
        print("no prompts in range")
        return 0
    turns = [float(prompt.turns) for prompt in prompts]
    contexts = [float(prompt.context_peak) for prompt in prompts]
    top = sorted(prompts, key=lambda prompt: prompt.input_tokens, reverse=True)[:15]
    cwds = {}  # type: Dict[Tuple[str, str], str]
    for (harness, session_id), summary in analysis.scan.sessions.items():
        cwds[(harness, session_id)] = cwd_label(summary.cwd)
    summary_json = {
        "schema": JSON_SCHEMA,
        "command": "fanout",
        "prompts": len(prompts),
        "turns_per_prompt": {
            "p50": percentile(turns, 0.5),
            "p90": percentile(turns, 0.9),
            "max": max(turns),
        },
        "context_at_prompt": {
            "p50": percentile(contexts, 0.5),
            "p90": percentile(contexts, 0.9),
            "max": max(contexts),
        },
        "turns_histogram": histogram(turns, (1, 2, 3, 5, 10, 20, 50, 100)),
        "context_histogram": histogram(
            contexts, (10_000, 50_000, 100_000, 200_000, 400_000, 800_000)
        ),
        "top_prompts": [
            dict(prompt.to_json(), cwd=cwds.get((prompt.harness, prompt.session_id), "-"))
            for prompt in top
        ],
    }
    if args.json:
        print(json.dumps(summary_json, indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    print(paint("%d prompts across %d sessions" % (len(prompts), len(analysis.prompts)), "bold"))
    print(
        "turns/prompt  p50 %.0f  p90 %.0f  max %.0f"
        % (percentile(turns, 0.5), percentile(turns, 0.9), max(turns))
    )
    print(
        "context/prompt p50 %s  p90 %s  max %s"
        % (
            format_tokens(percentile(contexts, 0.5)),
            format_tokens(percentile(contexts, 0.9)),
            format_tokens(max(contexts)),
        )
    )
    print("")
    render_histogram(paint, "API turns per prompt", summary_json["turns_histogram"],
                     terminal_width(args))
    print("")
    render_histogram(paint, "peak context per prompt", summary_json["context_histogram"],
                     terminal_width(args), tokens=True)
    print("")
    print(paint("top 15 single prompts by input tokens sent", "bold"))
    print(paint("%-7s %-10s %-18s %5s %6s %10s %10s" % (
        "harness", "session", "cwd", "#", "turns", "ctx peak", "input"), "bold"))
    for prompt in top:
        print(
            "%-7s %-10s %-18s %5d %6d %10s %10s"
            % (
                prompt.harness,
                short_id(prompt.session_id),
                cwds.get((prompt.harness, prompt.session_id), "-")[:18],
                prompt.index,
                prompt.turns,
                format_tokens(prompt.context_peak),
                format_tokens(prompt.input_tokens),
            )
        )
    return 0


def histogram(values: Sequence[float], edges: Sequence[float]) -> List[Dict[str, Any]]:
    buckets = []
    previous = 0.0
    for edge in edges:
        buckets.append({"lower": previous, "upper": edge, "count": 0})
        previous = edge
    buckets.append({"lower": previous, "upper": None, "count": 0})
    for value in values:
        placed = False
        for bucket in buckets[:-1]:
            if value < bucket["upper"]:
                bucket["count"] += 1
                placed = True
                break
        if not placed:
            buckets[-1]["count"] += 1
    return buckets


def render_histogram(paint: "Painter", title: str, buckets: Sequence[Mapping[str, Any]],
                     width: int, tokens: bool = False) -> None:
    print(paint(title, "bold"))
    peak = max([bucket["count"] for bucket in buckets] or [0])
    bar_width = max(10, width - 34)
    for bucket in buckets:
        lower = format_tokens(bucket["lower"]) if tokens else "%.0f" % bucket["lower"]
        upper = (
            (format_tokens(bucket["upper"]) if tokens else "%.0f" % bucket["upper"])
            if bucket["upper"] is not None
            else "+"
        )
        cells = int(round(bucket["count"] / peak * bar_width)) if peak else 0
        print("%9s - %-9s %7d %s" % (lower, upper, bucket["count"], "█" * cells))


def command_reductions(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    found = [
        reduction
        for reduction in detect_reductions_cached(analysis)
        if analysis.in_range(reduction.epoch)
    ][: args.top]
    if args.json:
        print(
            json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "command": "reductions",
                    "reductions": [reduction.to_json() for reduction in found],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    paint = make_painter(args)
    if not found:
        print("no context reductions detected in range")
        return 0
    print(
        paint(
            "context reductions (>%d%% drop between consecutive API calls)"
            % int(CONTEXT_REDUCTION_FRACTION * 100),
            "bold",
        )
    )
    print(
        paint(
            "%-7s %-10s %-9s %-16s %9s %9s %9s %6s %10s %10s"
            % ("harness", "session", "kind", "time", "before", "after", "removed",
               "turns", "saved", "upper"),
            "bold",
        )
    )
    for reduction in found:
        print(
            "%-7s %-10s %-9s %-16s %9s %9s %9s %6d %10.2f %10.2f"
            % (
                reduction.harness,
                short_id(reduction.session_id),
                reduction.kind,
                local_label(reduction.epoch),
                format_tokens(reduction.before),
                format_tokens(reduction.after),
                format_tokens(reduction.removed),
                reduction.turns_after,
                reduction.saved_units,
                reduction.saved_units_upper,
            )
        )
    return 0


def command_verify(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    scan, weights = analysis.scan, analysis.weights
    since, until = analysis.since, analysis.until
    report = []
    for (harness, session_id), summary in sorted(scan.sessions.items()):
        if since is not None and (summary.end or 0) < since:
            continue
        if harness == "claude":
            cost_state = summary.cost_state
            if not cost_state:
                continue
            for model, usage in (cost_state.get("modelUsage") or {}).items():
                normalized = normalize_claude_model(model)
                observed = summary.models.get(normalized) or {}
                report.append(
                    {
                        "harness": "claude",
                        "short_id": short_id(session_id),
                        "model": normalized,
                        "deduped": {
                            "input": observed.get("input", 0),
                            "output": observed.get("output", 0),
                            "cache_read": observed.get("cache_read", 0),
                            "cache_write": observed.get("cache_write_5m", 0)
                            + observed.get("cache_write_1h", 0),
                        },
                        "cost_state": {
                            "input": usage.get("inputTokens", 0),
                            "output": usage.get("outputTokens", 0),
                            "cache_read": usage.get("cacheReadInputTokens", 0),
                            "cache_write": usage.get("cacheCreationInputTokens", 0),
                        },
                    }
                )
        else:
            thread = summary.thread_usage
            if not thread:
                continue
            observed = empty_tokens(CODEX_KINDS)
            for tokens in summary.models.values():
                for kind in CODEX_KINDS:
                    observed[kind] += int(tokens.get(kind, 0))
            report.append(
                {
                    "harness": "codex",
                    "short_id": short_id(session_id),
                    "model": "-",
                    "deduped": observed,
                    "thread_token_usage": thread,
                }
            )
    if args.json:
        print(json.dumps({"schema": JSON_SCHEMA, "command": "verify", "rows": report},
                         indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    if not report:
        print("nothing to verify in range")
        return 0
    print(paint("%-7s %-10s %-20s %14s %14s %10s" %
                ("harness", "session", "model", "summed", "reported", "delta"), "bold"))
    for entry in report:
        reference = entry.get("cost_state") or entry.get("thread_token_usage") or {}
        summed = sum(entry["deduped"].get(kind, 0) for kind in ("input", "output"))
        reported = sum(int(reference.get(kind, 0) or 0) for kind in ("input", "output"))
        delta = summed - reported
        print(
            "%-7s %-10s %-20s %14d %14d %9.1f%%"
            % (
                entry["harness"],
                entry["short_id"],
                entry["model"],
                summed,
                reported,
                (100.0 * delta / reported) if reported else 0.0,
            )
        )
    return 0


def command_snapshot(args: argparse.Namespace) -> int:
    destination = state_dir() / "snapshots.jsonl"
    if args.stdin:
        return snapshot_from_stdin(destination)
    return snapshot_from_configs(destination)


def snapshot_from_stdin(destination: Path) -> int:
    raw = sys.stdin.buffer.read()
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    try:
        payload = json.loads(raw.decode("utf-8"))
        limits = payload.get("rate_limits")
        if not isinstance(limits, dict):
            return 0
        windows = {}
        for name in ("five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"):
            window = limits.get(name)
            if not isinstance(window, dict):
                continue
            used = window.get("used_percentage")
            if not isinstance(used, (int, float)):
                continue
            windows[name] = {"utilization_percent": float(used),
                             "resets_at": window.get("resets_at")}
        if not windows:
            return 0
        append_snapshot(
            destination,
            {"source": "statusline", "ts": time.time(), "windows": windows},
        )
    except Exception:  # never break the statusline pipeline
        return 0
    return 0


def snapshot_from_configs(destination: Path) -> int:
    last = last_snapshot_marks(destination)
    written = 0
    for config in sorted(home_dir().glob(".claude*/.claude.json")):
        payload = read_claude_config(config)
        if payload is None:
            continue
        cached = payload.get("cachedUsageUtilization")
        if not isinstance(cached, dict):
            continue
        fetched = cached.get("fetchedAtMs")
        if not isinstance(fetched, (int, float)):
            continue
        label = config.parent.name
        if last.get(label) is not None and fetched <= last[label]:
            continue
        utilization = cached.get("utilization")
        windows = {}
        if isinstance(utilization, dict):
            for name in ("five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"):
                window = utilization.get(name)
                if not isinstance(window, dict):
                    continue
                entry = {}
                percent = window.get("utilization")
                if isinstance(percent, (int, float)):
                    entry["utilization_percent"] = float(percent)
                for field in ("resets_at", "limit_dollars", "used_dollars",
                              "remaining_dollars"):
                    value = window.get(field)
                    if value is not None:
                        entry[field] = value
                if entry:
                    windows[name] = entry
        if not windows:
            continue
        append_snapshot(
            destination,
            {
                "source": "claude-json",
                "config_dir": label,
                "fetched_at_ms": fetched,
                "ts": float(fetched) / 1000.0,
                "windows": windows,
            },
        )
        written += 1
    print("appended %d snapshot(s) to %s" % (written, destination))
    return 0


def last_snapshot_marks(destination: Path) -> Dict[str, float]:
    marks = {}  # type: Dict[str, float]
    if not destination.is_file():
        return marks
    try:
        with open(destination, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                label = record.get("config_dir")
                fetched = record.get("fetched_at_ms")
                if isinstance(label, str) and isinstance(fetched, (int, float)):
                    if marks.get(label, 0.0) < fetched:
                        marks[label] = float(fetched)
    except OSError:
        return marks
    return marks


def append_snapshot(destination: Path, record: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# entry point


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--harness", choices=("claude", "codex", "all"), default="all")
    parser.add_argument("--since", default=None, metavar="WHEN")
    parser.add_argument("--until", default=None, metavar="WHEN")
    parser.add_argument("--claude-root", action="append", default=[], metavar="PATH")
    parser.add_argument("--codex-root", action="append", default=[], metavar="PATH")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--width", type=int, default=None, metavar="N")
    parser.add_argument("--claude-cache-read-weight", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--long-context-multiplier", type=float, default=1.0, metavar="FLOAT")
    parser.add_argument("--use-calibrated", action="store_true")
    parser.add_argument("--window", choices=("auto", "five_hour", "weekly"), default="auto")
    parser.add_argument("--top", type=int, default=25, metavar="N")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quota-drain", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")

    sessions = sub.add_parser("sessions", help="one row per session, ranked by quota drain")
    add_common(sessions)
    sessions.add_argument("--sort", choices=("drain", "tokens", "start"), default="drain")
    sessions.set_defaults(handler=command_sessions)

    timeline = sub.add_parser("timeline", help="one bar per time bucket")
    add_common(timeline)
    timeline.add_argument("--bucket", choices=("1h", "5h", "1d"), default="1h")
    timeline.set_defaults(handler=command_timeline)

    windows = sub.add_parser("windows", help="each observed Codex quota window")
    add_common(windows)
    windows.set_defaults(handler=command_windows)

    calibrate = sub.add_parser("calibrate", help="fit weights against measured drain")
    add_common(calibrate)
    calibrate.set_defaults(handler=command_calibrate)

    prompts = sub.add_parser("prompts", help="per-prompt fan-out and context growth for one session")
    add_common(prompts)
    prompts.add_argument("--session", required=True, metavar="ID_PREFIX")
    prompts.set_defaults(handler=command_prompts)

    fanout = sub.add_parser("fanout", help="turns-per-prompt and context distributions")
    add_common(fanout)
    fanout.set_defaults(handler=command_fanout)

    reductions = sub.add_parser("reductions", help="points where a session's context shrank")
    add_common(reductions)
    reductions.set_defaults(handler=command_reductions)

    verify = sub.add_parser("verify", help="cross-check parsed totals against in-band summaries")
    add_common(verify)
    verify.set_defaults(handler=command_verify)

    snapshot = sub.add_parser("snapshot", help="log a Claude quota utilisation observation")
    snapshot.add_argument("--stdin", action="store_true",
                          help="read statusline JSON on stdin and pass it through unchanged")
    snapshot.set_defaults(handler=command_snapshot)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # The statusline pipeline calls this thousands of times a day; get the
    # bytes moving before building the full parser.
    if arguments[:2] == ["snapshot", "--stdin"]:
        return snapshot_from_stdin(state_dir() / "snapshots.jsonl")
    parser = build_parser()
    args = parser.parse_args(arguments)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 2
    if getattr(args, "harness", None) == "claude" and args.command == "windows":
        warn("windows are a Codex-only measurement")
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
