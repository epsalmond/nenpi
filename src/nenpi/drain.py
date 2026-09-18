"""Report which Claude Code and Codex CLI sessions drained subscription quota.

Codex quota is measured from the `rate_limits` snapshots the CLI writes into
its own rollouts. Claude writes no quota data at all, so Claude sessions are
modelled as API list-price dollars and reported as a share of the observed
total; see docs/drain.md for what is official, community-sourced, and
unknown.

Test path overrides use the ``NENPI_*`` environment variables: ``HOME_DIR``,
``CACHE_DIR``, ``STATE_DIR``, ``CONFIG_DIR``, and ``CONFIG_FILE`` (the old
``QUOTA_DRAIN_*`` names still work for all but ``CONFIG_FILE``, with a
deprecation warning). See nenpi.config for root resolution and config.toml.
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
import statistics
import sys
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from nenpi.config import (
    Config,
    account_for_root,
    cache_dir,
    config_dir,
    config_path,
    discover_candidate_roots,
    load_config,
    migrate_dirs,
    resolve_roots,
    save_config,
    state_dir,
    warn,
    warn_once,
)

try:
    from .scanning import (
        Cancellation,
        CancellationToken,
        ProgressCallback,
        ScanCancelled,
        ScanStatus,
        ScannerRun,
        check_cancelled,
        json_progress,
        serialized_cache,
    )
except ImportError:  # bench can load drain.py as a standalone quota module.
    from nenpi.scanning import (  # type: ignore
        Cancellation,
        CancellationToken,
        ProgressCallback,
        ScanCancelled,
        ScanStatus,
        ScannerRun,
        check_cancelled,
        json_progress,
        serialized_cache,
    )

CACHE_SCHEMA = 3
JSON_SCHEMA = 1
LONG_CONTEXT_THRESHOLD = 200_000
CONTEXT_REDUCTION_FRACTION = 0.30
MIN_REDUCTION_CONTEXT = 20_000
REDUCTION_PERSISTENCE_TURNS = 3
DEFAULT_CALIBRATION_BUCKET_HOURS = 2.0
BOUNDARY_DEDUP_SECONDS = 5.0
MAX_CONFIG_JSON_BYTES = 128 * 1024 * 1024
OAUTH_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
OAUTH_BETA = "oauth-2025-04-20"
OAUTH_MIN_INTERVAL_SECONDS = 60.0
OAUTH_BACKOFF_SECONDS = 600.0
OAUTH_TIMEOUT_SECONDS = 15.0
OAUTH_WINDOWS = ("five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet")
SNAPSHOT_RETENTION_DAYS = 60
FALLBACK_CLAUDE_VERSION = "unknown"
JITTER_TOLERANCE = 2.0  # percent; a same-bucket decrease this small or smaller
# is vendor jitter, not a real drop, and must not become a new baseline (#16).
RESET_CLUSTER_TOLERANCE_SECONDS = 300.0  # 5 minutes; collapses a resets_at that
# drifts continuously (an idle pool re-stamping resets_at = now + 7d) into one
# window instead of one entry per drifting reading (#17).
CAP_FACTOR = 3.0  # a session's attributed share of an interval's drain is
# capped at this multiple of its plausible cost (rate * weighted units, #16).
MIN_RATE_INTERVALS = 3  # a pool needs at least this many non-rollover
# intervals before its own measured rate is trusted for capping.
MIN_RATE_UNITS = 1.0  # ...and at least this much weighted-unit coverage.
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
ASCII_GLYPHS = {
    "█": "#", "▄": "=", "▀": "-", "▒": ":", "▁": ".",
    "▏": "|", "▎": "|", "▍": "|", "▌": "|", "▋": "|", "▊": "|", "▉": "|",
}


# --------------------------------------------------------------------------
# time helpers


_TS_CACHE: Dict[str, float] = {}


FRACTIONAL_SECONDS = re.compile(r"\.(\d+)")


def parse_timestamp(value: Any) -> Optional[float]:
    """Parse an ISO-8601 transcript timestamp into a UTC epoch float.

    Python 3.9's `fromisoformat` accepts only 3- or 6-digit fractional
    seconds, and harnesses emit other widths, so the fraction is normalised
    to microseconds first.
    """
    if not isinstance(value, str) or not value:
        return None
    cached = _TS_CACHE.get(value)
    if cached is not None:
        return cached
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    text = FRACTIONAL_SECONDS.sub(
        lambda match: "." + match.group(1)[:6].ljust(6, "0"), text, count=1
    )
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        warn_once("unparseable timestamp %r; those records are skipped" % value[:40])
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
        raise SystemExit("nenpi: cannot parse time %r (use 7d, 12h, or 2026-09-10)" % value)
    if parsed.tzinfo is None:
        # astimezone() on a naive datetime reads it as local time and applies
        # the offset in force on that date, not today's.
        parsed = parsed.astimezone()
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


def claude_price(entry: Mapping[str, Any], model: str, kind: str) -> float:
    """One Claude price, falling back to the list price for a null coefficient.

    A calibrated table stores null for a coefficient the fit could not identify.
    Null means "not measured", so it falls back to the built-in price; reading
    it as zero would price that kind as free.
    """
    value = entry.get(kind)
    if isinstance(value, (int, float)):
        return float(value)
    price = CLAUDE_PRICES.get(normalize_claude_model(model))
    if price is None or kind not in CLAUDE_KINDS:
        return 0.0
    return float(price[CLAUDE_KINDS.index(kind)])


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
        cache_read_price = claude_price(entry, model, "cache_read")
        if cache_read_weight is not None:
            cache_read_price = claude_price(entry, model, "input") * cache_read_weight
        total = 0.0
        for kind in CLAUDE_KINDS:
            price = cache_read_price if kind == "cache_read" else claude_price(entry, model, kind)
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
        if not fit_file.is_file():
            warn("--use-calibrated: no fit at %s; run `quota-drain calibrate`" % fit_file)
            return Weights(table, sources)
        try:
            fit = json.loads(fit_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            warn("ignoring %s: %s" % (fit_file, error))
            return Weights(table, sources)
        if not isinstance(fit, Mapping) or not fit.get("usable"):
            warn("--use-calibrated: stored fit is not usable; keeping the rate card")
            return Weights(table, sources)
        table["codex"] = calibrated_codex_table(table.get("codex") or {}, fit)
        sources.append("calibrated")
    return Weights(table, sources)


def calibrated_codex_table(rate_card: Mapping[str, Any], fit: Mapping[str, Any]
                           ) -> Dict[str, Any]:
    """Express every Codex model in percent per Mtok, fitted or scaled.

    Half a table in fitted percents and half in rate-card credit units is not
    a scale: the two differ by orders of magnitude, so one model's event would
    absorb an interval and the rest would round to nothing.
    """
    scale = float(fit.get("fallback_scale") or 0.0)
    fitted = (fit.get("codex") or {}).get("models") or fit.get("models") or {}
    models = {}  # type: Dict[str, Dict[str, Any]]
    for name, entry in (rate_card.get("models") or {}).items():
        converted = {}  # type: Dict[str, Any]
        for kind in CODEX_FIT_KINDS:
            value = (fitted.get(name) or {}).get(kind)
            if isinstance(value, (int, float)):
                converted[kind] = float(value)
            else:
                converted[kind] = float(entry.get(kind, 0.0) or 0.0) * scale
        if entry.get("guessed"):
            converted["guessed"] = True
        models[name] = converted
    for name, entry in fitted.items():
        if name in models or not isinstance(entry, Mapping):
            continue
        models[name] = dict(
            (kind, float(value))
            for kind, value in entry.items()
            if isinstance(value, (int, float))
        )
    return {"unit": "percent_per_mtok", "models": models}


# --------------------------------------------------------------------------
# transcript discovery


def claude_transcripts(
    roots: Sequence[Path], cancellation: Cancellation = None
) -> Iterator[Tuple[Path, Path]]:
    """Yield (transcript path, resolved root) for every root's ``projects`` dir."""
    for root in roots:
        leaf = root / "projects"
        account_root = root
        if not leaf.is_dir():
            # Exact UI sources may be the leaf itself or one project below it.
            leaf = root
            if root.name == "projects":
                account_root = root.parent
            elif root.parent.name == "projects":
                account_root = root.parent.parent
        if not leaf.is_dir():
            continue
        found = []
        for directory, directories, filenames in os.walk(str(leaf)):
            check_cancelled(cancellation)
            directories.sort()
            for name in sorted(filenames):
                check_cancelled(cancellation)
                if name.endswith(".jsonl"):
                    path = Path(directory) / name
                    if path.is_file():
                        found.append(path)
        for path in sorted(found):
            check_cancelled(cancellation)
            yield path, account_root


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


def codex_transcripts(
    roots: Sequence[Path], since: Optional[float], cancellation: Cancellation = None
) -> Iterator[Tuple[Path, Path]]:
    """Yield (transcript path, resolved root) for every root's ``sessions`` dir."""
    for root in roots:
        leaf = root / "sessions"
        account_root = root
        if not leaf.is_dir():
            # Exact UI sources may be the leaf itself or one day below it.
            leaf = root
            if root.name == "sessions":
                account_root = root.parent
            elif root.parent.name in ("sessions", "rollouts"):
                account_root = root.parent.parent
        if not leaf.is_dir():
            continue
        found = []
        for directory, directories, filenames in os.walk(str(leaf)):
            check_cancelled(cancellation)
            directories.sort()
            for name in sorted(filenames):
                check_cancelled(cancellation)
                if not name.startswith("rollout-") or not name.endswith(".jsonl"):
                    continue
                path = Path(directory) / name
                if not path.is_file():
                    continue
                if since is not None:
                    day = codex_day_epoch(path)
                    # A rollout directory is named for the day it opened; allow one
                    # day of slack so a session that spans midnight is not pruned.
                    if day is not None and day < since - 86400:
                        continue
                found.append(path)
        for path in sorted(found):
            check_cancelled(cancellation)
            yield path, root


def read_lines_from(
    path: Path, offset: int, cancellation: Cancellation = None
) -> Iterator[Tuple[int, bytes]]:
    """Yield (offset-after-line, raw line) for every complete line from offset."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        position = offset
        for raw in handle:
            check_cancelled(cancellation)
            if not raw.endswith(b"\n"):
                # A transcript being appended to right now; stop before the
                # partial line so the next run re-reads it whole.
                break
            position += len(raw)
            yield position, raw


# --------------------------------------------------------------------------
# parsed records


class SessionSummary:
    """Per-session metadata. Token totals are derived from deduped events.

    Totals cannot be accumulated while parsing: a forked or resumed session
    replays the original's assistant lines into a new file under a new
    session id, so the same API call is seen more than once and can only be
    dropped once every file has been read.
    """

    def __init__(self, harness: str, session_id: str):
        self.harness = harness
        self.session_id = session_id
        self.cwd = ""
        self.cwd_hash = ""
        self.version = ""
        self.start = None  # type: Optional[float]
        self.end = None  # type: Optional[float]
        self.cost_state = None  # type: Optional[Dict[str, Any]]
        self.thread_usage = None  # type: Optional[Dict[str, int]]
        self.originator = ""
        # Filled by rebuild_totals from the surviving events.
        self.models = {}  # type: Dict[str, Dict[str, int]]
        self.sub_models = {}  # type: Dict[str, Dict[str, int]]
        self.requests = 0
        self.sub_requests = 0
        self.duplicate_turns = 0
        self.fork_of = ""

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

    def set_cwd(self, cwd: Any) -> None:
        """Keep the basename only; the full path is customer-identifying."""
        if not isinstance(cwd, str) or not cwd:
            return
        if not self.cwd:
            self.cwd = Path(cwd).name or cwd
            self.cwd_hash = hashlib.sha1(cwd.encode("utf-8")).hexdigest()[:12]

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "harness": self.harness,
            "session_id": self.session_id,
            "cwd": self.cwd,
            "cwd_hash": self.cwd_hash,
            "version": self.version,
            "start": self.start,
            "end": self.end,
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
        summary.cwd_hash = str(payload.get("cwd_hash", ""))
        summary.version = str(payload.get("version", ""))
        summary.start = payload.get("start")
        summary.end = payload.get("end")
        summary.cost_state = payload.get("cost_state")
        summary.thread_usage = payload.get("thread_usage")
        summary.originator = str(payload.get("originator", ""))
        return summary

    def merge(self, other: "SessionSummary") -> None:
        self.cwd = self.cwd or other.cwd
        self.cwd_hash = self.cwd_hash or other.cwd_hash
        self.version = other.version or self.version
        self.originator = self.originator or other.originator
        self.touch(other.start)
        self.touch(other.end)
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
        self.head_hash = ""
        self.tail_hash = ""

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
            "head_hash": self.head_hash,
            "tail_hash": self.tail_hash,
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
        index.head_hash = str(payload.get("head_hash", ""))
        index.tail_hash = str(payload.get("tail_hash", ""))
        return index

    def stamp(self, path: Path, stat: os.stat_result) -> None:
        self.size = stat.st_size
        self.mtime = stat.st_mtime
        self.head_hash, self.tail_hash = content_fingerprint(path, self.offset)


CACHE_HEAD_BYTES = 4096
CACHE_TAIL_BYTES = 256


def content_fingerprint(path: Path, offset: int) -> Tuple[str, str]:
    """Hash the head of a file and the bytes just before the resume offset.

    Size and mtime alone miss an in-place rewrite that happens to grow the
    file: resuming from the stored offset would then land mid-record and the
    rest of the file would be silently skipped.
    """
    # Only bytes already parsed are hashed. Covering more would change the
    # fingerprint on every append to a file shorter than the head window.
    head_bytes = min(CACHE_HEAD_BYTES, offset)
    try:
        with open(path, "rb") as handle:
            head = handle.read(head_bytes) if head_bytes else b""
            tail = b""
            if offset > 0:
                handle.seek(max(0, offset - CACHE_TAIL_BYTES))
                tail = handle.read(min(CACHE_TAIL_BYTES, offset))
    except OSError:
        return "", ""
    return (
        hashlib.sha1(head).hexdigest()[:16],
        hashlib.sha1(tail).hexdigest()[:16],
    )


# Event layout, one row per deduped API call:
# [session_id, model, timestamp, k0..k4, long_context, subagent,
#  turn_id, thread_id, call_id]
# assemble_prompts appends the owning prompt index as EVENT_PROMPT.
# The token area is a fixed five slots so one set of indices serves both
# harnesses; Codex uses four kinds and leaves the fifth zero.
EVENT_SESSION, EVENT_MODEL, EVENT_TS = 0, 1, 2
EVENT_KINDS = 3
EVENT_KIND_SLOTS = 5
EVENT_LONG = 8
EVENT_SUB = 9
EVENT_TURN = 10
EVENT_THREAD = 11
EVENT_ID = 12
EVENT_PROMPT = 13


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


def parse_claude_file(
    path: Path, index: FileIndex, cancellation: Cancellation = None
) -> FileIndex:
    is_subagent_file = "/subagents/" in str(path)
    seen_ids = set(index.carry_ids)
    recent = list(index.carry_ids)
    offset = index.offset
    for position, raw in read_lines_from(path, index.offset, cancellation):
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
        summary.set_cwd(record.get("cwd"))
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
                    tokens["output"],
                    1 if long_context else 0,
                    1 if sidechain else 0,
                    "",
                    "",
                    message_id if isinstance(message_id, str) else "",
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


def parse_codex_file(
    path: Path, index: FileIndex, cancellation: Cancellation = None
) -> FileIndex:
    fallback_events = []  # type: List[List[Any]]
    session_id = index.last_session
    model = index.last_model
    cumulative = index.cumulative
    offset = index.offset
    for position, raw in read_lines_from(path, index.offset, cancellation):
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
            summary.set_cwd(payload.get("cwd"))
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
            response_id = payload.get("response_id")
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
                call_id=response_id if isinstance(response_id, str) else "",
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
    call_id: str = "",
) -> None:
    context_size = tokens.get("input", 0) + tokens.get("cached_input", 0)
    long_context = context_size > LONG_CONTEXT_THRESHOLD
    summary.touch(epoch)
    if epoch is None:
        return
    row = [session_id, model, epoch]
    row.extend(int(tokens.get(kind, 0)) for kind in kinds)
    row.extend([0] * (EVENT_KIND_SLOTS - len(kinds)))
    row.append(1 if long_context else 0)
    row.append(1 if sidechain else 0)
    row.append(turn)
    row.append(thread)
    row.append(call_id)
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
        # About to resume from the stored offset, so check that the bytes
        # behind it are still the ones that were parsed. An in-place rewrite
        # that happens to grow the file moves size and mtime like an append
        # but leaves the offset pointing mid-record.
        if entry.head_hash and entry.offset > 0:
            if content_fingerprint(path, entry.offset) != (entry.head_hash, entry.tail_hash):
                entry = FileIndex(harness)
                self.entries[key] = entry
                return entry, True
        return entry, True

    def drop_old_schemas(self, cancellation: Cancellation = None) -> None:
        current = "v%d" % CACHE_SCHEMA
        with serialized_cache(self.root, cancellation):
            try:
                children = list(self.root.iterdir())
            except OSError:
                return
            for child in children:
                if child.is_dir() and child.name.startswith("v") and child.name != current:
                    shutil.rmtree(str(child), ignore_errors=True)

    def prune(self, live: Iterable[str], cancellation: Cancellation = None) -> None:
        """Delete shards for transcripts that no longer exist.

        Shard names are a pure function of the transcript path, so the live
        set can be compared by name; parsing every shard to read its `path`
        would cost more than the whole scan.
        """
        with serialized_cache(self.root, cancellation):
            root = self.root / ("v%d" % CACHE_SCHEMA)
            if not root.is_dir():
                return
            expected = set(self.shard_path(key).name for key in live)
            for shard in root.rglob("*.json"):
                check_cancelled(cancellation)
                if shard.name in expected:
                    continue
                try:
                    shard.unlink()
                except OSError:
                    continue

    def mark(self, path: Path) -> None:
        self.dirty.add(str(path))

    def forget(self, path: Path) -> None:
        self.entries.pop(str(path), None)

    def _flush_unlocked(self) -> None:
        for key in sorted(self.dirty):
            entry = self.entries.get(key)
            if entry is None:
                continue
            payload = entry.to_json()
            payload["path"] = key
            destination = self.shard_path(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                "%s.%d.%d.tmp" % (destination.name, os.getpid(), time.time_ns())
            )
            try:
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, separators=(",", ":"))
                os.replace(str(temporary), str(destination))
            finally:
                try:
                    temporary.unlink()
                except OSError:
                    pass
        self.dirty.clear()

    def flush(self) -> None:
        with serialized_cache(self.root):
            self._flush_unlocked()


class Progress:
    def __init__(self, enabled: bool, total: int, label: str):
        self.enabled = enabled and total > 0
        self.total = total
        self.label = label
        self.done = 0
        self.last = 0.0
        self.width = 0

    def step(self) -> None:
        self.done += 1
        if not self.enabled:
            return
        now = time.time()
        if now - self.last < PROGRESS_INTERVAL_SECONDS and self.done < self.total:
            return
        self.last = now
        line = "quota-drain: %s %d/%d" % (self.label, self.done, self.total)
        self.width = max(self.width, len(line))
        sys.stderr.write("\r" + line)
        sys.stderr.flush()

    def finish(self) -> None:
        if not self.enabled:
            return
        sys.stderr.write("\r" + " " * self.width + "\r")
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
        self.claimed = {}  # type: Dict[Tuple[str, str], str]
        self.forks = {}  # type: Dict[Tuple[str, str], Dict[str, int]]
        # (account key, account label) for every session seen, keyed the same
        # as `sessions`. Populated in `absorb` from the resolved root, never
        # from anything cached on disk - see the module docstring's note on
        # why account identity is not part of the per-file cache schema.
        self.session_accounts = {}  # type: Dict[Tuple[str, str], Tuple[str, str]]
        self.files_read = 0
        self.files_seen = 0
        self.bytes_read = 0


def collect(
    args: argparse.Namespace,
    since: Optional[float],
    run: Optional[ScannerRun] = None,
) -> Scan:
    run = run or ScannerRun()
    harness = args.harness
    cache = Cache(cache_dir(), args.rebuild_cache)
    cache.drop_old_schemas(run.cancellation)
    config = load_config()
    run.emit("discovery", message="discovering transcript roots")
    run.check()

    targets = []  # type: List[Tuple[Path, str, Path]]
    target_keys = set()

    def add_target(path: Path, kind: str, root: Path) -> None:
        key = (kind, str(path.resolve()))
        if key in target_keys:
            return
        target_keys.add(key)
        targets.append((path, kind, root))

    if harness in ("claude", "all"):
        roots = (
            [Path(value).expanduser() for value in args.claude_root]
            if not getattr(args, "discover", True)
            else resolve_roots("claude", args.claude_root, config)
        )
        for path, root in claude_transcripts(roots, run.cancellation):
            add_target(path, "claude", root)
    if harness in ("codex", "all"):
        roots = (
            [Path(value).expanduser() for value in args.codex_root]
            if not getattr(args, "discover", True)
            else resolve_roots("codex", args.codex_root, config)
        )
        for path, root in codex_transcripts(roots, since, run.cancellation):
            add_target(path, "codex", root)
    run.files_seen = len(targets)
    run.emit("discovery", message="found %d transcript files" % len(targets))

    stamped = []  # type: List[Tuple[float, str, Path, str, Path, os.stat_result]]
    live = set()
    for path, kind, root in targets:
        run.check()
        try:
            stat = path.stat()
        except OSError:
            continue
        live.add(str(path))
        if since is not None and stat.st_mtime < since:
            # A transcript last written before the window cannot hold events
            # inside it, so its shard is never opened.
            continue
        stamped.append((stat.st_mtime, str(path), path, kind, root, stat))
    # Oldest file first, so the session that recorded an API call originally
    # keeps it and a later fork that replays it is the one that loses.
    stamped.sort(key=lambda item: (item[0], item[1]))

    scan = Scan()
    scan.files_seen = len(targets)
    progress = Progress(
        sys.stderr.isatty()
        and not getattr(args, "no_color", False)
        and run.progress is None,
        len(stamped), "scanning")
    # One (label, key) lookup per root, however many files it holds - a
    # session's account never opens a second `auth.json`/`.claude.json` read.
    account_cache = {}  # type: Dict[Tuple[str, Path], Tuple[str, str]]
    for _, _, path, kind, root, stat in stamped:
        run.check()
        progress.step()
        run.emit("scanning", current_file=path.name, message="waiting for cache")
        with serialized_cache(cache.root, run.cancellation):
            entry, stale = cache.entry_for(path, kind, stat)
            if stale:
                previous_size = entry.size
                try:
                    if kind == "claude":
                        parse_claude_file(path, entry, run.cancellation)
                    else:
                        parse_codex_file(path, entry, run.cancellation)
                except OSError as error:
                    warn("skipping %s: %s" % (path.name, error.strerror or error.__class__.__name__))
                    continue
                scan.bytes_read += max(0, stat.st_size - previous_size)
                entry.stamp(path, stat)
                cache.mark(path)
                scan.files_read += 1
                run.files_parsed += 1
                run.cache_misses += 1
            else:
                run.cache_hits += 1
            account_key = (kind, root)
            if account_key not in account_cache:
                account_cache[account_key] = account_for_root(root, kind)
            account_label, account = account_cache[account_key]
            absorb(scan, entry, kind, account, account_label)
            try:
                cache._flush_unlocked()
            except OSError as error:
                warn("cache not written: %s" % error)
            cache.forget(path)
        run.bytes_read = scan.bytes_read
        run.emit("scanning", current_file=path.name)
    progress.finish()
    if since is None and harness == "all":
        # `live` holds only what this run looked at, so a harness-scoped or
        # range-limited sweep would delete every shard it never visited.
        cache.prune(live, run.cancellation)
    run.emit("scanning", message="scanned %d files" % len(stamped))
    return scan


def absorb(scan: Scan, entry: FileIndex, harness: str, account: str, account_label: str) -> None:
    """Merge one parsed (possibly cached) file into the scan.

    `account`/`account_label` come from the resolved root this file lives
    under, detected once per root in `collect` and never persisted to the
    per-file cache: a moved or relabelled root cannot leave a stale account
    behind, and a cache shard shared across users never leaks either.
    """
    for session_id, summary in entry.sessions.items():
        key = (harness, session_id)
        existing = scan.sessions.get(key)
        if existing is None:
            scan.sessions[key] = SessionSummary.from_json(summary.to_json())
        else:
            existing.merge(summary)
        scan.session_accounts[key] = (account, account_label)
    claimed = scan.claimed
    kept = scan.events[harness]
    for row in entry.events:
        # Every event's session gets an account stamp here too, not only the
        # ones with a `SessionSummary` above: `attribute()` treats a session
        # missing from `session_accounts` as "any pool" (fail-open), so a
        # session known only through its events - never seen in
        # `entry.sessions` - must never be credited to every account's pool.
        scan.session_accounts.setdefault((harness, row[EVENT_SESSION]), (account, account_label))
        call_id = (harness, row[EVENT_ID]) if row[EVENT_ID] else None
        if call_id is not None:
            owner = claimed.get(call_id)
            if owner is not None:
                # Same API call seen again: a resumed or forked transcript
                # replaying it. Count it once, against whoever recorded it first.
                if owner != row[EVENT_SESSION]:
                    forks = scan.forks.setdefault((harness, row[EVENT_SESSION]), {})
                    forks[owner] = forks.get(owner, 0) + 1
                continue
            claimed[call_id] = row[EVENT_SESSION]
        kept.append(row)
    for row in entry.snapshots:
        stamped_row = dict(row)
        stamped_row["account"] = account
        stamped_row["account_label"] = account_label
        scan.snapshots.append(stamped_row)
    for row in entry.boundaries:
        scan.boundaries.setdefault((harness, str(row[0])), []).append(float(row[1]))
    for row in entry.compactions:
        scan.compactions.setdefault((harness, str(row[0])), []).append(float(row[1]))


def window_events(
    scan: Scan, since: Optional[float], until: Optional[float], cancellation: Cancellation = None
) -> None:
    """Drop every event outside the reporting range.

    Without this a session selected by its end time still reported the tokens
    of its whole life, which overstates a narrow `--since` by orders of
    magnitude.
    """
    if since is None and until is None:
        return
    for harness, events in scan.events.items():
        check_cancelled(cancellation)
        scan.events[harness] = [
            row
            for row in events
            if (since is None or row[EVENT_TS] >= since)
            and (until is None or row[EVENT_TS] <= until)
        ]


def rebuild_totals(scan: Scan, cancellation: Cancellation = None) -> None:
    """Derive per-session token totals and weighted units from surviving events."""
    for summary in scan.sessions.values():
        check_cancelled(cancellation)
        summary.models = {}
        summary.sub_models = {}
        summary.requests = 0
        summary.sub_requests = 0
    for harness, events in scan.events.items():
        check_cancelled(cancellation)
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        for row in events:
            check_cancelled(cancellation)
            summary = scan.sessions.get((harness, row[EVENT_SESSION]))
            if summary is None:
                continue
            tokens = event_tokens(row, kinds)
            bucket = summary.sub_models if row[EVENT_SUB] else summary.models
            slot = bucket.setdefault(row[EVENT_MODEL], empty_tokens(kinds))
            for kind in kinds:
                slot[kind] += tokens[kind]
            if row[EVENT_SUB]:
                summary.sub_requests += 1
                rollup = summary.models.setdefault(row[EVENT_MODEL], empty_tokens(kinds))
                for kind in kinds:
                    rollup[kind] += tokens[kind]
            else:
                summary.requests += 1
    for (harness, session_id), forks in scan.forks.items():
        check_cancelled(cancellation)
        summary = scan.sessions.get((harness, session_id))
        if summary is None:
            continue
        summary.duplicate_turns = sum(forks.values())
        summary.fork_of = max(forks, key=lambda key: forks[key])


class Interval:
    def __init__(self, key: Tuple[str, str, str, Any], start: float, end: float, drain: float,
                 resets_at: Any, rollover: bool):
        # key = (account, limit_id, plan_type, window_minutes); keying the
        # timeline by account is what keeps two quota pools that share a
        # limit_id/plan_type/window (two Codex or two Claude roots) from
        # being read as one pool alternating readings, which used to be
        # charged as a window rollover on every alternation (#9).
        self.key = key
        self.account = key[0]
        self.start = start
        self.end = end
        self.drain = drain
        self.resets_at = resets_at
        self.rollover = rollover
        self.sessions = {}  # type: Dict[str, float]
        self.prompts = {}  # type: Dict[Tuple[str, Any], float]
        self.features = {}  # type: Dict[Tuple[str, str], float]
        # Drain the attribution cap could not place on any session - not a
        # foreign account (that's cross-account bleed, #9), but usage from a
        # client of THIS account that was never scanned (another machine,
        # another login copy). See attribute()/water_fill().
        self.unattributed = 0.0


UNSET = object()
NO_WINDOW = object()
ANY_WINDOW = object()
WINDOW_ALIASES = {"five_hour": 300, "weekly": 10080}
WINDOW_NAMES = {300: "five_hour", 10080: "weekly"}


def resets_bucket(value: Any) -> Any:
    """Collapse the jitter both vendors put in `resets_at`.

    Codex writes Unix seconds that move by a second or two between readings;
    Claude writes an ISO timestamp whose microseconds differ on every poll. A
    raw equality test reads either as a fresh window rollover on every line,
    which inflates measured drain by more than an order of magnitude.
    """
    epoch = resets_epoch(value)
    if epoch is not None:
        return int(round(epoch / 60.0))
    return NO_RESET if value is None else value


NO_RESET = "<no-reset>"


def resets_epoch(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return parse_timestamp(value)
    return None


def cluster_reset_keys(
    items: Iterable[Tuple[Any, Any, Any]],
    tolerance_seconds: float = RESET_CLUSTER_TOLERANCE_SECONDS,
) -> Dict[Tuple[Any, Any, Any], Any]:
    """Map each (account, window_minutes, resets_at) triple to a cluster key.

    `resets_bucket` rounds to the minute, which is enough for the second or
    two of jitter both vendors put on a genuinely stable window, but an idle
    Codex pool re-stamps `resets_at` to `now + 7d` on *every* snapshot, so it
    drifts by roughly a minute per reading and still produces one fresh
    bucket after another (#17) - dozens of `windows` rows for one
    continuously-idle window. Within one (account, window_minutes) pool,
    chain-cluster `resets_at` values sorted by time: a value within
    `tolerance_seconds` of its predecessor joins the same cluster as that
    predecessor, so a slow, continuous drift collapses into a single
    cluster while a real rollover (days away) starts a new one. Every member
    of a cluster is keyed on the earliest member's minute-rounded bucket, so
    the result is still a stable, comparable key.
    """
    groups = {}  # type: Dict[Tuple[Any, Any], List[Tuple[float, Any]]]
    result = {}  # type: Dict[Tuple[Any, Any, Any], Any]
    seen = set()  # type: set
    for account, window_minutes, resets_at in items:
        triple = (account, window_minutes, resets_at)
        if triple in seen:
            continue
        seen.add(triple)
        epoch = resets_epoch(resets_at)
        if epoch is None:
            result[triple] = resets_bucket(resets_at)
            continue
        groups.setdefault((account, window_minutes), []).append((epoch, resets_at))
    for group_key, values in groups.items():
        values.sort(key=lambda item: item[0])
        cluster_key = None  # type: Any
        last_epoch = None  # type: Optional[float]
        for epoch, resets_at in values:
            if cluster_key is None or epoch - last_epoch > tolerance_seconds:
                cluster_key = resets_bucket(resets_at)
            result[(group_key[0], group_key[1], resets_at)] = cluster_key
            last_epoch = epoch
    return result


def snapshot_windows(
    snapshots: Sequence[Mapping[str, Any]], cancellation: Cancellation = None
) -> Dict[Tuple[str, str, str, Any], List[Mapping[str, Any]]]:
    """Group snapshot rows into one timeline per quota pool.

    Keyed by account first: two roots on different accounts can report the
    same limit_id/plan_type/window_minutes (two Codex Pro seats, two Claude
    orgs on the same tier) and are still different pools whose readings must
    never be read as one timeline rolling over (#9).
    """
    grouped = {}  # type: Dict[Tuple[str, str, str, Any], List[Mapping[str, Any]]]
    for row in snapshots:
        check_cancelled(cancellation)
        key = (row.get("account") or "default", row.get("limit_id") or "codex",
               row.get("plan_type") or "unknown", row.get("window_minutes"))
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        check_cancelled(cancellation)
        rows.sort(key=lambda item: item["ts"])
    return grouped


def choose_window(snapshots: Sequence[Mapping[str, Any]], wanted: Optional[str]) -> Any:
    """Pick exactly one window so drains are never summed across denominators.

    Returns the `window_minutes` value to keep, whatever it is - including a
    value this tool has no name for. Summing a five-hour and a weekly window
    together doubles every session's drain.
    """
    if wanted and wanted != "auto":
        if wanted in WINDOW_ALIASES:
            return WINDOW_ALIASES[wanted]
        try:
            return int(wanted)
        except (TypeError, ValueError):
            return wanted
    counts = {}  # type: Dict[Any, int]
    for row in snapshots:
        minutes = row.get("window_minutes")
        key = minutes if isinstance(minutes, (int, float)) else NO_RESET
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return NO_WINDOW
    # Ties go to the shorter window: it is the tighter constraint, and an
    # arbitrary winner would make the same corpus report two different drains.
    ordered = sorted(counts, key=lambda key: (key if isinstance(key, (int, float)) else 1e18,
                                              repr(key)))
    return max(ordered, key=lambda key: counts[key])


def window_label(window: Any) -> str:
    if window is NO_WINDOW:
        return "none"
    if window is ANY_WINDOW:
        return "all"
    if isinstance(window, (int, float)):
        name = WINDOW_NAMES.get(int(window))
        return "%d min (%s)" % (int(window), name) if name else "%d min" % int(window)
    return str(window)


def window_matches(window_minutes: Any, wanted: Any) -> bool:
    if wanted is ANY_WINDOW:
        return True
    if wanted is NO_WINDOW:
        return False
    left = window_minutes if isinstance(window_minutes, (int, float)) else NO_RESET
    if isinstance(left, (int, float)) and isinstance(wanted, (int, float)):
        return int(left) == int(wanted)
    return left == wanted


def build_intervals(
    snapshots: Sequence[Mapping[str, Any]], window: Any,
    cancellation: Cancellation = None,
) -> List[Interval]:
    intervals = []
    groups = snapshot_windows(snapshots, cancellation)
    for key, rows in groups.items():
        check_cancelled(cancellation)
        if not window_matches(key[3], window):
            continue
        current = UNSET  # type: Any
        running = 0.0
        anchor_ts = None  # type: Optional[float]
        for row in rows:
            check_cancelled(cancellation)
            bucket = resets_bucket(row.get("resets_at"))
            used = row["used_percent"]
            if (
                current is not UNSET
                and isinstance(bucket, int)
                and isinstance(current, int)
                and bucket < current
            ):
                # A reading from a window that has already rolled over; two
                # sessions polling concurrently can interleave them.
                continue
            if current is UNSET:
                current, running, anchor_ts = bucket, used, row["ts"]
                continue
            if bucket != current:
                if used == running:
                    # No drain evidence: `resets_at` moved (an idle pool
                    # re-stamping resets_at = now + 7d, or a slide) but the
                    # reading itself did not, so this is not a rollover.
                    # Carry running/anchor_ts forward instead of resetting
                    # them to this reading, so the next real rise still spans
                    # back to the last reading that actually moved (#16/#17).
                    # `used == 0` with `running` already 0 (the only case an
                    # idle, never-active pool can hit) lands here too; a
                    # `used == 0` bucket change with `running > 0` is a real
                    # reset and falls through below instead.
                    current = bucket
                    continue
                drain = min(100.0, max(0.0, used))
                start = window_start(row, anchor_ts)
                if drain > 0 and start is not None and row["ts"] > start:
                    intervals.append(
                        Interval(key, start, row["ts"], drain, row.get("resets_at"), True)
                    )
                current, running, anchor_ts = bucket, used, row["ts"]
                continue
            delta = used - running
            if delta <= 0:
                # A same-bucket reset cannot happen, so a decrease is vendor
                # jitter unless it is bigger than JITTER_TOLERANCE. A jitter
                # decrease (or a flat reading, delta == 0) keeps both running
                # and anchor_ts; a real decrease takes the new, lower reading
                # as the baseline so the next rise is not re-charged for
                # percent already attributed against the old high-water mark.
                if delta < 0 and (running - used) > JITTER_TOLERANCE:
                    running, anchor_ts = used, row["ts"]
                continue
            # `used_percent` is reported in whole percent, so a step covers
            # everything since the last change, not just the last poll.
            if anchor_ts is not None and row["ts"] > anchor_ts:
                intervals.append(
                    Interval(key, anchor_ts, row["ts"], delta, row.get("resets_at"), False)
                )
            running, anchor_ts = used, row["ts"]
    intervals.sort(key=lambda item: item.start)
    if groups and not intervals:
        warn(
            "snapshots exist but none produced a measurable interval "
            "(window %s); Codex drain is unattributed" % window_label(window)
        )
    return intervals


def window_start(row: Mapping[str, Any], anchor_ts: Optional[float]) -> Optional[float]:
    """Where a rolled-over window's drain can have started.

    A new window's `used_percent` was accumulated inside that window, so the
    interval must not reach back past the reset into the previous one and
    charge sessions that had already finished.
    """
    reset = resets_epoch(row.get("resets_at"))
    minutes = row.get("window_minutes")
    if reset is not None and isinstance(minutes, (int, float)) and minutes > 0:
        opened = reset - float(minutes) * 60.0
        if anchor_ts is None:
            return opened
        return max(anchor_ts, opened)
    return anchor_ts


def attribute(
    intervals: Sequence[Interval], events: Sequence[Sequence[Any]], weights: Weights,
    args: argparse.Namespace, cancellation: Cancellation = None
) -> None:
    """Split each interval's measured drain across the sessions active in it.

    An interval belongs to one account's quota pool (`interval.account`); an
    event whose session is known to be on a *different* account is skipped so
    one pool's drain is never split across another account's sessions. The
    session -> (account, label) map is looked up from `args._session_accounts`,
    set by `prepare` just before this call - the signature stays
    `(intervals, events, weights, args)` and the EVENT_* tuple stays
    positional and unwidened; args is the one place with room for it.

    A sparse reading can still make one session's proportional share
    implausible: the whole jump between two readings is split only across
    whichever sessions had a token delta inside that interval, so a handful
    of turns that happened to fall inside a big jump can outrank a session
    that did a thousand times the work in a smaller one (#16). After the
    proportional split, each session's share is capped at `CAP_FACTOR` times
    its plausible cost (a per-account rate, in percent per weighted unit,
    times its own weighted units); drain the cap will not let any session
    absorb is `interval.unattributed` - usage from a client of this account
    that was never scanned, not cross-account bleed (that is #9, handled by
    the per-account `interval.account` filter above).
    """
    session_accounts = getattr(args, "_session_accounts", None) or {}
    ordered = sorted((event for event in events if event[EVENT_TS] is not None),
                     key=lambda event: event[EVENT_TS])
    if not ordered:
        # No Codex usage events at all (a pool with snapshots but nothing
        # scanned locally) still measured real drain across every interval;
        # bailing out here used to leave `interval.unattributed` at its 0.0
        # default, breaking sum(shares) + unattributed == drain. Route it
        # through the same "whole drain unattributed" outcome the
        # `total <= 0` path below gives an interval with events but no
        # local session active.
        for interval in intervals:
            interval.unattributed = interval.drain
        return
    stamps = [event[EVENT_TS] for event in ordered]
    computed = {}  # type: Dict[int, Dict[str, Any]]
    for interval in intervals:
        check_cancelled(cancellation)
        low = bisect.bisect_right(stamps, interval.start)
        high = bisect.bisect_right(stamps, interval.end)
        total = 0.0
        shares = {}  # type: Dict[str, float]
        prompt_shares = {}  # type: Dict[Tuple[str, Any], float]
        prompted_units = {}  # type: Dict[str, float]
        fallback_sessions = set()  # type: set
        for event in ordered[low:high]:
            check_cancelled(cancellation)
            account = session_accounts.get(("codex", event[EVENT_SESSION]))
            if account is not None and account[0] != interval.account:
                continue
            tokens = event_tokens(event, CODEX_KINDS)
            units = weighted_units(
                "codex", event[EVENT_MODEL], tokens, weights, args, bool(event[EVENT_LONG])
            )
            used_fallback = False
            if units <= 0:
                # A model with no weight, or one whose fitted coefficient is
                # zero, still ran inside this interval; fall back to raw
                # tokens so it is never treated as free. That fallback's
                # scale (tokens * 1e-9) is nowhere near the weighted-unit
                # scale, so it is exempted from the cap below rather than
                # capped to near-zero.
                units = float(sum(tokens.get(kind, 0) for kind in CODEX_FIT_KINDS)) * 1e-9
                used_fallback = True
            if units <= 0:
                continue
            session_id = event[EVENT_SESSION]
            total += units
            shares[session_id] = shares.get(session_id, 0.0) + units
            if used_fallback:
                fallback_sessions.add(session_id)
            if len(event) > EVENT_PROMPT and event[EVENT_PROMPT] is not None:
                prompt_key = (session_id, event[EVENT_PROMPT])
                prompt_shares[prompt_key] = prompt_shares.get(prompt_key, 0.0) + units
                prompted_units[session_id] = prompted_units.get(session_id, 0.0) + units
            for kind in CODEX_FIT_KINDS:
                feature = (event[EVENT_MODEL], kind)
                interval.features[feature] = (
                    interval.features.get(feature, 0.0) + tokens.get(kind, 0) / 1_000_000.0
                )
        computed[id(interval)] = {
            "total": total,
            "shares": shares,
            "prompt_shares": prompt_shares,
            "prompted_units": prompted_units,
            "fallback_sessions": fallback_sessions,
        }

    rates = attribution_rates(intervals, computed, weights)
    warned_no_rate = False
    for interval in intervals:
        data = computed[id(interval)]
        total = data["total"]
        if total <= 0:
            # No local session had a token event inside this interval at
            # all (a client nas cannot see did all of it) - the whole
            # drain is unattributed, not silently dropped, so
            # sum(shares) + unattributed == drain still holds for every
            # interval, not just the ones with local activity.
            interval.unattributed = interval.drain
            continue
        shares = data["shares"]
        fallback_sessions = data["fallback_sessions"]
        raw = dict(
            (session_id, interval.drain * units / total) for session_id, units in shares.items()
        )
        rate = rates.get(interval.account)
        if rate is None:
            if not warned_no_rate:
                warn_once(
                    "no attribution cap for one or more account pools: too little "
                    "non-rollover coverage to measure a plausible-cost rate, and no "
                    "calibrated or fitted rate is available; sessions in that pool "
                    "are not capped (see docs/drain.md)"
                )
                warned_no_rate = True
            attributed = raw
        else:
            allowed = dict(
                (session_id, None if session_id in fallback_sessions
                 else CAP_FACTOR * rate * units)
                for session_id, units in shares.items()
            )
            attributed = water_fill(raw, allowed)
        for session_id, value in attributed.items():
            interval.sessions[session_id] = interval.sessions.get(session_id, 0.0) + value
        interval.unattributed = max(0.0, interval.drain - sum(attributed.values()))
        for prompt_key, units in data["prompt_shares"].items():
            # Split the SESSION's attributed (possibly capped) share across
            # its own turns, not the interval's raw drain - otherwise a
            # capped session's prompts still summed to the uncapped figure
            # even though `sessions` reported the capped one. The
            # denominator is the session's PROMPT-KEYED units, not its whole
            # (all-event) units - only main-thread rows carry a prompt key,
            # so an event outside any prompt group (e.g. a sub-thread call
            # that never joined one) still counts toward `shares` but must
            # not dilute the prompt split, or the prompts sum to less than
            # the session's own attributed share.
            session_id = prompt_key[0]
            session_units = data["prompted_units"].get(session_id, 0.0)
            session_share = attributed.get(session_id, 0.0)
            interval.prompts[prompt_key] = (
                session_share * units / session_units if session_units > 0 else 0.0
            )


def water_fill(
    raw: Mapping[str, float], allowed: Mapping[str, Optional[float]]
) -> Dict[str, float]:
    """Clamp each session's share to its cap, redistributing what is freed.

    A session whose proportional share exceeds `allowed[session]` (`None`
    means no cap - the raw-token fallback, exempted because its unit scale
    is not comparable) is clamped to the cap; the drain that clamp frees is
    redistributed proportionally among the sessions still under their own,
    finite cap, which can push one of them over its own cap in turn, so
    this repeats until none are left over. Two guards keep it safe: it
    stops as soon as no capped session remains to receive the freed drain
    (the rest is `interval.unattributed`, not renormalized), and it never
    runs more than one round per session, so float residue cannot spin it.

    Exempt (`allowed is None`) sessions never join the redistribution pool.
    They have no cap to measure headroom against, so letting them soak up a
    capped session's overflow would just recreate #16 for whichever session
    happens to run an unweighted model: the freed drain becomes
    `interval.unattributed` instead. An exempt session keeps its raw
    proportional share untouched - including the degenerate case where it
    is the ONLY session in the interval, where there is no other session's
    cap to measure it against, so raw is the only defensible behaviour and
    is what falls out of this loop naturally (it is never `over`).
    """
    result = dict(raw)
    clamped = set()  # type: set
    for _ in range(len(raw) + 1):
        over = [
            session_id for session_id, value in result.items()
            if session_id not in clamped
            and allowed.get(session_id) is not None
            and value > allowed[session_id]
        ]
        if not over:
            break
        freed = 0.0
        for session_id in over:
            freed += result[session_id] - allowed[session_id]
            result[session_id] = allowed[session_id]
            clamped.add(session_id)
        # Only sessions with a finite, unmet cap can receive freed drain;
        # exempt sessions (allowed is None) are excluded even though they
        # are technically "unclamped", or they would become an uncapped
        # sink for everyone else's overflow.
        unclamped = [
            session_id for session_id in result
            if session_id not in clamped and allowed.get(session_id) is not None
        ]
        if not unclamped:
            break
        headroom_total = sum(result[session_id] for session_id in unclamped)
        if headroom_total <= 0:
            break
        for session_id in unclamped:
            result[session_id] += freed * (result[session_id] / headroom_total)
    return result


def attribution_rates(
    intervals: Sequence[Interval], computed: Mapping[int, Mapping[str, Any]], weights: Weights
) -> Dict[str, Optional[float]]:
    """One plausible-cost rate (percent of quota per weighted unit) per pool.

    The rate is the median of each qualifying interval's own drain/units
    ratio, not a pooled sum(drain)/sum(units): a single foreign-contaminated
    interval (bigger delta, same local tokens - the interval is still real,
    just not fully explained by the sessions this tool can see) drags a
    pooled ratio up and blunts the cap, while a median of ratios does not.
    Rollover intervals and intervals where any session's units came from the
    raw-token fallback are excluded - the latter is not on the weighted-unit
    scale, so mixing it in would corrupt the rate, not just that session's
    cap. A pool with fewer than MIN_RATE_INTERVALS qualifying intervals, or
    under MIN_RATE_UNITS of weighted-unit coverage across them, has too
    little evidence to measure its own rate and falls back to
    `fallback_rate`; `None` means capping is skipped for that pool.
    """
    ratios = {}  # type: Dict[str, List[float]]
    coverage = {}  # type: Dict[str, float]
    accounts = set()
    for interval in intervals:
        accounts.add(interval.account)
        if interval.rollover:
            continue
        data = computed.get(id(interval))
        if not data or data["fallback_sessions"] or data["total"] <= 0 or interval.drain <= 0:
            continue
        ratios.setdefault(interval.account, []).append(interval.drain / data["total"])
        coverage[interval.account] = coverage.get(interval.account, 0.0) + data["total"]
    rates = {}  # type: Dict[str, Optional[float]]
    for account in accounts:
        samples = ratios.get(account) or []
        if len(samples) >= MIN_RATE_INTERVALS and coverage.get(account, 0.0) >= MIN_RATE_UNITS:
            rate = statistics.median(samples)
        else:
            rate = fallback_rate(weights)
        rates[account] = rate if rate and rate > 0 else None
    return rates


def fallback_rate(weights: Weights) -> Optional[float]:
    """The rate to use when a pool has too little coverage to measure its own.

    A calibrated weights table is already expressed in percent per weighted
    unit (`calibrated_codex_table`), so its implied rate is exactly 1.0. Short
    of that, a usable stored fit's `fallback_scale` is "percent per
    rate-card unit" - the same conversion `calibrated_codex_table` applies to
    an unidentified model - so it converts too. With neither, the caller
    leaves the pool uncapped.
    """
    if "calibrated" in weights.sources:
        return 1.0
    fit_file = state_dir() / "codex-weights.json"
    if not fit_file.is_file():
        return None
    try:
        fit = json.loads(fit_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(fit, Mapping) or not fit.get("usable"):
        return None
    scale = fit.get("fallback_scale")
    return float(scale) if isinstance(scale, (int, float)) and scale > 0 else None


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


def weighted_units(
    harness: str,
    model: str,
    tokens: Mapping[str, int],
    weights: Weights,
    args: argparse.Namespace,
    long_context: bool = False,
) -> float:
    """The one place tokens turn into weighted units.

    Every view weighs the same way, so the long-context knob and the
    cache-read override cannot apply in one report and not another.
    """
    if harness == "claude":
        units = weights.claude_units(model, tokens, args.claude_cache_read_weight)
    else:
        units = weights.codex_units(model, tokens)
    if long_context:
        units *= args.long_context_multiplier
    return units


def event_units(
    harness: str, row: Sequence[Any], kinds: Sequence[str], weights: Weights,
    args: argparse.Namespace
) -> float:
    return weighted_units(
        harness, row[EVENT_MODEL], event_tokens(row, kinds), weights, args,
        bool(row[EVENT_LONG]),
    )


def input_side_units(harness: str, model: str, tokens: Mapping[str, int], weights: Weights,
                     cache_read_weight: Optional[float]) -> float:
    """Weighted cost of context sent to the API, excluding generated output."""
    if harness == "claude":
        trimmed = dict((kind, tokens.get(kind, 0)) for kind in CLAUDE_KINDS if kind != "output")
        return weights.claude_units(model, trimmed, cache_read_weight)
    trimmed = dict((kind, tokens.get(kind, 0)) for kind in CODEX_FIT_KINDS if kind != "output")
    return weights.codex_units(model, trimmed)


def assemble_prompts(
    scan: Scan, weights: Weights, args: argparse.Namespace,
    cancellation: Cancellation = None,
) -> Dict[Tuple[str, str], List[Prompt]]:
    """Group API calls under the user prompt that triggered them.

    Codex numbers its own turns, so `turn_id` is authoritative for the root
    thread. Claude has no turn id, so calls fall under the most recent
    user-prompt line. Subagent calls belong to whichever prompt was running
    when they started, in both harnesses.
    """
    assembled = {}  # type: Dict[Tuple[str, str], List[Prompt]]
    for harness, events in scan.events.items():
        check_cancelled(cancellation)
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        for session_id, rows in group_events_by_session(events, cancellation).items():
            check_cancelled(cancellation)
            bounds = sorted(scan.boundaries.get((harness, session_id), []))
            groups = group_rows_by_prompt(rows, bounds, harness)
            prompts = []
            for position, (start_ts, group) in enumerate(groups, start=1):
                check_cancelled(cancellation)
                prompts.append(
                    build_prompt(harness, session_id, position, start_ts, group, kinds,
                                 weights, args, cancellation)
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
    cancellation: Cancellation = None,
) -> Prompt:
    prompt = Prompt(harness, session_id, position)
    prompt.start = start_ts
    baseline = 0.0
    seen_main = False
    for row in group:
        check_cancelled(cancellation)
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
        prompt.units += weighted_units(
            harness, row[EVENT_MODEL], tokens, weights, args, bool(row[EVENT_LONG])
        )
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


def group_events_by_session(
    events: Sequence[Sequence[Any]], cancellation: Cancellation = None
) -> Dict[str, List[Sequence[Any]]]:
    grouped = {}  # type: Dict[str, List[Sequence[Any]]]
    for row in events:
        check_cancelled(cancellation)
        grouped.setdefault(row[EVENT_SESSION], []).append(row)
    for rows in grouped.values():
        check_cancelled(cancellation)
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


def aggregate_intervals(
    intervals: Sequence[Interval], bucket_hours: float
) -> List[Tuple[float, Dict[Tuple[str, str], float]]]:
    """Sum measured drain and token features into fixed time buckets.

    Both vendors report utilisation in whole percent, so a single interval's
    target is almost always exactly 1.0 while its token features vary by orders
    of magnitude; fitting that directly is dominated by quantisation noise.
    Buckets are fixed in *time*, not in drain: bucketing by drain would make
    every target equal by construction and destroy the variance the fit needs.
    """
    span = max(60.0, bucket_hours * 3600.0)
    buckets = {}  # type: Dict[int, Tuple[float, Dict[Tuple[str, str], float]]]
    for interval in intervals:
        slot = int(interval.end // span)
        drain, features = buckets.get(slot, (0.0, {}))
        drain += interval.drain
        for key, value in interval.features.items():
            features[key] = features.get(key, 0.0) + value
        buckets[slot] = (drain, features)
    return [buckets[slot] for slot in sorted(buckets) if buckets[slot][1]]


def column_correlation(matrix: Sequence[Sequence[float]], left: int, right: int) -> float:
    rows = len(matrix)
    if rows < 2:
        return 0.0
    a = [matrix[row][left] for row in range(rows)]
    b = [matrix[row][right] for row in range(rows)]
    mean_a = sum(a) / rows
    mean_b = sum(b) / rows
    covariance = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(rows))
    spread_a = math.sqrt(sum((value - mean_a) ** 2 for value in a))
    spread_b = math.sqrt(sum((value - mean_b) ** 2 for value in b))
    if spread_a <= 0 or spread_b <= 0:
        return 0.0
    return abs(covariance / (spread_a * spread_b))


def fit_percent_weights(
    intervals: Sequence[Interval],
    bucket_hours: float,
    rate_card: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Non-negative least squares of measured drain against token features.

    Least squares will happily return a number for a coefficient the data
    cannot separate - a model that only ever ran beside another, or one with
    too little token mass to move the target. Those are reported as
    `unidentified` and stored as null rather than as a confident zero, because
    a zero weight would price that model as free.
    """
    buckets = aggregate_intervals(intervals, bucket_hours)
    if not buckets:
        return None
    features = sorted(set(key for _, values in buckets for key in values))
    if not features:
        return None
    matrix = [[values.get(key, 0.0) for key in features] for _, values in buckets]
    target = [drain for drain, _ in buckets]
    coefficients = nnls(matrix, target)
    score = r_squared(matrix, target, coefficients)

    mass = [sum(matrix[row][column] for row in range(len(matrix)))
            for column in range(len(features))]
    total_mass = sum(mass) or 1.0
    diagnostics = {}  # type: Dict[str, Dict[str, Any]]
    fitted = {}  # type: Dict[str, Dict[str, Any]]
    identified = 0
    for column, (model, kind) in enumerate(features):
        nonzero = sum(1 for row in matrix if row[column] > 0)
        share = mass[column] / total_mass
        correlation = 0.0
        for other in range(len(features)):
            if other != column:
                correlation = max(correlation, column_correlation(matrix, column, other))
        at_boundary = coefficients[column] <= 0.0
        reasons = []
        if nonzero < 3:
            reasons.append("too few buckets")
        if correlation > 0.95:
            reasons.append("collinear")
        if at_boundary and share < 0.05:
            reasons.append("at zero with little token mass")
        entry = {
            "buckets": nonzero,
            "token_share": share,
            "max_correlation": correlation,
            "at_boundary": at_boundary,
            "unidentified": bool(reasons),
            "reasons": reasons,
        }
        diagnostics.setdefault(model, {})[kind] = entry
        # Only identified coefficients are stored as numbers; the rest are
        # null so the loader falls back to the rate card instead of treating
        # an unidentifiable model as free.
        fitted.setdefault(model, {})[kind] = None if reasons else coefficients[column]
        if not reasons:
            identified += 1

    scale = fit_fallback_scale(buckets, features, target, rate_card)
    scaled = scale > 0.0 if rate_card else True
    return {
        "samples": len(buckets),
        "intervals": len(intervals),
        "r_squared": score,
        "identified": identified,
        "usable": score >= 0.5 and identified > 0 and scaled,
        "fallback_scale": scale,
        "models": fitted,
        "diagnostics": diagnostics,
    }


def fit_fallback_scale(
    buckets: Sequence[Tuple[float, Mapping[Tuple[str, str], float]]],
    features: Sequence[Tuple[str, str]],
    target: Sequence[float],
    rate_card: Optional[Mapping[str, Any]],
) -> float:
    """Fit one percent-per-rate-card-unit scalar.

    Models the fit cannot identify still have to be priced in the same unit as
    the ones it can; mixing fitted percents with raw rate-card units would let
    one model's event absorb a whole interval.
    """
    if not rate_card:
        return 0.0
    models = rate_card.get("models") or {}
    numerator = 0.0
    denominator = 0.0
    for index, (_, values) in enumerate(buckets):
        predicted = 0.0
        for (model, kind), tokens in values.items():
            entry = models.get(model)
            if isinstance(entry, Mapping):
                predicted += tokens * float(entry.get(kind, 0.0) or 0.0)
        numerator += predicted * target[index]
        denominator += predicted * predicted
    if denominator <= 0:
        return 0.0
    return max(0.0, numerator / denominator)


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
        self.account = ""
        self.account_label = ""


def session_rows(
    scan: Scan,
    weights: Weights,
    args: argparse.Namespace,
    since: Optional[float],
    until: Optional[float],
) -> List[Row]:
    """Build one row per session from the events that survived windowing."""
    long_context = {}  # type: Dict[Tuple[str, str, str], Dict[str, int]]
    for harness, events in scan.events.items():
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        for event in events:
            if not event[EVENT_LONG]:
                continue
            slot = long_context.setdefault(
                (harness, event[EVENT_SESSION], event[EVENT_MODEL]), empty_tokens(kinds)
            )
            for kind, value in event_tokens(event, kinds).items():
                slot[kind] += value
    rows = []
    for (harness, session_id), summary in scan.sessions.items():
        if summary.end is None:
            continue
        if since is not None and summary.end < since:
            continue
        if until is not None and summary.start is not None and summary.start > until:
            continue
        if not summary.models and not summary.sub_models:
            continue
        row = Row(summary)
        account = scan.session_accounts.get((harness, session_id))
        if account is not None:
            row.account, row.account_label = account
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        row.tokens = empty_tokens(kinds)
        row.sub_tokens = empty_tokens(kinds)
        best = 0.0
        for model, tokens in summary.models.items():
            for kind in kinds:
                row.tokens[kind] += int(tokens.get(kind, 0))
            stretched = long_context.get((harness, session_id, model)) or {}
            units = weighted_units(harness, model, tokens, weights, args)
            units += weighted_units(harness, model, stretched, weights, args) * (
                args.long_context_multiplier - 1.0
            )
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
            row.sub_units += weighted_units(harness, model, tokens, weights, args)
        rows.append(row)
    return rows


def attach_fanout(rows: Sequence[Row], analysis: "Analysis") -> None:
    """Summarise each session's prompt fan-out onto its report row."""
    whole = getattr(analysis.args, "whole_session", False)
    for row in rows:
        prompts = [
            prompt
            for prompt in analysis.prompts_for(row.summary.harness, row.summary.session_id)
            if whole or analysis.in_range(prompt.start) or analysis.in_range(prompt.end)
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


def apply_codex_drain(rows: Sequence[Row], intervals: Sequence[Interval]) -> Dict[str, float]:
    """Write each session's measured drain, and return per-account unattributed.

    `share_of_window`'s denominator is `interval.drain`, not the sum of what
    got attributed to sessions: after the attribution cap (#16), a capped
    session's share can leave some of `interval.drain` unattributed, and that
    gap must show up as the window not summing to 100% rather than being
    silently renormalized away.
    """
    drained = {}  # type: Dict[str, float]
    window_totals = {}  # type: Dict[Any, float]
    session_window = {}  # type: Dict[str, Dict[Any, float]]
    labels = {}  # type: Dict[Any, Any]
    unattributed = {}  # type: Dict[str, float]
    # The account is part of the bucket key so two pools whose `resets_at`
    # happens to round to the same minute never share a window total; the
    # resets_at itself is clustered (not just minute-rounded) so an idle
    # pool's continuously drifting re-stamp does not fragment one window's
    # drain across many buckets (#17).
    cluster_map = cluster_reset_keys(
        (interval.account, interval.key[3], interval.resets_at) for interval in intervals
    )
    for interval in intervals:
        unattributed[interval.account] = (
            unattributed.get(interval.account, 0.0) + interval.unattributed
        )
        cluster_key = cluster_map.get(
            (interval.account, interval.key[3], interval.resets_at),
            resets_bucket(interval.resets_at),
        )
        bucket = (interval.account, cluster_key)
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
    return unattributed


def claude_dollars_per_percent(
    since: Optional[float] = None, until: Optional[float] = None
) -> Optional[float]:
    """Derive dollars-per-percent from logged Claude utilisation snapshots.

    Anthropic ships `limit_dollars` / `used_dollars` as null on every plan seen
    so far, so this normally returns None and Claude drain stays dollar-
    equivalent rather than a percent.
    """
    observations = []
    for record in iter_snapshots(since, until):
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
    def __init__(self, enabled: bool, ascii_only: bool = False):
        self.enabled = enabled
        self.ascii_only = ascii_only

    def __call__(self, text: str, *styles: str) -> str:
        text = self.glyphs(text)
        if not self.enabled or not styles:
            return text
        prefix = "".join(ANSI.get(style, "") for style in styles)
        return prefix + text + ANSI["reset"]

    def glyphs(self, text: str) -> str:
        if not self.ascii_only:
            return text
        for block, plain in ASCII_GLYPHS.items():
            text = text.replace(block, plain)
        return text


def ascii_output(args: argparse.Namespace) -> bool:
    if getattr(args, "ascii", False):
        return True
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" not in encoding


def make_painter(args: argparse.Namespace) -> Painter:
    return Painter(sys.stdout.isatty() and not args.no_color, ascii_output(args))


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


def account_labels(scan: Scan) -> Dict[str, str]:
    """Map each account key seen in this scan to its display label."""
    labels = {}  # type: Dict[str, str]
    for row in scan.snapshots:
        account = row.get("account")
        if account and account not in labels:
            labels[account] = row.get("account_label") or account
    return labels


def header_lines(
    scan: Scan, rows: Sequence[Row], weights: Weights, args: argparse.Namespace,
    dollars_per_percent: Optional[float], window: Optional[str] = None,
    unattributed: Optional[Mapping[str, float]] = None,
) -> List[str]:
    lines = []
    tiers = []
    config = load_config()
    claude_roots = resolve_roots(
        "claude",
        getattr(args, "claude_root", None) or [],
        config,
        quiet=getattr(args, "harness", "all") == "codex",
    )
    claude_tiers = read_claude_tier(claude_roots)
    if claude_tiers:
        for label in sorted(claude_tiers):
            tiers.append("claude[%s]=%s" % (label, claude_tiers[label]))
    elif config.plan_claude:
        tiers.append("claude=%s" % config.plan_claude)
    codex_plans = latest_codex_plan(scan.snapshots)
    if codex_plans:
        for label in sorted(codex_plans):
            tiers.append("codex[%s]=%s" % (label, codex_plans[label]))
    elif config.plan_codex:
        tiers.append("codex=%s" % config.plan_codex)
    lines.append(
        "plan: %s | weights: %s | codex window: %s | zone: %s"
        % (", ".join(tiers) or "unknown", weights.source_label, window_label(window),
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
    if unattributed:
        labels = account_labels(scan)
        for account in sorted(unattributed, key=lambda key: labels.get(key, key)):
            value = unattributed[account]
            if value <= 0:
                continue
            lines.append(
                "unattributed: %s %.1f%% (usage from clients not in the scanned roots)"
                % (labels.get(account, account), value)
            )
    return lines


def read_claude_tier(roots: Sequence[Path]) -> Dict[str, str]:
    """Map each resolved Claude root's basename to its detected rate-limit tier.

    Callers that only want the old combined string can still do
    ``"/".join(sorted(set(read_claude_tier(roots).values())))``.
    """
    tiers = {}
    for root in roots:
        payload = read_claude_config(root / ".claude.json")
        if payload is None:
            continue
        account = payload.get("oauthAccount")
        if isinstance(account, dict):
            tier = account.get("organizationRateLimitTier")
            if isinstance(tier, str) and tier:
                tiers[root.name] = tier
    return tiers


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


def latest_codex_plan(snapshots: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Map each account label to its most recently observed Codex plan_type."""
    latest = {}  # type: Dict[str, Mapping[str, Any]]
    for row in snapshots:
        label = str(row.get("account_label") or "default")
        current = latest.get(label)
        if current is None or row["ts"] > current["ts"]:
            latest[label] = row
    return dict((label, str(row.get("plan_type") or "")) for label, row in latest.items())


def render_sessions(rows: Sequence[Row], args: argparse.Namespace, paint: Painter,
                    width: int) -> List[str]:
    # Sized so the whole table plus a bar fits an 80x24 terminal's 120-column
    # descendant; anything wider goes to the bar.
    columns = [
        ("H", 1), ("session", 8), ("cwd", 12), ("model", 14), ("start", 11),
        ("dur", 6), ("in", 6), ("cached", 6), ("write", 6), ("out", 6),
        ("units", 8), ("drain", 9), ("prm", 3), ("resent", 6),
    ]
    fixed = sum(size for _, size in columns) + len(columns)
    bar_width = max(0, width - fixed - 1)
    header = " ".join(name.ljust(size)[:size] for name, size in columns)
    if bar_width:
        header += " " + "share".ljust(bar_width)[:bar_width]
    lines = [paint(header, "bold")]

    multi_account = len({row.account_label for row in rows if row.account_label}) > 1
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
            local_label(summary.start or summary.end or 0, "%m-%d %H:%M"),
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
        style = "claude" if summary.harness == "claude" else "codex"
        if bar_width:
            body += " " + paint(bar(row.relative, 1.0, bar_width), style)
        lines.append(body)
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
        if multi_account and row.account_label:
            detail.append("account %s" % row.account_label)
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
        "cwd_hash": summary.cwd_hash,
        "fork_of": short_id(summary.fork_of) if summary.fork_of else "",
        "duplicate_turns": summary.duplicate_turns,
        "prompts": row.prompt_count,
        "api_turns": row.api_turns,
        "turns_per_prompt_p90": row.turns_p90,
        "peak_context_tokens": row.peak_context,
        "resent_units": row.resent_units,
        "resent_share": resent_share(row),
        "share_of_window_percent": row.share_of_window,
        "window_resets_at": row.resets_at,
        "account": row.account,
        "account_label": row.account_label,
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
        self.reductions = None  # type: Optional[List[Reduction]]

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


def filter_by_account(scan: Scan, label: str) -> None:
    """Restrict a scan to one account's sessions, events, and quota rows.

    `label` is a root's basename (what `--account` takes and what `nenpi
    config` shows), but quota pools are keyed by account KEY, not label - two
    roots can share one key (a re-pointed root, or the same login copied to
    a second config dir) and both must stay in the pool together. So this
    first resolves the label to its key(s) via the sessions already seen in
    `scan.session_accounts`, then filters everything downstream by key.

    Filtering here, once, covers every report command that goes through
    `prepare` (sessions/prompts/fanout/windows/timeline/verify/calibrate)
    instead of adding a per-command re-check.
    """
    account_keys = {
        account_key
        for account_key, account_label in scan.session_accounts.values()
        if account_label == label
    }
    if not account_keys:
        known = sorted({
            account_label for _, account_label in scan.session_accounts.values()
        })
        warn(
            "%r matches no resolved root's account; known labels: %s (see `nenpi config`)"
            % (label, ", ".join(known) if known else "none")
        )
    keep = {
        session_key for session_key, (account_key, _) in scan.session_accounts.items()
        if account_key in account_keys
    }
    scan.sessions = dict(
        (key, value) for key, value in scan.sessions.items() if key in keep
    )
    for harness, events in scan.events.items():
        scan.events[harness] = [
            row for row in events if (harness, row[EVENT_SESSION]) in keep
        ]
    scan.snapshots = [row for row in scan.snapshots if row.get("account") in account_keys]
    scan.boundaries = dict(
        (key, value) for key, value in scan.boundaries.items() if key in keep
    )
    scan.compactions = dict(
        (key, value) for key, value in scan.compactions.items() if key in keep
    )


def prepare(
    args: argparse.Namespace,
    progress: Optional[ProgressCallback] = None,
    cancellation: Cancellation = None,
) -> Analysis:
    """Synchronously scan and analyze transcripts for CLI or a worker thread."""

    if progress is None and getattr(args, "json", False):
        progress = json_progress
    run = ScannerRun(progress, cancellation)
    now = time.time()
    try:
        run.check()
        since = parse_since(args.since, now)
        until = parse_since(args.until, now) if getattr(args, "until", None) else None
        weights = load_weights(getattr(args, "use_calibrated", False))
        scan = collect(args, since, run)
        if getattr(args, "account", None):
            filter_by_account(scan, args.account)
        run.emit("analysis", message="building analysis")
        run.check()
        if not getattr(args, "whole_session", False):
            window_events(scan, since, until, cancellation)
        rebuild_totals(scan, cancellation)
        analysis = Analysis(scan, weights, since, until, args)
        # Prompt keys must exist before attribution so measured drain can be split
        # down to the prompt as well as the session.
        analysis.prompts = assemble_prompts(scan, weights, args, cancellation)
        run.check()
        analysis.window = choose_window(scan.snapshots, args.window)
        analysis.intervals = [
            interval
            for interval in build_intervals(scan.snapshots, analysis.window, cancellation)
            if (since is None or interval.end >= since)
            and (until is None or interval.start <= until)
        ]
        for interval in analysis.intervals:
            run.check()
            if since is not None and interval.start < since:
                interval.start = since
        # attribute()'s signature stays compatible with its account-aware
        # implementation; cancellation is an optional final argument.
        args._session_accounts = scan.session_accounts
        attribute(analysis.intervals, scan.events["codex"], weights, args, cancellation)
        apply_prompt_drain(analysis, cancellation)
        run.emit("done", message="analysis ready")
        return analysis
    except (ScanCancelled, KeyboardInterrupt):
        run.emit("cancelled", message="scan cancelled")
        raise ScanCancelled() from None


def apply_prompt_drain(analysis: Analysis, cancellation: Cancellation = None) -> None:
    shares = {}  # type: Dict[Tuple[str, Any], float]
    for interval in analysis.intervals:
        check_cancelled(cancellation)
        for key, value in interval.prompts.items():
            shares[key] = shares.get(key, 0.0) + value
    for (harness, session_id), prompts in analysis.prompts.items():
        check_cancelled(cancellation)
        if harness != "codex":
            continue
        for prompt in prompts:
            check_cancelled(cancellation)
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
    unattributed = apply_codex_drain(rows, intervals)
    dollars_per_percent = claude_dollars_per_percent(since, until)
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
            "codex_window": window_label(window),
            "files_scanned": scan.files_seen,
            "files_parsed": scan.files_read,
            "sessions": [row_json(row) for row in rows],
            "pools": codex_pools(scan, intervals, since, window),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    width = terminal_width(args)
    for line in header_lines(scan, rows, weights, args, dollars_per_percent, window, unattributed):
        for wrapped in textwrap.wrap(line, width, subsequent_indent="  ") or [""]:
            print(paint(wrapped, "dim"))
    if not rows:
        print("no sessions in range")
        return 0
    print("")
    for line in render_sessions(rows, args, paint, width):
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
            units = event_units(harness, event, kinds, weights, args)
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
        if not window_matches(row.get("window_minutes"), timeline_window):
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


def codex_window_entries(
    scan: Scan, intervals: Sequence[Interval], since: Optional[float], window: Any
) -> List[Dict[str, Any]]:
    """One entry per (account, window instance): peak, sessions, unattributed.

    Grouped by `resets_at` cluster, not just `resets_bucket`'s minute
    rounding: an idle Codex pool re-stamps `resets_at` to `now + 7d` on every
    snapshot, drifting by roughly a minute per reading, which minute-rounding
    still slices into dozens of windows for one continuously-idle pool
    (#17). Account is still first in the key so two pools reporting the same
    window_minutes and cluster never merge into one window (#9).
    """
    filtered = [
        row for row in scan.snapshots
        if (since is None or row["ts"] >= since)
        and window_matches(row.get("window_minutes"), window)
    ]
    cluster_map = cluster_reset_keys(
        [
            (row.get("account") or "default", row.get("window_minutes"), row.get("resets_at"))
            for row in filtered
        ]
        + [
            (interval.account, interval.key[3], interval.resets_at)
            for interval in intervals
        ]
    )
    windows = {}  # type: Dict[Tuple[Any, Any, Any], Dict[str, Any]]
    for row in filtered:
        account = row.get("account") or "default"
        window_minutes = row.get("window_minutes")
        resets_at = row.get("resets_at")
        cluster_key = cluster_map.get(
            (account, window_minutes, resets_at), resets_bucket(resets_at)
        )
        key = (account, window_minutes, cluster_key)
        entry = windows.setdefault(
            key,
            {
                "account": account,
                "account_label": row.get("account_label") or "default",
                "window_minutes": window_minutes,
                "resets_at": resets_at,
                "start": row["ts"],
                "peak_used_percent": 0.0,
                "sessions": {},
                "attributed_percent": 0.0,
                "unattributed_percent": 0.0,
            },
        )
        entry["start"] = min(entry["start"], row["ts"])
        entry["peak_used_percent"] = max(entry["peak_used_percent"], row["used_percent"])
    for interval in intervals:
        cluster_key = cluster_map.get(
            (interval.account, interval.key[3], interval.resets_at),
            resets_bucket(interval.resets_at),
        )
        key = (interval.account, interval.key[3], cluster_key)
        entry = windows.get(key)
        if entry is None:
            continue
        for session_id, share in interval.sessions.items():
            entry["sessions"][session_id] = entry["sessions"].get(session_id, 0.0) + share
        entry["attributed_percent"] += sum(interval.sessions.values())
        entry["unattributed_percent"] += interval.unattributed
    return sorted(windows.values(), key=lambda item: item["start"], reverse=True)


def codex_pools(
    scan: Scan, intervals: Sequence[Interval], since: Optional[float], window: Any
) -> List[Dict[str, Any]]:
    """`--json`'s `pools`: one row per (account, window instance)."""
    return [
        {
            "account_label": entry["account_label"],
            "window_resets_at": entry["resets_at"],
            "peak": entry["peak_used_percent"],
            "attributed": entry["attributed_percent"],
            "unattributed": entry["unattributed_percent"],
        }
        for entry in codex_window_entries(scan, intervals, since, window)
    ]


def command_windows(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    scan = analysis.scan
    since = analysis.since
    window, intervals = analysis.window, analysis.intervals
    # `--account` is already applied to `scan` in `prepare` (`filter_by_account`),
    # so `intervals`/`scan.snapshots` here hold only the selected account's
    # pool; no re-check against `args.account` is needed.
    ordered = codex_window_entries(scan, intervals, since, window)
    if args.json:
        payload = {
            "schema": JSON_SCHEMA,
            "command": "windows",
            "windows": [
                {
                    "account": entry["account"],
                    "account_label": entry["account_label"],
                    "window_minutes": entry["window_minutes"],
                    "resets_at": entry["resets_at"],
                    "start": entry["start"],
                    "peak_used_percent": entry["peak_used_percent"],
                    "unattributed_percent": entry["unattributed_percent"],
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
    label_width = max(len(entry["account_label"]) for entry in ordered)
    for entry in ordered:
        resets = entry["resets_at"]
        resets_text = local_label(float(resets)) if isinstance(resets, (int, float)) else "-"
        print(
            paint(
                "window %s min  account %-*s  start %s  resets %s  peak %.1f%%"
                % (
                    entry["window_minutes"],
                    label_width,
                    entry["account_label"],
                    local_label(entry["start"]),
                    resets_text,
                    entry["peak_used_percent"],
                ),
                "bold",
            )
        )
        for item in top_session_list(entry["sessions"], args.top):
            print("    %-10s %6.3f%%" % (item["short_id"], item["drain_percent"]))
        if entry["unattributed_percent"] > 0:
            print("    unattributed %.1f%%" % entry["unattributed_percent"])
    return 0


def top_session_list(sessions: Mapping[str, float], top: int) -> List[Dict[str, Any]]:
    ordered = sorted(sessions.items(), key=lambda item: item[1], reverse=True)[:top]
    return [{"short_id": short_id(key), "drain_percent": value} for key, value in ordered]


def command_calibrate(args: argparse.Namespace) -> int:
    if args.harness == "all":
        # Fitting is per harness; scanning the other one buys nothing.
        args.harness = "codex"
    analysis = prepare(args)
    weights = analysis.weights
    if args.harness == "claude":
        return calibrate_claude(args, analysis)
    intervals = analysis.intervals
    usable = [interval for interval in intervals if not interval.rollover and interval.features]
    fit = fit_percent_weights(
        usable, args.calibrate_bucket_hours, weights.table.get("codex")
    )
    if fit is None:
        warn("no snapshot intervals with token activity in range")
        return 1
    if not fit["usable"]:
        warn(
            "fit is not usable (R^2 %.4f, %d identified coefficients); the window "
            "reports whole percents, so a longer range or a larger "
            "--calibrate-bucket-hours is needed before these weights mean anything"
            % (fit["r_squared"], fit["identified"])
        )
    # The fit target is measured drain, unchanged by the attribution cap: the
    # cap only redistributes ONE interval's drain across ITS OWN sessions and
    # never reaches `fit_percent_weights` (it reads `interval.drain`/
    # `interval.features`, never `interval.sessions`), and subtracting
    # unattributed from the target would be circular (unattributed depends on
    # the rate, which depends on these very weights). Report it instead, so a
    # corpus where a lot of drain has no local explanation is visible.
    total_drain = sum(interval.drain for interval in usable)
    total_unattributed = sum(interval.unattributed for interval in usable)
    unattributed_share = (total_unattributed / total_drain) if total_drain > 0 else 0.0
    payload = {
        "schema": JSON_SCHEMA,
        "version": 1,
        "fitted_at": time.time(),
        "samples": fit["samples"],
        "intervals": fit["intervals"],
        "r_squared": fit["r_squared"],
        "identified": fit["identified"],
        "usable": fit["usable"],
        "unit": "percent_per_mtok",
        "bucket_hours": args.calibrate_bucket_hours,
        "fallback_scale": fit["fallback_scale"],
        "unattributed_share": unattributed_share,
        "diagnostics": fit["diagnostics"],
        "codex": {"unit": "percent_per_mtok", "models": fit["models"]},
    }
    fitted = fit["models"]
    score = fit["r_squared"]
    destination = state_dir() / "codex-weights.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    print(
        paint(
            "codex calibration: %d intervals in %d buckets of %.1fh, R^2 %.4f%s"
            % (fit["intervals"], fit["samples"], args.calibrate_bucket_hours, score,
               "" if fit["usable"] else "  (NOT USABLE)"),
            "bold",
        )
    )
    print(
        "fallback scale %.8f %%/rate-card unit; unidentified coefficients use it"
        % fit["fallback_scale"]
    )
    if unattributed_share > 0:
        print(
            "diagnostic: %.1f%% of measured drain in this corpus is unattributed "
            "(usage from clients not in the scanned roots); the fit target is "
            "unchanged, but a large share here means a contaminated corpus"
            % (100.0 * unattributed_share)
        )
    print("%-22s %-14s %14s %8s %8s %6s %s" % (
        "model", "kind", "fit %/Mtok", "buckets", "share", "corr", "status"))
    for model in sorted(fitted):
        for kind in CODEX_FIT_KINDS:
            if kind not in fitted[model]:
                continue
            value = fitted[model][kind]
            note = fit["diagnostics"][model][kind]
            print(
                "%-22s %-14s %14s %8d %7.1f%% %6.2f %s"
                % (
                    model,
                    kind,
                    "-" if value is None else "%.6f" % value,
                    note["buckets"],
                    100.0 * note["token_share"],
                    note["max_correlation"],
                    ", ".join(note["reasons"]) or "identified",
                )
            )
    print("saved to %s" % destination)
    return 0


CLAUDE_WINDOW_MINUTES = {"five_hour": 300, "seven_day": 10080,
                         "seven_day_opus": 10080, "seven_day_sonnet": 10080}


def iter_snapshots(
    since: Optional[float] = None, until: Optional[float] = None
) -> Iterator[Dict[str, Any]]:
    """Yield logged snapshot records inside the requested range."""
    path = state_dir() / "snapshots.jsonl"
    if not path.is_file():
        return
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                epoch = record.get("ts")
                if not isinstance(epoch, (int, float)):
                    continue
                if since is not None and epoch < since:
                    continue
                if until is not None and epoch > until:
                    continue
                yield record
    except OSError:
        return


def load_claude_snapshots(
    window: str, since: Optional[float] = None, until: Optional[float] = None,
    claude_roots: Optional[Sequence[Path]] = None,
) -> List[Dict[str, Any]]:
    """Read logged Claude utilisation observations as snapshot rows.

    Shaped like the Codex `rate_limits` rows so the same interval builder and
    NNLS fit apply to both harnesses. Each record only ever carries
    `config_dir` (a root's basename, per the privacy note on
    `oauth_snapshot_record`'s docstring - it never stores an account id or
    org identifier); `account`/`account_label` are resolved here, in memory,
    by mapping that label through the currently resolved Claude roots.
    `plan_type` is the LIVE detected rate-limit tier for that root, not
    whatever tier was in effect when the record was written - `account` is
    what actually separates the pools now, so a present-tense tier label is
    enough, and there is no tier field to read out of the record itself.
    """
    roots = list(claude_roots) if claude_roots is not None else resolve_roots(
        "claude", [], load_config()
    )
    tiers = read_claude_tier(roots)
    accounts = {}  # type: Dict[str, str]
    for root in roots:
        label, key = account_for_root(root, "claude")
        accounts[label] = key
    default_label = roots[0].name if roots else "default"
    rows = []
    for record in iter_snapshots(since, until):
        entry = (record.get("windows") or {}).get(window)
        if not isinstance(entry, dict):
            continue
        used = entry.get("utilization_percent")
        if not isinstance(used, (int, float)):
            continue
        label = str(record.get("config_dir") or default_label)
        rows.append(
            {
                "ts": float(record["ts"]),
                "limit_id": "claude",
                "plan_type": tiers.get(label) or "unknown",
                "window_minutes": CLAUDE_WINDOW_MINUTES.get(window, 300),
                "used_percent": float(used),
                "resets_at": entry.get("resets_at"),
                "limit_dollars": entry.get("limit_dollars"),
                "used_dollars": entry.get("used_dollars"),
                "account": accounts.get(label, label),
                "account_label": label,
            }
        )
    rows.sort(key=lambda row: row["ts"])
    return rows


def collect_claude_features(intervals: Sequence[Interval], events: Sequence[Sequence[Any]]
                            ) -> None:
    ordered = sorted(events, key=lambda event: event[EVENT_TS])
    stamps = [event[EVENT_TS] for event in ordered]
    for interval in intervals:
        low = bisect.bisect_right(stamps, interval.start)
        high = bisect.bisect_right(stamps, interval.end)
        for event in ordered[low:high]:
            tokens = event_tokens(event, CLAUDE_KINDS)
            for kind in CLAUDE_KINDS:
                if not tokens[kind]:
                    continue
                feature = (event[EVENT_MODEL], kind)
                interval.features[feature] = (
                    interval.features.get(feature, 0.0) + tokens[kind] / 1_000_000.0
                )


def calibrate_claude(args: argparse.Namespace, analysis: "Analysis") -> int:
    results = {}
    claude_roots = resolve_roots("claude", getattr(args, "claude_root", None) or [], load_config())
    for window in ("five_hour", "seven_day"):
        rows = [
            row
            for row in load_claude_snapshots(
                window, analysis.since, analysis.until, claude_roots
            )
        ]
        if len(rows) < 2:
            continue
        intervals = build_intervals(rows, ANY_WINDOW)
        collect_claude_features(intervals, analysis.scan.events["claude"])
        usable = [
            interval for interval in intervals if not interval.rollover and interval.features
        ]
        fit = fit_percent_weights(usable, args.calibrate_bucket_hours)
        if fit is None:
            continue
        fit["models"] = dict(
            (model, dict((kind, value) for kind, value in entry.items()))
            for model, entry in fit["models"].items()
        )
        fit["dollars_per_percent"] = claude_dollars_per_percent(analysis.since, analysis.until)
        results[window] = fit
    if not results:
        warn(
            "no usable Claude snapshot intervals; sample utilisation with "
            "`quota-drain snapshot --oauth` (see docs/drain.md) and retry"
        )
        return 1
    payload = {
        "schema": JSON_SCHEMA,
        "command": "calibrate",
        "harness": "claude",
        "unit": "percent_per_mtok",
        "windows": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    for window, result in sorted(results.items()):
        print(
            paint(
                "claude %s: %d intervals in %d buckets, R^2 %.4f%s"
                % (window, result["intervals"], result["samples"], result["r_squared"],
                   "" if result["usable"] else "  (NOT USABLE)"),
                "bold",
            )
        )
        if result["dollars_per_percent"]:
            print("  %.4f USD per percent (from dollar fields)" % result["dollars_per_percent"])
        print("  %-22s %14s %14s %10s" % ("model", "input %/Mtok", "cache read %/Mtok",
                                          "implied x"))
        for model in sorted(result["models"]):
            entry = result["models"][model]
            uncached = entry.get("input")
            cached = entry.get("cache_read")
            implied = (
                (cached / uncached)
                if isinstance(uncached, (int, float))
                and isinstance(cached, (int, float))
                and uncached > 0
                else None
            )
            print(
                "  %-22s %14s %14s %10s"
                % (
                    model,
                    "-" if uncached is None else "%.6f" % uncached,
                    "-" if cached is None else "%.6f" % cached,
                    "-" if implied is None else "%.3f" % implied,
                )
            )
        print(
            "  list price puts cache reads at 0.1x input (0.025x on Fable 5.1); "
            "the implied column is the measured ratio"
        )
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
    harness_style = "claude" if harness == "claude" else "codex"
    peak_input = max([prompt.input_tokens for prompt in shown] or [0])
    print(
        paint("input tokens sent per prompt (", "bold")
        + paint("█", harness_style)
        + paint(" uncached, ", "bold")
        + paint("▒", "dim")
        + paint(" cached)", "bold")
    )
    for prompt in shown:
        cached = prompt.tokens.get("cache_read", 0) + prompt.tokens.get("cached_input", 0)
        total_cells = int(round(prompt.input_tokens / peak_input * bar_width)) if peak_input else 0
        cached_cells = int(round(total_cells * cached / prompt.input_tokens)) if prompt.input_tokens else 0
        glyphs = paint("█" * (total_cells - cached_cells), harness_style) + paint("▒" * cached_cells, "dim")
        print("%-4d %10s %s" % (prompt.index, format_tokens(prompt.input_tokens), glyphs))
    print("")
    peak_context = max([prompt.context_peak for prompt in shown] or [0])
    print(paint("context size at peak per prompt", "bold"))
    for prompt in shown:
        cells = int(round(prompt.context_peak / peak_context * bar_width)) if peak_context else 0
        print(
            "%-4d %10s %s"
            % (prompt.index, format_tokens(prompt.context_peak), paint("▁" * cells, harness_style))
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


def detect_reductions_cached(analysis: "Analysis") -> List[Reduction]:
    if analysis.reductions is None:
        analysis.reductions = detect_reductions(
            analysis.scan, analysis.weights, analysis.args
        )
    return analysis.reductions


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
        print("%9s - %-9s %7d %s" % (lower, upper, bucket["count"], paint("█" * cells)))


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
    scan = analysis.scan
    since = analysis.since
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


CLI_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def detect_claude_version() -> str:
    """Read the CLI version off the newest transcript, for the User-Agent.

    Only a top-level `version` field of a parsed line is accepted, and only
    when it looks like a version: a regex over the raw tail would let message
    content reach an outbound header.
    """
    newest = None
    newest_mtime = 0.0
    for root in resolve_roots("claude", [], load_config(), quiet=True):
        leaf = root / "projects"
        if not leaf.is_dir():
            continue
        for path in leaf.rglob("*.jsonl"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_mtime:
                newest, newest_mtime = path, mtime
    if newest is None:
        return FALLBACK_CLAUDE_VERSION
    try:
        with open(newest, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return FALLBACK_CLAUDE_VERSION
    for line in reversed(tail.splitlines()):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        version = record.get("version")
        if isinstance(version, str) and CLI_VERSION.match(version):
            return version
    return FALLBACK_CLAUDE_VERSION


def oauth_config_dirs(requested: Sequence[str]) -> List[Path]:
    if requested:
        return [Path(item).expanduser() for item in requested]
    found = []
    for root in resolve_roots("claude", [], load_config(), quiet=True):
        if (root / ".credentials.json").is_file():
            found.append(root)
    return found


def read_oauth_token(config: Path) -> Tuple[Optional[str], Optional[float]]:
    """Return (access token, expiry epoch). The token is never logged or stored."""
    try:
        payload = json.loads((config / ".credentials.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    block = payload.get("claudeAiOauth") if isinstance(payload, dict) else None
    if not isinstance(block, dict):
        return None, None
    token = block.get("accessToken")
    expires = block.get("expiresAt")
    if not isinstance(token, str) or not token:
        return None, None
    expiry = float(expires) / 1000.0 if isinstance(expires, (int, float)) else None
    return token, expiry


def load_poll_state() -> Dict[str, Dict[str, float]]:
    path = state_dir() / "oauth-poll.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_poll_state(state: Mapping[str, Mapping[str, float]]) -> None:
    path = state_dir() / "oauth-poll.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects would forward the Authorization header to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        raise urllib.error.HTTPError(
            req.full_url, code, "refusing redirect while holding a token", headers, fp
        )


def build_oauth_opener_handlers() -> List[Any]:
    return [RefuseRedirect()]


def build_oauth_opener() -> "urllib.request.OpenerDirector":
    return urllib.request.build_opener(*build_oauth_opener_handlers())


def fetch_oauth_usage(token: str, version: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        OAUTH_USAGE_URL,
        headers={
            "Authorization": "Bearer " + token,
            "anthropic-beta": OAUTH_BETA,
            "Content-Type": "application/json",
            "User-Agent": "claude-code/" + version,
        },
        method="GET",
    )
    opener = build_oauth_opener()
    response = opener.open(request, timeout=OAUTH_TIMEOUT_SECONDS)
    try:
        body = response.read().decode("utf-8")
    finally:
        response.close()
    payload = json.loads(body)
    return payload if isinstance(payload, dict) else {}


def oauth_snapshot_record(config: Path, payload: Mapping[str, Any], now: float
                          ) -> Optional[Dict[str, Any]]:
    """Keep only the quota shape; organization and account identifiers are dropped."""
    windows = {}
    for name in OAUTH_WINDOWS:
        window = payload.get(name)
        if not isinstance(window, Mapping):
            continue
        utilization = window.get("utilization")
        if not isinstance(utilization, (int, float)):
            continue
        entry = {"utilization_percent": float(utilization)}
        for field in ("resets_at", "limit_dollars", "used_dollars", "remaining_dollars"):
            value = window.get(field)
            if value is not None:
                entry[field] = value
        windows[name] = entry
    if not windows:
        return None
    limits = []
    raw_limits = payload.get("limits")
    if isinstance(raw_limits, list):
        for item in raw_limits:
            if not isinstance(item, Mapping):
                continue
            scope = item.get("scope")
            model = None
            if isinstance(scope, Mapping):
                scoped = scope.get("model")
                if isinstance(scoped, Mapping):
                    model = scoped.get("id") or scoped.get("display_name")
            limits.append(
                {
                    "kind": item.get("kind"),
                    "group": item.get("group"),
                    "percent": item.get("percent"),
                    "severity": item.get("severity"),
                    "is_active": item.get("is_active"),
                    "resets_at": item.get("resets_at"),
                    "scope_model": model,
                }
            )
    record = {
        "source": "oauth",
        "config_dir": config.name,
        "ts": now,
        "windows": windows,
    }
    if limits:
        record["limits"] = limits
    spend = payload.get("spend")
    if isinstance(spend, Mapping) and isinstance(spend.get("percent"), (int, float)):
        record["spend_percent"] = float(spend["percent"])
    return record


def snapshot_from_oauth(destination: Path, args: argparse.Namespace) -> int:
    configs = oauth_config_dirs(args.config_dir)
    if not configs:
        warn(
            "no config dir holds a .credentials.json with a claudeAiOauth block. "
            "On macOS the CLI keeps these in the login Keychain instead, which "
            "this tool does not read; see docs/drain.md"
        )
        return 1
    state = load_poll_state()
    version = detect_claude_version()
    now = time.time()
    written = 0
    for config in configs:
        label = config.name
        entry = dict(state.get(label) or {})
        if now < float(entry.get("blocked_until") or 0.0):
            warn("%s: backing off until %s" % (label, local_label(entry["blocked_until"])))
            continue
        if now - float(entry.get("last_attempt") or 0.0) < OAUTH_MIN_INTERVAL_SECONDS:
            warn("%s: polled less than %ds ago" % (label, int(OAUTH_MIN_INTERVAL_SECONDS)))
            continue
        token, expiry = read_oauth_token(config)
        if token is None:
            warn("%s: no usable claudeAiOauth token" % label)
            continue
        if expiry is not None and expiry <= now:
            warn("%s: oauth token expired; refresh is not attempted here" % label)
            entry["last_attempt"] = now
            state[label] = entry
            continue
        entry["last_attempt"] = now
        state[label] = entry
        try:
            payload = fetch_oauth_usage(token, version)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                entry["blocked_until"] = now + OAUTH_BACKOFF_SECONDS
                state[label] = entry
                warn("%s: rate limited; backing off %ds" % (label, int(OAUTH_BACKOFF_SECONDS)))
            else:
                warn("%s: usage request failed with HTTP %d" % (label, error.code))
            continue
        except Exception as error:  # network, timeout, malformed body
            warn("%s: usage request failed (%s)" % (label, type(error).__name__))
            continue
        finally:
            token = None
        record = oauth_snapshot_record(config, payload, now)
        if record is None:
            warn("%s: usage response carried no window utilisation" % label)
            continue
        append_snapshot(destination, record)
        written += 1
    save_poll_state(state)
    print("appended %d snapshot(s) to %s" % (written, destination))
    return 0


def command_snapshot(args: argparse.Namespace) -> int:
    destination = state_dir() / "snapshots.jsonl"
    if args.stdin:
        return snapshot_from_stdin(destination)
    if args.compact:
        return compact_snapshots(destination)
    if args.oauth:
        return snapshot_from_oauth(destination, args)
    return snapshot_from_configs(destination)


def snapshot_from_stdin(destination: Path) -> int:
    raw = b""
    try:
        raw = sys.stdin.buffer.read()
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
    except Exception:  # a closed pipe must not fail the statusline
        return 0
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
        config_env = os.environ.get("CLAUDE_CONFIG_DIR")
        config_label = (
            Path(config_env).expanduser().name if config_env else default_claude_config_label()
        )
        if windows == last_statusline_windows(destination, config_label):
            # The statusline runs on every prompt; only a change for THIS
            # account's config dir is news - comparing against the tail
            # record regardless of account would drop every other reading
            # once two config dirs start alternating (#9).
            return 0
        append_snapshot(
            destination,
            {
                "source": "statusline",
                "ts": time.time(),
                "windows": windows,
                "config_dir": config_label,
            },
        )
    except Exception:  # never break the statusline pipeline
        return 0
    return 0


def default_claude_config_label() -> str:
    """The basename of the first resolved Claude root.

    Used when `$CLAUDE_CONFIG_DIR` is unset, instead of hardcoding
    `.claude`: a host whose only configured Claude root is renamed (e.g.
    `.claude-arcade`) must still stamp its own label, not the literal
    default that happens to be right only for an unconfigured host.
    """
    roots = resolve_roots("claude", [], load_config(), quiet=True)
    return roots[0].name if roots else ".claude"


def last_statusline_windows(destination: Path, config_label: str) -> Optional[Dict[str, Any]]:
    """The windows of the newest statusline record for this config dir.

    Dedup must be account-blind-safe: two config dirs (two accounts) polling
    the statusline in alternation each keep their own "last seen" record, so
    comparing only within `config_label` never drops a genuine reading from
    the other account (#9).
    """
    try:
        with open(destination, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 65536))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if (
            isinstance(record, dict)
            and record.get("source") == "statusline"
            and record.get("config_dir") == config_label
        ):
            windows = record.get("windows")
            return windows if isinstance(windows, dict) else None
    return None


def compact_snapshots(destination: Path) -> int:
    """Drop repeats and anything past the retention window."""
    if not destination.is_file():
        print("no snapshots to compact")
        return 0
    cutoff = time.time() - SNAPSHOT_RETENTION_DAYS * 86400
    kept = []  # type: List[str]
    previous = {}  # type: Dict[Tuple[str, str], str]
    removed = 0
    try:
        with open(destination, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except ValueError:
                    removed += 1
                    continue
                epoch = record.get("ts")
                if isinstance(epoch, (int, float)) and epoch < cutoff:
                    removed += 1
                    continue
                key = (str(record.get("source") or ""), str(record.get("config_dir") or ""))
                shape = json.dumps(record.get("windows"), sort_keys=True)
                if previous.get(key) == shape:
                    removed += 1
                    continue
                previous[key] = shape
                kept.append(json.dumps(record, sort_keys=True))
    except OSError as error:
        warn("cannot compact %s: %s" % (destination, error))
        return 1
    temporary = destination.with_name(destination.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for line in kept:
            handle.write(line + "\n")
    os.replace(str(temporary), str(destination))
    print("kept %d snapshot(s), dropped %d" % (len(kept), removed))
    return 0


def snapshot_from_configs(destination: Path) -> int:
    last = last_snapshot_marks(destination)
    written = 0
    for root in resolve_roots("claude", [], load_config(), quiet=True):
        payload = read_claude_config(root / ".claude.json")
        if payload is None:
            continue
        cached = payload.get("cachedUsageUtilization")
        if not isinstance(cached, dict):
            continue
        fetched = cached.get("fetchedAtMs")
        if not isinstance(fetched, (int, float)):
            continue
        label = root.name
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
    parser.add_argument(
        "--harness", choices=("claude", "codex", "all"), default="all",
        help="transcript source to scan (default: all; calibrate defaults to codex "
             "unless overridden)",
    )
    parser.add_argument(
        "--since", default=None, metavar="WHEN",
        help="include events from this inclusive lower bound (7d, 12h, or ISO date/time); "
             "default: no lower bound",
    )
    parser.add_argument(
        "--until", default=None, metavar="WHEN",
        help="set the inclusive upper event-time cutoff (1d, 2026-09-10, or ISO date/time); "
             "default: no upper bound",
    )
    parser.add_argument("--claude-root", action="append", default=[], metavar="PATH",
                        help="a Claude home dir (holds .claude.json and projects/); "
                             "repeatable, replaces config.toml and the defaults")
    parser.add_argument("--codex-root", action="append", default=[], metavar="PATH",
                        help="a Codex home dir (holds auth.json and sessions/); "
                             "repeatable, replaces config.toml and the defaults")
    parser.add_argument("--account", default=None, metavar="LABEL",
                        help="limit to one account's pool, by root basename "
                             "(e.g. .codex-arcade); see `nenpi config`")
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="discard cached transcript parses and rebuild them",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="write the command result as indented JSON",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="disable ANSI colors, including the scan progress display",
    )
    parser.add_argument("--ascii", action="store_true",
                        help="draw bars with ASCII; automatic on non-UTF-8 stdout")
    parser.add_argument(
        "--width", type=int, default=None, metavar="COLUMNS",
        help="set text width in columns (default: terminal width, minimum 60)",
    )
    parser.add_argument(
        "--claude-cache-read-weight", type=float, default=None, metavar="RATIO",
        help="price Claude cache reads at this multiple of uncached input "
             "(default: API list ratio, usually 0.1)",
    )
    parser.add_argument(
        "--long-context-multiplier", type=float, default=1.0, metavar="MULTIPLIER",
        help="multiply requests over 200K tokens by this factor (default: 1.0)",
    )
    parser.add_argument(
        "--use-calibrated", action="store_true",
        help="use a stored usable calibration fit when one exists",
    )
    parser.add_argument("--window", default="auto", metavar="auto|five_hour|weekly|MINUTES",
                        help="quota window for drain attribution: auto, five_hour, weekly, "
                             "or window_minutes (default: auto)")
    parser.add_argument(
        "--top", type=int, default=25, metavar="N",
        help="limit rows or sessions shown (default: 25)",
    )
    parser.add_argument("--whole-session", action="store_true",
                        help="report each selected session's lifetime, not only events in the range")
    parser.add_argument(
        "--calibrate-bucket-hours", type=float,
        default=DEFAULT_CALIBRATION_BUCKET_HOURS, metavar="HOURS",
        help="time bucket size for calibration fitting (default: 2 hours)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nenpi",
        description=(
            "Measure subscription-plan quota used by Claude Code and Codex CLI sessions.\n"
            "Codex drain is measured from rate-limit snapshots; Claude drain is modelled\n"
            "from transcript usage at API list prices unless sampled."
        ),
        epilog=(
            "Examples:\n"
            "  nenpi sessions --since 7d --harness all\n"
            "  nenpi sessions --harness codex --window five_hour --sort tokens --top 10\n"
            "  nenpi prompts --session 0123abcd --since 7d\n"
            "  nenpi timeline --since 24h --bucket 1h\n"
            "\n"
            "Optional UI:\n"
            "  python -m pip install 'nenpi[ui]'\n"
            "  nenpi-ui                         # terminal UI (requires nenpi[ui])\n"
            "  python -m pip install 'nenpi[browser]'\n"
            "  nenpi-ui --browser --host 127.0.0.1 --port 8000\n"
            "  nenpi-web --host 127.0.0.1 --port 8000\n"
            "  Open http://127.0.0.1:8000/ in your browser.\n"
            "  HOST and PORT refer to the machine running nenpi."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sessions = sub.add_parser(
        "sessions", help="rank sessions by quota drain",
        description="Show one row per session, with fan-out and quota-drain details.",
        epilog="Example:\n  nenpi sessions --since 7d --sort tokens --top 20",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(sessions)
    sessions.add_argument(
        "--sort", choices=("drain", "tokens", "start"), default="drain",
        help="sort by within-harness drain share, token count, or newest start "
             "(default: drain)",
    )
    sessions.set_defaults(handler=command_sessions)

    timeline = sub.add_parser(
        "timeline", help="show quota activity over time",
        description="Aggregate weighted Claude units, Codex units, and observed quota by time bucket.",
        epilog="Example:\n  nenpi timeline --since 7d --bucket 5h",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(timeline)
    timeline.add_argument(
        "--bucket", choices=("1h", "5h", "1d"), default="1h",
        help="bucket width in hours or days (default: 1h)",
    )
    timeline.set_defaults(handler=command_timeline)

    windows = sub.add_parser(
        "windows", help="show observed Codex quota windows",
        description="List observed Codex windows, reset times, peak usage, and attributed sessions.",
        epilog="Example:\n  nenpi windows --since 30d --window weekly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(windows)
    windows.set_defaults(handler=command_windows)

    calibrate = sub.add_parser(
        "calibrate", help="fit token weights against measured drain",
        description="Fit percent-per-million-token weights from logged quota snapshots and save them.",
        epilog=(
            "Examples:\n"
            "  nenpi calibrate --harness codex --since 14d\n"
            "  nenpi calibrate --harness claude --since 30d --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(calibrate)
    calibrate.set_defaults(handler=command_calibrate, harness="codex")

    prompts = sub.add_parser(
        "prompts", help="inspect fan-out and context growth for one session",
        description="Show per-prompt input, context, turns, weighted units, and growth fits.",
        epilog="Example:\n  nenpi prompts --session 0123abcd --top 40 --since 7d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(prompts)
    prompts.add_argument(
        "--session", required=True, metavar="ID_PREFIX",
        help="required session ID prefix; if several match, use the busiest",
    )
    prompts.set_defaults(handler=command_prompts)

    fanout = sub.add_parser(
        "fanout", help="summarize turns and context per prompt",
        description="Show distributions and the 15 prompts with the most input tokens sent.",
        epilog="Example:\n  nenpi fanout --harness claude --since 7d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(fanout)
    fanout.set_defaults(handler=command_fanout)

    reductions = sub.add_parser(
        "reductions", help="find sharp context reductions",
        description="List points where a session context shrank by more than 30% across API calls.",
        epilog="Example:\n  nenpi reductions --since 30d --top 50",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(reductions)
    reductions.set_defaults(handler=command_reductions)

    verify = sub.add_parser(
        "verify", help="cross-check parsed totals",
        description="Compare parsed usage with each harness's in-band summary data.",
        epilog="Example:\n  nenpi verify --since 7d --harness all",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(verify)
    verify.set_defaults(handler=command_verify)

    snapshot = sub.add_parser(
        "snapshot", help="record a Claude quota-utilisation observation",
        description=(
            "Append Claude quota observations to the local snapshot log. "
            "Choose --stdin, --oauth, or --compact; with no mode, import cached CLI config data."
        ),
        epilog=(
            "Examples:\n"
            "  nenpi snapshot --stdin < statusline.json\n"
            "  nenpi snapshot --oauth --config-dir ~/.claude\n"
            "  nenpi snapshot --compact"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    snapshot.add_argument(
        "--stdin", action="store_true",
        help="read statusline JSON from stdin, echo it unchanged, and log changed rate limits",
    )
    snapshot.add_argument(
        "--oauth", action="store_true",
        help="read OAuth credentials and sample live utilisation (polls are at least 60s apart)",
    )
    snapshot.add_argument(
        "--compact", action="store_true",
        help="drop repeated, malformed, and older-than-60-day snapshot records",
    )
    snapshot.add_argument(
        "--config-dir", action="append", default=[], metavar="PATH",
        help="Claude config directory for --oauth; repeatable (default: discovered configs)",
    )
    snapshot.set_defaults(handler=command_snapshot)

    config_cmd = sub.add_parser(
        "config", help="show the config file and resolved harness roots in use"
    )
    config_cmd.add_argument("--claude-root", action="append", default=[], metavar="PATH")
    config_cmd.add_argument("--codex-root", action="append", default=[], metavar="PATH")
    config_cmd.add_argument("--init", action="store_true",
                            help="write a starter config.toml, seeded from every "
                                 "~/.claude*/~/.codex* dir found on this host")
    config_cmd.add_argument("--force", action="store_true",
                            help="with --init, overwrite an existing config.toml")
    config_cmd.add_argument("--json", action="store_true")
    config_cmd.set_defaults(handler=command_config)

    return parser


def command_config(args: argparse.Namespace) -> int:
    if args.init:
        return command_config_init(args)
    path = config_path()
    config = load_config()
    rows = []
    for harness in ("claude", "codex"):
        flags = args.claude_root if harness == "claude" else args.codex_root
        for root in resolve_roots(harness, flags, config, quiet=True, keep_missing=True):
            label, key = account_for_root(root, harness)
            rows.append(
                {
                    "harness": harness,
                    "path": str(root),
                    "exists": root.is_dir(),
                    "account_label": label,
                    "account_key": key,
                }
            )
    if args.json:
        print(
            json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "config_path": str(path),
                    "config_present": path.is_file(),
                    "roots": rows,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if path.is_file():
        print("config: %s" % path)
    else:
        print("config: %s (not present, using defaults)" % path)
    for row in rows:
        print(
            "%-6s %-40s exists=%-3s account=%-16s key=%s"
            % (
                row["harness"],
                row["path"],
                "yes" if row["exists"] else "no",
                row["account_label"],
                row["account_key"],
            )
        )
    return 0


def command_config_init(args: argparse.Namespace) -> int:
    path = config_path()
    if path.is_dir():
        warn("%s is a directory; cannot write config there" % path)
        return 1
    if path.exists() and not args.force:
        warn("%s already exists; pass --force to overwrite" % path)
        return 1
    existing = load_config(path) if path.is_file() else Config(path=path)
    # --init reseeds `roots` only: the user's other settings, the UI's
    # disabled/ignored bookkeeping, and any table this version does not know
    # about all survive an --init --force.
    fresh = Config(
        path=path,
        plan_claude=existing.plan_claude,
        plan_codex=existing.plan_codex,
        claude_disabled=existing.claude_disabled,
        codex_disabled=existing.codex_disabled,
        claude_ignored=existing.claude_ignored,
        codex_ignored=existing.codex_ignored,
        tables_present=existing.tables_present,
        extras=existing.extras,
    )
    comments = {}  # type: Dict[str, str]
    for harness in ("claude", "codex"):
        roots = fresh.roots(harness)
        excluded = {
            str(Path(item).expanduser())
            for item in list(fresh.disabled(harness)) + list(fresh.ignored(harness))
        }
        for root in discover_candidate_roots(harness):
            if str(root) in excluded:
                continue
            label, _key = account_for_root(root, harness)
            roots.append(str(root))
            comments[str(root)] = label
    save_config(
        fresh,
        path,
        comments=comments,
        header=[
            "# nenpi config, written by `nenpi config --init`.",
            "# Precedence: --claude-root/--codex-root flags > this file > defaults.",
            "# The Sources screen reads and writes this same file.",
        ],
    )
    print("wrote %s" % path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    migrate_dirs()
    # The statusline pipeline calls this thousands of times a day; get the
    # bytes moving before building the full parser.
    if arguments == ["snapshot", "--stdin"]:
        return snapshot_from_stdin(state_dir() / "snapshots.jsonl")
    # The Sources screen's old store is imported here too, so a CLI-only
    # user gets it as well (issue #23); it is a no-op once config.toml
    # exists, and stays out of the statusline path above.
    try:
        from .settings import migrate_json_store
    except ImportError:  # bench can load drain.py as a standalone module
        from nenpi.settings import migrate_json_store  # type: ignore
    migrate_json_store()
    parser = build_parser()
    args = parser.parse_args(arguments)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 2
    if getattr(args, "harness", None) == "claude" and args.command == "windows":
        warn("windows are a Codex-only measurement")
    try:
        return args.handler(args)
    except ScanCancelled:
        return 130


if __name__ == "__main__":
    sys.exit(main())
