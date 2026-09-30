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
import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import sys
import textwrap
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from nenpi.theme import DEFAULT_COLORS, ansi_styles, load_theme

from nenpi.config import (
    Config,
    account_for_root,
    cache_dir,
    config_dir,
    config_path,
    discover_candidate_roots,
    display_path,
    load_config,
    migrate_dirs,
    resolve_roots,
    save_config,
    set_notices_enabled,
    state_dir,
    unconfigured_roots,
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

CACHE_SCHEMA = 9
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

CLAUDE_KINDS = (
    "input", "cache_read", "cache_write_5m", "cache_write_1h", "output",
    "cache_write_unknown",
)
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

CODEX_LINE_MARKERS = (
    b'"session_meta"',
    b'"function_call',
    b'"custom_tool_call',
    b'"local_shell_call',
    b'"turn_context"',
    b'"token_usage_record"',
    b'"token_count"',
    b'"task_started"',
    b'"task_complete"',
    b'"compacted"',
    b'"role":"user"',
    b'"role": "user"',
    b'"role":"assistant"',
    b'"role": "assistant"',
)

# A Claude user line that carries a tool result is a fan-out step, not a new
# prompt, and its payload is the bulk of a transcript's bytes. Screening those
# out before json.loads keeps the scan cheap.
CLAUDE_USER_MARKERS = (b'"type":"user"', b'"type": "user"')
CLAUDE_NOT_A_PROMPT = (b'"toolUseResult"', b'"tool_result"')
# Tool bookkeeping reads the same lines the prompt screen rejects, but only
# for the tool's name and the SIZE of its result; no input or output text is
# kept, hashed, or printed.
# Only the result marker: an assistant line issuing a tool call already
# passes the `"usage"` screen, so admitting on `"tool_use"` never let a line
# through that was otherwise dropped - it only cost a second scan per line.
CLAUDE_TOOL_MARKERS = (b'"tool_result"',)
# A prompt label is the one human-typed first line of a user prompt, cut to
# PROMPT_LABEL_CHARS and scrubbed of anything secret-shaped. The prompt itself
# is never stored, printed, or hashed - see docs/drain.md, "What is stored".
PROMPT_LABEL_CHARS = 120
LABEL_SCAN_LINES = 64
REDACTED = "[redacted]"
# Blocks a harness injects into the user turn. Only these are dropped: any
# other `<...>` in a prompt is something the person typed and is kept.
LABEL_INJECTED_TAGS = frozenset((
    "system-reminder",
    "user_instructions",
    "user-instructions",
    "environment_context",
    "recommended_plugins",
    "pasted_content",
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
    "local-command-stderr",
    "local-command-caveat",
    "ide_selection",
    "ide_opened_file",
    "task-notification",
    "cross-session-message",
))
LABEL_TAG_OPEN = re.compile(r"<\s*([A-Za-z][\w.:-]*)[^>]*>")
# How much a label is worth, carried beside it rather than read back off its
# shape: a line the person typed that happens to be parenthesised is still
# typed text. Codex writes its injected context as its own user message just
# before the typed one, so both compete for the same prompt.
LABEL_RANK_NONE = 0
LABEL_RANK_BLOCK = 1
LABEL_RANK_TYPED = 2
# Attachment placeholders the harness substitutes for pasted bulk. Whatever
# follows one is the paste itself, so the scan stops there.
LABEL_PASTED_LINE = re.compile(
    r"^\[(?:pasted|image|attachment|screenshot|file|request interrupted)",
    re.IGNORECASE,
)
# Ordered: a broader pattern must not eat a narrower one's prefix.
LABEL_REDACTIONS = (
    re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{8,}"),
    re.compile(r"\bxox[abprse]-[A-Za-z0-9-]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]"),
    # Lookarounds, not \b: `_` is a word character, so `api_key_<32 hex>`
    # has no boundary before the run and would otherwise survive.
    re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{32,}(?![0-9a-fA-F])"),
    re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9+/])"),
)
CHARS_PER_TOKEN = 4.0
# Tools whose call starts a child agent, so the child's usage can be read back
# against the call that spawned it.
SPAWN_TOOL_NAMES = frozenset(
    ("task", "agent", "spawn_agent", "collaboration.spawn_agent")
)
MAX_PENDING_TOOLS = 512

ANSI = ansi_styles(DEFAULT_COLORS)

BAR_GLYPHS = "▏▎▍▌▋▊▉█"
ASCII_GLYPHS = {
    "█": "#", "▄": "=", "▀": "-", "▒": ":", "▁": ".",
    "▏": "|", "▎": "|", "▍": "|", "▌": "|", "▋": "|", "▊": "|", "▉": "|",
}


# --------------------------------------------------------------------------
# time helpers


_TS_CACHE: Dict[str, float] = {}


FRACTIONAL_SECONDS = re.compile(r"\.(\d+)")


def redact_label(text: str) -> str:
    """Replace secret-shaped substrings with `[redacted]`."""
    for pattern in LABEL_REDACTIONS:
        text = pattern.sub(REDACTED, text)
    return text


def strip_injected_spans(line: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Remove whole injected blocks from one line.

    Returns the text a person typed around them, the closing tag still owed
    when a block opened without closing on this line, and the name of the
    first injected block seen.
    """
    position = 0
    seen = None  # type: Optional[str]
    while True:
        match = LABEL_TAG_OPEN.search(line, position)
        if match is None:
            return line.strip(), None, seen
        name = match.group(1)
        if name.lower() not in LABEL_INJECTED_TAGS:
            # Something the person typed that happens to look like a tag.
            position = match.end()
            continue
        if seen is None:
            seen = name.lower()
        if match.group(0).rstrip().endswith("/>"):
            # `<ide_selection ... />` is a block with no body; drop the tag
            # and keep whatever the person typed around it.
            line = line[: match.start()] + " " + line[match.end():]
            position = match.start()
            continue
        closing = "</%s>" % name
        end = line.find(closing, match.end())
        if end < 0:
            return line[: match.start()].strip(), closing, seen
        line = line[: match.start()] + " " + line[end + len(closing):]
        position = match.start()


def prompt_label(text: Any) -> str:
    """The label alone; see `prompt_label_parts` for how it is built."""
    return prompt_label_parts(text)[0]


def prompt_label_parts(text: Any) -> Tuple[str, int]:
    """Return a short, redacted label for one user prompt, and its rank.

    Only what a person typed on the first such line survives: injected blocks
    are stripped, a pasted-content placeholder ends the scan so the paste can
    never become the label, the rest of the prompt is dropped, whitespace is
    collapsed, obvious secrets are redacted, and the result is cut to
    PROMPT_LABEL_CHARS. The full prompt is never returned, so nothing longer
    can reach the cache or the terminal.
    """
    if not isinstance(text, str) or not text:
        return "", LABEL_RANK_NONE
    lines = text.split("\n")[:LABEL_SCAN_LINES]
    skip_until = None  # type: Optional[str]
    # A turn that is nothing but an injected block - a task notification, a
    # slash-command expansion - is labelled with the block's name and nothing
    # from inside it. The name comes from LABEL_INJECTED_TAGS, so the label
    # stays a fixed vocabulary rather than transcript text.
    injected = None  # type: Optional[str]
    for position, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        if skip_until is not None:
            # Only a closing tag at the start of a line ends a block. A body
            # that quotes its own closing tag must not hand the rest of that
            # line back as a label.
            if not line.startswith(skip_until):
                continue
            line = line[len(skip_until):].strip()
            skip_until = None
            if not line:
                continue
        if LABEL_PASTED_LINE.match(line):
            # The next lines are the pasted body, not a prompt.
            return "", LABEL_RANK_NONE
        line, unterminated, tag = strip_injected_spans(line)
        if line:
            return (
                redact_label(" ".join(line.split()))[:PROMPT_LABEL_CHARS],
                LABEL_RANK_TYPED,
            )
        if injected is None and tag is not None:
            injected = tag
        if unterminated is not None:
            # If it never closes in what we are willing to read, the loop ends
            # and the block's NAME is the label; no line of its body can be.
            skip_until = unterminated
    return ("(%s)" % injected, LABEL_RANK_BLOCK) if injected else ("", LABEL_RANK_NONE)


def claude_prompt_text(message: Mapping[str, Any]) -> str:
    """The text blocks of one Claude user message, for `prompt_label` only."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            value = block.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


def codex_prompt_text(payload: Mapping[str, Any]) -> str:
    """The text blocks of one Codex user message, for `prompt_label` only."""
    content = payload.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("input_text", "text", "output_text"):
            value = block.get("text")
            if isinstance(value, str):
                parts.append(value)
    return "\n".join(parts)


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
            # Legacy aggregate-only transcripts did not expose a TTL split;
            # keep their historical 5-minute pricing as an explicit estimate.
            "cache_write_unknown": price[2],
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
    if kind == "cache_write_unknown":
        known_5m = entry.get("cache_write_5m")
        if isinstance(known_5m, (int, float)):
            return float(known_5m)
        price = CLAUDE_PRICES.get(normalize_claude_model(model))
        return float(price[2]) if price is not None else 0.0
    price = CLAUDE_PRICES.get(normalize_claude_model(model))
    if price is None or kind not in CLAUDE_KINDS:
        return 0.0
    return float(price[CLAUDE_KINDS.index(kind)])


class Weights:
    """Per-model price tables plus the provenance shown in the report header.

    Every lookup here is a pure function of `table`, and the analysis phase
    asks the same handful of questions a million times (one model per API
    call, a few dozen distinct models in a whole corpus). So the resolved
    entry and the per-kind price row are memoized; `invalidate()` drops the
    memos for anyone who mutates `table` in place after construction.
    """

    def __init__(self, table: Dict[str, Any], sources: Sequence[str]):
        # Treat this as frozen once constructed: the memos below are only
        # correct while it does not change, so any in-place edit must be
        # followed by `invalidate()`. Nothing in the tool edits it today.
        self.table = table
        self.sources = list(sources)
        self._entries = {}  # type: Dict[Tuple[str, str], Optional[Mapping[str, Any]]]
        self._price_rows = {}  # type: Dict[Tuple[str, str, Any], Optional[Tuple[float, ...]]]
        self._vectors = {}  # type: Dict[Tuple[str, str, Any, bool], Optional[Tuple[Any, ...]]]

    def invalidate(self) -> None:
        """Forget every memoized lookup; call after mutating `table` in place."""
        self._entries.clear()
        self._price_rows.clear()
        self._vectors.clear()

    @property
    def source_label(self) -> str:
        return " + ".join(self.sources) if self.sources else "default"

    def model_entry(self, harness: str, model: str) -> Optional[Mapping[str, Any]]:
        key = (harness, model)
        entries = self._entries
        if key in entries:
            return entries[key]
        section = self.table.get(harness) or {}
        models = section.get("models") or {}
        entry = models.get(model)
        resolved = entry if isinstance(entry, Mapping) else None
        entries[key] = resolved
        return resolved

    def price_row(
        self, harness: str, model: str, cache_read_weight: Optional[float] = None
    ) -> Optional[Tuple[float, ...]]:
        """Per-kind prices for one model, aligned to the harness' fit kinds.

        Claude rows line up with `CLAUDE_KINDS`, Codex rows with
        `CODEX_FIT_KINDS`. `None` means the model has no weight at all, which
        the callers report as zero units rather than as a zero price.
        """
        key = (harness, model, cache_read_weight)
        rows = self._price_rows
        if key in rows:
            return rows[key]
        entry = self.model_entry(harness, model)
        if entry is None:
            row = None  # type: Optional[Tuple[float, ...]]
        elif harness == "claude":
            cache_read = claude_price(entry, model, "cache_read")
            if cache_read_weight is not None:
                cache_read = claude_price(entry, model, "input") * cache_read_weight
            row = tuple(
                cache_read if kind == "cache_read" else claude_price(entry, model, kind)
                for kind in CLAUDE_KINDS
            )
        else:
            row = tuple(float(entry.get(kind, 0.0)) for kind in CODEX_FIT_KINDS)
        rows[key] = row
        return row

    def event_vector(
        self, harness: str, model: str, cache_read_weight: Optional[float] = None,
        input_only: bool = False,
    ) -> Optional[Tuple[Tuple[int, float], ...]]:
        """`(token slot, price)` pairs for pricing an event row in place.

        The slot is an absolute index into the event row's token area, so a
        hot loop can weigh a call without first materializing a token dict.
        `input_only` prices generated output at zero, which is exactly what
        `input_side_units` does by dropping "output" from its token dict.
        """
        key = (harness, model, cache_read_weight, input_only)
        vectors = self._vectors
        if key in vectors:
            return vectors[key]
        row = self.price_row(harness, model, cache_read_weight)
        if row is None:
            vector = None  # type: Optional[Tuple[Tuple[int, float], ...]]
        else:
            kinds = CLAUDE_KINDS if harness == "claude" else CODEX_FIT_KINDS
            slots = CLAUDE_FIT_SLOTS if harness == "claude" else CODEX_FIT_SLOTS
            vector = tuple(
                (slots[index], 0.0 if (input_only and kind == "output") else row[index])
                for index, kind in enumerate(kinds)
            )
        vectors[key] = vector
        return vector

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
        prices = self.price_row("claude", model, cache_read_weight)
        if prices is None:
            return 0.0
        total = 0.0
        for index, kind in enumerate(CLAUDE_KINDS):
            total += tokens.get(kind, 0) / 1_000_000.0 * prices[index]
        return total

    def codex_units(self, model: str, tokens: Mapping[str, int]) -> float:
        prices = self.price_row("codex", model)
        if prices is None:
            return 0.0
        total = 0.0
        for index, kind in enumerate(CODEX_FIT_KINDS):
            total += tokens.get(kind, 0) / 1_000_000.0 * prices[index]
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
            warn("--use-calibrated: no fit at %s; run `nenpi calibrate`" % fit_file)
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


def claude_agent_id(path: Path) -> str:
    """Return the opaque ID from ``subagents/agent-<id>.jsonl`` paths."""
    if path.parent.name != "subagents" or not path.stem.startswith("agent-"):
        return ""
    value = path.stem[len("agent-"):]
    return value if re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value) else ""


def claude_sidecar(path: Path) -> Tuple[str, Dict[str, Any]]:
    """Read only the safe structural fields from an agent metadata sidecar.

    The digest is used solely to notice any sidecar edit. No raw sidecar,
    prompt, or tool argument text enters the cache.
    """
    if not claude_agent_id(path):
        return "", {}
    sidecar = path.with_suffix(".meta.json")
    try:
        raw = sidecar.read_bytes()
    except OSError:
        return "", {}
    signature = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return signature, {}
    if not isinstance(payload, Mapping):
        return signature, {}

    def safe_id(value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value):
            return ""
        return value

    metadata: Dict[str, Any] = {}
    tool_use_id = safe_id(payload.get("toolUseId"))
    if tool_use_id:
        metadata["tool_use_id"] = tool_use_id
    depth = payload.get("spawnDepth")
    if isinstance(depth, int) and not isinstance(depth, bool) and 0 <= depth <= 128:
        metadata["depth"] = depth
    for source, target in (("agentType", "agent_type"), ("model", "model")):
        value = safe_id(payload.get(source))
        if value:
            metadata[target] = value
    return signature, metadata


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


def codex_transcripts(
    roots: Sequence[Path], cancellation: Cancellation = None
) -> Iterator[Tuple[Path, Path]]:
    """Yield every rollout file under resolved roots.

    The date directory records when a rollout opened, not when its latest
    event occurred. A session may be resumed days later, and imported roots
    may use another timezone, so the directory date cannot safely filter a
    `--since` query. `collect()` applies the file-mtime fast path before it
    opens or loads a transcript shard.
    """
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
        self.thread_usages = {}  # type: Dict[str, Dict[str, int]]
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
        if self.thread_usages:
            payload["thread_usages"] = self.thread_usages
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
        thread_usages = payload.get("thread_usages")
        if isinstance(thread_usages, Mapping):
            for thread_id, usage in thread_usages.items():
                if not isinstance(usage, Mapping):
                    continue
                summary.thread_usages[str(thread_id)] = dict(
                    (kind, int(usage.get(kind, 0) or 0)) for kind in CODEX_KINDS
                )
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
        for thread_id, usage in other.thread_usages.items():
            previous = self.thread_usages.get(thread_id)
            if previous is None:
                self.thread_usages[thread_id] = dict(usage)
            else:
                self.thread_usages[thread_id] = dict(
                    (kind, max(previous.get(kind, 0), usage.get(kind, 0)))
                    for kind in CODEX_KINDS
                )


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
        # One row per completed tool call; see the TOOL_* layout below.
        # Tool rows are a fifth of the cached bytes and only `tools`,
        # `prompts` and `fanout` ever look at them, so they live in a
        # sibling shard that `tools_loader` reads on first touch. `sessions`
        # and `timeline` never decode them.
        self._tools = []  # type: List[List[Any]]
        self.tools_loader = None  # type: Optional[Callable[[], List[List[Any]]]]
        # Row count and serialized size of the sibling, written into the
        # main shard so a missing, truncated or stale sibling is caught
        # instead of read back as "this transcript called no tools".
        self.tool_count = 0
        self.tool_bytes = 0
        # tool-call id -> [name, sidechain, spawn] for calls whose result has
        # not been read yet. Kept across incremental parses because a call and
        # its result can straddle the resume offset.
        self.pending_tools = []  # type: List[List[Any]]
        # A Codex user message arrives just before the task_started /
        # turn_context pair that opens its prompt, and can straddle the resume
        # offset, so the label waits here until a boundary claims it.
        self.pending_label = ""
        self.pending_label_rank = LABEL_RANK_NONE
        self.thread_id = ""
        self.is_subagent = False
        self.thread_metadata = {}  # type: Dict[str, Dict[str, Any]]
        self.message_rows = []  # type: List[List[Any]]
        # A sidecar edit invalidates its transcript shard even when the
        # transcript JSONL itself has not changed.
        self.claude_sidecar_signature = ""
        self.claude_agent_metadata = {}  # type: Dict[str, Any]
        self.pending_actions = []  # type: List[str]
        self.pending_response_seen = False
        self.head_hash = ""
        self.tail_hash = ""

    @property
    def tools(self) -> List[List[Any]]:
        loader = self.tools_loader
        if loader is not None:
            self.tools_loader = None
            self._tools = loader()
        return self._tools

    @tools.setter
    def tools(self, rows: List[List[Any]]) -> None:
        self.tools_loader = None
        self._tools = rows

    @property
    def tools_pending(self) -> bool:
        """True when tool rows exist on disk but have not been read."""
        return self.tools_loader is not None

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
            "tool_count": self.tool_count,
            "tool_bytes": self.tool_bytes,
            "pending_tools": self.pending_tools,
            "pending_label": self.pending_label,
            "pending_label_rank": self.pending_label_rank,
            "thread_id": self.thread_id,
            "is_subagent": self.is_subagent,
            "thread_metadata": self.thread_metadata,
            "message_rows": self.message_rows,
            "claude_sidecar_signature": self.claude_sidecar_signature,
            "claude_agent_metadata": self.claude_agent_metadata,
            "pending_actions": self.pending_actions,
            "pending_response_seen": self.pending_response_seen,
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
        index.tool_count = int(payload.get("tool_count") or 0)
        index.tool_bytes = int(payload.get("tool_bytes") or 0)
        index.pending_tools = list(payload.get("pending_tools") or [])
        index.pending_label = str(payload.get("pending_label", ""))
        index.pending_label_rank = int(payload.get("pending_label_rank") or 0)
        index.thread_id = str(payload.get("thread_id", ""))
        index.is_subagent = bool(payload.get("is_subagent"))
        index.thread_metadata = dict(payload.get("thread_metadata") or {})
        index.message_rows = list(payload.get("message_rows") or [])
        index.claude_sidecar_signature = str(payload.get("claude_sidecar_signature", ""))
        index.claude_agent_metadata = dict(payload.get("claude_agent_metadata") or {})
        index.pending_actions = [str(item) for item in payload.get("pending_actions") or []]
        index.pending_response_seen = bool(payload.get("pending_response_seen"))
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
EVENT_KIND_SLOTS = 6
EVENT_LONG = 9
EVENT_SUB = 10
EVENT_TURN = 11
EVENT_THREAD = 12
EVENT_ID = 13
EVENT_PROMPT = 14
EVENT_REASONING = 15
EVENT_ACTIONS = 16
EVENT_WAIT = 17
EVENT_REASONING_KNOWN = 18

# Absolute token-area indices for the kinds each harness actually prices, in
# the order its `*_units` method sums them. `Weights.event_vector` pairs these
# with prices so a hot loop can weigh a row without building a token dict.
CLAUDE_FIT_SLOTS = tuple(EVENT_KINDS + offset for offset in range(len(CLAUDE_KINDS)))
CODEX_FIT_SLOTS = tuple(EVENT_KINDS + CODEX_KINDS.index(kind) for kind in CODEX_FIT_KINDS)


def event_tokens(event: Sequence[Any], kinds: Sequence[str]) -> Dict[str, int]:
    return dict((kind, int(event[EVENT_KINDS + offset])) for offset, kind in enumerate(kinds))


def vector_units(vector: Sequence[Tuple[int, float]], event: Sequence[Any]) -> float:
    """Weighted units for one event row, summed in the harness' kind order.

    Arithmetic is term-for-term what `Weights.claude_units`/`codex_units` do
    on the equivalent token dict, so results stay bit-identical.
    """
    total = 0.0
    for slot, price in vector:
        total += int(event[slot]) / 1_000_000.0 * price
    return total


# Tool-call layout, one row per completed call:
# [session_id, timestamp, tool name, result characters, subagent, spawn, id]
# Only the name and the size are stored; the result text never leaves the
# parser, and the id is the harness' own opaque call id, used to count a
# replayed call once.
TOOL_SESSION, TOOL_TS, TOOL_NAME = 0, 1, 2
TOOL_CHARS, TOOL_SUB, TOOL_SPAWN, TOOL_ID, TOOL_THREAD = 3, 4, 5, 6, 7

# Command text is deliberately not part of a cached tool row.  `tools
# --explain` rereads the source transcript and uses this transient provenance
# to associate that text with the already deduplicated, scoped row.
EXPLAIN_MAX_EXAMPLES = 3
EXPLAIN_MAX_TOP_CALLS = 10
EXPLAIN_EXAMPLE_CHARS = 180
EXPLAIN_SHAPE_CHARS = 180
COMMAND_VALUE_OPTIONS = frozenset(
    (
        "-C", "-A", "-B", "-g", "-t", "-j", "-m", "-M", "-f", "-c",
        "--after-context", "--before-context", "--context", "--glob",
        "--type", "--type-not", "--threads", "--max-count", "--max-depth",
        "--color", "--sort", "--strip-ansi", "--exclude", "--include",
        "--file", "--regexp", "--pattern", "--workdir", "--timeout",
    )
)
COMMAND_OPTION_ALIASES = {
    "--line-number": "-n",
    "--no-heading": "-N",
    "--fixed-strings": "-F",
    "--hidden": "--hidden",
}
SHELL_TOOL_NAMES = frozenset(
    ("bash", "shell", "exec", "exec_command", "run_shell_command", "local_shell")
)
SHELL_OPERATORS = frozenset(("|", "|&", "&&", "||", ";", "&", ">", ">>", "<", "<<"))


def content_chars(value: Any) -> int:
    """Size of a tool result in characters, without keeping any of it.

    Text parts are measured directly; a structured part is measured by its
    compact JSON length, which is what the harness sends back to the model.
    """
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (int, float, bool)):
        return len(str(value))
    if isinstance(value, list):
        total = 0
        for part in value:
            total += content_chars(part)
        return total
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            return len(text)
        output = value.get("output")
        if isinstance(output, str):
            return len(output)
        # Everything else is measured by walking its strings rather than by
        # serializing it: a re-serialized megabyte of structured output costs
        # more than the parse did. The walk under-counts JSON punctuation and
        # key quoting by a few percent, which the /4 token estimate absorbs.
        total = 0
        for key, part in value.items():
            if isinstance(key, str):
                total += len(key)
            total += content_chars(part)
        return total
    return len(str(value))


def command_text(value: Any) -> str:
    """Turn a transcript command field into one shell-like string.

    Codex has emitted both a string command and argv-shaped lists over time;
    accepting both here keeps explanation mode useful across rollout versions.
    This value exists only during the requested source reread.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = [part for part in value if isinstance(part, (str, int, float))]
        if not parts:
            return ""
        return shlex.join([str(part) for part in parts])
    if isinstance(value, Mapping):
        for key in ("command", "cmd", "argv", "args"):
            if key in value:
                result = command_text(value.get(key))
                if result:
                    return result
    return ""


def command_from_payload(payload: Mapping[str, Any], item: str) -> str:
    """Extract a shell command from Claude or Codex's call payload."""
    if item == "local_shell_call":
        action = payload.get("action")
        result = command_text(action)
        if result:
            return result
    for key in ("command", "cmd", "argv"):
        result = command_text(payload.get(key))
        if result:
            return result
    for key in ("arguments", "input", "parameters"):
        value = payload.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                # A plain string is itself useful for custom shell tools.
                return value if item != "function_call" else ""
        result = command_text(value)
        if result:
            return result
    return ""


def source_tool_commands(
    path: Path, harness: str
) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(call id, session id, command)`` from one live transcript.

    This is intentionally a small, command-only rereader rather than a second
    full parser.  It never writes the extracted strings to a cache.
    """
    session_id = ""
    try:
        lines = read_lines_from(path, 0)
        for _position, raw in lines:
            if harness == "claude":
                # Result lines can contain arbitrarily large tool output;
                # the tool-use record itself carries the session id needed for
                # the provenance key, so no other Claude record is needed.
                if b'"tool_use"' not in raw:
                    continue
            else:
                # Keep the reread command-only.  In particular, do not JSON
                # decode function/custom/local-shell outputs, whose output
                # fields may be much larger than the call record.
                if any(marker in raw for marker in (
                    b'"function_call_output"',
                    b'"custom_tool_call_output"',
                    b'"local_shell_call_output"',
                )):
                    continue
                if (
                    b'"session_meta"' not in raw
                    and b'"function_call"' not in raw
                    and b'"local_shell_call"' not in raw
                ):
                    continue
            try:
                record = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(record, Mapping):
                continue
            if harness == "claude":
                candidate = record.get("sessionId")
                if isinstance(candidate, str) and candidate:
                    session_id = candidate
                if record.get("type") != "assistant":
                    continue
                message = record.get("message")
                if not isinstance(message, Mapping):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                        continue
                    name = block.get("name")
                    call_id = block.get("id")
                    if not isinstance(call_id, str) or not isinstance(name, str):
                        continue
                    if name.lower() != "bash":
                        continue
                    command = command_text(block.get("input"))
                    if command:
                        yield call_id, session_id, command
                continue
            kind = record.get("type")
            payload = record.get("payload")
            if not isinstance(payload, Mapping):
                continue
            if kind == "session_meta":
                candidate = payload.get("session_id") or payload.get("id")
                if isinstance(candidate, str) and candidate:
                    session_id = candidate
                continue
            if kind != "response_item":
                continue
            item = payload.get("type")
            if item not in CODEX_TOOL_CALL_ITEMS:
                continue
            call_id = payload.get("call_id") or payload.get("id")
            name = codex_tool_name(payload, str(item))
            if not isinstance(call_id, str) or not call_id:
                continue
            # local_shell_call is the canonical Codex shell item.  The
            # function-call names cover older and provider-backed shell APIs.
            if item != "local_shell_call" and (
                item != "function_call"
                or name.lower().split(".")[-1] not in SHELL_TOOL_NAMES
            ):
                continue
            command = command_from_payload(payload, str(item))
            if command:
                yield call_id, session_id, command
    except OSError:
        return


def escaped_command(value: str, limit: int = EXPLAIN_EXAMPLE_CHARS) -> str:
    """Make a command one-line and terminal-safe before displaying it."""
    output = []
    length = 0
    for char in value:
        code = ord(char)
        if char == "\\":
            piece = "\\\\"
        elif char == "\n":
            piece = "\\n"
        elif char == "\r":
            piece = "\\r"
        elif char == "\t":
            piece = "\\t"
        elif code < 0x20 or code == 0x7F:
            piece = "\\x%02x" % code
        else:
            piece = char
        if length + len(piece) > limit:
            return "".join(output)[: max(0, limit - 3)] + "..."
        output.append(piece)
        length += len(piece)
    return "".join(output)


def _command_tokens(command: str) -> Optional[List[str]]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>()")
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return None


def _simple_command_shape(tokens: Sequence[str]) -> str:
    if not tokens:
        return "<empty>"
    tokens = list(tokens)
    # The development wrapper is also commonly present in captured prompts;
    # grouping the underlying command makes `rtk rg` comparable with `rg`.
    while tokens and tokens[0] in ("rtk", "command"):
        wrapper = tokens.pop(0)
        if wrapper == "rtk" and tokens and tokens[0] == "proxy":
            tokens.pop(0)
    if tokens and tokens[0] == "env":
        tokens.pop(0)
        while tokens and ("=" in tokens[0] or tokens[0].startswith("-")):
            tokens.pop(0)
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)
    if not tokens:
        return "<empty>"
    executable = os.path.basename(tokens[0]) or tokens[0]
    options = []  # type: List[str]
    positional = []  # type: List[str]
    git_subcommand = False
    git_action = ""
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in SHELL_OPERATORS:
            index += 1
            continue
        if token.startswith("-") and token != "-":
            option = token
            value = None
            if "=" in option and option.startswith("--"):
                option, _ = option.split("=", 1)
                value = "<arg>"
            elif option in COMMAND_OPTION_ALIASES:
                option = COMMAND_OPTION_ALIASES[option]
            elif option.startswith("-") and not option.startswith("--") and len(option) > 2:
                # Short flag clusters have no semantic order; sorting their
                # letters makes `rg -nH` and `rg -Hn` one shape.
                option = "-" + "".join(sorted(option[1:]))
            if value is None and option in COMMAND_VALUE_OPTIONS and index + 1 < len(tokens):
                index += 1
                value = "<arg>"
            options.append(option if value is None else option + " <arg>")
        else:
            if executable == "git" and not git_subcommand:
                # Git's first non-option token is an actionable subcommand;
                # retain it while normalizing paths and revisions after it.
                git_action = token
                git_subcommand = True
            else:
                positional.append("<arg>")
        index += 1
    options.sort()
    prefix = [executable]
    if git_action:
        prefix.append(git_action)
    return " ".join(prefix + options + positional)[:EXPLAIN_SHAPE_CHARS]


def command_shape(command: str) -> str:
    """Normalize a shell command into a deterministic, readable shape."""
    normalized = command.strip()
    tokens = _command_tokens(normalized)
    if not tokens:
        # Keep a stable fallback for malformed/unterminated shell syntax.
        return "compound: " + escaped_command(command, EXPLAIN_SHAPE_CHARS - 10)
    inner = list(tokens)
    while inner and inner[0] in ("rtk", "command"):
        wrapper = inner.pop(0)
        if wrapper == "rtk" and inner and inner[0] == "proxy":
            inner.pop(0)
    if (len(inner) == 3 and os.path.basename(inner[0]) in ("sh", "bash", "zsh")
            and re.fullmatch(r"-[a-z]*c[a-z]*", inner[1])):
        return command_shape(inner[2])
    if "\n" in normalized:
        # shlex treats newlines as whitespace. Split them before tokenizing so
        # `rg foo\nsed bar` cannot silently turn into one `rg` shape.
        parts = [command_shape(line) for line in normalized.splitlines() if line.strip()]
        return "compound: " + " ; ".join(parts)[: EXPLAIN_SHAPE_CHARS - 10]
    if any(token in SHELL_OPERATORS for token in tokens):
        parts = []
        segment = []
        for token in tokens:
            if token in SHELL_OPERATORS:
                if segment:
                    parts.append(_simple_command_shape(segment))
                    segment = []
                parts.append(token)
            else:
                segment.append(token)
        if segment:
            parts.append(_simple_command_shape(segment))
        return "compound: " + " ".join(parts)[: EXPLAIN_SHAPE_CHARS - 10]
    return _simple_command_shape(tokens)


def explain_command_calls(
    scan: Scan, calls: Sequence[ToolCall]
) -> Tuple[List[Dict[str, Any]], List[ToolCall]]:
    """Attach live command text to already-scoped calls, without persistence."""
    from .tool_activity import _safe_example

    wanted = {
        (call.harness, call.call_id, call.session_id, call.source)
        for call in calls
        if call.call_id and call.source
    }
    found = {}  # type: Dict[Tuple[str, str, str, str], str]
    wanted_sources = {call.source for call in calls if call.call_id and call.source}
    sources = {
        (harness, str(source))
        for _entry, harness, source in scan._tool_sources
        if source is not None and str(source) in wanted_sources
    }
    for harness, source in sorted(sources):
        path = Path(source)
        for call_id, session_id, command in source_tool_commands(path, harness):
            key = (harness, call_id, session_id, source)
            if key in wanted and key not in found:
                found[key] = command
    records = []  # type: List[Tuple[ToolCall, str, str]]
    for call in calls:
        command = found.get((call.harness, call.call_id, call.session_id, call.source))
        if command:
            records.append((call, command, _safe_example(command_shape(command), EXPLAIN_SHAPE_CHARS)))
    grouped = {}  # type: Dict[str, Dict[str, Any]]
    for call, command, shape in records:
        row = grouped.get(shape)
        if row is None:
            row = {
                "shape": shape,
                "calls": 0,
                "result_chars": 0,
                "est_tokens": 0.0,
                "measured_tokens": 0.0,
                "mean_result_chars": 0.0,
                "max_result_chars": 0,
                "mean_est_tokens": 0.0,
                "max_est_tokens": 0.0,
                "examples": [],
            }
            grouped[shape] = row
        row["calls"] += 1
        row["result_chars"] += call.chars
        row["est_tokens"] += call.est_tokens
        row["measured_tokens"] += call.measured
        row["max_result_chars"] = max(row["max_result_chars"], call.chars)
        row["max_est_tokens"] = max(row["max_est_tokens"], call.est_tokens)
        if len(row["examples"]) < EXPLAIN_MAX_EXAMPLES:
            example = _safe_example(command, EXPLAIN_EXAMPLE_CHARS)
            if example not in row["examples"]:
                row["examples"].append(example)
    for row in grouped.values():
        calls_count = row["calls"]
        row["mean_result_chars"] = row["result_chars"] / calls_count if calls_count else 0.0
        row["mean_est_tokens"] = row["est_tokens"] / calls_count if calls_count else 0.0
    shapes = sorted(
        grouped.values(), key=lambda row: (row["est_tokens"], row["calls"], row["shape"]),
        reverse=True,
    )
    top_calls = sorted(
        records, key=lambda item: (item[0].chars, item[0].ts, item[1]), reverse=True
    )[:EXPLAIN_MAX_TOP_CALLS]
    top = []
    for call, command, shape in top_calls:
        top.append(
            {
                "command": _safe_example(command, EXPLAIN_EXAMPLE_CHARS),
                "shape": shape,
                "harness": call.harness,
                "session_id": call.session_id,
                "short_id": short_id(call.session_id),
                "prompt": call.prompt,
                "result_chars": call.chars,
                "est_tokens": call.est_tokens,
                "measured_tokens": call.measured,
            }
        )
    return shapes, top


def is_spawn_tool(name: str) -> bool:
    return name.lower() in SPAWN_TOOL_NAMES


def remember_tool(index: "FileIndex", call_id: str, name: str, sidechain: bool, thread_id: str = "") -> None:
    """Note an issued tool call so its result can be named when it arrives."""
    if not call_id or not name:
        return
    for row in index.pending_tools:
        if row[0] == call_id:
            return
    index.pending_tools.append(
        [call_id, name, 1 if sidechain else 0, 1 if is_spawn_tool(name) else 0, thread_id]
    )
    if len(index.pending_tools) > MAX_PENDING_TOOLS:
        del index.pending_tools[: len(index.pending_tools) - MAX_PENDING_TOOLS]


def expire_pending_tools(index: "FileIndex") -> None:
    """Forget issued calls whose result can no longer arrive.

    A user prompt starts a new turn, and a tool result for a call issued
    before it is never written afterwards - the harness has either recorded
    the result already or abandoned the call (interrupt, crash, rejected
    permission). Dropping them there keeps the pending map bounded by one
    turn's calls instead of decaying through the 512-entry FIFO trim, so a
    later result cannot be named after a long-abandoned call of the same id.
    """
    if index.pending_tools:
        del index.pending_tools[:]


def resolve_tool(index: "FileIndex", call_id: str) -> Tuple[str, int, int, str]:
    """Name the call a result answers, or report it as unmatched."""
    for position in range(len(index.pending_tools) - 1, -1, -1):
        row = index.pending_tools[position]
        if row[0] == call_id:
            del index.pending_tools[position]
            return str(row[1]), int(row[2]), int(row[3]), str(row[4]) if len(row) > 4 else ""
    return "unknown", 0, 0, ""


def record_tool(
    index: "FileIndex",
    session_id: str,
    epoch: Optional[float],
    call_id: str,
    chars: int,
    sidechain: bool,
    thread_id: str = "",
) -> None:
    if epoch is None or not session_id:
        return
    name, was_sub, spawn, pending_thread = resolve_tool(index, call_id)
    index.tools.append(
        [
            session_id,
            epoch,
            name,
            int(chars),
            1 if (sidechain or was_sub) else 0,
            spawn,
            call_id,
            pending_thread or thread_id or index.thread_id,
        ]
    )


def event_context(event: Sequence[Any], harness: str) -> int:
    """Tokens the harness re-sent to the API for this one call."""
    if harness == "claude":
        # input + cache_read + both known cache-write TTLs + unknown writes
        return int(event[3]) + int(event[4]) + int(event[5]) + int(event[6]) + int(event[8])
    # Codex splits input into uncached and cached; together they are the context.
    return int(event[3]) + int(event[4])


# --------------------------------------------------------------------------
# Claude parsing


def claude_cache_write_tokens(usage: Mapping[str, Any]) -> Tuple[int, int, int]:
    """Split cache creation while retaining any unclassified aggregate.

    Subdivisions are evidence only when the transcript actually carries them.
    Any aggregate remainder stays in ``cache_write_unknown`` and is not
    priced as though a TTL had been observed.
    """
    creation = usage.get("cache_creation")
    aggregate_value = usage.get("cache_creation_input_tokens")
    try:
        aggregate = max(0, int(aggregate_value or 0))
    except (TypeError, ValueError):
        aggregate = 0
    if not isinstance(creation, Mapping):
        return 0, 0, aggregate

    def bucket(key: str) -> Tuple[bool, int]:
        value = creation.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, 0
        return True, max(0, int(value))

    has_5m, write_5m = bucket("ephemeral_5m_input_tokens")
    has_1h, write_1h = bucket("ephemeral_1h_input_tokens")
    observed_total = write_5m + write_1h
    if aggregate == 0:
        # Older records sometimes omit the aggregate but carry TTL details.
        return write_5m, write_1h, 0
    if not has_5m and not has_1h:
        return 0, 0, aggregate
    if observed_total > aggregate:
        # The fields disagree, so preserve the authoritative disjoint total
        # and leave its TTL unknown instead of silently overstating a bucket.
        return 0, 0, aggregate
    return write_5m, write_1h, aggregate - observed_total


def derived_payload_bytes(value: Any) -> Optional[int]:
    """Measure a payload without retaining or printing its content."""
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, (Mapping, list, tuple)):
        try:
            encoded = json.dumps(
                value, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        return len(encoded)
    return None


def claude_tool_use_actions(content: Any) -> List[Dict[str, Any]]:
    """Keep tool names, opaque IDs, safe sizes, and explicit wait semantics."""
    if not isinstance(content, list):
        return []
    actions = []
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "tool_use":
            continue
        raw_name = block.get("name")
        name = (
            raw_name if isinstance(raw_name, str)
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", raw_name)
            else "unknown"
        )
        raw_id = block.get("id")
        call_id = (
            raw_id if isinstance(raw_id, str)
            and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", raw_id)
            else ""
        )
        arguments = block.get("input")
        action: Dict[str, Any] = {
            "name": name,
            "call_id": call_id,
            "input_bytes": derived_payload_bytes(arguments),
        }
        if name == "TaskOutput" and isinstance(arguments, Mapping):
            block_wait = arguments.get("block")
            if isinstance(block_wait, bool):
                action["blocking"] = block_wait
        actions.append(action)
    return actions


def merge_claude_actions(
    existing: Any, additions: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    merged = [dict(item) for item in existing if isinstance(item, Mapping)] if isinstance(existing, list) else []
    positions = {
        item.get("call_id"): index
        for index, item in enumerate(merged)
        if item.get("call_id")
    }
    for action in additions:
        item = dict(action)
        call_id = item.get("call_id")
        if call_id and call_id in positions:
            previous = merged[positions[call_id]]
            # Later streamed fragments may complete a size that was missing
            # in the first copy. Preserve the original ID and tool name.
            if isinstance(item.get("input_bytes"), int) and (
                not isinstance(previous.get("input_bytes"), int)
                or item["input_bytes"] > previous["input_bytes"]
            ):
                previous["input_bytes"] = item["input_bytes"]
            if "blocking" in item:
                previous["blocking"] = item["blocking"]
            continue
        merged.append(item)
        if call_id:
            positions[call_id] = len(merged) - 1
    return merged


def claude_wait_for_actions(actions: Any) -> bool:
    """Recognize only a lone TaskOutput call with explicit block=true."""
    return bool(
        isinstance(actions, list)
        and len(actions) == 1
        and isinstance(actions[0], Mapping)
        and actions[0].get("name") == "TaskOutput"
        and actions[0].get("blocking") is True
    )


def merge_claude_event_metadata(
    retained: List[Any], duplicate: Sequence[Any]
) -> None:
    """Merge safe metadata from a streamed/resumed duplicate into its first call."""
    if len(retained) <= EVENT_ACTIONS:
        retained.extend([None] * (EVENT_ACTIONS + 1 - len(retained)))
    additions = duplicate[EVENT_ACTIONS] if len(duplicate) > EVENT_ACTIONS else None
    retained[EVENT_ACTIONS] = merge_claude_actions(retained[EVENT_ACTIONS], additions or [])
    if len(retained) <= EVENT_WAIT:
        retained.extend([None] * (EVENT_WAIT + 1 - len(retained)))
    retained[EVENT_WAIT] = claude_wait_for_actions(retained[EVENT_ACTIONS])

    duplicate_known = (
        len(duplicate) > EVENT_REASONING_KNOWN
        and duplicate[EVENT_REASONING_KNOWN] is True
    )
    retained_known = (
        len(retained) > EVENT_REASONING_KNOWN
        and retained[EVENT_REASONING_KNOWN] is True
    )
    if duplicate_known and not retained_known:
        if len(retained) <= EVENT_REASONING_KNOWN:
            retained.extend([None] * (EVENT_REASONING_KNOWN + 1 - len(retained)))
        output_slot = EVENT_KINDS + CLAUDE_KINDS.index("output")
        retained_output = max(0, int(retained[output_slot]))
        retained[EVENT_REASONING] = min(
            max(0, int(duplicate[EVENT_REASONING])), retained_output
        )
        retained[EVENT_REASONING_KNOWN] = True


def claude_file_thread(path: Path, session_id: str, sidechain: bool) -> Tuple[str, str]:
    """Return a safe thread ID and its initial lineage class for one record."""
    child_id = claude_agent_id(path)
    if child_id:
        return child_id, "unknown"
    if sidechain:
        return "unknown:" + path.stem, "unknown"
    return session_id, "root"


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
    agent_id = claude_agent_id(path)
    is_subagent_file = bool(agent_id)
    seen_ids = set(index.carry_ids)
    event_by_id = {
        str(row[EVENT_ID]): row
        for row in index.events
        if len(row) > EVENT_ID and row[EVENT_ID]
    }
    recent = list(index.carry_ids)
    offset = index.offset
    for position, raw in read_lines_from(path, index.offset, cancellation):
        offset = position
        wants_usage = b'"usage"' in raw or b'"cost-state"' in raw
        wants_prompt = (
            any(marker in raw for marker in CLAUDE_USER_MARKERS)
            and not any(marker in raw for marker in CLAUDE_NOT_A_PROMPT)
        )
        # Tool lines are no longer dropped on the raw-bytes screen: they are
        # parsed for the tool's name and the size of its result only.
        wants_tool = any(marker in raw for marker in CLAUDE_TOOL_MARKERS)
        if not wants_usage and not wants_prompt and not wants_tool:
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
                    label, rank = prompt_label_parts(
                        claude_prompt_text(record.get("message") or {})
                    )
                    index.boundaries.append([session_id, epoch, label, rank])
                    summary.touch(epoch)
                expire_pending_tools(index)
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            epoch = parse_timestamp(record.get("timestamp"))
            sidechain = bool(record.get("isSidechain")) or is_subagent_file
            thread_id, classification = claude_file_thread(path, session_id, sidechain)
            if thread_id:
                index.thread_metadata.setdefault(thread_id, {
                    "thread_id": thread_id,
                    "session_id": session_id,
                    "parent_thread_id": None,
                    "parent_call_id": None,
                    "depth": None,
                    "classification": classification,
                })
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                call_id = block.get("tool_use_id")
                record_tool(
                    index,
                    session_id,
                    epoch,
                    call_id if isinstance(call_id, str) else "",
                    content_chars(block.get("content")),
                    sidechain,
                    thread_id,
                )
            continue
        if kind != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        sidechain = bool(record.get("isSidechain")) or is_subagent_file
        thread_id, classification = claude_file_thread(path, session_id, sidechain)
        actions = claude_tool_use_actions(content)
        if isinstance(content, list):
            for action in actions:
                call_id = action.get("call_id")
                name = action.get("name")
                if call_id and name:
                    remember_tool(index, str(call_id), str(name), sidechain, thread_id)
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        message_id = message.get("id") or record.get("requestId")
        if isinstance(message_id, str) and message_id:
            if message_id in seen_ids:
                existing = event_by_id.get(message_id)
                if existing is not None:
                    existing[EVENT_ACTIONS] = merge_claude_actions(
                        existing[EVENT_ACTIONS] if len(existing) > EVENT_ACTIONS else None,
                        actions,
                    )
                    if len(existing) <= EVENT_WAIT:
                        existing.extend([None] * (EVENT_WAIT + 1 - len(existing)))
                    existing[EVENT_WAIT] = claude_wait_for_actions(existing[EVENT_ACTIONS])
                    detail = usage.get("output_tokens_details")
                    thinking = detail.get("thinking_tokens") if isinstance(detail, Mapping) else None
                    if (
                        len(existing) <= EVENT_REASONING_KNOWN
                        or not existing[EVENT_REASONING_KNOWN]
                    ) and isinstance(thinking, (int, float)) and not isinstance(thinking, bool):
                        if len(existing) <= EVENT_REASONING_KNOWN:
                            existing.extend([None] * (EVENT_REASONING_KNOWN + 1 - len(existing)))
                        output_slot = EVENT_KINDS + CLAUDE_KINDS.index("output")
                        retained_output = max(0, int(existing[output_slot]))
                        existing[EVENT_REASONING] = min(
                            max(0, int(thinking)), retained_output
                        )
                        existing[EVENT_REASONING_KNOWN] = True
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

        write_5m, write_1h, write_unknown = claude_cache_write_tokens(usage)
        tokens = {
            "input": int(usage.get("input_tokens") or 0),
            "cache_read": int(usage.get("cache_read_input_tokens") or 0),
            "cache_write_5m": write_5m,
            "cache_write_1h": write_1h,
            "output": int(usage.get("output_tokens") or 0),
            "cache_write_unknown": write_unknown,
        }
        context_size = (
            tokens["input"] + tokens["cache_read"] + write_5m + write_1h + write_unknown
        )
        long_context = context_size > LONG_CONTEXT_THRESHOLD
        if epoch is not None:
            detail = usage.get("output_tokens_details")
            thinking = detail.get("thinking_tokens") if isinstance(detail, Mapping) else None
            reasoning_known = isinstance(thinking, (int, float)) and not isinstance(thinking, bool)
            reasoning = (
                min(max(0, int(thinking)), max(0, tokens["output"]))
                if reasoning_known else 0
            )
            event = [
                session_id, model, epoch,
                tokens["input"], tokens["cache_read"], write_5m, write_1h,
                tokens["output"], write_unknown,
                1 if long_context else 0, 1 if sidechain else 0,
                "", thread_id, message_id if isinstance(message_id, str) else "",
                None, reasoning, actions,
                claude_wait_for_actions(actions),
                bool(reasoning_known),
            ]
            index.events.append(event)
            if isinstance(message_id, str) and message_id:
                event_by_id[message_id] = event
            info = {
                "thread_id": thread_id,
                "session_id": session_id,
                "parent_thread_id": None,
                "parent_call_id": None,
                "depth": None,
                "classification": classification,
            }
            if agent_id:
                info["parent_call_id"] = index.claude_agent_metadata.get("tool_use_id")
                info["depth"] = index.claude_agent_metadata.get("depth")
                info["agent_type"] = index.claude_agent_metadata.get("agent_type")
                info["model"] = index.claude_agent_metadata.get("model") or model
                if info["parent_call_id"]:
                    info["classification"] = "lineage_pending"
                else:
                    info["classification"] = "unknown"
            index.thread_metadata[thread_id] = info
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
        "reasoning_output": min(
            int(usage.get("reasoning_output_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        ),
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
            first_thread_metadata = not index.thread_id
            if first_thread_metadata and isinstance(thread, str) and thread:
                index.thread_id = thread
            umbrella = payload.get("session_id")
            if isinstance(umbrella, str) and umbrella:
                session_id = umbrella
            elif isinstance(thread, str) and thread:
                session_id = thread
            if not session_id:
                session_id = path.stem
            if first_thread_metadata:
                index.is_subagent = is_codex_subagent(payload)
            parent = payload.get("parent_thread_id")
            source = payload.get("source")
            if isinstance(source, Mapping):
                spawn = source.get("subagent")
                if isinstance(spawn, Mapping):
                    parent = spawn.get("thread_spawn", {}).get("parent_thread_id", parent)
                    depth = spawn.get("thread_spawn", {}).get("depth")
                else:
                    depth = None
            else:
                depth = None
            if not isinstance(depth, int):
                depth = 1 if parent else 0
            metadata_is_subagent = is_codex_subagent(payload)
            if isinstance(thread, str) and thread and (
                first_thread_metadata or (metadata_is_subagent and not index.is_subagent)
            ):
                index.thread_id = thread
                index.is_subagent = metadata_is_subagent
            classification = "descendant" if isinstance(parent, str) and parent else (
                "unknown" if metadata_is_subagent else "root"
            )
            if isinstance(thread, str) and thread:
                index.thread_metadata[thread] = {
                    "thread_id": thread,
                    "session_id": session_id,
                    "parent_thread_id": parent if isinstance(parent, str) else None,
                    "depth": depth,
                    "classification": classification,
                }
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

        if kind == "response_item":
            item = payload.get("type")
            payload_thread = payload.get("thread_id")
            event_thread = (
                payload_thread
                if isinstance(payload_thread, str) and (not index.is_subagent or payload_thread != session_id)
                else (index.thread_id or path.stem)
            )
            action = codex_response_action(payload, item)
            if action:
                index.pending_response_seen = True
                index.pending_actions.append(action)
                if is_message_action(action):
                    target, payload_size = codex_message_metadata(payload)
                    index.message_rows.append([
                        session_id,
                        epoch,
                        event_thread,
                        action,
                        target,
                        payload_size,
                        str(payload.get("call_id") or payload.get("id") or ""),
                    ])
            if item in CODEX_TOOL_CALL_ITEMS:
                call_id = payload.get("call_id") or payload.get("id")
                name = codex_tool_name(payload, item)
                if isinstance(call_id, str):
                    remember_tool(index, call_id, name, index.is_subagent, event_thread)
            elif item == "message" and payload.get("role") == "user":
                # Only the label is taken; the message body is never kept.
                if not index.is_subagent:
                    label, rank = prompt_label_parts(codex_prompt_text(payload))
                    if rank > index.pending_label_rank:
                        index.pending_label = label
                        index.pending_label_rank = rank
            elif item in CODEX_TOOL_OUTPUT_ITEMS:
                call_id = payload.get("call_id") or payload.get("id")
                record_tool(
                    index,
                    session_id,
                    epoch if epoch is not None else codex_item_epoch(payload),
                    call_id if isinstance(call_id, str) else "",
                    content_chars(payload.get("output")),
                    index.is_subagent,
                    event_thread,
                )
            continue

        if kind == "turn_context":
            if epoch is not None and not index.is_subagent:
                add_boundary(index.boundaries, session_id, epoch,
                             index.pending_label, index.pending_label_rank)
                index.pending_label = ""
                index.pending_label_rank = LABEL_RANK_NONE
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
            actions = list(index.pending_actions) if index.pending_response_seen else None
            pure_wait = (
                bool(actions)
                and len(actions) == 1
                and actions[0].split(".")[-1] == "wait_agent"
            ) if actions is not None else None
            thread_usage = payload.get("thread_token_usage")
            payload_thread = payload.get("thread_id")
            event_thread = (
                payload_thread
                if isinstance(payload_thread, str) and (not index.is_subagent or payload_thread != session_id)
                else (index.thread_id or path.stem)
            )
            event_metadata = index.thread_metadata.get(event_thread) or {}
            event_sidechain = event_metadata.get("classification") != "root" if event_metadata else index.is_subagent
            if isinstance(thread_usage, dict):
                summary.thread_usage = codex_usage_tokens(thread_usage)
                summary.thread_usages[event_thread] = codex_usage_tokens(thread_usage)
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
                thread=event_thread,
                sidechain=event_sidechain,
                call_id=response_id if isinstance(response_id, str) else "",
                reasoning_output=tokens.get("reasoning_output", 0),
                actions=actions,
                pure_wait=pure_wait,
            )
            index.pending_actions = []
            index.pending_response_seen = False
            continue

        if kind == "event_msg" and payload.get("type") == "task_complete":
            # The turn is over. A user message that never opened a boundary of
            # its own - an interjection queued mid-turn - must not label the
            # next prompt.
            index.pending_label = ""
            index.pending_label_rank = LABEL_RANK_NONE
            continue

        if kind == "event_msg" and payload.get("type") == "task_started":
            if epoch is not None and not index.is_subagent:
                add_boundary(index.boundaries, session_id, epoch,
                             index.pending_label, index.pending_label_rank)
                index.pending_label = ""
                index.pending_label_rank = LABEL_RANK_NONE
            expire_pending_tools(index)
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


CODEX_TOOL_CALL_ITEMS = frozenset(
    ("function_call", "custom_tool_call", "local_shell_call")
)
CODEX_TOOL_OUTPUT_ITEMS = frozenset(
    ("function_call_output", "custom_tool_call_output", "local_shell_call_output")
)


def codex_tool_name(payload: Mapping[str, Any], item: str) -> str:
    """Tool name for a Codex call, namespace-qualified when one is given.

    MCP and collaboration tools arrive as a `namespace` plus a bare `name`;
    both are kept so `mcp.search` and `collaboration.send_message` stay
    distinct from a local tool of the same name.
    """
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        name = "shell" if item == "local_shell_call" else item
    namespace = payload.get("namespace")
    if isinstance(namespace, str) and namespace:
        return "%s.%s" % (namespace, name)
    return name


def codex_item_epoch(payload: Mapping[str, Any]) -> Optional[float]:
    metadata = payload.get("internal_chat_message_metadata_passthrough")
    if isinstance(metadata, Mapping):
        created = metadata.get("create_time")
        if isinstance(created, (int, float)):
            return float(created)
    return None


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


def codex_response_action(payload: Mapping[str, Any], item: Any) -> str:
    if item in CODEX_TOOL_CALL_ITEMS:
        return codex_tool_name(payload, str(item))
    if item == "message" and payload.get("role") == "assistant":
        return "message"
    return ""


def is_message_action(action: str) -> bool:
    return action.split(".")[-1] in {"send_message", "spawn_agent"}


def _decode_structural_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return None
        return decoded if isinstance(decoded, Mapping) else None
    return None


def codex_message_metadata(payload: Mapping[str, Any]) -> Tuple[str, Optional[int]]:
    missing = object()
    request = missing
    for key in ("arguments", "input", "action"):
        if key in payload and payload[key] is not None:
            request = payload[key]
            break
    if request is missing:
        return "unknown", None
    request_map = _decode_structural_payload(request)
    if request_map is None:
        request_map = {}
    nested = request_map
    for key in ("command", "input", "payload", "request"):
        decoded = _decode_structural_payload(nested.get(key))
        if decoded is not None:
            nested = decoded
    target = "unknown"
    for key in ("target_thread_id", "recipient_thread_id", "thread_id", "agent_id", "target"):
        value = nested.get(key)
        if isinstance(value, str) and value and "@" not in value and " " not in value:
            target = value[:128]
            break
    try:
        if isinstance(request, str):
            payload_size = len(request.encode("utf-8"))
        else:
            payload_size = len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError, UnicodeEncodeError):
        payload_size = None
    return target, payload_size


def add_boundary(
    boundaries: List[List[Any]], session_id: str, epoch: float, label: str = "",
    rank: int = LABEL_RANK_NONE,
) -> None:
    """Record a prompt start, collapsing the task_started/turn_context pair.

    Codex writes both within a second or two of each other for the same user
    turn, so a naive append would double every prompt. A label seen after the
    first of the pair still lands on the boundary it belongs to.
    """
    if boundaries:
        last = boundaries[-1]
        if last[0] == session_id and abs(epoch - last[1]) <= BOUNDARY_DEDUP_SECONDS:
            if len(last) > 3 and rank > last[3]:
                last[2], last[3] = label, rank
            return
    boundaries.append([session_id, epoch, label, rank])


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
    reasoning_output: int = 0,
    actions: Optional[Sequence[str]] = None,
    pure_wait: Optional[bool] = None,
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
    row.append(None)
    row.append(int(reasoning_output))
    row.append(list(actions) if actions is not None else None)
    row.append(pure_wait)
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
# phase timing


# `--profile` is a stderr-only stopwatch, not a profiler: it names the phases
# a run spends its time in so a regression shows up without cProfile. Off by
# default and free when off, so the TUI and the statusline path pay nothing.
_PROFILE = {"on": False, "phases": []}  # type: Dict[str, Any]


def profile_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "profile", False)) or os.environ.get("NENPI_PROFILE") == "1"


def profile_start(enabled: bool) -> None:
    """Arm the timer and drop whatever the previous run recorded.

    The TUI calls `prepare` once per refresh in one long-lived process, so
    the phase list has to be per-run; appending to a process-global list
    would grow without bound and report every refresh since startup.
    """
    _PROFILE["on"] = enabled
    _PROFILE["phases"] = []


@contextlib.contextmanager
def profile_phase(name: str) -> Iterator[None]:
    if not _PROFILE["on"]:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        _PROFILE["phases"].append((name, time.perf_counter() - started))


def profile_report() -> None:
    """Print this run's phases and clear them, so nothing accumulates."""
    phases = _PROFILE["phases"]
    _PROFILE["phases"] = []
    if not _PROFILE["on"] or not phases:
        return
    width = max(len(name) for name, _ in phases)
    for name, seconds in phases:
        sys.stderr.write("nenpi: %-*s %8.3fs\n" % (width, name, seconds))
    sys.stderr.flush()


# --------------------------------------------------------------------------
# cache


class CacheShardError(RuntimeError):
    """A cache shard disagrees with the sibling file it points at."""


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

    def tools_path(self, key: str) -> Path:
        """Sibling shard holding this transcript's tool rows.

        Tool rows are a fifth of the cached bytes and most commands never
        read one, so they are parked beside the shard instead of inside it:
        `sessions` then decodes only what it reports on.
        """
        return self.shard_path(key).with_suffix(".tools.json")

    @staticmethod
    def _load_tool_rows(path: Path, expected: int) -> List[List[Any]]:
        """Decode a tool sibling, or refuse to guess at what it should hold.

        `stored` already checked the sibling's size, so getting here with
        the wrong content means the cache was corrupted in a way a stat
        cannot see. Returning `[]` would report a transcript that called
        hundreds of tools as one that called none, for as long as its
        size and mtime stay put - so this raises instead.
        """
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CacheShardError(
                "cached tool rows in %s are unreadable (%s); "
                "re-run with --rebuild-cache" % (path, error)
            ) from None
        if not isinstance(rows, list) or len(rows) != expected:
            raise CacheShardError(
                "cached tool rows in %s do not match the shard that names "
                "them; re-run with --rebuild-cache" % path
            )
        return list(rows)

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
        if entry.tool_count:
            # The sibling holds rows this shard is accountable for, so it is
            # checked here rather than when something finally reads it: a
            # deleted, truncated, half-written or stale sibling has to make
            # the whole entry miss, so the transcript is parsed again from
            # the start. One stat keeps that check off the hot path - the
            # rows themselves are still only decoded on demand.
            tools_file = self.tools_path(key)
            try:
                if tools_file.stat().st_size != entry.tool_bytes:
                    return None
            except OSError:
                return None
            entry.tools_loader = (
                lambda path=tools_file, count=entry.tool_count: self._load_tool_rows(
                    path, count
                )
            )
        self.entries[key] = entry
        return entry

    def entry_for(
        self, path: Path, harness: str, stat: os.stat_result,
        sidecar_signature: str = "",
    ) -> Tuple[FileIndex, bool]:
        key = str(path)
        entry = self.stored(key, harness)
        if entry is None:
            entry = FileIndex(harness)
            self.entries[key] = entry
            return entry, True
        if harness == "claude" and entry.claude_sidecar_signature != sidecar_signature:
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
            expected = set()  # type: set
            for key in live:
                expected.add(self.shard_path(key).name)
                expected.add(self.tools_path(key).name)
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
            destination = self.shard_path(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            tools_file = self.tools_path(key)
            # A dirty entry was parsed, and parsing appends to `tools`, so
            # its rows are always loaded here; the guard only keeps an
            # untouched entry from being rewritten from an unread loader.
            if not entry.tools_pending:
                rows = entry.tools
                entry.tool_count = len(rows)
                if rows:
                    entry.tool_bytes = self._write_atomic(tools_file, rows)
                else:
                    # An empty list would cost a file and a decode per
                    # transcript that never called a tool.
                    entry.tool_bytes = 0
                    try:
                        tools_file.unlink()
                    except OSError:
                        pass
            payload = entry.to_json()
            payload["path"] = key
            # The main shard is written last so it is the commit point: it
            # is the only file that records what the sibling must contain,
            # so an interrupted flush leaves an orphan sibling (harmless,
            # pruned or overwritten later) rather than a shard pointing at
            # rows that were never written.
            self._write_atomic(destination, payload)
        self.dirty.clear()

    @staticmethod
    def _write_atomic(destination: Path, payload: Any) -> int:
        """Replace `destination` with `payload`; return the bytes written."""
        temporary = destination.with_name(
            "%s.%d.%d.tmp" % (destination.name, os.getpid(), time.time_ns())
        )
        try:
            blob = json.dumps(payload, separators=(",", ":"))
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(blob)
            os.replace(str(temporary), str(destination))
            return len(blob.encode("utf-8"))
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

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
        line = "nenpi: %s %d/%d" % (self.label, self.done, self.total)
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
        # Tool rows stay in their per-file shards until something asks for
        # them. `absorb` only records where they are; the dedup merge and
        # the range/account filters below replay on first read, in the same
        # order they were applied, so the rows come out identical to the
        # eager build. See `FileIndex.tools`.
        self._tool_sources = []  # type: List[Tuple[FileIndex, str, Optional[Path]]]
        self._tool_filters = []  # type: List[Callable[[str, List[List[Any]]], List[List[Any]]]]
        self._tools = None  # type: Optional[Dict[str, List[List[Any]]]]
        # `id(row)` is stable for the in-memory list owned by a FileIndex and
        # lets explain mode recover the exact source file after deduplication.
        # It is intentionally transient and never serialized.
        self.tool_provenance = {}  # type: Dict[int, str]
        self.claimed_tools = {}  # type: Dict[Tuple[str, str], str]
        self.snapshots = []  # type: List[Dict[str, Any]]
        self.boundaries = {}  # type: Dict[Tuple[str, str], List[float]]
        # (harness, session) -> {boundary epoch: redacted prompt label}. One
        # short line per prompt; see `prompt_label`.
        self.prompt_labels = {}  # type: Dict[Tuple[str, str], Dict[float, str]]
        self.compactions = {}  # type: Dict[Tuple[str, str], List[float]]
        self.claimed = {}  # type: Dict[Tuple[str, str], str]
        # First in-memory event row for a deduplicated message ID. Claude
        # resumed files can add safe tool-use metadata to that surviving row.
        self.claimed_event_rows = {}  # type: Dict[Tuple[str, str], List[Any]]
        self.forks = {}  # type: Dict[Tuple[str, str], Dict[str, int]]
        self.thread_metadata = {}  # type: Dict[Tuple[str, str, str], Dict[str, Any]]
        self.messages = []  # type: List[List[Any]]
        self.claimed_messages = set()  # type: set
        # (account key, account label) for every session seen, keyed the same
        # as `sessions`. Populated in `absorb` from the resolved root, never
        # from anything cached on disk - see the module docstring's note on
        # why account identity is not part of the per-file cache schema.
        self.session_accounts = {}  # type: Dict[Tuple[str, str], Tuple[str, str]]
        self.files_read = 0
        self.files_seen = 0
        self.bytes_read = 0

    def defer_tools(
        self, entry: "FileIndex", harness: str, source: Optional[Path] = None
    ) -> None:
        if self._tools is not None:
            # The merge has already run, so this file's rows would never
            # reach it and the report would quietly be short a transcript.
            raise RuntimeError(
                "tool rows were absorbed after the merged list was read"
            )
        self._tool_sources.append((entry, harness, source))

    def filter_tools(
        self, keep: "Callable[[str, List[List[Any]]], List[List[Any]]]"
    ) -> None:
        """Record a filter to apply when the tool rows are finally read."""
        if self._tools is None:
            self._tool_filters.append(keep)
            return
        for harness, rows in self._tools.items():
            self._tools[harness] = keep(harness, rows)

    @property
    def tools(self) -> Dict[str, List[List[Any]]]:
        if self._tools is None:
            merged = {"claude": [], "codex": []}  # type: Dict[str, List[List[Any]]]
            claimed = self.claimed_tools
            for entry, harness, source in self._tool_sources:
                kept = merged.setdefault(harness, [])
                for row in entry.tools:
                    tool_id = (harness, row[TOOL_ID]) if row[TOOL_ID] else None
                    if tool_id is not None:
                        if tool_id in claimed:
                            # The same call replayed into a resumed or
                            # forked transcript.
                            continue
                        claimed[tool_id] = row[TOOL_SESSION]
                    if source is not None:
                        self.tool_provenance[id(row)] = str(source)
                    kept.append(row)
            for keep in self._tool_filters:
                for harness, rows in merged.items():
                    merged[harness] = keep(harness, rows)
            self._tool_filters = []
            self._tools = merged
        return self._tools

    @tools.setter
    def tools(self, value: Dict[str, List[List[Any]]]) -> None:
        self._tools = value
        self._tool_filters = []


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
        for path, root in codex_transcripts(roots, run.cancellation):
            add_target(path, "codex", root)
    run.files_seen = len(targets)
    run.emit("discovery", message="found %d transcript files" % len(targets))

    stamped = []  # type: List[Tuple[float, str, Path, str, Path, os.stat_result, str, Dict[str, Any]]]
    live = set()
    for path, kind, root in targets:
        run.check()
        try:
            stat = path.stat()
        except OSError:
            continue
        live.add(str(path))
        sidecar_signature, sidecar_metadata = (
            claude_sidecar(path) if kind == "claude" else ("", {})
        )
        if since is not None and stat.st_mtime < since:
            # A transcript last written before the window cannot hold events
            # inside it, so its shard is never opened.
            continue
        stamped.append((stat.st_mtime, str(path), path, kind, root, stat,
                        sidecar_signature, sidecar_metadata))
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
    for _, _, path, kind, root, stat, sidecar_signature, sidecar_metadata in stamped:
        run.check()
        progress.step()
        run.emit("scanning", current_file=path.name, message="waiting for cache")
        with serialized_cache(cache.root, run.cancellation):
            entry, stale = cache.entry_for(path, kind, stat, sidecar_signature)
            if stale:
                previous_size = entry.size
                try:
                    if kind == "claude":
                        entry.claude_sidecar_signature = sidecar_signature
                        entry.claude_agent_metadata = sidecar_metadata
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
            absorb(scan, entry, kind, account, account_label, path)
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
    resolve_claude_thread_metadata(scan)
    run.emit("scanning", message="scanned %d files" % len(stamped))
    return scan


def resolve_claude_thread_metadata(scan: Scan) -> None:
    """Join Claude agent sidecars to the exact parent Agent/Task call ID."""
    call_threads = {}  # type: Dict[Tuple[str, str], str]
    for row in scan.events.get("claude", []):
        if len(row) <= EVENT_ACTIONS or not isinstance(row[EVENT_ACTIONS], list):
            continue
        session_id = str(row[EVENT_SESSION])
        thread_id = str(row[EVENT_THREAD] or "")
        for action in row[EVENT_ACTIONS]:
            if not isinstance(action, Mapping):
                continue
            name = str(action.get("name") or "")
            call_id = str(action.get("call_id") or "")
            if call_id and is_spawn_tool(name):
                call_threads[(session_id, call_id)] = thread_id
    for (harness, session_id, thread_id), info in scan.thread_metadata.items():
        if harness != "claude" or not info.get("parent_call_id"):
            continue
        parent = call_threads.get((session_id, str(info["parent_call_id"])))
        if parent:
            info["parent_thread_id"] = parent
            info["classification"] = "descendant"
        else:
            info["parent_thread_id"] = None
            info["classification"] = "unknown"


def absorb(
    scan: Scan, entry: FileIndex, harness: str, account: str, account_label: str,
    source: Optional[Path] = None,
) -> None:
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
    stamp = (account, account_label)
    session_accounts = scan.session_accounts
    stamped_sessions = set()  # type: set
    for row in entry.events:
        # Every event's session gets an account stamp here too, not only the
        # ones with a `SessionSummary` above: `attribute()` treats a session
        # missing from `session_accounts` as "any pool" (fail-open), so a
        # session known only through its events - never seen in
        # `entry.sessions` - must never be credited to every account's pool.
        row_session = row[EVENT_SESSION]
        if row_session not in stamped_sessions:
            stamped_sessions.add(row_session)
            session_accounts.setdefault((harness, row_session), stamp)
        call_id = (harness, row[EVENT_ID]) if row[EVENT_ID] else None
        if call_id is not None:
            owner = claimed.get(call_id)
            if owner is not None:
                # Same API call seen again: a resumed or forked transcript
                # replaying it. Count it once, against whoever recorded it first.
                if owner != row_session:
                    forks = scan.forks.setdefault((harness, row_session), {})
                    forks[owner] = forks.get(owner, 0) + 1
                elif harness == "claude":
                    retained = scan.claimed_event_rows.get(call_id)
                    if retained is not None:
                        merge_claude_event_metadata(retained, row)
                continue
            claimed[call_id] = row_session
            if harness == "claude":
                scan.claimed_event_rows[call_id] = row
        kept.append(row)
    for thread_id, metadata in entry.thread_metadata.items():
        metadata = dict(metadata)
        scan.thread_metadata[
            (harness, str(metadata.get("session_id") or entry.last_session or next(iter(entry.sessions), "")), str(thread_id))
        ] = metadata
    for row in entry.message_rows:
        message_id = row[6] if len(row) > 6 and row[6] else None
        key = (harness, message_id) if message_id else None
        if key is not None and key in scan.claimed_messages:
            continue
        if key is not None:
            scan.claimed_messages.add(key)
        scan.messages.append([harness] + list(row))
    # Keep the path only in this process.  It is needed by the opt-in command
    # explanation to reread command arguments, but must never enter a cache
    # shard or make a default report touch transcript content.
    scan.defer_tools(entry, harness, source)
    for row in entry.snapshots:
        stamped_row = dict(row)
        stamped_row["account"] = account
        stamped_row["account_label"] = account_label
        scan.snapshots.append(stamped_row)
    for row in entry.boundaries:
        key = (harness, str(row[0]))
        epoch = float(row[1])
        scan.boundaries.setdefault(key, []).append(epoch)
        label = str(row[2]) if len(row) > 2 and row[2] else ""
        if label:
            scan.prompt_labels.setdefault(key, {})[epoch] = label
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
    scan.filter_tools(
        lambda _harness, rows: [
            row
            for row in rows
            if (since is None or row[TOOL_TS] >= since)
            and (until is None or row[TOOL_TS] <= until)
        ]
    )


def rebuild_totals(scan: Scan, cancellation: Cancellation = None) -> None:
    """Derive per-session token totals and weighted units from surviving events."""
    for summary in scan.sessions.values():
        check_cancelled(cancellation)
        summary.models = {}
        summary.sub_models = {}
        summary.requests = 0
        summary.sub_requests = 0
    sessions = scan.sessions
    watch = cancellation is not None
    for harness, events in scan.events.items():
        check_cancelled(cancellation)
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        kind_slots = tuple(enumerate(kinds, start=EVENT_KINDS))
        for row in events:
            if watch:
                check_cancelled(cancellation)
            summary = sessions.get((harness, row[EVENT_SESSION]))
            if summary is None:
                continue
            model = row[EVENT_MODEL]
            bucket = summary.sub_models if row[EVENT_SUB] else summary.models
            slot = bucket.get(model)
            if slot is None:
                slot = empty_tokens(kinds)
                bucket[model] = slot
            for index, kind in kind_slots:
                slot[kind] += int(row[index])
            if row[EVENT_SUB]:
                summary.sub_requests += 1
                rollup = summary.models.get(model)
                if rollup is None:
                    rollup = empty_tokens(kinds)
                    summary.models[model] = rollup
                for index, kind in kind_slots:
                    rollup[kind] += int(row[index])
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
    watch = cancellation is not None
    for row in snapshots:
        if watch:
            check_cancelled(cancellation)
        key = (row.get("account") or "default", row.get("limit_id") or "codex",
               row.get("plan_type") or "unknown", row.get("window_minutes"))
        rows = grouped.get(key)
        if rows is None:
            grouped[key] = [row]
        else:
            rows.append(row)
    for rows in grouped.values():
        if watch:
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
    watch = cancellation is not None
    groups = snapshot_windows(snapshots, cancellation)
    for key, rows in groups.items():
        check_cancelled(cancellation)
        if not window_matches(key[3], window):
            continue
        current = UNSET  # type: Any
        running = 0.0
        anchor_ts = None  # type: Optional[float]
        for row in rows:
            if watch:
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
    # One price vector and one set of feature keys per model, resolved once
    # for the whole call: a corpus has a handful of Codex models and millions
    # of calls. Memoizing per *event* was tried and lost - the table costs
    # more in allocation than re-pricing a row costs in arithmetic.
    per_model = {}  # type: Dict[str, Tuple[Any, Tuple[Tuple[str, str], ...]]]
    cache_read_weight = args.claude_cache_read_weight
    multiplier = args.long_context_multiplier
    watch = cancellation is not None
    for interval in intervals:
        check_cancelled(cancellation)
        low = bisect.bisect_right(stamps, interval.start)
        high = bisect.bisect_right(stamps, interval.end)
        total = 0.0
        shares = {}  # type: Dict[str, float]
        prompt_shares = {}  # type: Dict[Tuple[str, Any], float]
        prompted_units = {}  # type: Dict[str, float]
        fallback_sessions = set()  # type: set
        features = interval.features
        account_label = interval.account
        for index in range(low, high):
            if watch:
                check_cancelled(cancellation)
            event = ordered[index]
            session_id = event[EVENT_SESSION]
            account = session_accounts.get(("codex", session_id))
            if account is not None and account[0] != account_label:
                continue
            model = event[EVENT_MODEL]
            resolved = per_model.get(model)
            if resolved is None:
                resolved = (
                    weights.event_vector("codex", model, cache_read_weight),
                    tuple((model, kind) for kind in CODEX_FIT_KINDS),
                )
                per_model[model] = resolved
            vector, feature_keys = resolved
            units = 0.0 if vector is None else vector_units(vector, event)
            if event[EVENT_LONG]:
                units *= multiplier
            used_fallback = False
            if units <= 0:
                # A model with no weight, or one whose fitted coefficient is
                # zero, still ran inside this interval; fall back to raw
                # tokens so it is never treated as free. That fallback's
                # scale (tokens * 1e-9) is nowhere near the weighted-unit
                # scale, so it is exempted from the cap below rather than
                # capped to near-zero.
                units = float(
                    sum(int(event[slot]) for slot in CODEX_FIT_SLOTS)
                ) * 1e-9
                used_fallback = True
            if units <= 0:
                continue
            total += units
            shares[session_id] = shares.get(session_id, 0.0) + units
            if used_fallback:
                fallback_sessions.add(session_id)
            if len(event) > EVENT_PROMPT and event[EVENT_PROMPT] is not None:
                prompt_key = (session_id, event[EVENT_PROMPT])
                prompt_shares[prompt_key] = prompt_shares.get(prompt_key, 0.0) + units
                prompted_units[session_id] = prompted_units.get(session_id, 0.0) + units
            for offset, feature in enumerate(feature_keys):
                features[feature] = (
                    features.get(feature, 0.0)
                    + int(event[CODEX_FIT_SLOTS[offset]]) / 1_000_000.0
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
        # The redacted one-line label of the user prompt; see `prompt_label`.
        self.label = ""
        # The session's working directory, basename only, as `cwd_label` cuts
        # it. "-" when the transcript never named one.
        self.cwd = "-"
        # Tool aggregates, filled by attribute_tools. Sizes only.
        self.tool_calls = 0
        self.tool_chars = 0
        self.tool_measured = 0.0
        self.top_tool = ""
        self.top_tool_chars = 0

    @property
    def kinds(self) -> Sequence[str]:
        return CLAUDE_KINDS if self.harness == "claude" else CODEX_KINDS

    @property
    def wall_seconds(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return max(0.0, self.end - self.start)

    @property
    def tool_est_tokens(self) -> float:
        """Approximate tokens added by this prompt's tool results."""
        return self.tool_chars / CHARS_PER_TOKEN

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
            "label": self.label,
            "cwd": self.cwd,
            "tool_calls": self.tool_calls,
            "tool_result_chars": self.tool_chars,
            "tool_est_tokens": self.tool_est_tokens,
            "tool_measured_tokens": self.tool_measured,
            "largest_tool": self.top_tool,
            "largest_tool_chars": self.top_tool_chars,
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


class ToolCall:
    """One completed tool call: its name, its result size, nothing else.

    `est_tokens` is an ESTIMATE from the result size (characters divided by
    CHARS_PER_TOKEN). `measured` is this call's share of the context growth
    actually billed on the next API call of the same thread, split across the
    results of that turn in proportion to their sizes; it is 0 when no later
    call was recorded and so nothing was measured.
    """

    __slots__ = (
        "harness", "session_id", "prompt", "ts", "name", "chars",
        "measured", "subagent", "spawn", "call_id", "source", "thread_id",
    )

    def __init__(self, harness: str, session_id: str, prompt: int, ts: float,
                 name: str, chars: int, subagent: bool, spawn: bool,
                 call_id: str = "", source: str = "", thread_id: str = ""):
        self.harness = harness
        self.session_id = session_id
        self.prompt = prompt
        self.ts = ts
        self.name = name
        self.chars = chars
        self.measured = 0.0
        self.subagent = subagent
        self.spawn = spawn
        self.call_id = call_id
        self.source = source
        self.thread_id = thread_id

    @property
    def est_tokens(self) -> float:
        return self.chars / CHARS_PER_TOKEN

    def to_json(self) -> Dict[str, Any]:
        return {
            "harness": self.harness,
            "session_id": self.session_id,
            "short_id": short_id(self.session_id),
            "prompt": self.prompt,
            "ts": self.ts,
            "tool": self.name,
            "result_chars": self.chars,
            "est_tokens": self.est_tokens,
            "measured_tokens": self.measured,
            "subagent": self.subagent,
            "spawned_subagent": self.spawn,
        }


def attribute_tools(
    scan: Scan, analysis: "Analysis", cancellation: Cancellation = None
) -> List[ToolCall]:
    """Tie each tool result to the prompt and the context growth it caused.

    Estimated context is the result size over CHARS_PER_TOKEN. Measured
    context is the input-token growth between two consecutive API calls of the
    same thread, split across the tool results recorded between them in
    proportion to their sizes - so a turn whose growth came from somewhere
    else (a pasted message, a re-read file) is not blamed on the tools, and a
    turn with no following call measures nothing.
    """
    calls = []  # type: List[ToolCall]
    for harness, tools in scan.tools.items():
        check_cancelled(cancellation)
        if not tools:
            continue
        streams = {}  # type: Dict[Tuple[str, int], List[List[Any]]]
        for row in tools:
            stream_id = (
                str(row[TOOL_THREAD]) if len(row) > TOOL_THREAD and row[TOOL_THREAD]
                else str(int(row[TOOL_SUB]))
            )
            streams.setdefault((row[TOOL_SESSION], stream_id), []).append(row)
        events = {}  # type: Dict[Tuple[str, int], List[List[Any]]]
        for row in scan.events.get(harness, []):
            stream_id = (
                str(row[EVENT_THREAD])
                if len(row) > EVENT_THREAD and row[EVENT_THREAD]
                else str(int(row[EVENT_SUB]))
            )
            events.setdefault((row[EVENT_SESSION], stream_id), []).append(row)
        for key, rows in events.items():
            check_cancelled(cancellation)
            rows.sort(key=lambda item: item[EVENT_TS])
            total = 0
            for position in range(1, len(rows)):
                step = (
                    event_context(rows[position], harness)
                    - event_context(rows[position - 1], harness)
                )
                if step > 0:
                    total += step
            growth_key = (harness, key[0])
            analysis.growth_totals[growth_key] = (
                analysis.growth_totals.get(growth_key, 0) + total
            )
        for key, rows in streams.items():
            check_cancelled(cancellation)
            rows.sort(key=lambda item: item[TOOL_TS])
            calls.extend(
                attribute_tool_stream(
                    harness, key[0], rows, events.get(key, []),
                    analysis.scan.tool_provenance,
                    key[1],
                )
            )
    calls.sort(key=lambda call: call.ts)
    attach_prompt_tools(analysis, calls)
    return calls


def attribute_tool_stream(
    harness: str, session_id: str, rows: Sequence[List[Any]],
    stream_events: Sequence[List[Any]],
    provenance: Optional[Mapping[int, str]] = None,
    thread_id: str = "",
) -> List[ToolCall]:
    starts = [row[EVENT_TS] for row in stream_events]
    calls = []  # type: List[ToolCall]
    buckets = {}  # type: Dict[int, List[ToolCall]]
    for row in rows:
        # bisect_left: a result stamped exactly at a call's time belongs to
        # that call, which is the one that carried it into context.
        position = bisect.bisect_left(starts, row[TOOL_TS])
        prompt = 0
        if stream_events:
            anchor = stream_events[min(position, len(stream_events) - 1)]
            index = anchor[EVENT_PROMPT] if len(anchor) > EVENT_PROMPT else None
            prompt = int(index) if isinstance(index, int) else 0
        call = ToolCall(
            harness, session_id, prompt, float(row[TOOL_TS]), str(row[TOOL_NAME]),
            int(row[TOOL_CHARS]), bool(row[TOOL_SUB]), bool(row[TOOL_SPAWN]),
            str(row[TOOL_ID]) if len(row) > TOOL_ID and row[TOOL_ID] else "",
            (provenance or {}).get(id(row), ""),
            thread_id or (str(row[TOOL_THREAD]) if len(row) > TOOL_THREAD else ""),
        )
        calls.append(call)
        if position < len(stream_events):
            buckets.setdefault(position, []).append(call)
    for position, bucket in buckets.items():
        if position == 0:
            # Nothing before it to measure growth against.
            continue
        growth = (
            event_context(stream_events[position], harness)
            - event_context(stream_events[position - 1], harness)
        )
        if growth <= 0:
            continue
        total = sum(call.chars for call in bucket)
        if total <= 0:
            continue
        for call in bucket:
            call.measured = growth * (call.chars / total)
    return calls


def _drill_bucket(thread_id: str, classification: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    metadata = metadata or {}
    return {
        "thread_id": thread_id or None,
        "parent_thread_id": metadata.get("parent_thread_id"),
        "depth": metadata.get("depth"),
        "classification": classification,
        "api_calls": 0,
        "uncached_input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "input_context_tokens": 0,
        "reasoning_output_tokens": 0,
        "cache_write_input_tokens": 0,
        "weighted_units": 0.0,
        "action_counts": {},
        "tool_family_rankings": [],
        "messages": {
            "calls": 0,
            "payload_bytes": 0,
            "known_payload_bytes": 0,
            "unknown_payload_calls": 0,
            "unknown_target_calls": 0,
            "routes": [],
        },
        "wait_streaks": [],
    }


def _drill_add_event(bucket: Dict[str, Any], row: Sequence[Any], weights: Weights) -> None:
    uncached = int(row[EVENT_KINDS])
    cached = int(row[EVENT_KINDS + 1])
    output = int(row[EVENT_KINDS + 3])
    bucket["api_calls"] += 1
    bucket["uncached_input_tokens"] += uncached
    bucket["cached_input_tokens"] += cached
    bucket["input_context_tokens"] += uncached + cached
    bucket["output_tokens"] += output
    bucket["reasoning_output_tokens"] += int(row[EVENT_REASONING]) if len(row) > EVENT_REASONING else 0
    bucket["cache_write_input_tokens"] += int(row[EVENT_KINDS + 2])
    actions = row[EVENT_ACTIONS] if len(row) > EVENT_ACTIONS else None
    if isinstance(actions, list):
        for action in actions:
            name = str(action)
            counts = bucket["action_counts"]
            counts[name] = counts.get(name, 0) + 1
    bucket["weighted_units"] += weights.codex_units(row[EVENT_MODEL], {
        "input": uncached, "cached_input": cached, "cache_write": int(row[EVENT_KINDS + 2]),
        "output": output,
    })


def _drill_wait_streaks(events: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    streaks = []
    current = []

    def flush() -> None:
        if not current:
            return
        first = event_context(current[0], "codex")
        last = event_context(current[-1], "codex")
        delta = last - first
        streaks.append({
            "length": len(current),
            "first_context": first,
            "last_context": last,
            "context_delta": delta,
            "mean_context_delta_per_transition": delta / (len(current) - 1) if len(current) > 1 else None,
            "uncached_input_tokens": sum(int(row[EVENT_KINDS]) for row in current),
            "cached_replay_tokens": sum(int(row[EVENT_KINDS + 1]) for row in current),
            "start": current[0][EVENT_TS],
            "end": current[-1][EVENT_TS],
        })
        del current[:]

    for row in sorted(events, key=lambda item: item[EVENT_TS]):
        if len(row) > EVENT_WAIT and row[EVENT_WAIT] is True:
            current.append(row)
        else:
            flush()
    flush()
    return streaks


def _drill_add_tools(bucket: Dict[str, Any], calls: Sequence[ToolCall]) -> None:
    grouped = {}
    for call in calls:
        key = call.name
        item = grouped.setdefault(key, {"tool": key, "calls": 0, "result_chars": 0, "measured_tokens": 0.0})
        item["calls"] += 1
        item["result_chars"] += call.chars
        item["measured_tokens"] += call.measured
    bucket["tool_family_rankings"] = sorted(
        grouped.values(), key=lambda item: (item["measured_tokens"], item["result_chars"], item["calls"]), reverse=True
    )


def _drill_add_messages(bucket: Dict[str, Any], rows: Sequence[Sequence[Any]]) -> None:
    routes = {}
    messages = bucket["messages"]
    for row in rows:
        target = row[5] if len(row) > 5 else "unknown"
        size = row[6] if len(row) > 6 else None
        messages["calls"] += 1
        if target == "unknown":
            messages["unknown_target_calls"] += 1
        if isinstance(size, int):
            messages["known_payload_bytes"] += size
        else:
            messages["unknown_payload_calls"] += 1
        key = str(target)
        route = routes.setdefault(key, {
            "target": target,
            "calls": 0,
            "payload_bytes": 0,
            "known_payload_bytes": 0,
            "unknown_payload_calls": 0,
        })
        route["calls"] += 1
        if isinstance(size, int):
            route["known_payload_bytes"] += size
        else:
            route["unknown_payload_calls"] += 1
    messages["payload_bytes"] = (
        None if messages["unknown_payload_calls"] else messages["known_payload_bytes"]
    )
    for route in routes.values():
        route["payload_bytes"] = (
            None if route["unknown_payload_calls"] else route["known_payload_bytes"]
        )
    messages["routes"] = sorted(
        routes.values(),
        key=lambda item: (item["calls"], item["known_payload_bytes"]),
        reverse=True,
    )


def _drill_thread_classification(
    metadata: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    session_id: str,
    thread_id: str,
    memo: Dict[str, str],
    visiting: Optional[set] = None,
    harness: str = "codex",
) -> str:
    if thread_id == session_id:
        return "root"
    if thread_id in memo:
        return memo[thread_id]
    visiting = set(visiting or ())
    if thread_id in visiting:
        memo[thread_id] = "unknown"
        return "unknown"
    visiting.add(thread_id)
    info = metadata.get((harness, session_id, thread_id))
    if not info:
        classification = "unknown"
    else:
        parent = info.get("parent_thread_id")
        if not isinstance(parent, str) or not parent:
            classification = "root" if info.get("classification") == "root" else "unknown"
        else:
            parent_classification = _drill_thread_classification(
                metadata, session_id, parent, memo, visiting, harness
            )
            classification = (
                "descendant"
                if parent_classification in {"root", "descendant"}
                else "unknown"
            )
    memo[thread_id] = classification
    return classification


def drilldown_for_prompt(analysis: "Analysis", prompt: Prompt) -> Dict[str, Any]:
    scan = analysis.scan
    rows = [
        row for row in scan.events.get("codex", [])
        if row[EVENT_SESSION] == prompt.session_id
        and len(row) > EVENT_PROMPT and row[EVENT_PROMPT] == prompt.index
    ]
    metadata = scan.thread_metadata
    partitions = {}
    classifications = {}

    def classify(thread: str) -> str:
        return _drill_thread_classification(
            metadata, prompt.session_id, thread, classifications
        )

    def ensure_bucket(thread: str) -> Dict[str, Any]:
        classification = classify(thread)
        key = (classification, thread)
        if key not in partitions:
            info = metadata.get(("codex", prompt.session_id, thread))
            if info is None and thread == prompt.session_id:
                info = {"thread_id": thread, "depth": 0, "classification": "root"}
            partitions[key] = _drill_bucket(thread, classification, info)
        return partitions[key]

    for row in rows:
        thread = str(row[EVENT_THREAD] or "")
        _drill_add_event(ensure_bucket(thread), row, analysis.weights)

    calls = [
        call for call in analysis.tool_calls
        if call.harness == "codex" and call.session_id == prompt.session_id and call.prompt == prompt.index
    ]
    for call in calls:
        ensure_bucket(call.thread_id)

    for key, bucket in partitions.items():
        thread = key[1]
        bucket["wait_streaks"] = _drill_wait_streaks(
            [row for row in rows if str(row[EVENT_THREAD] or "") == thread]
        )
        _drill_add_tools(bucket, [call for call in calls if call.thread_id == thread])
        message_rows = [
            row for row in scan.messages
            if row[0] == "codex" and row[1] == prompt.session_id and str(row[3] or "") == thread
            and prompt.start is not None and prompt.end is not None and prompt.start <= float(row[2]) <= prompt.end
        ]
        _drill_add_messages(bucket, message_rows)

    root = next((bucket for (kind, _), bucket in partitions.items() if kind == "root"), _drill_bucket("", "root"))
    descendants = [
        bucket for (kind, _), bucket in sorted(partitions.items()) if kind == "descendant"
    ]
    unknown_threads = [
        bucket for (kind, _), bucket in sorted(partitions.items()) if kind == "unknown"
    ]
    unknown = _drill_bucket("", "unknown")
    for bucket in unknown_threads:
        for field in ("api_calls", "uncached_input_tokens", "cached_input_tokens", "input_context_tokens", "output_tokens", "reasoning_output_tokens", "cache_write_input_tokens"):
            unknown[field] += bucket[field]
        unknown["weighted_units"] += bucket["weighted_units"]
        for action, count in bucket["action_counts"].items():
            unknown["action_counts"][action] = unknown["action_counts"].get(action, 0) + count
        unknown["wait_streaks"].extend(bucket["wait_streaks"])
    unknown["wait_streaks"].sort(key=lambda streak: (streak["start"], streak["end"]))
    unknown_calls = [call for call in calls if classify(call.thread_id) == "unknown"]
    _drill_add_tools(unknown, unknown_calls)
    unknown_message_rows = [
        row for row in scan.messages
        if row[0] == "codex" and row[1] == prompt.session_id
        and classify(str(row[3] or "")) == "unknown"
        and prompt.start is not None and prompt.end is not None
        and prompt.start <= float(row[2]) <= prompt.end
    ]
    _drill_add_messages(unknown, unknown_message_rows)
    all_buckets = [root] + descendants + [unknown]
    combined = _drill_bucket("combined", "combined")
    for bucket in all_buckets:
        for field in ("api_calls", "uncached_input_tokens", "cached_input_tokens", "input_context_tokens", "output_tokens", "reasoning_output_tokens", "cache_write_input_tokens"):
            combined[field] += bucket[field]
        combined["weighted_units"] += bucket["weighted_units"]
        for action, count in bucket["action_counts"].items():
            combined["action_counts"][action] = combined["action_counts"].get(action, 0) + count
    combined_message_rows = [
        row for row in scan.messages
        if row[0] == "codex" and row[1] == prompt.session_id
        and prompt.start is not None and prompt.end is not None
        and prompt.start <= float(row[2]) <= prompt.end
    ]
    _drill_add_messages(combined, combined_message_rows)
    prompt_tokens = prompt.tokens
    reconciliation = {
        "matches_prompt": combined["api_calls"] == prompt.turns
        and combined["uncached_input_tokens"] == prompt_tokens.get("input", 0)
        and combined["cached_input_tokens"] == prompt_tokens.get("cached_input", 0)
        and combined["output_tokens"] == prompt_tokens.get("output", 0)
        and abs(combined["weighted_units"] - prompt.units) < 1e-9,
        "prompt_api_calls": prompt.turns,
        "prompt_tokens": dict(prompt_tokens),
        "combined_tokens": {key: combined[key] for key in ("uncached_input_tokens", "cached_input_tokens", "output_tokens")},
    }
    ranking_sources = [root] + descendants
    ranking_sources = [bucket for bucket in ranking_sources if bucket["api_calls"] or bucket["thread_id"]]
    ranking_sources.sort(key=lambda bucket: (bucket["weighted_units"], bucket["uncached_input_tokens"]), reverse=True)
    agent_rankings = []
    for rank, bucket in enumerate(ranking_sources, 1):
        agent_rankings.append({
            "rank": rank,
            "classification": bucket["classification"],
            "thread_id": bucket["thread_id"],
            "weighted_units": bucket["weighted_units"],
            "uncached_input_tokens": bucket["uncached_input_tokens"],
            "api_calls": bucket["api_calls"],
        })
    return {"root": root, "descendants": descendants, "unknown": unknown,
            "unknown_threads": unknown_threads,
            "combined": combined, "reconciliation": reconciliation,
            "agent_rankings": agent_rankings,
            "caveats": ["descendant membership is temporal", "measured tool growth is an upper bound", "unknown lineage is not inferred"],
            "attribution": "explicit_lineage_with_temporal_prompt_membership"}


def _claude_drill_bucket(
    thread_id: str, classification: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    return {
        "thread_id": thread_id or None,
        "parent_thread_id": metadata.get("parent_thread_id"),
        "parent_tool_use_id": metadata.get("parent_call_id"),
        "depth": metadata.get("depth"),
        "agent_type": metadata.get("agent_type"),
        "model": metadata.get("model"),
        "classification": classification,
        "api_calls": 0,
        "tokens": empty_tokens(CLAUDE_KINDS),
        "uncached_input_tokens": 0,
        "cached_input_tokens": 0,
        "input_context_tokens": 0,
        "cache_write_input_tokens": 0,
        "cache_write_5m_input_tokens": 0,
        "cache_write_1h_input_tokens": 0,
        "cache_write_unknown_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "known_reasoning_output_tokens": 0,
        "reasoning_unknown_calls": 0,
        "fallback_priced_cache_write_tokens": 0,
        "weighted_units": 0.0,
        "action_counts": {},
        "tool_uses": {
            "calls": 0,
            "known_input_bytes": 0,
            "unknown_input_sizes": 0,
            "families": [],
        },
        "tool_family_rankings": [],
        "wait_evidence": {
            "task_output_calls": 0,
            "blocking_task_output_calls": 0,
            "unclassified_task_output_calls": 0,
        },
        "wait_streaks": [],
    }


def _claude_drill_wait_streaks(events: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    streaks = []
    by_thread = {}  # type: Dict[str, List[Sequence[Any]]]
    for row in events:
        thread_id = str(row[EVENT_THREAD] or "")
        by_thread.setdefault(thread_id, []).append(row)
    for thread_id, thread_events in by_thread.items():
        current = []

        def flush() -> None:
            if not current:
                return
            first = event_context(current[0], "claude")
            last = event_context(current[-1], "claude")
            delta = last - first
            streaks.append({
                "thread_id": thread_id or None,
                "kind": "TaskOutput(block=true)",
                "length": len(current),
                "first_context": first,
                "last_context": last,
                "context_delta": delta,
                "mean_context_delta_per_transition": (
                    delta / (len(current) - 1) if len(current) > 1 else None
                ),
                "uncached_input_tokens": sum(int(row[3]) for row in current),
                "cached_replay_tokens": sum(int(row[4]) for row in current),
                "cache_write_tokens": sum(
                    int(row[5]) + int(row[6]) + int(row[8]) for row in current
                ),
                "start": current[0][EVENT_TS],
                "end": current[-1][EVENT_TS],
            })
            current.clear()

        for row in sorted(thread_events, key=lambda item: item[EVENT_TS]):
            if len(row) > EVENT_WAIT and row[EVENT_WAIT] is True:
                current.append(row)
            else:
                flush()
        flush()
    return sorted(streaks, key=lambda item: (item["start"], item["end"], item["thread_id"] or ""))


def _claude_drill_add_events(
    bucket: Dict[str, Any], events: Sequence[Sequence[Any]],
    weights: Weights, args: argparse.Namespace,
    children_by_call: Mapping[Tuple[str, str], str],
) -> None:
    families = {}
    for row in events:
        tokens = event_tokens(row, CLAUDE_KINDS)
        bucket["api_calls"] += 1
        for kind, amount in tokens.items():
            bucket["tokens"][kind] += amount
        bucket["uncached_input_tokens"] += tokens["input"]
        bucket["cached_input_tokens"] += tokens["cache_read"]
        write_total = (
            tokens["cache_write_5m"] + tokens["cache_write_1h"]
            + tokens["cache_write_unknown"]
        )
        context = tokens["input"] + tokens["cache_read"] + write_total
        bucket["input_context_tokens"] += context
        bucket["cache_write_input_tokens"] += write_total
        bucket["cache_write_5m_input_tokens"] += tokens["cache_write_5m"]
        bucket["cache_write_1h_input_tokens"] += tokens["cache_write_1h"]
        bucket["cache_write_unknown_input_tokens"] += tokens["cache_write_unknown"]
        bucket["output_tokens"] += tokens["output"]
        bucket["fallback_priced_cache_write_tokens"] += tokens["cache_write_unknown"]
        known_reasoning = (
            len(row) > EVENT_REASONING_KNOWN and row[EVENT_REASONING_KNOWN] is True
        )
        if known_reasoning:
            bucket["known_reasoning_output_tokens"] += int(row[EVENT_REASONING])
        else:
            bucket["reasoning_unknown_calls"] += 1
        unit = weights.claude_units(
            row[EVENT_MODEL], tokens, args.claude_cache_read_weight
        )
        if row[EVENT_LONG]:
            unit *= args.long_context_multiplier
        bucket["weighted_units"] += unit

        actions = row[EVENT_ACTIONS] if len(row) > EVENT_ACTIONS else None
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            name = str(action.get("name") or "unknown")
            counts = bucket["action_counts"]
            counts[name] = counts.get(name, 0) + 1
            item = families.setdefault(name, {
                "tool": name, "calls": 0, "known_input_bytes": 0,
                "unknown_input_sizes": 0, "targets": [],
                "unknown_targets": 0,
            })
            item["calls"] += 1
            payload_size = action.get("input_bytes")
            if isinstance(payload_size, int):
                item["known_input_bytes"] += payload_size
                bucket["tool_uses"]["known_input_bytes"] += payload_size
            else:
                item["unknown_input_sizes"] += 1
                bucket["tool_uses"]["unknown_input_sizes"] += 1
            bucket["tool_uses"]["calls"] += 1
            if name.lower() in SPAWN_TOOL_NAMES:
                call_id = str(action.get("call_id") or "")
                target = children_by_call.get((str(row[EVENT_SESSION]), call_id))
                if target:
                    if target not in item["targets"]:
                        item["targets"].append(target)
                else:
                    item["unknown_targets"] += 1
            if name == "TaskOutput":
                evidence = bucket["wait_evidence"]
                evidence["task_output_calls"] += 1
                if action.get("blocking") is True:
                    evidence["blocking_task_output_calls"] += 1
                elif action.get("blocking") is not False:
                    evidence["unclassified_task_output_calls"] += 1
    bucket["tool_uses"]["families"] = sorted(
        families.values(),
        key=lambda item: (item["calls"], item["known_input_bytes"]),
        reverse=True,
    )
    bucket["wait_streaks"] = _claude_drill_wait_streaks(events)
    bucket["reasoning_output_tokens"] = (
        None if bucket["reasoning_unknown_calls"]
        else bucket["known_reasoning_output_tokens"]
    )


def drilldown_for_claude_prompt(
    analysis: "Analysis", prompt: Prompt
) -> Dict[str, Any]:
    scan = analysis.scan
    events = [
        row for row in scan.events.get("claude", [])
        if row[EVENT_SESSION] == prompt.session_id
        and len(row) > EVENT_PROMPT and row[EVENT_PROMPT] == prompt.index
    ]
    metadata = scan.thread_metadata
    classifications = {}

    def classify(thread: str) -> str:
        return _drill_thread_classification(
            metadata, prompt.session_id, thread, classifications, harness="claude"
        )

    def ensure_bucket(thread: str) -> Dict[str, Any]:
        classification = classify(thread)
        key = (classification, thread)
        if key not in partitions:
            info = metadata.get(("claude", prompt.session_id, thread))
            if info is None and thread == prompt.session_id:
                info = {"thread_id": thread, "depth": 0, "classification": "root"}
            partitions[key] = _claude_drill_bucket(thread, classification, info)
        return partitions[key]

    partitions = {}
    for row in events:
        ensure_bucket(str(row[EVENT_THREAD] or ""))
    calls = [
        call for call in analysis.tool_calls
        if call.harness == "claude" and call.session_id == prompt.session_id
        and call.prompt == prompt.index
    ]
    for call in calls:
        ensure_bucket(call.thread_id)

    children_by_call = {}
    for (harness, session_id, thread_id), info in metadata.items():
        if harness != "claude" or session_id != prompt.session_id:
            continue
        call_id = info.get("parent_call_id")
        if call_id and info.get("parent_thread_id"):
            children_by_call[(session_id, str(call_id))] = thread_id

    for (classification, thread), bucket in partitions.items():
        thread_events = [row for row in events if str(row[EVENT_THREAD] or "") == thread]
        thread_calls = [call for call in calls if call.thread_id == thread]
        _claude_drill_add_events(
            bucket, thread_events, analysis.weights, analysis.args, children_by_call
        )
        _drill_add_tools(bucket, thread_calls)

    root = next(
        (bucket for (kind, _), bucket in partitions.items() if kind == "root"),
        _claude_drill_bucket(prompt.session_id, "root", {"depth": 0}),
    )
    descendants = [
        bucket for (kind, _), bucket in sorted(partitions.items()) if kind == "descendant"
    ]
    unknown_threads = [
        bucket for (kind, _), bucket in sorted(partitions.items()) if kind == "unknown"
    ]
    unknown = _claude_drill_bucket("", "unknown")
    unknown_events = [row for row in events if classify(str(row[EVENT_THREAD] or "")) == "unknown"]
    unknown_calls = [call for call in calls if classify(call.thread_id) == "unknown"]
    _claude_drill_add_events(
        unknown, unknown_events, analysis.weights, analysis.args, children_by_call
    )
    _drill_add_tools(unknown, unknown_calls)

    combined = _claude_drill_bucket("combined", "combined")
    _claude_drill_add_events(
        combined, events, analysis.weights, analysis.args, children_by_call
    )
    _drill_add_tools(combined, calls)
    prompt_tokens = dict(prompt.tokens)
    combined_tokens = dict(combined["tokens"])
    reconciliation = {
        "matches_prompt": combined["api_calls"] == prompt.turns
        and all(combined_tokens.get(kind, 0) == prompt_tokens.get(kind, 0)
                for kind in CLAUDE_KINDS)
        and abs(combined["weighted_units"] - prompt.units) < 1e-9,
        "prompt_api_calls": prompt.turns,
        "prompt_tokens": prompt_tokens,
        "combined_tokens": combined_tokens,
    }
    ranking_sources = [root] + descendants
    ranking_sources = [bucket for bucket in ranking_sources if bucket["api_calls"] or bucket["thread_id"]]
    ranking_sources.sort(
        key=lambda bucket: (bucket["weighted_units"], bucket["uncached_input_tokens"]),
        reverse=True,
    )
    agent_rankings = [
        {
            "rank": rank,
            "classification": bucket["classification"],
            "thread_id": bucket["thread_id"],
            "parent_thread_id": bucket["parent_thread_id"],
            "agent_type": bucket["agent_type"],
            "model": bucket["model"],
            "weighted_units": bucket["weighted_units"],
            "input_context_tokens": bucket["input_context_tokens"],
            "api_calls": bucket["api_calls"],
        }
        for rank, bucket in enumerate(ranking_sources, 1)
    ]
    caveats = [
        "agent lineage uses exact sidecar toolUseId matches; missing or unmatched lineage remains unknown",
        "agent membership is temporal to this prompt; exact lineage does not establish prompt ownership for background calls",
        "tool result context growth is an upper bound and size-based token counts are estimates",
        "wait streaks include only explicit TaskOutput calls with block=true",
    ]
    if combined["cache_write_unknown_input_tokens"]:
        caveats.append(
            "cache creation without a complete TTL breakdown uses the historical 5-minute price as an estimate"
        )
    if combined["reasoning_unknown_calls"]:
        caveats.append(
            "thinking output is a known subtotal because some usage records omit its token count"
        )
    return {
        "root": root,
        "descendants": descendants,
        "unknown": unknown,
        "unknown_threads": unknown_threads,
        "combined": combined,
        "reconciliation": reconciliation,
        "agent_rankings": agent_rankings,
        "caveats": caveats,
        "attribution": "explicit_toolUseId_lineage_with_temporal_prompt_membership",
        "wait_semantics": "TaskOutput block=true only; missing block evidence is unclassified",
    }


def render_claude_drilldown(drilldown: Mapping[str, Any]) -> None:
    print("prompt drilldown: exact sidecar toolUseId lineage with temporal prompt membership")
    entries = [("root", drilldown["root"])]
    entries += [("descendant", item) for item in drilldown["descendants"]]
    entries += [("unknown thread", item) for item in drilldown.get("unknown_threads", [])]
    entries.append(("unknown aggregate", drilldown["unknown"]))
    for label, bucket in entries:
        tokens = bucket["tokens"]
        thinking = (
            str(bucket["reasoning_output_tokens"])
            if bucket["reasoning_output_tokens"] is not None
            else "partial"
        )
        print("%s %s parent=%s depth=%s agent_type=%s model=%s" % (
            label, bucket["thread_id"] or "-", bucket["parent_thread_id"] or "-",
            bucket["depth"] if bucket["depth"] is not None else "unknown",
            bucket["agent_type"] or "-", bucket["model"] or "-",
        ))
        print("  totals: calls=%d input=%d cache_read=%d write_5m=%d write_1h=%d write_unknown=%d output=%d thinking=%s known_thinking=%d thinking_unknown_calls=%d units=%.4f" % (
            bucket["api_calls"], tokens["input"], tokens["cache_read"],
            tokens["cache_write_5m"], tokens["cache_write_1h"],
            tokens["cache_write_unknown"], tokens["output"], thinking,
            bucket["known_reasoning_output_tokens"], bucket["reasoning_unknown_calls"],
            bucket["weighted_units"],
        ))
        print("  actions: %s" % (
            ", ".join("%s=%d" % item for item in sorted(bucket["action_counts"].items()))
            or "none"
        ))
        uses = bucket["tool_uses"]
        families = ", ".join(
            "%s calls=%d input_bytes=%d unknown_input=%d targets=%s unknown_targets=%d" % (
                item["tool"], item["calls"], item["known_input_bytes"],
                item["unknown_input_sizes"], "/".join(item["targets"]) or "-",
                item["unknown_targets"],
            ) for item in uses["families"]
        ) or "none"
        print("  tool uses: calls=%d known_input_bytes=%d unknown_input_sizes=%d families=%s" % (
            uses["calls"], uses["known_input_bytes"], uses["unknown_input_sizes"], families,
        ))
        print("  tool results: %s" % (
            ", ".join("%s calls=%d measured=%.1f chars=%d" % (
                item["tool"], item["calls"], item["measured_tokens"], item["result_chars"]
            ) for item in bucket["tool_family_rankings"]) or "none"
        ))
        evidence = bucket["wait_evidence"]
        print("  wait evidence: TaskOutput calls=%d blocking=%d unclassified=%d" % (
            evidence["task_output_calls"], evidence["blocking_task_output_calls"],
            evidence["unclassified_task_output_calls"],
        ))
        for streak in bucket["wait_streaks"]:
            print("  wait streak: kind=%s length=%d first_context=%d last_context=%d context_delta=%d start=%s end=%s" % (
                streak["kind"], streak["length"], streak["first_context"],
                streak["last_context"], streak["context_delta"],
                streak["start"], streak["end"],
            ))
    combined = drilldown["combined"]
    tokens = combined["tokens"]
    thinking = (
        str(combined["reasoning_output_tokens"])
        if combined["reasoning_output_tokens"] is not None else "partial"
    )
    print("combined totals: calls=%d input=%d cache_read=%d write_5m=%d write_1h=%d write_unknown=%d output=%d thinking=%s known_thinking=%d thinking_unknown_calls=%d units=%.4f" % (
        combined["api_calls"], tokens["input"], tokens["cache_read"],
        tokens["cache_write_5m"], tokens["cache_write_1h"],
        tokens["cache_write_unknown"], tokens["output"],
        thinking, combined["known_reasoning_output_tokens"],
        combined["reasoning_unknown_calls"], combined["weighted_units"],
    ))
    reconciliation = drilldown["reconciliation"]
    print("reconciliation: %s" % ("ok" if reconciliation["matches_prompt"] else "mismatch"))
    print("caveats: " + "; ".join(drilldown["caveats"]))


def render_drilldown(drilldown: Mapping[str, Any]) -> None:
    if str(drilldown.get("attribution", "")).startswith("explicit_toolUseId_"):
        render_claude_drilldown(drilldown)
        return
    print("prompt drilldown: explicit lineage with temporal prompt membership")
    print("agent rankings: weighted units (then uncached input)")
    rankings = drilldown.get("agent_rankings") or []
    for item in rankings:
        print("  #%d %-10s thread=%s units=%.4f input=%d calls=%d" % (
            item["rank"], item["classification"], item["thread_id"] or "unknown",
            item["weighted_units"], item["uncached_input_tokens"], item["api_calls"],
        ))
    entries = [("root", drilldown["root"])]
    entries += [("descendant", item) for item in drilldown["descendants"]]
    entries += [("unknown thread", item) for item in drilldown.get("unknown_threads", [])]
    entries.append(("unknown aggregate", drilldown["unknown"]))
    for label, bucket in entries:
        print("%s agent/thread=%s parent=%s depth=%s" % (
            label, bucket.get("thread_id") or "unknown", bucket.get("parent_thread_id") or "unknown", bucket.get("depth", "unknown")
        ))
        print("  totals: calls=%d uncached_input=%d cached_replay=%d context=%d output=%d reasoning_output=%d cache_write=%d units=%.4f" % (
            bucket["api_calls"], bucket["uncached_input_tokens"], bucket["cached_input_tokens"], bucket["input_context_tokens"],
            bucket["output_tokens"], bucket["reasoning_output_tokens"], bucket["cache_write_input_tokens"], bucket["weighted_units"],
        ))
        print("  actions: %s" % (", ".join("%s=%d" % item for item in sorted(bucket["action_counts"].items())) or "none"))
        print("  tools: %s" % (", ".join("%s calls=%d measured=%.1f chars=%d" % (
            item["tool"], item["calls"], item["measured_tokens"], item["result_chars"]
        ) for item in bucket["tool_family_rankings"]) or "none"))
        messages = bucket["messages"]
        payload_bytes = messages["payload_bytes"] if messages["payload_bytes"] is not None else "unknown/partial"
        routes = ", ".join(
            "%s calls=%d payload_bytes=%s known_payload_bytes=%d unknown_payloads=%d" % (
                route["target"], route["calls"],
                route["payload_bytes"] if route["payload_bytes"] is not None else "unknown/partial",
                route["known_payload_bytes"], route["unknown_payload_calls"],
            ) for route in messages["routes"]
        ) or "none"
        print("  messages: calls=%d payload_bytes=%s known_payload_bytes=%d unknown_targets=%d unknown_payloads=%d routes=%s" % (
            messages["calls"], payload_bytes, messages["known_payload_bytes"],
            messages["unknown_target_calls"], messages["unknown_payload_calls"], routes,
        ))
        for streak in bucket["wait_streaks"]:
            print("  wait streak: length=%d first_context=%d last_context=%d context_delta=%d mean_delta=%s uncached_input=%d cached_replay=%d start=%s end=%s" % (
                streak["length"], streak["first_context"], streak["last_context"], streak["context_delta"],
                streak["mean_context_delta_per_transition"] if streak["mean_context_delta_per_transition"] is not None else "n/a",
                streak["uncached_input_tokens"], streak["cached_replay_tokens"], streak["start"], streak["end"],
            ))
    reconciliation = drilldown["reconciliation"]
    print("combined totals: calls=%d uncached_input=%d cached_replay=%d output=%d reasoning_output=%d units=%.4f" % (
        drilldown["combined"]["api_calls"], drilldown["combined"]["uncached_input_tokens"], drilldown["combined"]["cached_input_tokens"],
        drilldown["combined"]["output_tokens"], drilldown["combined"]["reasoning_output_tokens"], drilldown["combined"]["weighted_units"],
    ))
    print("reconciliation: %s" % ("ok" if reconciliation["matches_prompt"] else "mismatch"))
    print("caveats: descendant membership is temporal; measured tool growth is an upper bound; unknown lineage is not inferred")


def attach_prompt_tools(analysis: "Analysis", calls: Sequence[ToolCall]) -> None:
    for call in calls:
        prompts = analysis.prompts.get((call.harness, call.session_id))
        if not prompts or call.prompt <= 0 or call.prompt > len(prompts):
            continue
        prompt = prompts[call.prompt - 1]
        prompt.tool_calls += 1
        prompt.tool_chars += call.chars
        prompt.tool_measured += call.measured
        if call.chars > prompt.top_tool_chars:
            prompt.top_tool_chars = call.chars
            prompt.top_tool = call.name


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
    watch = cancellation is not None
    for harness, events in scan.events.items():
        check_cancelled(cancellation)
        kinds = CLAUDE_KINDS if harness == "claude" else CODEX_KINDS
        for session_id, rows in group_events_by_session(events, cancellation).items():
            check_cancelled(cancellation)
            bounds = sorted(scan.boundaries.get((harness, session_id), []))
            labels = scan.prompt_labels.get((harness, session_id), {})
            summary = scan.sessions.get((harness, session_id))
            cwd = cwd_label(summary.cwd) if summary is not None else "-"
            groups = group_rows_by_prompt(rows, bounds, harness)
            prompts = []
            for position, (start_ts, group) in enumerate(groups, start=1):
                if watch:
                    check_cancelled(cancellation)
                prompt = build_prompt(harness, session_id, position, start_ts, group,
                                      kinds, weights, args, cancellation)
                if labels:
                    prompt.label = labels.get(start_ts, "")
                prompt.cwd = cwd
                prompts.append(prompt)
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
    # One price vector per model, not per call: `group` is usually dozens of
    # calls against one or two models, and the vector lets each row be weighed
    # straight off its token slots instead of through a fresh dict.
    cache_read_weight = args.claude_cache_read_weight
    multiplier = args.long_context_multiplier
    watch = cancellation is not None
    is_claude = harness == "claude"
    totals = prompt.tokens
    kind_slots = tuple(enumerate(kinds, start=EVENT_KINDS))
    vectors = {}  # type: Dict[str, Tuple[Any, Any]]
    for row in group:
        if watch:
            check_cancelled(cancellation)
        if len(row) <= EVENT_PROMPT:
            row.extend([None] * (EVENT_PROMPT + 1 - len(row)))
        row[EVENT_PROMPT] = position
        model = row[EVENT_MODEL]
        pair = vectors.get(model)
        if pair is None:
            pair = (
                weights.event_vector(harness, model, cache_read_weight, False),
                weights.event_vector(harness, model, cache_read_weight, True),
            )
            vectors[model] = pair
        unit_vector, input_vector = pair
        if is_claude:
            context = int(row[3]) + int(row[4]) + int(row[5]) + int(row[6]) + int(row[8])
        else:
            context = int(row[3]) + int(row[4])
        resent = 0.0 if input_vector is None else vector_units(input_vector, row)
        prompt.end = row[EVENT_TS]
        prompt.turns += 1
        if row[EVENT_SUB]:
            prompt.sub_turns += 1
        else:
            if context > prompt.context_peak:
                prompt.context_peak = context
            if not seen_main:
                prompt.context_start = context
                prompt.model = model
                baseline = resent
                seen_main = True
        for slot, kind in kind_slots:
            totals[kind] += int(row[slot])
        prompt.input_tokens += context
        units = 0.0 if unit_vector is None else vector_units(unit_vector, row)
        if row[EVENT_LONG]:
            units *= multiplier
        prompt.units += units
        prompt.resent_units += resent
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
            "session_id": self.session_id,
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
    watch = cancellation is not None
    for row in events:
        if watch:
            check_cancelled(cancellation)
        session = row[EVENT_SESSION]
        rows = grouped.get(session)
        if rows is None:
            grouped[session] = [row]
        else:
            rows.append(row)
    for rows in grouped.values():
        if watch:
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
        kind_slots = tuple(enumerate(kinds, start=EVENT_KINDS))
        for event in events:
            if not event[EVENT_LONG]:
                continue
            key = (harness, event[EVENT_SESSION], event[EVENT_MODEL])
            slot = long_context.get(key)
            if slot is None:
                slot = empty_tokens(kinds)
                long_context[key] = slot
            for index, kind in kind_slots:
                slot[kind] += int(event[index])
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
    def __init__(self, enabled: bool, ascii_only: bool = False, force_color: bool = False):
        self.enabled = enabled and (force_color or not os.environ.get("NO_COLOR"))
        self.ascii_only = ascii_only
        self.styles = ansi_styles(load_theme()[1]) if self.enabled else ANSI

    def __call__(self, text: str, *styles: str) -> str:
        text = self.glyphs(text)
        if not self.enabled or not styles:
            return text
        prefix = "".join(self.styles.get(style, "") for style in styles)
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
    color = getattr(args, "color", "auto")
    enabled = color == "always" or (color == "auto" and sys.stdout.isatty())
    return Painter(enabled and not args.no_color, ascii_output(args), force_color=color == "always")


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
        if any(
            row.summary.harness == "claude"
            and row.tokens.get("cache_write_unknown", 0) > 0
            for row in rows
        ):
            lines.append(
                "caveat: Claude cache writes without a full TTL split use the 5-minute rate as an estimate"
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
            write = (
                row.tokens.get("cache_write_5m", 0)
                + row.tokens.get("cache_write_1h", 0)
                + row.tokens.get("cache_write_unknown", 0)
            )
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
# "what to run next" footer (issue #27)
#
# Every command ends by naming the next command worth running, built from
# the rows it just printed. The footer goes to STDERR so `nenpi sessions |
# tee` and every `--json` pipeline stay byte-clean; under `--json` the same
# suggestions ride along as the payload's additive `next` key instead.


def hint(cmd: str, why: str) -> Dict[str, str]:
    """One footer entry. An empty `cmd` prints `why` as a plain note."""
    return {"cmd": cmd, "why": why}


def quiet_output(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "quiet", False)) or os.environ.get("NENPI_QUIET") == "1"


def scope_flags(args: argparse.Namespace) -> List[str]:
    """The range/harness/account flags the user passed, to repeat verbatim.

    A suggestion that dropped them would report on a different corpus than
    the rows it was derived from.
    """
    flags = []  # type: List[str]
    for name, flag in (("since", "--since"), ("until", "--until")):
        value = getattr(args, name, None)
        if value:
            flags += [flag, str(value)]
    harness = getattr(args, "harness", "all")
    if harness and harness != "all":
        flags += ["--harness", str(harness)]
    account = getattr(args, "account", None)
    if account:
        flags += ["--account", str(account)]
    for attr, flag in (("project", "--project"), ("exclude_project", "--exclude-project")):
        for value in getattr(args, attr, []) or []:
            flags += [flag, value]
    return flags


def suggest(args: argparse.Namespace, command: str, *flags: Any) -> str:
    """`nenpi <command> <flags>` with the user's scope flags carried through.

    A flag the caller set itself (a narrowed `--since`, say) wins over the
    same flag from the invocation, so a suggestion is never self-contradictory.
    """
    given = [str(flag) for flag in flags]
    carried = scope_flags(args)
    extra = []  # type: List[str]
    for index in range(0, len(carried), 2):
        if carried[index] not in given:
            extra += [carried[index], carried[index + 1]]
    # Quoted because --since/--until/--account carry free text: a printed
    # suggestion has to be safe to paste into a shell as it stands.
    return " ".join(["nenpi", command] + [shlex.quote(part) for part in given + extra])


def analysis_session_ids(analysis: "Analysis") -> List[str]:
    """Every session id the run could resolve a `--session` prefix against."""
    return sorted({session_id for _harness, session_id in analysis.scan.sessions})


def pick_id(session_id: str, known: Sequence[str]) -> str:
    """The shortest prefix of `session_id` no other session in scope shares.

    `short_id`'s eight characters are fine to read but not to run: on a real
    corpus several sessions share them and the suggested `--session` would
    exit 1 as ambiguous (#27). Grows from eight characters, and falls back to
    the full id when even that is only a prefix of another id.
    """
    flat = session_id.replace("-", "")
    rivals = [
        other.replace("-", "") for other in known
        if other != session_id and other.replace("-", "") != flat
    ]
    for length in range(min(8, len(flat)), len(flat) + 1):
        prefix = flat[:length]
        if not any(rival.startswith(prefix) for rival in rivals):
            return prefix
    return session_id


def footer_painter(args: argparse.Namespace) -> Painter:
    """Color the footer only when stderr is a tty of its own."""
    color = getattr(args, "color", "auto")
    enabled = color == "always" or (color == "auto" and sys.stderr.isatty())
    return Painter(enabled and not getattr(args, "no_color", False), force_color=color == "always")


def footer(args: argparse.Namespace, hints: Sequence[Mapping[str, str]]) -> None:
    """Write up to three "what to run next" lines to stderr."""
    if getattr(args, "json", False) or quiet_output(args):
        return
    rows = [entry for entry in hints if entry and (entry.get("cmd") or entry.get("why"))][:3]
    if not rows:
        return
    paint = footer_painter(args)
    for entry in rows:
        if not entry.get("cmd"):
            sys.stderr.write(paint("next: " + entry["why"], "dim") + "\n")
            continue
        sys.stderr.write(
            "%s %s  %s\n"
            % (paint("next:", "dim"), paint(entry["cmd"], "bold"),
               paint("# " + entry["why"], "dim"))
        )
    sys.stderr.flush()


def widen_hints(args: argparse.Namespace, what: str) -> List[Dict[str, str]]:
    """The empty-result footer: widen the range, then check the roots."""
    return [
        hint(suggest(args, "sessions", "--since", "30d"),
             "no %s in range; widen the window" % what),
        hint("nenpi config", "check which harness roots are being scanned"),
    ]


def session_weight(row: "Row") -> float:
    return row.relative


def sessions_hints(
    args: argparse.Namespace, shown: Sequence["Row"], scoped: Sequence["Row"],
    known: Sequence[str]
) -> List[Dict[str, str]]:
    if not shown:
        return widen_hints(args, "sessions")
    top = shown[:3]
    lead = pick_id(top[0].summary.session_id, known)
    # The note has to name the order actually on screen: --sort start ranks
    # by recency, and calling that "by drain" would be a lie (#27).
    note = "top %d by %s: %s" % (
        len(top), getattr(args, "sort", "drain"),
        ", ".join(pick_id(row.summary.session_id, known) for row in top))
    # The median is over every session in scope, not the --top slice, and
    # the multiple is always the drain leader's, whatever the sort is.
    weights = sorted(session_weight(row) for row in scoped)
    median = weights[len(weights) // 2] if weights else 0.0
    leader = max(scoped, key=session_weight) if scoped else None
    if leader is not None and median > 0 and session_weight(leader) >= 3.0 * median:
        note += " (%s drains %.1fx the median session here)" % (
            pick_id(leader.summary.session_id, known), session_weight(leader) / median)
    return [
        hint("", note),
        hint(suggest(args, "prompts", "--session", lead), "which prompts drove %s" % lead),
        hint(suggest(args, "tools", "--session", lead), "which tools filled its context"),
    ]


def timeline_hints(
    args: argparse.Namespace, buckets: Mapping[int, Mapping[str, float]], bucket_seconds: int
) -> List[Dict[str, str]]:
    if not buckets:
        return widen_hints(args, "activity")
    slot = max(buckets, key=lambda key: buckets[key]["claude"] + buckets[key]["codex"])
    entry = buckets[slot]
    return [
        hint("", "busiest bucket %s: %.1f claude + %.1f codex units"
             % (local_label(slot), entry["claude"], entry["codex"])),
        hint(suggest(args, "sessions", "--since", iso_arg(slot),
                     "--until", iso_arg(slot + bucket_seconds)),
             "the sessions running in that bucket"),
    ]


def iso_arg(epoch: float) -> str:
    """A local ISO stamp `--since`/`--until` parse and a shell needs no quotes."""
    return local_label(epoch, "%Y-%m-%dT%H:%M")


def windows_hints(
    args: argparse.Namespace, ordered: Sequence[Mapping[str, Any]], known: Sequence[str]
) -> List[Dict[str, str]]:
    if not ordered:
        return widen_hints(args, "Codex quota windows")
    busiest = max(ordered, key=lambda entry: entry["peak_used_percent"])
    start = busiest["start"]
    resets = busiest["resets_at"]
    end = float(resets) if isinstance(resets, (int, float)) else (
        start + 60.0 * float(busiest["window_minutes"] or 0))
    hints = [
        hint("", "busiest window started %s, peak %.1f%%"
             % (local_label(start), busiest["peak_used_percent"])),
        hint(suggest(args, "sessions", "--since", iso_arg(start), "--until", iso_arg(end)),
             "the sessions that drained it"),
    ]
    top = top_session_list(busiest["sessions"], 1)
    if top:
        hints.append(
            hint(suggest(args, "prompts", "--session", pick_id(top[0]["session_id"], known)),
                 "its biggest session, prompt by prompt")
        )
    return hints


def prompts_ranked_hints(
    args: argparse.Namespace, shown: Sequence["Prompt"], known: Sequence[str]
) -> List[Dict[str, str]]:
    if not shown:
        return widen_hints(args, "prompts")
    lead = shown[0]
    session = pick_id(lead.session_id, known)
    return [
        hint("", "#1 is prompt %d of session %s (%d turns, %s peak context)"
             % (lead.index, session, lead.turns, format_tokens(lead.context_peak))),
        hint(suggest(args, "prompts", "--session", session),
             "that session's per-prompt breakdown"),
        hint(suggest(args, "tools", "--session", session),
             "the tools those prompts ran"),
    ]


def prompts_session_hints(
    args: argparse.Namespace, session_id: str, prompts: Sequence["Prompt"],
    known: Sequence[str]
) -> List[Dict[str, str]]:
    if not prompts:
        return widen_hints(args, "prompts")
    session = pick_id(session_id, known)
    busiest = max(prompts, key=lambda prompt: (prompt.turns, prompt.index))
    largest = max(prompts, key=lambda prompt: (prompt.context_peak, prompt.index))
    hints = [
        hint("", "prompt %d ran the most turns (%d); prompt %d held the most context (%s)"
             % (busiest.index, busiest.turns, largest.index,
                format_tokens(largest.context_peak))),
        hint(suggest(args, "tools", "--session", session, "--prompt", busiest.index),
             "what prompt %d's turns were reading" % busiest.index),
    ]
    marked = [prompt for prompt in prompts if prompt.reduction]
    if marked:
        hints.append(hint(suggest(args, "reductions"),
                          "a context reduction landed in this session"))
    else:
        hints.append(hint(suggest(args, "prompts", "--sort", "context"),
                          "compare these prompts against every other session's"))
    return hints


def tools_hints(
    args: argparse.Namespace, session_id: str, calls: Sequence["ToolCall"],
    shown: Sequence[Mapping[str, Any]], known: Sequence[str]
) -> List[Dict[str, str]]:
    if not calls or not shown:
        return widen_hints(args, "tool calls")
    top = shown[0]
    ranking = {"calls": "call count", "mean": "mean result size"}.get(getattr(args, "sort", "context"), "total result size")
    note = hint("", "top tool by %s: %s (%s estimated result tokens over %d calls)"
                % (ranking, top["tool"], format_tokens(top["est_tokens"]), top["calls"]))
    if session_id:
        session = pick_id(session_id, known)
        busiest = max(calls, key=lambda call: (call.measured, call.prompt)).prompt
        if getattr(args, "prompt", None) and not getattr(args, "explain", False):
            return [
                note,
                hint(suggest(args, "tools", "--session", session, "--prompt", args.prompt, "--explain"),
                     "unpack exec activity and rank its associated model usage"),
                hint(suggest(args, "prompts", "--session", session, "--prompt", args.prompt, "--drilldown"),
                     "compare root and subagent usage for this prompt"),
            ]
        return [
            note,
            hint(suggest(args, "tools", "--session", session, "--prompt", busiest),
                 "the same tools inside prompt %d alone" % busiest),
            hint(suggest(args, "prompts", "--session", session, "--tools"),
                 "which prompts those calls belong to"),
        ]
    session = pick_id(heaviest_session(calls), known)
    return [
        note,
        hint(suggest(args, "tools", "--session", session),
             "the same ranking inside %s, the heaviest session" % session),
        hint(suggest(args, "prompts", "--session", session, "--tools"),
             "per-prompt tool counts for that session"),
    ]


def heaviest_session(calls: Sequence["ToolCall"]) -> str:
    totals = {}  # type: Dict[str, float]
    for call in calls:
        totals[call.session_id] = totals.get(call.session_id, 0.0) + call.measured + call.chars
    return max(sorted(totals), key=lambda key: totals[key]) if totals else ""


def fanout_hints(
    args: argparse.Namespace, prompts: Sequence["Prompt"], known: Sequence[str]
) -> List[Dict[str, str]]:
    if not prompts:
        return widen_hints(args, "prompts")
    totals = {}  # type: Dict[str, float]
    for prompt in prompts:
        totals[prompt.session_id] = totals.get(prompt.session_id, 0.0) + prompt.sub_turns
    busiest = max(sorted(totals), key=lambda key: totals[key])
    if totals[busiest] <= 0:
        busiest = max(prompts, key=lambda prompt: (prompt.turns, prompt.index)).session_id
        why = "the session with the widest single prompt"
    else:
        why = "the session with the most sub-agent turns (%d)" % int(totals[busiest])
    session = pick_id(busiest, known)
    return [
        hint(suggest(args, "prompts", "--session", session), why),
        hint(suggest(args, "tools", "--session", session),
             "what its sub-agents were reading"),
    ]


def reductions_hints(
    args: argparse.Namespace, found: Sequence["Reduction"], known: Sequence[str]
) -> List[Dict[str, str]]:
    if not found:
        return [
            hint(suggest(args, "prompts", "--sort", "context"),
                 "no reductions in range; see which prompts carry the most context"),
        ]
    biggest = max(found, key=lambda reduction: reduction.removed)
    session = pick_id(biggest.session_id, known)
    return [
        hint("", "largest drop: %s removed from %s at %s"
             % (format_tokens(biggest.removed), session, local_label(biggest.epoch))),
        hint(suggest(args, "prompts", "--session", session),
             "where that session's context went"),
    ]


def verify_hints(
    args: argparse.Namespace, report: Sequence[Mapping[str, Any]], known: Sequence[str]
) -> List[Dict[str, str]]:
    if not report:
        return widen_hints(args, "sessions to verify")

    def gap(entry: Mapping[str, Any]) -> float:
        reference = entry.get("cost_state") or entry.get("thread_token_usage") or {}
        summed = sum(entry["deduped"].get(kind, 0) for kind in ("input", "output"))
        reported = sum(int(reference.get(kind, 0) or 0) for kind in ("input", "output"))
        return abs(summed - reported)

    worst = max(report, key=gap)
    session = pick_id(worst["session_id"], known)
    return [
        hint("", "largest gap between parsed and reported totals: %s" % session),
        hint(suggest(args, "prompts", "--session", session),
             "read that session prompt by prompt"),
    ]


def calibrate_hints(args: argparse.Namespace, usable: bool) -> List[Dict[str, str]]:
    hints = [
        hint(suggest(args, "sessions", "--use-calibrated"),
             "price sessions with the fit just saved"),
    ]
    if not usable:
        hints.insert(0, hint("", "the fit is not usable yet; a longer --since or a "
                                 "larger --calibrate-bucket-hours may identify it"))
    return hints


def snapshot_hints(args: argparse.Namespace) -> List[Dict[str, str]]:
    return [
        hint("nenpi sessions --since 7d",
             "rank sessions against the drain these snapshots measure"),
        hint("nenpi calibrate --harness claude --since 30d",
             "fit Claude weights once the log spans a few windows"),
    ]


def config_hints(
    args: argparse.Namespace, present: bool, unconfigured: Sequence[Mapping[str, Any]]
) -> List[Dict[str, str]]:
    if unconfigured or not present:
        return [
            hint("nenpi config --init",
                 "write a config.toml seeded with the roots found on this host"),
        ]
    return [hint("nenpi sessions --since 7d", "rank the sessions these roots hold")]


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
        # (harness, session) -> total positive input growth across its calls,
        # the denominator for a tool's share of context growth.
        self.growth_totals = {}  # type: Dict[Tuple[str, str], int]
        self.reductions = None  # type: Optional[List[Reduction]]
        self.prompts = {}  # type: Dict[Tuple[str, str], List[Prompt]]
        # Tool attribution walks every recorded tool result and every API
        # call to split context growth between them. Only `tools`, `prompts`
        # and `fanout` ever read the answer, so it is computed on first use
        # rather than by `prepare`: `sessions` and `timeline` were paying for
        # a whole second pass over the corpus they never looked at.
        #
        # Prompt assembly stays eager because `attribute()` reads the prompt
        # index that `assemble_prompts` writes onto each codex event, so
        # every command that reports drain already depends on it.
        self.cancellation = None  # type: Cancellation
        self._tool_calls = None  # type: Optional[List[ToolCall]]

    @property
    def tool_calls(self) -> List[ToolCall]:
        if self._tool_calls is None:
            with profile_phase("tools"):
                self._tool_calls = attribute_tools(self.scan, self, self.cancellation)
        return self._tool_calls

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
    scan.filter_tools(
        lambda harness, rows: [
            row for row in rows if (harness, row[TOOL_SESSION]) in keep
        ]
    )
    scan.snapshots = [row for row in scan.snapshots if row.get("account") in account_keys]
    scan.boundaries = dict(
        (key, value) for key, value in scan.boundaries.items() if key in keep
    )
    scan.prompt_labels = dict(
        (key, value) for key, value in scan.prompt_labels.items() if key in keep
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
        with profile_phase("scan"):
            scan = collect(args, since, run)
        if getattr(args, "account", None):
            filter_by_account(scan, args.account)
        run.emit("analysis", message="building analysis")
        run.check()
        with profile_phase("totals"):
            if not getattr(args, "whole_session", False):
                window_events(scan, since, until, cancellation)
            rebuild_totals(scan, cancellation)
        analysis = Analysis(scan, weights, since, until, args)
        analysis.cancellation = cancellation
        # Prompt keys must exist before attribution so measured drain can be split
        # down to the prompt as well as the session.
        with profile_phase("prompts"):
            analysis.prompts = assemble_prompts(scan, weights, args, cancellation)
        run.check()
        with profile_phase("intervals"):
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
        with profile_phase("attribute"):
            attribute(analysis.intervals, scan.events["codex"], weights, args, cancellation)
            apply_prompt_drain(analysis, cancellation)
        from .project_filter import filter_analysis
        filter_analysis(analysis)
        run.emit("done", message="analysis ready")
        return analysis
    except (ScanCancelled, KeyboardInterrupt):
        run.emit("cancelled", message="scan cancelled")
        raise ScanCancelled() from None


def apply_prompt_drain(analysis: Analysis, cancellation: Cancellation = None) -> None:
    shares = {}  # type: Dict[Tuple[str, Any], float]
    watch = cancellation is not None
    for interval in analysis.intervals:
        check_cancelled(cancellation)
        for key, value in interval.prompts.items():
            shares[key] = shares.get(key, 0.0) + value
    for (harness, session_id), prompts in analysis.prompts.items():
        check_cancelled(cancellation)
        if harness != "codex":
            continue
        for prompt in prompts:
            if watch:
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
    scoped = sort_rows(rows, args.sort)
    rows = scoped[: args.top]
    hints = sessions_hints(args, rows, scoped, analysis_session_ids(analysis))
    if args.json:
        payload = {
            "schema": JSON_SCHEMA,
            "command": "sessions",
            "next": hints,
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
        footer(args, hints)
        return 0
    print("")
    for line in render_sessions(rows, args, paint, width):
        print(line)
    footer(args, hints)
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
    hints = timeline_hints(args, buckets, bucket_seconds)
    if args.json:
        payload = {
            "schema": JSON_SCHEMA,
            "command": "timeline",
            "next": hints,
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
        footer(args, hints)
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
    footer(args, hints)
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
    hints = windows_hints(args, ordered, analysis_session_ids(analysis))
    if args.json:
        payload = {
            "schema": JSON_SCHEMA,
            "command": "windows",
            "next": hints,
            "windows": [
                {
                    "account": entry["account"],
                    "account_label": entry["account_label"],
                    "window_minutes": entry["window_minutes"],
                    "resets_at": entry["resets_at"],
                    "start": entry["start"],
                    "peak_used_percent": entry["peak_used_percent"],
                    "attributed_drain_percent": entry["attributed_percent"],
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
        footer(args, hints)
        return 0
    label_width = max(len(entry["account_label"]) for entry in ordered)
    for entry in ordered:
        resets = entry["resets_at"]
        resets_text = local_label(float(resets)) if isinstance(resets, (int, float)) else "-"
        print(
            paint(
                "window %s min  account %-*s  start %s  resets %s  peak %.1f%%  attributed drain %.1f%%"
                % (
                    entry["window_minutes"],
                    label_width,
                    entry["account_label"],
                    local_label(entry["start"]),
                    resets_text,
                    entry["peak_used_percent"],
                    entry["attributed_percent"],
                ),
                "bold",
            )
        )
        for item in top_session_list(entry["sessions"], args.top):
            print("    %-10s %6.3f%%" % (item["short_id"], item["drain_percent"]))
        if entry["unattributed_percent"] > 0:
            print("    unattributed %.1f%%" % entry["unattributed_percent"])
    footer(args, hints)
    return 0


def top_session_list(sessions: Mapping[str, float], top: int) -> List[Dict[str, Any]]:
    ordered = sorted(sessions.items(), key=lambda item: item[1], reverse=True)[:top]
    return [
        {"session_id": key, "short_id": short_id(key), "drain_percent": value}
        for key, value in ordered
    ]


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
    hints = calibrate_hints(args, bool(fit["usable"]))
    payload = {
        "schema": JSON_SCHEMA,
        "command": "calibrate",
        "next": hints,
        "harness": "codex",
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
    footer(args, hints)
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
            "`nenpi snapshot --oauth` (see docs/drain.md) and retry"
        )
        return 1
    hints = calibrate_hints(
        args, any(result["usable"] for result in results.values())
    )
    payload = {
        "schema": JSON_SCHEMA,
        "command": "calibrate",
        "next": hints,
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
    footer(args, hints)
    return 0


def session_prefix_match(session_id: str, prefix: str) -> bool:
    """The prefix rule `prompts` has always used, shared with `tools`."""
    return session_id.replace("-", "").startswith(
        prefix.replace("-", "")
    ) or session_id.startswith(prefix)


def resolve_session_prefix(
    candidates: Mapping[Tuple[str, str], float], prefix: str, first: bool
) -> Tuple[Optional[Tuple[str, str]], int]:
    """Resolve one `--session` prefix against `{(harness, id): weight}`.

    Shared by `prompts` and `tools` so a scripted drill-down never reads a
    different session in one command than in the other. A full id (with or
    without dashes) wins outright; an ambiguous prefix is an error listing
    the candidates, unless `--first` asks for the busiest of them.
    Returns the chosen key, or `None` and the exit code to return.
    """
    matches = [key for key in candidates if session_prefix_match(key[1], prefix)]
    if not matches:
        warn("no session matching %r; run `nenpi sessions` for ids" % prefix)
        return None, 1
    wanted = prefix.replace("-", "")
    exact = [key for key in matches if key[1].replace("-", "") == wanted]
    if len(exact) == 1:
        return exact[0], 0
    if len(matches) > 1 and not first:
        warn(
            "%r matches %d sessions (%s); pass a longer prefix, a full id, "
            "or --first for the busiest"
            % (prefix, len(matches),
               ", ".join(short_id(key[1]) for key in sorted(matches)[:5]))
        )
        return None, 1
    return max(matches, key=lambda key: (candidates[key], key)), 0


def session_unit_totals(analysis: "Analysis") -> Dict[Tuple[str, str], float]:
    """Weighted units per session, the `--first` tie-break over ALL sessions."""
    totals = {}  # type: Dict[Tuple[str, str], float]
    for key, summary in analysis.scan.sessions.items():
        totals[key] = sum(
            weighted_units(summary.harness, model, tokens, analysis.weights, analysis.args)
            for model, tokens in summary.models.items()
        )
    return totals


PROMPTS_SESSION_TOP = 25
PROMPTS_RANK_TOP = 10
PROMPT_RANK_KEYS = {
    "turns": lambda prompt: (prompt.turns, prompt.sub_turns, prompt.units),
    "context": lambda prompt: (prompt.context_peak, prompt.turns),
    "drain": lambda prompt: (prompt.drain_percent or 0.0, prompt.units),
    "tokens": lambda prompt: (prompt.input_tokens, prompt.turns),
    "units": lambda prompt: (prompt.units, prompt.turns),
}


def rank_prompts(
    analysis: "Analysis", sort: str, top: int
) -> List[Prompt]:
    """The prompts in range, busiest first by `sort`, cut to `top`."""
    prompts = [
        prompt
        for session_prompts in analysis.prompts.values()
        for prompt in session_prompts
        if analysis.in_range(prompt.start) or analysis.in_range(prompt.end)
    ]
    # Sorted by identity first so the metric sort, which is stable, breaks
    # its own ties the same way on every run.
    prompts.sort(key=lambda prompt: (prompt.harness, prompt.session_id, prompt.index))
    prompts.sort(key=PROMPT_RANK_KEYS[sort], reverse=True)
    return prompts[:top] if top else prompts


def format_prompt_drain(prompt: Prompt) -> str:
    if prompt.drain_percent is None:
        return "-"
    return "%.2f%%" % prompt.drain_percent


def command_prompts_ranked(args: argparse.Namespace, analysis: "Analysis") -> int:
    """`prompts` with no --session: one row per prompt, across sessions."""
    top = args.top if args.top is not None else PROMPTS_RANK_TOP
    shown = rank_prompts(analysis, args.sort, top)
    hints = prompts_ranked_hints(args, shown, analysis_session_ids(analysis))
    if args.json:
        print(
            json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "command": "prompts",
                    "next": hints,
                    "sort": args.sort,
                    "top": top,
                    "prompts": [
                        dict(prompt.to_json(), prompt_index=prompt.index)
                        for prompt in shown
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    paint = make_painter(args)
    print(
        paint(
            "top %d prompts by %s across %d sessions"
            % (top or len(shown), args.sort, len(analysis.prompts)),
            "bold",
        )
    )
    if not shown:
        print("no prompts in range")
        footer(args, hints)
        return 0
    print("")
    columns = "%4s %-7s %-10s %-14s %5s %6s %6s %10s %8s" % (
        "rank", "harness", "session", "cwd", "#", "turns", "sub", "ctx peak",
        "drain")
    with_label = args.label is not False
    label_width = max(20, terminal_width(args) - len(columns) - 1)
    if with_label:
        columns += " %-*s" % (label_width, "prompt")
    print(paint(columns.rstrip(), "bold"))
    for rank, prompt in enumerate(shown, start=1):
        line = "%4d %-7s %-10s %-14s %5d %6d %6d %10s %8s" % (
            rank,
            prompt.harness,
            short_id(prompt.session_id),
            prompt.cwd[:14],
            prompt.index,
            prompt.turns,
            prompt.sub_turns,
            format_tokens(prompt.context_peak),
            format_prompt_drain(prompt),
        )
        if with_label:
            line += " " + prompt.label[:label_width]
        print(line.rstrip())
    footer(args, hints)
    return 0


def command_prompts(args: argparse.Namespace) -> int:
    prompt_index = getattr(args, "prompt", None)
    if prompt_index is not None and not getattr(args, "session", None):
        warn("--prompt requires --session")
        return 2
    if getattr(args, "drilldown", False) and prompt_index is None:
        warn("--drilldown requires --prompt")
        return 2
    if getattr(args, "drilldown", False) and not getattr(args, "session", None):
        warn("--drilldown requires --session")
        return 2
    analysis = prepare(args)
    # Every prompt row this command prints or serializes carries its tool
    # counts, which `attribute_tools` writes onto the prompts as a side
    # effect, so the lazy attribution is forced before any prompt is read.
    _ = analysis.tool_calls
    if not getattr(args, "session", None):
        return command_prompts_ranked(args, analysis)
    candidates = dict(
        (key, sum(prompt.units for prompt in prompts))
        for key, prompts in analysis.prompts.items()
    )
    key, code = resolve_session_prefix(candidates, args.session, args.first)
    if key is None:
        return code
    harness, session_id = key
    prompts = analysis.prompts[key]
    mark_reductions(prompts, analysis)
    growth = fit_growth(prompts)
    top = args.top if args.top is not None else PROMPTS_SESSION_TOP
    shown = prompts[-top:] if top and len(prompts) > top else prompts
    if prompt_index is not None:
        if prompt_index < 1 or prompt_index > len(prompts):
            warn("prompt index %d is out of range (1..%d)" % (prompt_index, len(prompts)))
            return 2
        shown = [prompts[prompt_index - 1]]
    drilldown = None
    if getattr(args, "drilldown", False):
        selected_prompt = prompts[prompt_index - 1]
        if harness == "codex":
            drilldown = drilldown_for_prompt(analysis, selected_prompt)
        elif harness == "claude":
            drilldown = drilldown_for_claude_prompt(analysis, selected_prompt)
        else:
            warn("--drilldown requires a Claude or Codex session")
            return 2
    hints = prompts_session_hints(
        args, session_id, shown, analysis_session_ids(analysis))
    if args.json:
        print(
            json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "command": "prompts",
                    "next": hints,
                    "harness": harness,
                    "session_id": session_id,
                    "short_id": short_id(session_id),
                    "growth": growth,
                    "prompts": [prompt.to_json() for prompt in shown],
                    **({"drilldown": drilldown} if drilldown is not None else {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    paint = make_painter(args)
    if drilldown is not None:
        render_drilldown(drilldown)
        footer(args, hints)
        return 0
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
    with_tools = getattr(args, "tools", False)
    with_label = getattr(args, "label", None) is True
    label_width = max(20, width - 110)
    header = "%-4s %-16s %6s %6s %10s %10s %10s %9s %-8s" % (
        "#", "start", "wall", "turns", "ctx start", "ctx peak", "input sent", "units", "note")
    if with_tools:
        header += " %6s %10s %-16s" % ("tools", "tool est", "largest tool")
    if with_label:
        header += " %-*s" % (label_width, "prompt")
    print(paint(header, "bold"))
    for prompt in shown:
        line = (
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
        if with_tools:
            line += " %6d %10s %-16s" % (
                prompt.tool_calls,
                format_tokens(prompt.tool_est_tokens),
                prompt.top_tool[:16],
            )
        if with_label:
            line += " " + prompt.label[:label_width]
            line = line.rstrip()
        print(line)
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
    footer(args, hints)
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


def tool_totals(calls: Sequence[ToolCall]) -> Dict[str, Dict[str, Any]]:
    """Aggregate calls by tool name. Names and sizes only."""
    totals = {}  # type: Dict[str, Dict[str, Any]]
    for call in calls:
        row = totals.get(call.name)
        if row is None:
            row = {
                "tool": call.name,
                "calls": 0,
                "result_chars": 0,
                "est_tokens": 0.0,
                "measured_tokens": 0.0,
                "max_result_chars": 0,
                "spawns": 0,
                "harness": call.harness,
            }
            totals[call.name] = row
        row["calls"] += 1
        row["result_chars"] += call.chars
        row["est_tokens"] += call.est_tokens
        row["measured_tokens"] += call.measured
        row["spawns"] += 1 if call.spawn else 0
        if call.chars > row["max_result_chars"]:
            row["max_result_chars"] = call.chars
    for row in totals.values():
        row["mean_result_chars"] = row["result_chars"] / row["calls"] if row["calls"] else 0.0
    return totals


def sort_tool_rows(rows: Sequence[Mapping[str, Any]], key: str) -> List[Dict[str, Any]]:
    def order(row: Mapping[str, Any]) -> Tuple[float, float]:
        if key == "calls":
            return (row["calls"], row["est_tokens"])
        if key == "mean":
            return (row["mean_result_chars"], row["calls"])
        return (row["est_tokens"], row["calls"])

    return sorted(rows, key=order, reverse=True)


def command_tools(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    calls = analysis.tool_calls
    session_id = ""
    if getattr(args, "session", None):
        key, code = resolve_session_prefix(
            session_unit_totals(analysis), args.session, args.first
        )
        if key is None:
            return code
        session_id = key[1]
        calls = [call for call in calls if call.session_id == session_id]
    if getattr(args, "prompt", None):
        calls = [call for call in calls if call.prompt == args.prompt]
    sessions_seen = {(call.harness, call.session_id) for call in calls}
    growth_total = sum(
        analysis.growth_totals.get(key, 0) for key in sessions_seen
    )
    rows = sort_tool_rows(list(tool_totals(calls).values()), args.sort)
    shown = rows[: args.top] if args.top else rows
    largest = sorted(calls, key=lambda call: call.chars, reverse=True)[:5]
    measured_total = sum(call.measured for call in calls)
    command_shapes = []  # type: List[Dict[str, Any]]
    top_commands = []  # type: List[Dict[str, Any]]
    explained_calls = 0
    activity = None
    if getattr(args, "explain", False):
        from .tool_activity import explain_activity

        command_shapes, top_commands = explain_command_calls(analysis.scan, calls)
        explained_calls = sum(int(row["calls"]) for row in command_shapes)
        activity = explain_activity(analysis, calls)
    explain_scope = (
        "selected prompt calls; context_growth_tokens remains session-wide"
        if getattr(args, "prompt", None) is not None
        else "selected sessions"
    )
    hints = tools_hints(
        args, session_id, calls, shown, analysis_session_ids(analysis))
    if args.json:
        print(
            json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "command": "tools",
                    "next": hints,
                    "generated_at": time.time(),
                    "session": short_id(session_id) if session_id else "",
                    "session_id": session_id,
                    "tool_calls": len(calls),
                    "result_chars": sum(call.chars for call in calls),
                    "est_tokens": sum(call.est_tokens for call in calls),
                    "measured_tokens": measured_total,
                    "measured_is_upper_bound": True,
                    "measured_note": (
                        "a turn's whole input growth is split across its tool "
                        "results, so output and prompt text are charged here too"
                    ),
                    "context_growth_tokens": growth_total,
                    "tools": [
                        dict(row, share_of_growth=(
                            row["measured_tokens"] / growth_total if growth_total else None
                        ))
                        for row in shown
                    ],
                    "largest_results": [call.to_json() for call in largest],
                    **({"context_growth_scope": explain_scope}
                       if getattr(args, "explain", False) else {}),
                    **({
                        # Keep the drilldown easy to consume alongside the
                        # existing top-level `tools` and `largest_results`
                        # fields; the nested object carries the scope note.
                        "command_shapes": command_shapes[: args.top] if args.top else command_shapes,
                        "top_commands": top_commands,
                        "activity": activity,
                    } if getattr(args, "explain", False) else {}),
                    **({
                        "explain": {
                            "shell_calls": explained_calls,
                            "unexplained_tool_calls": max(0, len(calls) - explained_calls),
                            "command_shapes": command_shapes[: args.top] if args.top else command_shapes,
                            "top_commands": top_commands,
                            "context_growth_tokens": growth_total,
                            "context_growth_scope": explain_scope,
                        }
                    } if getattr(args, "explain", False) else {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    paint = make_painter(args)
    width = terminal_width(args)
    scope = "session %s" % short_id(session_id) if session_id else "all sessions"
    if activity and activity["tool_calls"]:
        from .report_render import render_activity

        print(paint("tool usage - " + scope, "bold", "codex"))
        print(render_activity(activity, args))
        if all(call.harness == "codex" for call in calls):
            flags = ["--session", pick_id(session_id, analysis_session_ids(analysis))] if session_id else []
            if getattr(args, "prompt", None):
                flags += ["--prompt", args.prompt]
            footer(args, [hint(suggest(args, "auto", *flags), "find repeated large-context responses and Shake opportunities")])
            return 0
    print(paint(
        "tool calls - %s, %d calls, %s est tokens added (estimate: result chars / %d)"
        % (scope, len(calls), format_tokens(sum(call.est_tokens for call in calls)),
           int(CHARS_PER_TOKEN)),
        "bold",
    ))
    if not calls:
        print("no tool calls in range")
        footer(args, hints)
        return 0
    print("")
    print(paint("%-28s %6s %10s %10s %9s %9s %6s" % (
        "tool", "calls", "est tokens", "measured", "mean", "max", "share"), "bold"))
    print(paint(
        "measured is an UPPER BOUND: a turn's whole input growth is split across "
        "its tool results, so output and prompt text land here too",
        "dim",
    ))
    peak = max(row["est_tokens"] for row in shown)
    bar_width = max(8, min(30, width - 90))
    for row in shown:
        share = row["measured_tokens"] / growth_total if growth_total else None
        style = "claude" if row["harness"] == "claude" else "codex"
        print(
            "%-28s %6d %10s %10s %9s %9s %6s %s"
            % (
                row["tool"][:28],
                row["calls"],
                format_tokens(row["est_tokens"]),
                format_tokens(row["measured_tokens"]),
                format_tokens(row["mean_result_chars"] / CHARS_PER_TOKEN),
                format_tokens(row["max_result_chars"] / CHARS_PER_TOKEN),
                "-" if share is None else "%5.1f%%" % (100.0 * share),
                paint(bar(row["est_tokens"], peak, bar_width), style),
            )
        )
    print("")
    print(paint("largest single results", "bold"))
    for call in largest:
        print(
            "%-28s %10s  %-10s prompt %d"
            % (
                call.name[:28],
                format_tokens(call.est_tokens),
                short_id(call.session_id),
                call.prompt,
            )
        )
    if getattr(args, "explain", False):
        print("")
        print(paint("command-shape drilldown (shell calls only)", "bold"))
        print(
            paint(
                "command text is reread from live transcripts for this report; "
                "nothing is cached",
                "dim",
            )
        )
        if not command_shapes:
            print("no direct shell calls found; commands inside exec are included in the activity report above"
                  if activity and activity["explained_calls"] else "no shell command text found in the selected calls")
        else:
            print(
                paint(
                    "%4s %-34s %6s %9s %9s %9s %10s %10s %s"
                    % ("#", "shape", "calls", "chars", "est total", "est mean", "est max", "measured", "examples"),
                    "bold",
                )
            )
            for rank, row in enumerate(command_shapes[: args.top] if args.top else command_shapes, 1):
                examples = "; ".join(row["examples"])
                print(
                    "%4d %-34s %6d %9s %9s %9s %10s %10s %s"
                    % (
                        rank,
                        row["shape"][:34],
                        row["calls"],
                        format_tokens(row["result_chars"]),
                        format_tokens(row["est_tokens"]),
                        format_tokens(row["mean_est_tokens"]),
                        format_tokens(row["max_est_tokens"]),
                        format_tokens(row["measured_tokens"]),
                        examples[:80],
                    )
                )
            print("")
            print(paint("largest individual shell calls", "bold"))
            for row in top_commands:
                print(
                    "%10s %-10s prompt %-4d %s"
                    % (
                        format_tokens(row["est_tokens"]),
                        short_id(row["session_id"]),
                        row["prompt"],
                        row["command"],
                    )
                )
        print(
            paint(
                "context_growth_tokens is provider input growth for the %s; "
                "est tokens are result characters / %d" % (explain_scope, int(CHARS_PER_TOKEN)),
                "dim",
            )
        )
    print("")
    print(paint(
        "measured = this turn's input-token growth split across its tool results; "
        "est = result size / %d" % int(CHARS_PER_TOKEN),
        "dim",
    ))
    footer(args, hints)
    return 0


def command_fanout(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    # Every prompt row this command prints or serializes carries its tool
    # counts, which `attribute_tools` writes onto the prompts as a side
    # effect, so the lazy attribution is forced before any prompt is read.
    _ = analysis.tool_calls
    prompts = [
        prompt
        for prompts in analysis.prompts.values()
        for prompt in prompts
        if analysis.in_range(prompt.start) or analysis.in_range(prompt.end)
    ]
    sort = getattr(args, "sort", "tokens")
    if not prompts:
        hints = widen_hints(args, "prompts")
        if args.json:
            print(json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "command": "fanout",
                    "next": hints,
                    "prompts": 0,
                    "sort": sort,
                    "turns_histogram": [],
                    "context_histogram": [],
                    "top_prompts": [],
                },
                indent=2, sort_keys=True))
            return 0
        print("no prompts in range")
        footer(args, hints)
        return 0
    turns = [float(prompt.turns) for prompt in prompts]
    contexts = [float(prompt.context_peak) for prompt in prompts]
    top = sorted(prompts, key=PROMPT_RANK_KEYS[sort], reverse=True)[:15]
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
        "sort": sort,
        "turns_histogram": histogram(turns, (1, 2, 3, 5, 10, 20, 50, 100)),
        "context_histogram": histogram(
            contexts, (10_000, 50_000, 100_000, 200_000, 400_000, 800_000)
        ),
        "top_prompts": [
            dict(prompt.to_json(), cwd=cwds.get((prompt.harness, prompt.session_id), "-"))
            for prompt in top
        ],
        "next": fanout_hints(args, prompts, analysis_session_ids(analysis)),
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
    print(paint(
        "top 15 single prompts by %s"
        % ("input tokens sent" if sort == "tokens" else sort), "bold"))
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
    footer(args, summary_json["next"])
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
    hints = reductions_hints(args, found, analysis_session_ids(analysis))
    if args.json:
        print(
            json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "command": "reductions",
                    "next": hints,
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
        footer(args, hints)
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
    footer(args, hints)
    return 0


def command_verify(args: argparse.Namespace) -> int:
    analysis = prepare(args)
    scan = analysis.scan
    since = analysis.since
    until = analysis.until
    report = []
    for (harness, session_id), summary in sorted(scan.sessions.items()):
        if since is not None and (summary.end or 0) < since:
            continue
        if until is not None and (summary.start or 0) > until:
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
                        "session_id": session_id,
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
            threads = summary.thread_usages
            if not threads and summary.thread_usage:
                # Compatibility for cache entries written before per-thread
                # reports were retained. Fresh scans always populate the map.
                threads = {"": summary.thread_usage}
            if not threads:
                continue
            observed = empty_tokens(CODEX_KINDS)
            for tokens in summary.models.values():
                for kind in CODEX_KINDS:
                    observed[kind] += int(tokens.get(kind, 0))
            reported = empty_tokens(CODEX_KINDS)
            for thread in threads.values():
                for kind in CODEX_KINDS:
                    reported[kind] += int(thread.get(kind, 0))
            report.append(
                {
                    "harness": "codex",
                    "session_id": session_id,
                    "short_id": short_id(session_id),
                    "model": "-",
                    "deduped": observed,
                    "thread_token_usage": reported,
                }
            )
    hints = verify_hints(args, report, analysis_session_ids(analysis))
    if args.json:
        print(json.dumps(
            {"schema": JSON_SCHEMA, "command": "verify", "next": hints, "rows": report},
            indent=2, sort_keys=True))
        return 0
    paint = make_painter(args)
    if not report:
        print("nothing to verify in range")
        footer(args, hints)
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
    footer(args, hints)
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
        # The statusline passthrough writes nothing of its own on either
        # stream; no footer here, ever (#27).
        return snapshot_from_stdin(destination)
    if args.compact:
        code = compact_snapshots(destination)
    elif args.oauth:
        code = snapshot_from_oauth(destination, args)
    else:
        code = snapshot_from_configs(destination)
    if code == 0:
        footer(args, snapshot_hints(args))
    return code


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
    parser.add_argument("--project", action="append", default=[], metavar="PATH",
                        help="include a project directory and its descendants; repeat for alternatives; quoted globs allowed")
    parser.add_argument("--exclude-project", action="append", default=[], metavar="PATH",
                        help="exclude a project directory and its descendants; repeatable; exclusions win")
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
        "-q", "--quiet", action="store_true",
        help="drop the stderr \"what to run next\" footer; NENPI_QUIET=1 does the same",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="print phase timings (scan, totals, prompts, intervals, attribute, "
             "render) to stderr; NENPI_PROFILE=1 does the same",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="disable ANSI colors, including the scan progress display",
    )
    parser.add_argument(
        "--color", choices=("auto", "always", "never"), default="auto",
        help="use semantic report colors on a terminal, always (overrides NO_COLOR), or never; --no-color takes precedence",
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

    from .activity_report import command_activities

    activities = sub.add_parser(
        "activities", help="rank activities and repeated checks across projects (default)",
        description="Most expensive activities in the last 72 hours, ranked by input processed across harnesses.",
    )
    add_common(activities)
    activities.add_argument("--activity", metavar="NAME", help="inspect an activity by name")
    activities.add_argument("--session", default=None, metavar="ID_PREFIX")
    activities.add_argument("--prompt", type=int, default=None, metavar="N")
    activities.add_argument("--first", action="store_true")
    activities.set_defaults(handler=command_activities, since="72h", harness="all", top=5)
    for action in activities._actions:
        if action.dest == "since":
            action.help = "include events since this lower bound (default: 72h)"
        elif action.dest == "top":
            action.help = "number of activities; 0 shows all (default: 5)"

    from .auto_report import command_auto

    auto = sub.add_parser(
        "auto", help="find repeated model work and opportunities to reduce usage",
        description="Rank recent prompts by cumulative input processed, exposing repeated large-context responses even when tool results are tiny.",
    )
    add_common(auto)
    auto.add_argument("--session", default=None, metavar="ID_PREFIX", help="inspect one session")
    auto.add_argument("--first", action="store_true", help="use the busiest match for an ambiguous session prefix")
    auto.add_argument("--prompt", type=int, default=None, metavar="N", help="inspect one prompt within --session")
    auto.set_defaults(handler=command_auto, since="7d", harness="codex", top=5)
    for action in auto._actions:
        if action.dest == "since":
            action.help = "include events since this lower bound (default: 7d)"
        elif action.dest == "harness":
            action.help = "transcript source to scan (default: codex)"
        elif action.dest == "top":
            action.help = "number of findings to show; 0 shows all (default: 5)"

    sessions = sub.add_parser(
        "sessions", help="rank sessions by quota drain",
        description="Show one row per session, with fan-out and drain details.",
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
        description=(
            "List observed Codex windows, reset times, peak usage, and cumulative "
            "attributed drain. Drain can exceed peak usage when reported usage "
            "falls and later rises within the selected range."
        ),
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
        "prompts", help="rank prompts across sessions, or break down one session",
        description=(
            "Without --session, rank every prompt in range by turns (or context, "
            "drain, tokens) with a short redacted label. With --session, show that "
            "session's per-prompt input, context, turns, weighted units, and "
            "growth fits."
        ),
        epilog="Examples:\n"
               "  nenpi prompts --since 7d --sort turns --top 20\n"
               "  nenpi prompts --session 0123abcd --top 40 --since 7d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(prompts)
    prompts.add_argument(
        "--session", default=None, metavar="ID_PREFIX",
        help="break down one session by ID prefix; an ambiguous prefix is an error. "
             "Without it, prompts from every session in range are ranked together",
    )
    prompts.add_argument(
        "--prompt", type=int, default=None, metavar="N",
        help="select one prompt within --session; requires --session",
    )
    prompts.add_argument(
        "--drilldown", action="store_true",
        help="show thread lineage, tool, and wait details for the selected --prompt",
    )
    prompts.add_argument(
        "--first", action="store_true",
        help="with an ambiguous --session prefix, use the busiest match instead of failing",
    )
    prompts.add_argument(
        "--sort", choices=tuple(sorted(PROMPT_RANK_KEYS)), default="turns",
        help="ranking key without --session: API turns (default), peak context, "
             "measured drain, input tokens sent, or weighted units",
    )
    prompts.add_argument(
        "--tools", action="store_true",
        help="add per-prompt tool columns: call count, estimated tokens added, "
             "and the largest single result's tool",
    )
    label_flag = prompts.add_mutually_exclusive_group()
    label_flag.add_argument(
        "--label", dest="label", action="store_true", default=None,
        help="show the redacted prompt label column (the default for the "
             "cross-session ranking; opt-in with --session)",
    )
    label_flag.add_argument(
        "--no-label", dest="label", action="store_false",
        help="hide the redacted prompt label column",
    )
    # `--top` means "last N of this session" with --session and "top N of the
    # ranking" without, so its default is resolved in the handler.
    prompts.set_defaults(handler=command_prompts, top=None)

    tools = sub.add_parser(
        "tools", help="rank tool calls by the context they add",
        description=(
            "Rank tool names by estimated context added. Only tool names and "
            "result sizes are used by default. With --explain, reread live "
            "transcripts to group shell commands and unpack Codex exec activity, "
            "ranked by associated model usage."
        ),
        epilog="Example:\n  nenpi tools --since 7d --sort context --top 15",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(tools)
    tools.add_argument(
        "--session", default=None, metavar="ID_PREFIX",
        help="limit to one session by ID prefix; an ambiguous prefix is an error",
    )
    tools.add_argument(
        "--first", action="store_true",
        help="with an ambiguous --session prefix, use the busiest match instead of failing",
    )
    tools.add_argument(
        "--prompt", type=int, default=None, metavar="N",
        help="limit to one prompt index within the selected session",
    )
    tools.add_argument(
        "--explain", action="store_true",
        help="reread live commands and exec code; show activities, model usage, and bounded examples without caching inputs",
    )
    tools.add_argument(
        "--sort", choices=("context", "calls", "mean"), default="context",
        help="sort by estimated context added, call count, or mean result size "
             "(default: context)",
    )
    tools.set_defaults(handler=command_tools)

    fanout = sub.add_parser(
        "fanout", help="summarize turns and context per prompt",
        description="Show distributions and the 15 prompts with the most input tokens sent.",
        epilog="Example:\n  nenpi fanout --harness claude --since 7d",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common(fanout)
    fanout.add_argument(
        "--sort", choices=tuple(sorted(PROMPT_RANK_KEYS)), default="tokens",
        help="rank the top-prompts table by input tokens sent (default), API "
             "turns, peak context, measured drain, or weighted units",
    )
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
        "-q", "--quiet", action="store_true",
        help="drop the stderr \"what to run next\" footer",
    )
    snapshot.add_argument("--no-color", action="store_true",
                          help="disable ANSI colors")
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
    config_cmd.add_argument(
        "-q", "--quiet", action="store_true",
        help="drop the stderr \"what to run next\" footer",
    )
    config_cmd.add_argument("--no-color", action="store_true",
                            help="disable ANSI colors")
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
    unconfigured = unconfigured_roots(config)
    unconfigured_rows = [
        {"harness": harness, "path": str(root)}
        for harness in ("claude", "codex")
        for root in unconfigured.get(harness, [])
    ]
    hints = config_hints(args, path.is_file(), unconfigured_rows)
    if args.json:
        print(
            json.dumps(
                {
                    "schema": JSON_SCHEMA,
                    "next": hints,
                    "config_path": str(path),
                    "config_present": path.is_file(),
                    "roots": rows,
                    "unconfigured_roots": unconfigured_rows,
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
    if unconfigured_rows:
        print(
            "unconfigured: %s (run `nenpi config --init`, add them to "
            "[claude]/[codex] roots, or set [general] ignore_unconfigured = true)"
            % ", ".join(display_path(Path(row["path"])) for row in unconfigured_rows)
        )
    footer(args, hints)
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
        ignore_unconfigured=existing.ignore_unconfigured,
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
    footer(args, [hint("nenpi config", "check the roots it resolved"),
                  hint("nenpi sessions --since 7d", "rank the sessions they hold")])
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
    if not arguments or (arguments[0].startswith("-") and arguments[0] not in {"-h", "--help"}):
        arguments.insert(0, "activities")
    args = parser.parse_args(arguments)
    # Advisory notes belong on a human's stderr, not in a --json run.
    set_notices_enabled(not getattr(args, "json", False))
    if not getattr(args, "handler", None):
        parser.print_help()
        return 2
    if getattr(args, "harness", None) == "claude" and args.command == "windows":
        warn("windows are a Codex-only measurement")
    profile_start(profile_enabled(args))
    try:
        with profile_phase("total"):
            return args.handler(args)
    except ScanCancelled:
        return 130
    finally:
        profile_report()


if __name__ == "__main__":
    sys.exit(main())
