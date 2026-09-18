"""Shared scanner controls for CLI and threaded UI callers.

The scanner remains synchronous so existing command handlers keep their
simple call shape.  A Textual caller can run ``prepare`` in a worker thread,
pass a :class:`CancellationToken`, and receive immutable :class:`ScanStatus`
events through the progress callback.
"""

from __future__ import annotations

import errno
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported hosts are POSIX.
    fcntl = None  # type: ignore


class ScanCancelled(Exception):
    """Raised when a scanner caller requests cooperative cancellation."""


class CancellationToken:
    """Thread-safe-enough cancellation flag for a scanner worker.

    Setting a boolean is atomic under CPython and is sufficient for the
    worker/UI handoff.  The token deliberately has no waiting or blocking
    behavior: scanner loops call ``is_cancelled`` frequently and stop at the
    next record boundary.
    """

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled


Cancellation = Optional[Any]
ProgressCallback = Callable[["ScanStatus"], None]


def cancelled(token: Cancellation) -> bool:
    if token is None:
        return False
    if callable(token):
        return bool(token())
    checker = getattr(token, "is_cancelled", None)
    if checker is not None:
        return bool(checker())
    return bool(token)


def check_cancelled(token: Cancellation) -> None:
    if cancelled(token):
        raise ScanCancelled()


@dataclass(frozen=True)
class ScanStatus:
    """One structured scanner update.

    ``phase`` is one of ``discovery``, ``scanning``, ``analysis``,
    ``done``, or ``cancelled``.  Counts are cumulative for this prepare call.
    ``current_file`` is a basename only for local UI use; callers should not
    treat it as a stable transcript identifier.
    """

    phase: str
    files_seen: int = 0
    files_parsed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_read: int = 0
    elapsed: float = 0.0
    current_file: Optional[str] = None
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "files_seen": self.files_seen,
            "files_parsed": self.files_parsed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "bytes_read": self.bytes_read,
            "elapsed": self.elapsed,
            "current_file": self.current_file,
            "message": self.message,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


class ScannerRun:
    """Stateful reporter/cancellation adapter used by ``drain.prepare``."""

    def __init__(
        self,
        progress: Optional[ProgressCallback] = None,
        cancellation: Cancellation = None,
    ) -> None:
        self.progress = progress
        self.cancellation = cancellation
        self.started = time.monotonic()
        self.files_seen = 0
        self.files_parsed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.bytes_read = 0

    def check(self) -> None:
        check_cancelled(self.cancellation)

    def emit(
        self,
        phase: str,
        *,
        current_file: Optional[str] = None,
        message: str = "",
    ) -> None:
        if self.progress is None:
            return
        self.progress(
            ScanStatus(
                phase=phase,
                files_seen=self.files_seen,
                files_parsed=self.files_parsed,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                bytes_read=self.bytes_read,
                elapsed=max(0.0, time.monotonic() - self.started),
                current_file=current_file,
                message=message,
            )
        )


@contextmanager
def serialized_cache(root: Path, cancellation: Cancellation = None) -> Iterator[None]:
    """Serialize cache reads/writes across scanner processes.

    The lock file lives beside versioned shards and is intentionally separate
    from any transcript.  Atomic shard replacement still protects readers,
    while this lock prevents two processes from reading, parsing, and writing
    the same stale shard concurrently.
    """

    root.mkdir(parents=True, exist_ok=True)
    handle = open(root / ".lock", "a+")
    locked = False
    try:
        if fcntl is not None:
            while not locked:
                check_cancelled(cancellation)
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    time.sleep(0.05)
        else:
            check_cancelled(cancellation)
        yield
    finally:
        if fcntl is not None and locked:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def json_progress(status: ScanStatus) -> None:
    """Write one status event per line to stderr, keeping stdout machine-safe."""

    print(status.as_json(), file=os.sys.stderr, flush=True)


def scan_sources(
    sources: Sequence[Tuple[str, str]],
    progress: Optional[Callable[[int, int], None]] = None,
    cancelled: Cancellation = None,
    status_callback: Optional[Callable[[ScanStatus], None]] = None,
) -> Mapping[str, Any]:
    """Scan exactly the enabled ``(harness, absolute_root)`` pairs.

    This adapter is intentionally UI-shaped: it reports ``done, total`` to
    the small Textual status line and returns JSON-compatible session rows.
    The mapping contains ``prompts`` and ``reductions`` drilldown lists;
    prompt rows expose ``turns``, ``input_tokens`` (also
    ``input_side_tokens``), ``context_start``, ``context_peak``, and
    ``context_growth``, ``label`` (the redacted one-line prompt label), and
    per-prompt tool counts and result SIZES.  No other prompt or tool content
    is returned.
    Callers that need full structured lifecycle events should call
    ``nenpi.drain.prepare`` with a ``ScanStatus`` callback instead.
    """

    from . import drain

    args = drain.build_parser().parse_args(["sessions", "--harness", "all", "--json"])
    args.claude_root = []
    args.codex_root = []
    roots_by_harness: dict[str, List[str]] = {"claude": [], "codex": []}
    for harness, root in sources:
        if harness == "claude":
            roots_by_harness["claude"].append(root)
        elif harness == "codex":
            roots_by_harness["codex"].append(root)
        else:
            raise ValueError("unknown transcript harness: %s" % harness)
    for harness, roots in roots_by_harness.items():
        # A user can add both a profile root and one of its children. Keep the
        # broadest root once so a transcript is parsed exactly once.
        unique: List[Path] = []
        for raw in sorted({str(Path(root).expanduser().resolve()) for root in roots}, key=lambda item: (len(Path(item).parts), item)):
            path = Path(raw)
            if any(path == parent or parent in path.parents for parent in unique):
                continue
            unique.append(path)
        setattr(args, harness + "_root", [str(path) for path in unique])
    args.discover = False
    args.top = 1_000_000

    final_status: List[Optional[ScanStatus]] = [None]

    def report(status: ScanStatus) -> None:
        if status_callback is not None and status.phase == "done":
            final_status[0] = status
        elif status_callback is not None:
            status_callback(status)
        if progress is None:
            return
        done = status.files_parsed + status.cache_hits
        total = status.files_seen
        if status.phase in ("analysis", "done"):
            done = total
        progress(done, total)

    analysis = drain.prepare(args, progress=report, cancellation=cancelled)

    def check_facade_cancelled() -> None:
        if callable(cancelled) and cancelled():
            if status_callback is not None:
                status_callback(
                    ScanStatus(
                        phase="cancelled",
                        files_seen=analysis.scan.files_seen,
                        files_parsed=analysis.scan.files_read,
                        message="scan cancelled",
                    )
                )
            raise ScanCancelled()

    check_facade_cancelled()

    rows = drain.session_rows(
        analysis.scan, analysis.weights, args, analysis.since, analysis.until
    )
    drain.attach_fanout(rows, analysis)
    drain.apply_codex_drain(rows, analysis.intervals)
    drain.apply_claude_estimate(
        rows, drain.claude_dollars_per_percent(analysis.since, analysis.until)
    )
    drain.score_rows(rows)
    rows = drain.sort_rows(rows, args.sort)[: args.top]
    check_facade_cancelled()
    reductions_found = drain.detect_reductions_cached(analysis)
    check_facade_cancelled()
    all_prompts = [
        prompt
        for prompt_rows in analysis.prompts.values()
        for prompt in prompt_rows
    ]
    drain.mark_reductions(all_prompts, analysis)
    prompts = [_prompt_detail(prompt) for prompt in all_prompts]
    reductions = [_reduction_detail(reduction) for reduction in reductions_found]
    prompts_by_session: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    reductions_by_session: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for prompt in prompts:
        check_facade_cancelled()
        key = (prompt["harness"], prompt["session_id"])
        prompts_by_session.setdefault(key, []).append(prompt)
    for reduction in reductions:
        check_facade_cancelled()
        key = (reduction["harness"], reduction["session_id"])
        reductions_by_session.setdefault(key, []).append(reduction)
    session_payload = []
    for row in rows:
        check_facade_cancelled()
        payload = drain.row_json(row)
        key = (payload["harness"], payload["session_id"])
        details = prompts_by_session.get(key, [])
        payload["prompt_details"] = details
        payload["reductions"] = reductions_by_session.get(key, [])
        # Tool names and sizes only, rolled up from the prompt rows.
        payload["tools"] = {
            "calls": sum(int(item["tool_calls"]) for item in details),
            "result_chars": sum(int(item["tool_result_chars"]) for item in details),
            "est_tokens": sum(float(item["tool_est_tokens"]) for item in details),
            "measured_tokens": sum(
                float(item["tool_measured_tokens"]) for item in details
            ),
        }
        session_payload.append(payload)
    if status_callback is not None and final_status[0] is not None:
        status_callback(final_status[0])
    return {
        "sessions": session_payload,
        "prompts": prompts,
        "files_seen": analysis.scan.files_seen,
        "files_parsed": analysis.scan.files_read,
        "reductions": reductions,
    }


def _prompt_detail(prompt: Any) -> Dict[str, Any]:
    """Serialize prompt metrics without transcript text or tool payloads."""

    raw = prompt.to_json()
    context_start = int(raw.get("context_start", 0) or 0)
    context_peak = int(raw.get("context_peak", 0) or 0)
    return {
        "harness": raw["harness"],
        "session_id": raw["session_id"],
        "short_id": raw["short_id"],
        "index": raw["index"],
        "start": raw["start"],
        "end": raw["end"],
        "wall_seconds": raw["wall_seconds"],
        "turns": raw["api_turns"],
        "api_turns": raw["api_turns"],
        "subagent_turns": raw["subagent_turns"],
        "input_tokens": raw["input_side_tokens"],
        "input_side_tokens": raw["input_side_tokens"],
        "context_start": context_start,
        "context_peak": context_peak,
        "context_growth": context_peak - context_start,
        "tokens": dict(raw["tokens"]),
        "weighted_units": raw["weighted_units"],
        "resent_units": raw["resent_units"],
        "drain_percent": raw["drain_percent"],
        "model": raw["model"],
        "reduction": raw["reduction"],
        # The redacted one-line prompt label, never the prompt itself.
        "label": raw.get("label", ""),
        "tool_calls": raw["tool_calls"],
        "tool_result_chars": raw["tool_result_chars"],
        "tool_est_tokens": raw["tool_est_tokens"],
        "tool_measured_tokens": raw["tool_measured_tokens"],
        "largest_tool": raw["largest_tool"],
        "largest_tool_chars": raw["largest_tool_chars"],
    }


def _reduction_detail(reduction: Any) -> Dict[str, Any]:
    """Serialize context reductions with metrics only, never transcript text."""

    raw = reduction.to_json()
    return {
        "harness": raw["harness"],
        "session_id": reduction.session_id,
        "short_id": raw["short_id"],
        "time": raw["time"],
        "kind": raw["kind"],
        "before": raw["before"],
        "after": raw["after"],
        "removed_tokens": raw["removed_tokens"],
        "turns_after": raw["turns_after"],
        "model": raw["model"],
        "saved_units": raw["saved_units"],
        "saved_units_upper_bound": raw["saved_units_upper_bound"],
    }
