"""Optional Textual browser for nenpi transcript analysis."""

from __future__ import annotations

import sys
import datetime
import argparse
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .settings import Source, SourceSettings


@dataclass
class SessionRecord:
    """UI-safe session shape.  Scanner implementations may supply extra fields."""

    session_id: str
    harness: str
    project: str = "-"
    model: str = "-"
    start: Optional[float] = None
    end: Optional[float] = None
    weighted_units: float = 0.0
    drain_percent: Optional[float] = None
    requests: int = 0
    prompts: int = 0
    context_peak: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    sessions: List[SessionRecord] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    reductions: List[Dict[str, Any]] = field(default_factory=list)
    files_seen: int = 0
    files_parsed: int = 0


def _record(value: Any) -> SessionRecord:
    if isinstance(value, SessionRecord):
        return value
    value = dict(value)
    return SessionRecord(
        session_id=str(value.get("session_id", "")), harness=str(value.get("harness", "-")),
        project=str(value.get("cwd", value.get("project", "-"))),
        model=str(value.get("primary_model", value.get("model", "-"))),
        start=value.get("start"), end=value.get("end"),
        weighted_units=float(value.get("weighted_units", value.get("units", 0.0)) or 0.0),
        drain_percent=value.get("drain_percent"), requests=int(value.get("requests", 0) or 0),
        prompts=int(value.get("prompts", 0) or 0),
        context_peak=int(value.get("peak_context_tokens", value.get("context_peak", 0)) or 0),
        payload=value,
    )


def _when(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return datetime.datetime.fromtimestamp(value, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def normalize_result(value: Any) -> ScanResult:
    """Accept a scanner result mapping without coupling the UI to its internals."""
    if isinstance(value, ScanResult):
        return value
    if isinstance(value, dict):
        return ScanResult(
            sessions=[_record(row) for row in value.get("sessions", [])],
            prompts=list(value.get("prompts", [])),
            reductions=list(value.get("reductions", [])),
            files_seen=int(value.get("files_scanned", value.get("files_seen", 0)) or 0),
            files_parsed=int(value.get("files_parsed", 0) or 0),
        )
    return ScanResult(sessions=[_record(row) for row in value])


def filter_sessions(rows: Sequence[SessionRecord], query: str = "", sort: str = "start") -> List[SessionRecord]:
    """Apply small, URL/bookmark-friendly filters used by the landing table.

    Syntax is ``harness:claude project:name since:YYYY-MM-DD text``. Unknown
    words remain a free-text match against project, model, and session ID.
    """
    harness = project = since = until = None
    text: List[str] = []
    for token in query.split():
        if token.startswith("harness:"):
            harness = token[8:].lower()
        elif token.startswith("project:"):
            project = token[8:].lower()
        elif token.startswith("since:"):
            since = token[6:]
        elif token.startswith("until:"):
            until = token[6:]
        else:
            text.append(token.lower())

    def epoch(value: str) -> Optional[float]:
        try:
            return datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            return None

    lower, upper = epoch(since) if since else None, epoch(until) if until else None
    selected = []
    for row in rows:
        haystack = "%s %s %s" % (row.project, row.model, row.session_id)
        if harness and row.harness.lower() != harness:
            continue
        if project and project not in row.project.lower():
            continue
        if text and any(word not in haystack.lower() for word in text):
            continue
        if lower is not None and (row.end is None or row.end < lower):
            continue
        if upper is not None and (row.start is None or row.start > upper):
            continue
        selected.append(row)
    if sort in ("start", "recent"):
        return sorted(selected, key=lambda row: row.start or 0, reverse=True)
    if sort == "project":
        return sorted(selected, key=lambda row: (row.project.lower(), row.session_id))
    if sort == "harness":
        return sorted(selected, key=lambda row: (row.harness, row.project.lower(), row.session_id))
    return sorted(selected, key=lambda row: (row.harness, -(row.drain_percent or 0), row.session_id))


def scan_enabled_sources(
    settings: SourceSettings,
    progress: Optional[Callable[[int, int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> ScanResult:
    """Invoke the scanner foundation using only enabled source paths.

    The foundation is intentionally imported lazily: the base CLI remains
    importable without UI dependencies, and scanner tests can inject a fake.
    """
    try:
        from . import scanning  # type: ignore
    except ImportError as error:
        raise RuntimeError("nenpi scanning foundation is unavailable") from error
    sources = settings.enabled_sources()
    if not sources:
        return ScanResult()
    scan = getattr(scanning, "scan_sources", None)
    if scan is None:
        raise RuntimeError("nenpi.scanning.scan_sources is unavailable")
    return normalize_result(scan(
        [(source.harness, str(source.expanded_path())) for source in sources],
        progress=progress, cancelled=cancelled,
        status_callback=getattr(progress, "status_callback", None),
    ))


def _textual_imports():
    try:
        from textual.app import App, ComposeResult, Screen
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static, Select
        return App, ComposeResult, Horizontal, Vertical, Button, Footer, Header, Input, Label, ListItem, ListView, Static, Screen, Select
    except ImportError as error:
        raise RuntimeError("install the nenpi[ui] extra to use the browser") from error


def build_app(settings: Optional[SourceSettings] = None, scanner: Callable[..., ScanResult] = scan_enabled_sources):
    (App, ComposeResult, Horizontal, Vertical, Button, Footer, Header, Input, Label,
     ListItem, ListView, Static, Screen, Select) = _textual_imports()

    class SourcesScreen(Screen):
        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield Static("Transcript sources on the machine running nenpi (disabled sources are not scanned)", id="source-help")
            yield ListView(id="source-list")
            with Horizontal(id="source-form"):
                yield Select((("claude", "claude"), ("codex", "codex")), value="claude", id="source-harness")
                yield Input(placeholder="absolute transcript directory", id="source-path")
                yield Button("Add", id="source-add")
                yield Button("Toggle", id="source-toggle")
                yield Button("Remove", id="source-remove")
                yield Button("Done", id="source-done")

        def on_mount(self) -> None:
            self.refresh_sources()

        def refresh_sources(self) -> None:
            view = self.query_one("#source-list", ListView)
            view.clear()
            for source in self.app.source_settings.sources:  # type: ignore[attr-defined]
                marker = "on" if source.enabled else "off"
                view.append(ListItem(Label("[%s] %s %s: %s" %
                                          (marker, source.harness, source.name, source.path))))

        def selected_source(self) -> Optional[Source]:
            index = self.query_one("#source-list", ListView).index
            sources = self.app.source_settings.sources  # type: ignore[attr-defined]
            return sources[index] if index is not None and index < len(sources) else None

        def on_button_pressed(self, event: Any) -> None:
            app = self.app  # type: ignore[attr-defined]
            if event.button.id == "source-done":
                app.source_settings.save()
                self.dismiss()
            elif event.button.id == "source-add":
                path = self.query_one("#source-path", Input).value.strip()
                harness = str(self.query_one("#source-harness", Select).value)
                try:
                    app.source_settings.add(Path(path), harness)
                    app.source_settings.save()
                    self.query_one("#source-path", Input).value = ""
                    self.refresh_sources()
                except ValueError as error:
                    self.query_one("#source-help", Static).update(str(error))
            elif event.button.id == "source-toggle":
                source = self.selected_source()
                if source is not None:
                    app.source_settings.set_enabled(source.id, not source.enabled)
                    app.source_settings.save()
                    self.refresh_sources()
            elif event.button.id == "source-remove":
                source = self.selected_source()
                if source is not None:
                    app.source_settings.remove(source.id)
                    app.source_settings.save()
                    self.refresh_sources()

    class NenpiApp(App):
        TITLE = "nenpi"
        CSS = """
        #sessions { height: 1fr; width: 2fr; border: solid $accent; }
        #detail { width: 1fr; border: solid $accent; padding: 1; }
        #detail { overflow-y: auto; }
        #status { height: 1; }
        #toolbar { height: 3; }
        #filter { width: 1fr; }
        #source-path { width: 1fr; }
        """

        def __init__(self, source_settings: Optional[SourceSettings] = None, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.source_settings = source_settings or SourceSettings.load()
            self.scanner = scanner
            self.result = ScanResult()
            self._scan_generation = 0
            self._scan_worker = None
            self._sort = "start"
            self._visible_sessions: List[SessionRecord] = []

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="toolbar"):
                yield Input(placeholder="filter harness, project, or session", id="filter")
                yield Button("Scan", id="scan")
                yield Button("Cancel", id="cancel", disabled=True)
                yield Button("Sort: recent", id="sort")
                yield Button("Sources", id="sources")
            yield Static("Ready", id="status")
            with Horizontal():
                yield ListView(id="sessions")
                yield Static("Select a session", id="detail")
            yield Footer()

        def on_mount(self) -> None:
            self.start_scan()

        def action_quit(self) -> None:
            self.exit()

        def on_button_pressed(self, event: Any) -> None:
            if event.button.id == "scan":
                self.start_scan()
            elif event.button.id == "cancel":
                self.cancel_scan()
            elif event.button.id == "sort":
                self._sort = {"start": "project", "project": "harness",
                              "harness": "start"}[self._sort]
                event.button.label = "Sort: %s" % self._sort
                self.render_sessions(self.query_one("#filter", Input).value)
            elif event.button.id == "sources":
                self.push_screen(SourcesScreen())

        def on_input_changed(self, event: Any) -> None:
            if event.input.id == "filter":
                self.render_sessions(event.value)

        def start_scan(self) -> None:
            if self._scan_worker is not None:
                self.cancel_scan()
            self._scan_generation += 1
            generation = self._scan_generation
            self.query_one("#status", Static).update("Scanning…")
            self.query_one("#cancel", Button).disabled = False

            def progress(done: int, total: int) -> None:
                if generation != self._scan_generation:
                    return
                self.call_from_thread(self._progress, done, total)

            def status_callback(status: Any) -> None:
                if generation != self._scan_generation:
                    return
                self.call_from_thread(self._status_progress, status)
            progress.status_callback = status_callback  # type: ignore[attr-defined]

            def cancelled() -> bool:
                return generation != self._scan_generation

            def work() -> ScanResult:
                return normalize_result(self.scanner(self.source_settings, progress, cancelled))

            self._scan_worker = self.run_worker(work, thread=True, exclusive=True,
                                                exit_on_error=False)

        def _progress(self, done: int, total: int) -> None:
            self.query_one("#status", Static).update("Scanning %d/%d" % (done, total))

        def _status_progress(self, status: Any) -> None:
            self.query_one("#status", Static).update(
                "Scanning %s: %d/%d files, %d cache hits (%.1fs)" %
                (status.phase, status.files_parsed, status.files_seen,
                 status.cache_hits, status.elapsed))

        def cancel_scan(self) -> None:
            self._scan_generation += 1
            if self._scan_worker is not None:
                self._scan_worker.cancel()
            self._scan_worker = None
            self.query_one("#cancel", Button).disabled = True
            self.query_one("#status", Static).update("Scan cancelled; previous results kept")

        def on_unmount(self) -> None:
            self._scan_generation += 1
            if self._scan_worker is not None:
                self._scan_worker.cancel()
                self._scan_worker = None

        def on_worker_state_changed(self, event: Any) -> None:
            worker = event.worker
            if worker is not self._scan_worker or event.state.name in ("PENDING", "RUNNING"):
                return
            self._scan_worker = None
            self.query_one("#cancel", Button).disabled = True
            if event.state.name == "SUCCESS":
                self.result = normalize_result(worker.result)
                self.render_sessions(self.query_one("#filter", Input).value)
                self.query_one("#status", Static).update(
                    "%d sessions (%d files)" % (len(self.result.sessions), self.result.files_parsed))
            elif event.state.name == "ERROR":
                self.query_one("#status", Static).update("Scan failed: %s" % worker.error)

        def render_sessions(self, query: str = "") -> None:
            needle = query.strip().lower()
            rows = filter_sessions(self.result.sessions, query, self._sort)
            self._visible_sessions = rows
            view = self.query_one("#sessions", ListView)
            view.clear()
            for row in rows:
                if row.harness == "claude":
                    metric = "$%.2f modelled" % row.weighted_units
                elif row.drain_percent is not None:
                    qualifier = "estimated" if row.payload.get("drain_is_estimate") else "measured"
                    metric = "%.2f%% drain %s" % (row.drain_percent, qualifier)
                else:
                    metric = "Codex drain unavailable"
                view.append(ListItem(Label("%s  %s  %s" % (row.harness, row.project, metric))))

        def on_list_view_selected(self, event: Any) -> None:
            index = event.list_view.index
            if index is None or index >= len(self._visible_sessions):
                return
            row = self._visible_sessions[index]
            tokens = ", ".join("%s=%s" % item for item in sorted(row.payload.get("tokens", {}).items()))
            reductions = row.payload.get("reductions", [])
            prompts = row.payload.get("prompt_details", [])
            prompt_lines = ["  %s: turns=%s input=%s context=%s" %
                            (_when(item.get("start")), item.get("api_turns", "-"),
                             item.get("input_tokens", "-"), item.get("context_peak", "-"))
                            for item in prompts]
            reduction_lines = ["  %s: %s %s -> %s, removed=%s, saved=%.3f" %
                               (_when(item.get("time")), item.get("kind", "reduction"),
                                item.get("before", "-"), item.get("after", "-"),
                                item.get("removed_tokens", "-"), item.get("saved_units", 0.0))
                               for item in reductions]
            metric = ("$%.2f modelled" % row.weighted_units if row.harness == "claude"
                      else "%.2f%% drain %s" % (row.drain_percent, "estimated" if row.payload.get("drain_is_estimate") else "measured")
                      if row.drain_percent is not None else "Codex drain unavailable")
            self.query_one("#detail", Static).update(
                "session %s\n%s / %s\nmodel %s\n%s\n%s to %s\nrequests %d  prompts %d\n"
                "peak context %d\ntokens %s\nprompt turns:\n%s\nreductions:\n%s"
                % (row.session_id, row.harness, row.project, row.model, metric,
                   _when(row.start), _when(row.end), row.requests, row.prompts,
                   row.context_peak, tokens or "-", "\n".join(prompt_lines) or "  -",
                   "\n".join(reduction_lines) or "  -")
            )

    return NenpiApp(settings)


def serve_browser(host: str = "127.0.0.1", port: int = 8000) -> int:
    """Serve the app through textual-serve's real websocket bridge."""
    try:
        from textual_serve.server import Server
    except ImportError:
        print("nenpi: install the nenpi[browser] extra to use --browser", file=sys.stderr)
        return 2
    command = "%s -c %s" % (
        shlex.quote(sys.executable),
        shlex.quote("from nenpi.tui import build_app; build_app().run()"),
    )

    Server(command, host=host, port=port, title="nenpi").serve()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Browse nenpi reports")
    parser.add_argument("--browser", action="store_true", help="serve a browser app")
    parser.add_argument("--host", default="127.0.0.1", help="browser bind host")
    parser.add_argument("--port", default=8000, type=int, help="browser port")
    args = parser.parse_args(argv)
    if args.browser:
        return serve_browser(args.host, args.port)
    try:
        app = build_app()
    except RuntimeError as error:
        print("nenpi: %s" % error, file=sys.stderr)
        return 2
    app.run()
    return 0


def web_main(argv: Optional[Sequence[str]] = None) -> int:
    return main(["--browser"] + list(argv or []))


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    raise SystemExit(main())
