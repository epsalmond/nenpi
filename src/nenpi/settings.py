"""Persistent transcript-source settings for the optional browser UI.

The command line remains stdlib-only.  This module deliberately contains no
Textual imports so source management can be tested and used by integrations
without installing the UI extra.

Sources live in the same ``config.toml`` the CLI reads (issue #23): enabled
sources are ``[claude].roots`` / ``[codex].roots``, sources switched off in
the UI are ``disabled``, and discovered roots the user removed are
``ignored``.  A pre-existing ``config.json`` is imported once and renamed to
``config.json.migrated``.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import config as core_config

LEAVES = {"claude": "projects", "codex": "sessions"}


def config_path() -> Path:
    """The single store: ``config.toml``.

    ``$NENPI_CONFIG`` (the UI's old override) still works and is now a
    deprecated alias of ``$NENPI_CONFIG_FILE``; a ``.json`` value is read as
    the ``.toml`` beside it so an old export keeps working.
    """

    override = os.environ.get("NENPI_CONFIG")
    if override:
        path = Path(override).expanduser()
        if path.suffix == ".json":
            core_config.warn_once(
                "NENPI_CONFIG points at %s; sources now live in config.toml, "
                "using %s" % (path, path.with_suffix(".toml"))
            )
            path = path.with_suffix(".toml")
        return path
    return core_config.config_path()


def legacy_json_path(path: Path) -> Path:
    """Where the pre-#23 ``config.json`` sits next to ``path``."""

    return path.with_name("config.json") if path.name == "config.toml" else path.with_suffix(".json")


def infer_harness(path: Path, use_name: bool = True) -> Optional[str]:
    """Which harness a directory looks like, by layout then by name.

    ``~/.codex-arcade`` was being saved with ``harness: "claude"`` by the
    Sources form; the layout (``projects/`` vs ``sessions/``) is the truth,
    and the name is only consulted when ``use_name`` is set, because an
    empty directory named ``codex-notes`` is not evidence of anything.
    """

    path = Path(path).expanduser()
    claude = (path / "projects").is_dir()
    codex = (path / "sessions").is_dir()
    if claude and not codex:
        return "claude"
    if codex and not claude:
        return "codex"
    if not use_name:
        return None
    name = path.name
    if name.startswith(".codex") or name.startswith("codex"):
        return "codex"
    if name.startswith(".claude") or name.startswith("claude"):
        return "claude"
    return None


CONFIG_HEADER = (
    "# nenpi config. Shared by the CLI and the Sources screen.",
    "# Precedence: --claude-root/--codex-root flags > this file > defaults.",
)


def _json_source_rows(payload: Any) -> List[Dict[str, Any]]:
    rows = payload.get("sources") if isinstance(payload, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def migrate_json_store(path: Optional[Path] = None) -> Optional[Path]:
    """Import a pre-#23 ``config.json`` into ``config.toml``, once.

    Runs only when the TOML file is absent and a JSON one is present. The
    harness of each imported source is re-derived from the directory layout,
    which fixes the ``~/.codex-*`` entries the Sources form saved as
    ``claude``. The JSON file is renamed to ``config.json.migrated``.
    """

    path = path or config_path()
    legacy = legacy_json_path(path)
    if legacy == path or path.exists() or not legacy.is_file():
        return None
    try:
        payload = json.loads(legacy.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    config = core_config.Config(path=path)
    buckets = {
        ("claude", True): config.claude_roots,
        ("claude", False): config.claude_disabled,
        ("codex", True): config.codex_roots,
        ("codex", False): config.codex_disabled,
    }
    seen = set()
    for row in _json_source_rows(payload):
        source_path = str(row.get("path") or "")
        if not source_path:
            continue
        harness = str(row.get("harness") or "")
        inferred = infer_harness(Path(source_path))
        if inferred is not None and inferred != harness:
            if harness in ("claude", "codex"):
                core_config.warn(
                    "config.json listed %s as %s; importing it as %s"
                    % (source_path, harness, inferred)
                )
            harness = inferred
        if harness not in ("claude", "codex"):
            continue
        key = (harness, str(Path(source_path).expanduser()))
        if key in seen:
            continue
        seen.add(key)
        buckets[(harness, bool(row.get("enabled", True)))].append(source_path)

    ignored = payload.get("ignored_discovered") if isinstance(payload, dict) else None
    for value in ignored or []:
        if isinstance(value, dict):
            pairs = [(value.get("harness"), value.get("path"))]
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            pairs = [(value[0], value[1])]
        elif isinstance(value, str):
            pairs = [("claude", value), ("codex", value)]
        else:
            continue
        for harness, item in pairs:
            if harness in ("claude", "codex") and isinstance(item, str) and item:
                config.ignored(harness).append(item)

    core_config.save_config(config, path, header=list(CONFIG_HEADER))
    migrated = legacy.with_name(legacy.name + ".migrated")
    try:
        os.replace(str(legacy), str(migrated))
    except OSError:
        migrated = legacy
    core_config.warn("imported %s into %s (old file kept as %s)"
                     % (legacy, path, migrated))
    return path


@dataclass
class Source:
    id: str
    name: str
    harness: str
    path: str
    enabled: bool = True
    discovered: bool = False

    def expanded_path(self) -> Path:
        return Path(self.path).expanduser()


def _source_id(harness: str, path: Path) -> str:
    return "%s-%s" % (harness, uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))


def _config_for_home(home: Path) -> core_config.Config:
    """Load the CLI config for a test home or the process's real home."""

    if home.resolve() == core_config.home_dir().resolve():
        return core_config.load_config()
    return core_config.load_config(home / ".config" / "nenpi" / "config.toml")


def _resolved_roots(home: Path, harness: str) -> List[Path]:
    config = _config_for_home(home)
    if home.resolve() == core_config.home_dir().resolve():
        return core_config.resolve_roots(harness, [], config, quiet=True)
    configured = config.claude_roots if harness == "claude" else config.codex_roots
    if configured:
        return core_config.resolve_roots(harness, configured, config, quiet=True)
    leaf = "projects" if harness == "claude" else "sessions"
    candidates = sorted(home.glob(".%s*" % harness))
    return [candidate for candidate in candidates if (candidate / leaf).is_dir()]


def discover_sources(home: Optional[Path] = None) -> List[Source]:
    home = (home or Path.home()).expanduser()
    found: List[Source] = []
    for harness, leaf in (("claude", "projects"), ("codex", "sessions")):
        for root in _resolved_roots(home, harness):
            if not (root / leaf).is_dir():
                continue
            found.append(Source(_source_id(harness, root), root.name,
                                harness, str(root), discovered=True))
    return found


class SourceSettings:
    """User sources plus discovered defaults.

    Disabled entries are retained so a user's choice survives rediscovery.
    Explicit roots are never silently replaced by a discovered root.
    """

    VERSION = 1

    def __init__(
        self,
        sources: Iterable[Source] = (),
        ignored_discovered: Iterable[Any] = (),
        path: Optional[Path] = None,
    ) -> None:
        self.sources: List[Source] = list(sources)
        self.ignored_discovered = self._normalise_ignored(ignored_discovered)
        self.path = path

    @staticmethod
    def _normalise_ignored(values: Iterable[Any]) -> set[tuple[str, str]]:
        result = set()
        for value in values:
            if isinstance(value, dict):
                harness = value.get("harness")
                path = value.get("path")
                if harness in ("claude", "codex") and isinstance(path, str):
                    result.add((harness, str(Path(path).expanduser().resolve())))
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                harness, path = value
                if harness in ("claude", "codex") and isinstance(path, str):
                    result.add((harness, str(Path(path).expanduser().resolve())))
            elif isinstance(value, str):
                # Legacy path-only tombstones suppressed either harness.
                resolved = str(Path(value).expanduser().resolve())
                result.update((harness, resolved) for harness in ("claude", "codex"))
        return result

    @classmethod
    def load(cls, path: Optional[Path] = None, home: Optional[Path] = None) -> "SourceSettings":
        if path is None:
            path = (
                home / ".config" / "nenpi" / "config.toml"
                if home is not None
                else config_path()
            )
        migrate_json_store(path)
        config = core_config.load_config(path)
        loaded: List[Source] = []
        for harness in ("claude", "codex"):
            for enabled, values in (
                (True, config.roots(harness)),
                (False, config.disabled(harness)),
            ):
                for value in values:
                    loaded.append(
                        Source(
                            _source_id(harness, Path(value).expanduser()),
                            Path(value).expanduser().name,
                            harness,
                            value,
                            enabled,
                        )
                    )
        ignored = [
            {"harness": harness, "path": value}
            for harness in ("claude", "codex")
            for value in config.ignored(harness)
        ]
        deduped: Dict[tuple[str, str], Source] = {}
        for source in loaded:
            deduped[(source.harness, str(source.expanded_path().resolve()))] = source
        settings = cls(deduped.values(), ignored)
        settings.path = path
        settings.merge_discovered(home)
        return settings

    def merge_discovered(self, home: Optional[Path] = None) -> None:
        known = {(source.harness, str(source.expanded_path().resolve())): source
                 for source in self.sources}
        for discovered in discover_sources(home):
            key = str(discovered.expanded_path().resolve())
            identity = (discovered.harness, key)
            if identity in self.ignored_discovered:
                continue
            if identity in known:
                known[identity].discovered = True
                continue
            self.sources.append(discovered)
            known[identity] = discovered

    def add(self, path: Path, harness: str, name: Optional[str] = None) -> Source:
        if harness not in ("claude", "codex"):
            raise ValueError("harness must be claude or codex")
        path = path.expanduser()
        if not path.is_absolute():
            raise ValueError("source path must be absolute")
        if not path.is_dir():
            raise ValueError("source path is not a directory: %s" % path)
        inferred = infer_harness(path, use_name=False)
        if inferred is not None and inferred != harness:
            raise ValueError(
                "%s looks like a %s root, not %s" % (path, inferred, harness)
            )
        for source in self.sources:
            if source.harness == harness and source.expanded_path().resolve() == path.resolve():
                source.enabled = True
                return source
        source = Source(_source_id(harness, path), name or path.name, harness, str(path))
        self.sources.append(source)
        return source

    def get(self, source_id: str) -> Source:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(source_id)

    def remove(self, source_id: str) -> None:
        removed = [source for source in self.sources if source.id == source_id]
        for source in removed:
            if source.discovered:
                self.ignored_discovered.add(
                    (source.harness, str(source.expanded_path().resolve()))
                )
        self.sources = [source for source in self.sources if source.id != source_id]

    def set_enabled(self, source_id: str, enabled: bool) -> None:
        self.get(source_id).enabled = bool(enabled)

    def enabled_sources(self) -> List[Source]:
        return [source for source in self.sources if source.enabled]

    def save(self, path: Optional[Path] = None) -> Path:
        """Write the sources into ``config.toml``, keeping the other tables."""

        path = path or self.path or config_path()
        config = core_config.load_config(path)
        config.path = path
        for harness in ("claude", "codex"):
            roots = [source.path for source in self.sources
                     if source.harness == harness and source.enabled]
            disabled = [source.path for source in self.sources
                        if source.harness == harness and not source.enabled]
            ignored = sorted(
                item for kind, item in self.ignored_discovered if kind == harness
            )
            if harness == "claude":
                config.claude_roots, config.claude_disabled = roots, disabled
                config.claude_ignored = ignored
            else:
                config.codex_roots, config.codex_disabled = roots, disabled
                config.codex_ignored = ignored
        self.path = path
        return core_config.save_config(config, path, header=list(CONFIG_HEADER))
