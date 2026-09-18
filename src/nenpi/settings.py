"""Persistent transcript-source settings for the optional browser UI.

The command line remains stdlib-only.  This module deliberately contains no
Textual imports so source management can be tested and used by integrations
without installing the UI extra.
"""

from __future__ import annotations

import json
import os
import tomllib
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import config as core_config


def config_path() -> Path:
    override = os.environ.get("NENPI_CONFIG")
    if override:
        return Path(override).expanduser()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return config_home / "nenpi" / "config.json"


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
    path = home / ".config" / "nenpi" / "config.toml"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return core_config.Config(path=path)
    claude = raw.get("claude") if isinstance(raw.get("claude"), dict) else {}
    codex = raw.get("codex") if isinstance(raw.get("codex"), dict) else {}
    return core_config.Config(
        claude_roots=claude.get("roots") if isinstance(claude.get("roots"), list) else [],
        codex_roots=codex.get("roots") if isinstance(codex.get("roots"), list) else [],
        path=path,
    )


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
    ) -> None:
        self.sources: List[Source] = list(sources)
        self.ignored_discovered = self._normalise_ignored(ignored_discovered)

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
        path = path or config_path()
        loaded: List[Source] = []
        payload: Dict[str, Any] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for raw in payload.get("sources", []):
                if not isinstance(raw, dict):
                    continue
                harness = str(raw.get("harness", ""))
                source_path = str(raw.get("path", ""))
                if harness not in ("claude", "codex") or not source_path:
                    continue
                loaded.append(Source(
                    str(raw.get("id") or _source_id(harness, Path(source_path).expanduser())),
                    str(raw.get("name") or Path(source_path).name), harness, source_path,
                    bool(raw.get("enabled", True)), bool(raw.get("discovered", False)),
                ))
        except (OSError, ValueError, TypeError):
            pass
        ignored = payload.get("ignored_discovered", [])
        deduped: Dict[tuple[str, str], Source] = {}
        for source in loaded:
            deduped[(source.harness, str(source.expanded_path().resolve()))] = source
        settings = cls(deduped.values(), ignored)
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
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "version": self.VERSION,
            "sources": [asdict(source) for source in self.sources],
            "ignored_discovered": [
                {"harness": harness, "path": path}
                for harness, path in sorted(self.ignored_discovered)
            ],
        }
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        os.replace(str(temporary), str(path))
        return path
