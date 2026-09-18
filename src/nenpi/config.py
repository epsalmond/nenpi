"""Root resolution and ``config.toml`` for nenpi.

State-dir layout, precedence between flags/config/defaults, and per-root
account detection all live here so ``drain.py`` can stay focused on parsing
transcripts. See docs/drain.md's Roots section for the user-facing picture.

Precedence for Claude/Codex roots: flags (non-empty) replace config;
config (non-empty) replaces defaults; defaults are
``[$CLAUDE_CONFIG_DIR, ~/.claude]`` / ``[$CODEX_HOME, ~/.codex]`` with the
env var first, deduplicated, and missing directories skipped silently. A
harness that resolves to zero roots gets one warning.

``config.toml`` is also the Textual UI's source store (issue #23): enabled
sources are ``roots``, sources switched off in the UI are ``disabled``, and
discovered roots the user removed are ``ignored``. An empty ``roots`` in a
table the file has means "scan nothing"; only a missing table falls back to
the defaults. Writing the file keeps unknown tables and keys (so a newer or
hand-written setting survives) but not comments.

Test path overrides use the ``NENPI_*`` environment variables (``HOME_DIR``,
``CACHE_DIR``, ``STATE_DIR``, ``CONFIG_DIR``, ``CONFIG_FILE``). The old
``QUOTA_DRAIN_*`` names are still honoured for ``HOME_DIR``, ``CACHE_DIR``,
``STATE_DIR``, and ``CONFIG_DIR``, with a deprecation warning;
``QUOTA_DRAIN_CONFIG_FILE`` has no legacy fallback.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PREFIX = "NENPI_"
LEGACY_PREFIX = "QUOTA_DRAIN_"

MAX_JSON_BYTES = 128 * 1024 * 1024

_WARNED = set()  # type: set


def warn(message: str) -> None:
    sys.stderr.write("nenpi: %s\n" % message)


def warn_once(message: str) -> None:
    if message in _WARNED:
        return
    _WARNED.add(message)
    warn(message)


# --------------------------------------------------------------------------
# state dirs


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(PREFIX + name)
    if value is None:
        legacy = os.environ.get(LEGACY_PREFIX + name)
        if legacy is not None:
            warn_once(
                "%s%s is deprecated; use %s%s" % (LEGACY_PREFIX, name, PREFIX, name)
            )
            value = legacy
    if value is None:
        value = str(default)
    return Path(value).expanduser()


def home_dir() -> Path:
    return env_path("HOME_DIR", Path.home())


def cache_dir() -> Path:
    return env_path("CACHE_DIR", home_dir() / ".cache" / "nenpi")


def state_dir() -> Path:
    return env_path("STATE_DIR", home_dir() / ".local" / "state" / "nenpi")


def config_dir() -> Path:
    return env_path("CONFIG_DIR", home_dir() / ".config" / "nenpi")


def migrate_dirs() -> None:
    """Move ``quota-drain`` state dirs to their ``nenpi`` names, once.

    Only runs against the default location for a dir whose environment
    variable (new or legacy name) is not set; an explicit override is left
    alone; there is nothing to migrate for it.
    """
    home = home_dir()
    for name, parts in (
        ("CACHE_DIR", (".cache",)),
        ("STATE_DIR", (".local", "state")),
        ("CONFIG_DIR", (".config",)),
    ):
        if os.environ.get(PREFIX + name) or os.environ.get(LEGACY_PREFIX + name):
            continue
        new_dir = home.joinpath(*parts, "nenpi")
        old_dir = home.joinpath(*parts, "quota-drain")
        if new_dir.exists() or not old_dir.is_dir():
            continue
        try:
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            os.rename(str(old_dir), str(new_dir))
            warn("moved %s to %s" % (old_dir, new_dir))
        except OSError as error:
            warn(
                "could not move %s to %s (%s); move it by hand: mv %s %s"
                % (old_dir, new_dir, error, old_dir, new_dir)
            )


# --------------------------------------------------------------------------
# config.toml


class Config:
    """The parsed ``config.toml``, shared by the CLI and the Textual UI.

    ``roots`` is what the CLI scans. ``disabled`` holds roots the UI knows
    about but has switched off, and ``ignored`` holds discovered roots the
    user removed, so rediscovery does not bring them back. Both are UI
    bookkeeping the CLI only has to round-trip.
    """

    def __init__(
        self,
        claude_roots: Optional[Sequence[str]] = None,
        codex_roots: Optional[Sequence[str]] = None,
        plan_claude: Optional[str] = None,
        plan_codex: Optional[str] = None,
        path: Optional[Path] = None,
        claude_disabled: Optional[Sequence[str]] = None,
        codex_disabled: Optional[Sequence[str]] = None,
        claude_ignored: Optional[Sequence[str]] = None,
        codex_ignored: Optional[Sequence[str]] = None,
        tables_present: Optional[Sequence[str]] = None,
        extras: Optional[Dict[str, Any]] = None,
    ):
        self.claude_roots = list(claude_roots or [])
        self.codex_roots = list(codex_roots or [])
        self.plan_claude = plan_claude
        self.plan_codex = plan_codex
        self.path = path
        self.claude_disabled = list(claude_disabled or [])
        self.codex_disabled = list(codex_disabled or [])
        self.claude_ignored = list(claude_ignored or [])
        self.codex_ignored = list(codex_ignored or [])
        # Which harness tables the file actually had: an empty ``roots`` in a
        # table the user wrote is a choice ("scan nothing"), while an absent
        # table means "no opinion" and falls back to the defaults.
        self.tables_present = set(tables_present or ())
        # Tables and keys this version does not know about, kept verbatim so
        # saving never drops a newer (or hand-written) setting.
        self.extras: Dict[str, Any] = dict(extras or {})

    def roots(self, harness: str) -> List[str]:
        return self.claude_roots if harness == "claude" else self.codex_roots

    def disabled(self, harness: str) -> List[str]:
        return self.claude_disabled if harness == "claude" else self.codex_disabled

    def ignored(self, harness: str) -> List[str]:
        return self.claude_ignored if harness == "claude" else self.codex_ignored

    def has_table(self, name: str) -> bool:
        return name in self.tables_present


_KNOWN_TOP_KEYS = ("claude", "codex", "plan")
_KNOWN_SUBKEYS = {
    "claude": ("roots", "disabled", "ignored"),
    "codex": ("roots", "disabled", "ignored"),
    "plan": ("claude", "codex"),
}


def config_path() -> Path:
    override = os.environ.get(PREFIX + "CONFIG_FILE")
    if override:
        return Path(override).expanduser()
    return config_dir() / "config.toml"


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def load_config(path: Optional[Path] = None) -> Config:
    path = path or config_path()
    try:
        raw = path.read_bytes()
    except OSError:
        return Config(path=path)
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise SystemExit("nenpi: config %s: %s" % (path, error))
    if not isinstance(data, dict):
        raise SystemExit("nenpi: config %s: expected a table at the top level" % path)

    tables_present = []
    extras = {}  # type: Dict[str, Any]
    for key in data:
        if key not in _KNOWN_TOP_KEYS:
            warn_once("config %s: unknown key [%s]" % (path, key))
            extras[key] = data[key]
            continue
        block = data.get(key)
        if not isinstance(block, dict):
            continue
        tables_present.append(key)
        for subkey in block:
            if subkey not in _KNOWN_SUBKEYS[key]:
                warn_once("config %s: unknown key %s.%s" % (path, key, subkey))
                extras.setdefault(key, {})[subkey] = block[subkey]

    claude_block = data.get("claude") if isinstance(data.get("claude"), dict) else {}
    codex_block = data.get("codex") if isinstance(data.get("codex"), dict) else {}
    plan_block = data.get("plan") if isinstance(data.get("plan"), dict) else {}

    plan_claude = plan_block.get("claude")
    plan_codex = plan_block.get("codex")

    return Config(
        claude_roots=_string_list(claude_block.get("roots")),
        codex_roots=_string_list(codex_block.get("roots")),
        plan_claude=plan_claude if isinstance(plan_claude, str) else None,
        plan_codex=plan_codex if isinstance(plan_codex, str) else None,
        path=path,
        claude_disabled=_string_list(claude_block.get("disabled")),
        codex_disabled=_string_list(codex_block.get("disabled")),
        claude_ignored=_string_list(claude_block.get("ignored")),
        codex_ignored=_string_list(codex_block.get("ignored")),
        tables_present=tables_present,
        extras=extras,
    )


# --------------------------------------------------------------------------
# writing config.toml
#
# The standard library reads TOML (``tomllib``) but cannot write it, and
# nenpi stays dependency-free, so this is the smallest writer that covers
# the schema above: basic strings and arrays of them, plus one boolean.


def toml_escape(value: str) -> str:
    """Quote a string as a TOML basic string, without the quotes."""

    out = []
    for char in value:
        if char == "\\":
            out.append("\\\\")
        elif char == '"':
            out.append('\\"')
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        elif char < " " or char == "\x7f":
            out.append("\\u%04x" % ord(char))
        else:
            out.append(char)
    return "".join(out)


def toml_string(value: str) -> str:
    return '"%s"' % toml_escape(value)


def toml_value(value: Any) -> Optional[str]:
    """Render a scalar or flat array; ``None`` when this writer cannot."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return toml_string(value)
    if isinstance(value, int) or isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        rendered = [toml_value(item) for item in value]
        if any(item is None for item in rendered):
            return None
        return "[%s]" % ", ".join(rendered)  # type: ignore[arg-type]
    return None


def _extra_lines(prefix: str, block: Any) -> List[str]:
    """Re-emit a preserved table (one level of sub-tables) as TOML."""

    lines = []
    nested = []
    if not isinstance(block, dict):
        rendered = toml_value(block)
        return ["%s = %s" % (prefix, rendered)] if rendered is not None else []
    lines.append("[%s]" % prefix)
    for key, value in block.items():
        if isinstance(value, dict):
            nested.append((key, value))
            continue
        rendered = toml_value(value)
        if rendered is None:
            warn_once("config: cannot rewrite %s.%s; it is dropped" % (prefix, key))
            continue
        lines.append("%s = %s" % (key, rendered))
    lines.append("")
    for key, value in nested:
        lines.extend(_extra_lines("%s.%s" % (prefix, key), value))
    return lines


def _root_array(
    key: str, values: Sequence[str], comments: Optional[Dict[str, str]] = None
) -> List[str]:
    if not values:
        return ["%s = []" % key]
    lines = ["%s = [" % key]
    for value in values:
        comment = (comments or {}).get(value)
        lines.append(
            "    %s,%s" % (toml_string(value), ("  # %s" % comment) if comment else "")
        )
    lines.append("]")
    return lines


def dump_config(
    config: Config,
    comments: Optional[Dict[str, str]] = None,
    header: Optional[Sequence[str]] = None,
) -> str:
    """Render a :class:`Config` as ``config.toml`` text."""

    lines = list(header or ())
    if lines:
        lines.append("")
    for harness in ("claude", "codex"):
        lines.append("[%s]" % harness)
        lines.extend(_root_array("roots", config.roots(harness), comments))
        if config.disabled(harness):
            lines.append("# disabled in the UI; kept so the choice survives rediscovery")
            lines.extend(_root_array("disabled", config.disabled(harness)))
        if config.ignored(harness):
            lines.append("# removed in the UI; not offered again by discovery")
            lines.extend(_root_array("ignored", config.ignored(harness)))
        for key, value in (config.extras.get(harness) or {}).items():
            rendered = toml_value(value)
            if rendered is None:
                warn_once("config: cannot rewrite %s.%s; it is dropped" % (harness, key))
                continue
            lines.append("%s = %s" % (key, rendered))
        lines.append("")
    plan_extras = config.extras.get("plan") or {}
    if config.plan_claude or config.plan_codex or plan_extras:
        lines.append("[plan]  # display only")
        if config.plan_claude:
            lines.append("claude = %s" % toml_string(config.plan_claude))
        if config.plan_codex:
            lines.append("codex = %s" % toml_string(config.plan_codex))
        for key, value in plan_extras.items():
            rendered = toml_value(value)
            if rendered is not None:
                lines.append("%s = %s" % (key, rendered))
        lines.append("")
    else:
        lines.append("# [plan]  # optional, display only")
        lines.append('# claude = "max_20x"')
        lines.append('# codex = "pro"')
        lines.append("")
    for key, value in config.extras.items():
        if key in _KNOWN_TOP_KEYS:
            continue
        lines.extend(_extra_lines(key, value))
    return "\n".join(lines)


def save_config(
    config: Config,
    path: Optional[Path] = None,
    comments: Optional[Dict[str, str]] = None,
    header: Optional[Sequence[str]] = None,
) -> Path:
    """Write ``config.toml`` atomically and return the path written."""

    path = path or config.path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(dump_config(config, comments, header), encoding="utf-8")
    os.replace(str(temporary), str(path))
    config.tables_present.update(("claude", "codex"))
    return path


# --------------------------------------------------------------------------
# roots


_LEAVES = {"claude": "projects", "codex": "sessions"}


def normalize_root_flag(value: str, harness: str) -> Path:
    """Accept a ``sessions``/``projects`` leaf path for one release, as a shim.

    Before this change ``--claude-root``/``--codex-root`` took the leaf
    directory the tool actually scans; they now take the harness home dir,
    which is also where the account lives. A leaf path is still accepted,
    with the leaf stripped and a deprecation warning.
    """
    path = Path(value).expanduser()
    leaf = _LEAVES[harness]
    if path.name == leaf:
        warn_once(
            "--%s-root given a %r path; pass the home dir instead (this still "
            "works but is deprecated)" % (harness, leaf)
        )
        return path.parent
    return path


def _dedup_existing(paths: Sequence[Path], keep_missing: bool = False) -> List[Path]:
    seen = set()
    result = []
    for path in paths:
        resolved = path.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_dir() or keep_missing:
            result.append(resolved)
    return result


def _default_root_candidates(harness: str) -> List[Path]:
    home = home_dir()
    if harness == "claude":
        env_value = os.environ.get("CLAUDE_CONFIG_DIR")
        literal = home / ".claude"
    else:
        env_value = os.environ.get("CODEX_HOME")
        literal = home / ".codex"
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(literal)
    return candidates


def resolve_roots(
    harness: str,
    flags: Sequence[str],
    config: Config,
    quiet: bool = False,
    keep_missing: bool = False,
) -> List[Path]:
    flag_paths = [normalize_root_flag(value, harness) for value in flags if value]
    if flag_paths:
        resolved = _dedup_existing(flag_paths, keep_missing=keep_missing)
    else:
        configured = (
            config.claude_roots if harness == "claude" else config.codex_roots
        )
        if configured:
            resolved = _dedup_existing(
                [Path(item).expanduser() for item in configured], keep_missing=keep_missing
            )
        elif config.has_table(harness):
            # The file has a [claude]/[codex] table and it resolves to
            # nothing: every root was disabled or removed on purpose, so the
            # defaults must not creep back in.
            resolved = []
        else:
            resolved = _dedup_existing(
                _default_root_candidates(harness), keep_missing=keep_missing
            )
    if not resolved and not quiet:
        warn_once("no %s roots found; see `nenpi config`" % harness)
    return resolved


def discover_candidate_roots(harness: str) -> List[Path]:
    """The old ``~/.claude*`` / ``~/.codex*`` glob, for ``config --init`` only.

    Default resolution no longer globs (see ``resolve_roots``); this is kept
    solely to seed a starter config with what the old implicit scan used to
    pick up, so switching to a config file does not silently narrow what is
    measured.
    """
    home = home_dir()
    leaf = _LEAVES[harness]
    pattern = ".claude*" if harness == "claude" else ".codex*"
    try:
        candidates = sorted(home.glob(pattern))
    except OSError:
        candidates = []
    seen = {str(candidate) for candidate in candidates}
    for candidate in _default_root_candidates(harness):
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)
    return [candidate for candidate in candidates if (candidate / leaf).is_dir()]


# --------------------------------------------------------------------------
# accounts


def _read_json_field(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def account_for_root(root: Path, harness: str) -> Tuple[str, str]:
    """Return (account label, account key) for a resolved root.

    The label is always the root's basename, for display. The key is the
    stable id of the quota pool behind the root when one can be read, and
    falls back to the label otherwise. Never reads or returns a token.
    """
    label = root.name
    key = label
    if harness == "codex":
        payload = _read_json_field(root / "auth.json")
        if payload is not None:
            tokens = payload.get("tokens")
            account_id = tokens.get("account_id") if isinstance(tokens, dict) else None
            if isinstance(account_id, str) and account_id:
                key = account_id
    else:
        payload = _read_json_field(root / ".claude.json")
        if payload is not None:
            account = payload.get("oauthAccount")
            if isinstance(account, dict):
                org_uuid = account.get("organizationUuid")
                account_uuid = account.get("accountUuid")
                if isinstance(org_uuid, str) and org_uuid:
                    key = org_uuid
                elif isinstance(account_uuid, str) and account_uuid:
                    key = account_uuid
    return label, key
