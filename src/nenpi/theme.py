"""Shared, stdlib-only theme settings for reports and the optional UI."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .config import load_config, warn_once

# Codex tui/src/style.rs uses the terminal palette: cyan accents, default
# text/background, green/red status colors, and #005f87 accents on light themes.
# Keep terminal colors symbolic so the user's terminal palette remains in charge.
COLORS = {
    "default": 39, "black": 30, "red": 31, "green": 32, "yellow": 33,
    "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
}
DEFAULT_COLORS = {
    "primary": "cyan", "accent": "cyan", "foreground": "default",
    "background": "default", "surface": "default", "panel": "default",
    "success": "green", "warning": "default", "error": "red",
    "claude": "default", "codex": "cyan",
}


def load_theme(path: Path | None = None) -> tuple[str, dict[str, str]]:
    settings = load_config(path).theme
    name = os.environ.get("NENPI_THEME", settings.get("name", "codex"))
    if name not in ("codex", "codex-light"):
        warn_once("unknown theme %r; using codex" % name)
        name = "codex"
    colors = dict(DEFAULT_COLORS)
    if name == "codex-light":
        colors.update(primary="#005f87", accent="#005f87", codex="#005f87")
    overrides = settings.get("colors", {})
    if not isinstance(overrides, dict):
        warn_once("theme.colors must be a table; using theme defaults")
        overrides = {}
    for key, value in overrides.items():
        if key not in colors or not isinstance(value, str) or not (
            value in COLORS or re.fullmatch(r"#[0-9a-fA-F]{6}", value)
        ):
            warn_once("invalid theme color %s=%r; using theme default" % (key, value))
            continue
        colors[key] = value
    return name, colors


def ansi_color(value: str) -> str:
    if value in COLORS:
        return "\033[%dm" % COLORS[value]
    return "\033[38;2;%d;%d;%dm" % tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


def ansi_styles(colors: dict[str, str]) -> dict[str, str]:
    return {
        "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
        "claude": ansi_color(colors["claude"]), "codex": ansi_color(colors["codex"]),
        "warn": ansi_color(colors["warning"]),
    }


def textual_theme(name: str, colors: dict[str, str]):
    # Lazy import: the CLI keeps its zero-dependency installation.
    from textual.theme import BUILTIN_THEMES, Theme

    def color(key: str) -> str:
        value = colors[key]
        return "ansi_" + value if value in COLORS else value

    accent = color("accent")
    return Theme(
        name=name, ansi=True, dark=name != "codex-light",
        **{key: color(key) for key in DEFAULT_COLORS if key not in ("claude", "codex")},
        secondary=accent,
        variables={
            **BUILTIN_THEMES["ansi-dark" if name == "codex" else "ansi-light"].variables,
            "surface": color("surface"), "panel": color("panel"),
            "text": color("foreground"),
            "text-muted": color("foreground") + " 50%",
            "text-disabled": color("foreground") + " 50%",
            "footer-background": color("background"),
            "footer-foreground": color("foreground"),
            "footer-description-foreground": color("foreground"),
            "border": accent, "border-blurred": "ansi_default",
            "block-cursor-background": accent,
            "block-cursor-foreground": "ansi_black" if name == "codex" else "ansi_white",
            "input-selection-background": accent,
            "input-selection-foreground": "ansi_black" if name == "codex" else "ansi_white",
            "footer-key-foreground": accent,
            "scrollbar": accent, "scrollbar-hover": accent, "scrollbar-active": accent,
        },
    )
