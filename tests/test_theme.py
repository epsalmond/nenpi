import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nenpi.config import load_config, save_config
from nenpi.drain import Painter
from nenpi.theme import TEXT_ROLES, load_theme


class ThemeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config.toml"
        self.env = patch.dict(os.environ, {"NENPI_CONFIG_FILE": str(self.path)})
        self.env.start()
        self.addCleanup(self.env.stop)
        for key in ("NENPI_THEME", "NO_COLOR"):
            os.environ.pop(key, None)

    def test_default_and_light_report_colors(self):
        self.assertEqual(Painter(True)("report", "codex"), "\033[36mreport\033[0m")
        with patch.dict(os.environ, {"NENPI_THEME": "codex-light"}):
            self.assertEqual(Painter(True)("report", "codex"),
                             "\033[38;2;0;95;135mreport\033[0m")

    def test_auto_report_semantic_colors(self):
        painter = Painter(True)
        self.assertEqual(painter("context", "replay"), "\033[36mcontext\033[0m")
        self.assertEqual(painter("repeat", "repeat"), "\033[33mrepeat\033[0m")
        self.assertEqual(painter("output", "output"), "\033[35moutput\033[0m")
        self.assertEqual(painter("next", "next"), "\033[32mnext\033[0m")

    def test_config_roundtrip_and_report_overrides(self):
        self.path.write_text('[theme]\nname = "codex-light"\n'
                             '[theme.colors]\ncodex = "#123456"\naccent = "blue"\n'
                             'replay = "#654321"\nrepeat = "blue"\n'
                             'output = "yellow"\nnext = "red"\n')
        before = load_config().theme
        save_config(load_config())
        self.assertEqual(load_config().theme, before)
        self.assertEqual(Painter(True)("report", "codex"),
                         "\033[38;2;18;52;86mreport\033[0m")
        self.assertEqual(Painter(True)("context", "replay"),
                         "\033[38;2;101;67;33mcontext\033[0m")
        self.assertEqual(Painter(True)("repeat", "repeat"), "\033[34mrepeat\033[0m")
        self.assertEqual(Painter(True)("output", "output"), "\033[33moutput\033[0m")
        self.assertEqual(Painter(True)("next", "next"), "\033[31mnext\033[0m")
        self.assertEqual(Painter(False)("report", "codex"), "report")
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(Painter(True)("report", "codex"), "report")

    def test_report_classifications_are_theme_configurable(self):
        self.path.write_text('[theme.colors]\nlabel = "blue"\ntotal = "magenta"\n'
                             'instruction = "#123456"\n')
        painter = Painter(True)
        self.assertEqual(painter("Name", "label"), "\033[34mName\033[0m")
        self.assertEqual(painter("40M", "total"), "\033[1m\033[35m40M\033[0m")
        self.assertEqual(painter("nenpi auto", "instruction"),
                         "\033[38;2;18;52;86mnenpi auto\033[0m")
        for role in TEXT_ROLES:
            self.assertEqual(Painter(False)("text", role), "text")

    def test_invalid_color_falls_back_without_escape_injection(self):
        self.path.write_text('[theme]\nname = "unknown"\n'
                             '[theme.colors]\ncodex = "invalid"\n')
        with patch("nenpi.theme.warn_once") as warn:
            name, colors = load_theme()
        self.assertEqual(name, "codex")
        self.assertEqual(colors["codex"], "cyan")
        self.assertEqual(warn.call_count, 2)
