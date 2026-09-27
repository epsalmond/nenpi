import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nenpi.config import load_config, save_config
from nenpi.drain import Painter
from nenpi.theme import load_theme


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

    def test_config_roundtrip_and_report_overrides(self):
        self.path.write_text('[theme]\nname = "codex-light"\n'
                             '[theme.colors]\ncodex = "#123456"\naccent = "blue"\n')
        before = load_config().theme
        save_config(load_config())
        self.assertEqual(load_config().theme, before)
        self.assertEqual(Painter(True)("report", "codex"),
                         "\033[38;2;18;52;86mreport\033[0m")
        self.assertEqual(Painter(False)("report", "codex"), "report")
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            self.assertEqual(Painter(True)("report", "codex"), "report")

    def test_invalid_color_falls_back_without_escape_injection(self):
        self.path.write_text('[theme]\nname = "unknown"\n'
                             '[theme.colors]\ncodex = "invalid"\n')
        with patch("nenpi.theme.warn_once") as warn:
            name, colors = load_theme()
        self.assertEqual(name, "codex")
        self.assertEqual(colors["codex"], "cyan")
        self.assertEqual(warn.call_count, 2)
