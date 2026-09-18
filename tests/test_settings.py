import json
import tempfile
import unittest
from pathlib import Path

from nenpi.settings import SourceSettings, discover_sources
from nenpi.tui import SessionRecord, filter_sessions


class SourceSettingsTests(unittest.TestCase):
    def test_discovery_finds_claude_and_codex_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / ".claude" / "projects").mkdir(parents=True)
            (home / ".codex-work" / "sessions").mkdir(parents=True)
            sources = discover_sources(home)
            self.assertEqual({source.harness for source in sources}, {"claude", "codex"})

    def test_discovery_honors_configured_custom_root(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            custom = home / "profiles" / "claude-work"
            (custom / "projects").mkdir(parents=True)
            config = home / ".config" / "nenpi" / "config.toml"
            config.parent.mkdir(parents=True)
            config.write_text('[claude]\nroots = ["%s"]\n' % custom)
            sources = discover_sources(home)
            self.assertEqual(
                [(source.harness, Path(source.path)) for source in sources],
                [("claude", custom)],
            )

    def test_disabled_configured_root_stays_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            custom = home / "profiles" / "claude-work"
            (custom / "projects").mkdir(parents=True)
            config = home / ".config" / "nenpi" / "config.toml"
            config.parent.mkdir(parents=True)
            config.write_text('[claude]\nroots = ["%s"]\n' % custom)
            settings_path = home / "browser.json"
            source = SourceSettings(
                [SourceSettings.load(settings_path, home).sources[0]]
            ).sources[0]
            source.enabled = False
            settings = SourceSettings([source])
            settings.save(settings_path)
            restored = SourceSettings.load(settings_path, home)
            self.assertEqual(restored.enabled_sources(), [])
            self.assertEqual(restored.sources[0].path, str(custom))

    def test_ignored_discovered_roots_are_keyed_by_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared"
            settings = SourceSettings(
                ignored_discovered=[{"harness": "claude", "path": str(path)}]
            )
            identity = str(path.resolve())
            self.assertIn(("claude", identity), settings.ignored_discovered)
            self.assertNotIn(("codex", identity), settings.ignored_discovered)

    def test_disabled_source_is_persisted_and_not_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "transcripts"
            root.mkdir()
            path = Path(directory) / "config.json"
            settings = SourceSettings()
            source = settings.add(root, "codex")
            settings.set_enabled(source.id, False)
            settings.save(path)
            restored = SourceSettings.load(path, Path(directory) / "empty-home")
            self.assertEqual(restored.enabled_sources(), [])
            self.assertFalse(restored.get(source.id).enabled)
            self.assertEqual(json.loads(path.read_text())["version"], 1)

    def test_relative_and_missing_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = SourceSettings()
            with self.assertRaises(ValueError):
                settings.add(Path("relative"), "codex")
            with self.assertRaises(ValueError):
                settings.add(Path(directory) / "missing", "codex")

    def test_removed_discovered_root_is_tombstoned(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            root = home / ".claude" / "projects"
            root.mkdir(parents=True)
            path = home / "config.json"
            settings = SourceSettings.load(path, home)
            source = settings.sources[0]
            settings.remove(source.id)
            settings.save(path)
            restored = SourceSettings.load(path, home)
            self.assertEqual(restored.sources, [])

    def test_filter_supports_harness_project_dates_and_sort(self):
        rows = [
            SessionRecord("a", "claude", "Shop", start=100, end=200, weighted_units=4),
            SessionRecord("b", "codex", "Lab", start=300, end=400, weighted_units=8),
        ]
        self.assertEqual([row.session_id for row in filter_sessions(rows, "harness:codex")], ["b"])
        self.assertEqual([row.session_id for row in filter_sessions(rows, "project:shop")], ["a"])
        self.assertEqual([row.session_id for row in filter_sessions(rows, "since:1970-01-01 until:1970-01-01")], [])
        self.assertEqual([row.session_id for row in filter_sessions(rows, "", "start")], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
