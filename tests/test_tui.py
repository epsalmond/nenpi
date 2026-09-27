import asyncio
import contextlib
import io
import sys
import unittest
from unittest.mock import patch

try:
    import textual  # noqa: F401
except ImportError:  # pragma: no cover - exercised only without the optional extra
    textual = None

from nenpi.tui import ScanResult, SessionRecord, build_app, normalize_result
import nenpi.tui as tui


class EntryPointHelpTests(unittest.TestCase):
    def test_web_help_uses_console_arguments_without_starting_server(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["nenpi-web", "--help"]), \
             patch.object(tui, "serve_browser", side_effect=AssertionError("server started")), \
             contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                tui.web_main()
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("usage: nenpi-web", output.getvalue())

    def test_web_forwards_host_and_port_to_browser_server(self):
        with patch.object(tui, "serve_browser", return_value=7) as serve:
            result = tui.web_main(["--host", "127.0.0.1", "--port", "8123"])
        self.assertEqual(result, 7)
        serve.assert_called_once_with("127.0.0.1", 8123)


class SessionRecordTests(unittest.TestCase):
    def test_account_label_falls_back_to_account(self):
        result = normalize_result({"sessions": [{
            "session_id": "same", "harness": "claude", "account": "alias"
        }]})
        self.assertEqual(result.sessions[0].account_label, "alias")


@unittest.skipUnless(textual is not None, "nenpi[ui] is optional")
class BrowserPilotTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_theme_applies_and_survives_source_save(self):
        import tempfile
        from pathlib import Path
        from nenpi.config import load_config
        from nenpi.settings import SourceSettings

        for name in ("codex", "codex-light"):
            with self.subTest(theme=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text('[theme]\nname = "%s"\n'
                                '[theme.colors]\naccent = "#123456"\n'
                                'foreground = "#eeeeee"\nsurface = "#181818"\n' % name)
                settings = SourceSettings(path=path)
                with patch.dict("os.environ", {"NENPI_THEME": name}):
                    app = build_app(settings=settings, scanner=lambda *args: ScanResult())
                async with app.run_test() as pilot:
                    await pilot.pause()
                    self.assertEqual(app.theme, name)
                    self.assertEqual(app.current_theme.accent, "#123456")
                    self.assertEqual(app.query_one("#sessions").styles.border_left[1].hex,
                                     "#123456")
                    self.assertEqual(app.get_css_variables()["text"], "#eeeeee")
                    self.assertEqual(app.get_css_variables()["surface"], "#181818")
                    settings.save()
                    self.assertEqual(load_config(path).theme["colors"]["accent"], "#123456")

    async def test_worker_progress_result_and_filter_keep_ui_responsive(self):
        def scanner(settings, progress, cancelled):
            for done in range(1, 4):
                if cancelled():
                    return ScanResult()
                progress(done, 3)
                asyncio.run(asyncio.sleep(0.01))
            return ScanResult([
                SessionRecord("one", "codex", "lab", weighted_units=8),
                SessionRecord("two", "claude", "shop", weighted_units=2),
            ])

        app = build_app(scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause(0.15)
            self.assertIn("2 sessions", str(app.query_one("#status").render()))
            await pilot.click("#filter")
            await pilot.press("h", "a", "r", "n", "e", "s", "s", ":", "c", "o", "d", "e", "x")
            await pilot.pause()
            self.assertEqual(len(app.query_one("#sessions").children), 1)

    async def test_cancel_preserves_previous_result(self):
        calls = [0]

        def scanner(settings, progress, cancelled):
            calls[0] += 1
            if calls[0] == 1:
                return ScanResult([SessionRecord("old", "codex", "lab")])
            for _ in range(100):
                if cancelled():
                    return ScanResult()
                asyncio.run(asyncio.sleep(0.01))
            return ScanResult([SessionRecord("new", "codex", "lab")])

        app = build_app(scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            self.assertIn("1 sessions", str(app.query_one("#status").render()))
            app.start_scan()
            await pilot.pause(0.03)
            app.cancel_scan()
            await pilot.pause(0.05)
            self.assertEqual(len(app.result.sessions), 1)
            self.assertEqual(app.result.sessions[0].session_id, "old")

    async def test_detail_renders_all_prompt_rows_and_reductions(self):
        prompts = [{"short_id": "detail", "start": 1700000000 + index,
                    "api_turns": index + 1, "input_tokens": index * 10,
                    "context_peak": index * 20} for index in range(9)]
        reductions = [{"time": 1700000090, "kind": "unmarked", "before": 900,
                       "after": 400, "removed_tokens": 500, "saved_units": 1.25}]
        row = SessionRecord("detail-session", "claude", "project", model="opus",
                            payload={"short_id": "detail", "prompt_details": prompts,
                                     "reductions": reductions, "tokens": {"input": 90}})
        app = build_app(scanner=lambda *args: ScanResult([row]))
        async with app.run_test() as pilot:
            await pilot.pause(.1)
            await pilot.click("#sessions", offset=(10, 1))
            await pilot.press("enter")
            await pilot.pause()
            detail = str(app.query_one("#detail").render())
            self.assertIn("input=80", detail)
            self.assertIn("removed=500", detail)

    async def test_account_alias_is_visible_for_same_session_project(self):
        rows = [
            SessionRecord("same", "claude", "project", account_label="work"),
            SessionRecord("same", "claude", "project", account_label="personal"),
        ]
        app = build_app(scanner=lambda *args: ScanResult(rows))
        async with app.run_test() as pilot:
            await pilot.pause(.1)
            from textual.widgets import Label

            labels = [str(item.query_one(Label).render())
                      for item in app.query_one("#sessions").children]
            self.assertTrue(any("work" in label for label in labels))
            self.assertTrue(any("personal" in label for label in labels))
            await pilot.click("#sessions", offset=(10, 1))
            await pilot.press("enter")
            await pilot.pause()
            detail = str(app.query_one("#detail").render())
            self.assertIn("account", detail)
            self.assertTrue("work" in detail or "personal" in detail)
