import asyncio
import unittest

try:
    import textual  # noqa: F401
except ImportError:  # pragma: no cover - exercised only without the optional extra
    textual = None

from nenpi.tui import ScanResult, SessionRecord, build_app


@unittest.skipUnless(textual is not None, "nenpi[ui] is optional")
class BrowserPilotTests(unittest.IsolatedAsyncioTestCase):
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
