"""Optional Chromium smoke: real dashboard training, replay, and responsive screenshots.

Install requirements-ui-qa.lock and Playwright Chromium first. The offline
default test command does not need a browser. Outputs stay under artifacts/.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from threading import Thread

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import participant_ui as ui


def check() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Install requirements-ui-qa.lock for this optional browser check") from exc
    output = ui.ROOT / "artifacts" / "ui-polish"
    output.mkdir(parents=True, exist_ok=True)
    server = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="browser-run-", dir=output) as directory, sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.wait_for_function("document.querySelector('#watch-map').options.length === 3")
            page.locator("#output-dir").fill(str(Path(directory).relative_to(ui.ROOT)))
            for field, value in (("steps", "8"), ("budget-steps", "8"), ("n-envs", "1"),
                                 ("rollout-length", "8"), ("ray-count", "3"), ("ray-angles", "-90,0,90")):
                page.locator(f"#{field}").fill(value)
            page.locator("#reward-speed").fill("0.25")
            page.locator("#ray-angles").fill("0,0,90")
            assert page.locator("#start").is_disabled()
            assert "unique" in page.locator("#run-error").inner_text()
            page.locator("#ray-angles").fill("-90,0,90")
            assert page.locator("#ray-lines line").count() == 3
            # SVG object-bounding-box filters clip horizontal/vertical lines
            # (one bounding-box dimension is zero). These rays must stay visible.
            assert page.locator("#ray-lines line").evaluate_all(
                "lines => lines.every(line => getComputedStyle(line).filter === 'none')"
            )
            arc_bounds = page.locator(".range-arc").evaluate("arc => {const b = arc.getBBox(); return {y:b.y, height:b.height}}")
            assert round(arc_bounds["y"]) == 76 and round(arc_bounds["height"]) == 194
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(output / "dashboard-desktop.png"), full_page=True)
            page.locator("#ray-diagram").screenshot(path=str(output / "rays-desktop.png"))
            page.locator("#start").click()
            page.wait_for_function("document.querySelector('#state-chip').textContent === 'completed'", timeout=90_000)
            assert not page.locator("#trainer").is_disabled()
            assert page.locator("#metric-update").inner_text() == "1"
            assert page.locator("#run-error").inner_text() == ""
            state = json.loads((Path(directory) / "training_state.json").read_text())
            assert state["budget_used"] == 8 and state["ray_angles_deg"] == [-90.0, 0.0, 90.0]
            page.locator("#watch-source").select_option("reference")
            page.locator("#watch-renderer").select_option("browser")
            page.locator("#watch-map").select_option("synthetic_oval")
            page.locator("#watch").click()
            link = page.get_by_role("link", name="Open browser fallback")
            page.wait_for_function("""() => document.querySelector('#watch-error').textContent.trim()
                || [...document.querySelectorAll('a')].some(a => a.textContent === 'Open browser fallback'
                    && a.getClientRects().length)""", timeout=120_000)
            assert page.locator("#watch-error").inner_text() == ""
            link.wait_for(state="visible")
            with page.expect_popup() as popup:
                link.click()
            replay = popup.value
            replay.on("pageerror", lambda error: errors.append(str(error)))
            replay.wait_for_function("!document.querySelector('#next-tick').disabled")
            replay.locator("#next-tick").click()
            assert replay.locator("#current-tick").inner_text() == "1"
            replay.screenshot(path=str(output / "browser-replay.png"), full_page=True)
            for width in (390, 800):
                page.set_viewport_size({"width": width, "height": 900})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"overflow at {width}px"
                page.screenshot(path=str(output / f"dashboard-{width}.png"), full_page=True)
                if width == 390:
                    page.locator("#ray-count").scroll_into_view_if_needed()
                    page.screenshot(path=str(output / "rays-mobile.png"))
            browser.close()
        assert not errors, errors
        print("Chromium dashboard smoke passed: custom rays/weights, 8 transitions, replay, 390/800/1440px; no JS errors")
        print("Screenshots: artifacts/ui-polish/")
    finally:
        ui.PROCESS.stop()
        ui.RENDER_PROCESS.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    check()
