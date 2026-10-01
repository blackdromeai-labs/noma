"""Record the playground being used: screenshots (PNG) and screen recordings (WebM).

    pip install playwright && playwright install chromium
    noma serve --port 8000 &
    python scripts/record_playground.py --url http://127.0.0.1:8000 --out media

Everything on screen comes from the running server; nothing is mocked. Convert a recording
to a GIF with ffmpeg (see docs/SERVING.md).
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

W, H = 1280, 920


def wait_ready(page, url: str) -> None:
    page.goto(url, wait_until="networkidle")
    page.wait_for_function("fetch('/v1/info').then(r => r.json()).then(i => i.ready)", timeout=600_000)
    page.wait_for_function("!document.querySelector('#statusText').textContent.includes('Connecting')")
    page.wait_for_timeout(2200)  # entrance animation


def decide(page, hold: int = 2600) -> None:
    btn = page.locator("#decideBtn")
    btn.scroll_into_view_if_needed()
    btn.hover()
    page.wait_for_timeout(350)
    page.evaluate("document.querySelector('#answers').innerHTML = ''")
    btn.click()
    page.wait_for_selector("#answers .card", timeout=120_000)
    page.wait_for_timeout(hold)


def preset(page, name: str) -> None:
    b = page.locator("#presets button", has_text=name)
    b.scroll_into_view_if_needed()
    b.hover()
    page.wait_for_timeout(250)
    b.click()
    page.wait_for_timeout(700)


def scene_presets(page, url):
    wait_ready(page, url)
    decide(page)
    for name in ("Agent step check", "Model routing", "Contract clause"):
        preset(page, name)
        decide(page)


def scene_custom(page, url):
    """Type a new state and a new question by hand."""
    wait_ready(page, url)
    preset(page, "Agent step check")
    state = page.locator("#state")
    state.click()
    state.fill("")
    state.type("Step 4 of 6: ran `terraform apply` on prod-eu.\n"
               "Output: Apply complete! Resources: 3 added, 0 changed, 1 destroyed.\n"
               "Destroyed: aws_db_instance.orders_replica\n"
               "The plan approved in review listed 3 additions and no deletions.", delay=9)
    page.wait_for_timeout(400)
    page.locator("#addQ").click()
    q = page.locator("#questions .q").last
    q.scroll_into_view_if_needed()
    q.locator(".q-key").fill("matches_plan")
    q.locator(".q-ins").type("Did the apply match the approved plan?", delay=14)
    page.wait_for_timeout(400)
    decide(page, hold=3200)


def scene_code(page, url):
    wait_ready(page, url)
    decide(page, hold=1200)
    page.locator("#codeH").scroll_into_view_if_needed()
    page.wait_for_timeout(900)
    for t in ("python", "jev", "curl"):
        page.locator(f".tabs button[data-tab={t}]").click()
        page.wait_for_timeout(1300)
    page.locator("#copyBtn").click()
    page.wait_for_timeout(900)
    page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
    page.wait_for_timeout(900)
    page.locator("#themeBtn").click()
    page.wait_for_timeout(1800)
    page.locator("#themeBtn").click()
    page.wait_for_timeout(1200)


def full_page(page, path: Path) -> None:
    """The background is fixed to the viewport, so grow the viewport instead of stitching."""
    h = page.evaluate("document.documentElement.scrollHeight")
    page.set_viewport_size({"width": W, "height": h})
    page.wait_for_timeout(400)
    page.screenshot(path=str(path))
    page.set_viewport_size({"width": W, "height": H})


SCENES = {"presets": scene_presets, "custom": scene_custom, "code": scene_code}


def screenshots(browser, url: str, out: Path) -> None:
    ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=2)
    page = ctx.new_page()
    wait_ready(page, url)
    page.screenshot(path=str(out / "playground-hero.png"))
    decide(page, hold=1800)
    full_page(page, out / "playground-triage.png")
    page.locator(".panel.answers").screenshot(path=str(out / "answers-triage.png"))
    for name, slug in (("Agent step check", "agent"), ("Moderation", "moderation"),
                       ("Model routing", "routing"), ("Contract clause", "contract")):
        preset(page, name)
        decide(page, hold=1800)
        page.locator(".grid").screenshot(path=str(out / f"playground-{slug}.png"))
    page.locator("section.code").screenshot(path=str(out / "playground-code.png"))
    page.evaluate("window.scrollTo(0, 0)")
    page.locator("#themeBtn").click()
    page.wait_for_timeout(900)
    full_page(page, out / "playground-light.png")
    ctx.close()

    ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True)
    page = ctx.new_page()
    wait_ready(page, url)
    decide(page, hold=1800)
    page.locator("#aH").scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    page.screenshot(path=str(out / "playground-mobile.png"))
    ctx.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="media")
    ap.add_argument("--channel", default=None, help="e.g. 'chrome' to use an installed browser")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.channel)
        screenshots(browser, args.url, out)
        for name, scene in SCENES.items():
            tmp = out / f"_video_{name}"
            ctx = browser.new_context(viewport={"width": W, "height": H},
                                      record_video_dir=str(tmp), record_video_size={"width": W, "height": H})
            ctx.grant_permissions(["clipboard-read", "clipboard-write"])
            page = ctx.new_page()
            t = time.time()
            scene(page, args.url)
            ctx.close()  # flushes the video
            shutil.move(str(next(tmp.glob("*.webm"))), str(out / f"playground-{name}.webm"))
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"recorded {name} ({time.time() - t:.0f}s)", flush=True)
        browser.close()
    print("done:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
