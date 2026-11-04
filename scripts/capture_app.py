"""Screenshots and animated GIFs of the Live Sky app for the README (written to assets/).

Start the app first (`solace serve`), then:  .venv/bin/python scripts/capture_app.py
Needs `pip install playwright` and a local Google Chrome (no browser download).
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000"
OUT = Path("assets")
SIZE = {"width": 1280, "height": 760}


def gif(page, path, n, gap_ms, size=(720, 428), crop=None, ms=90):
    frames = []
    for _ in range(n):
        im = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
        frames.append((im.crop(crop) if crop else im).resize(size))
        page.wait_for_timeout(gap_ms)
    frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--enable-gpu", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport=SIZE, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("#reentry tr.click", timeout=120000)
        page.wait_for_timeout(6000)
        page.screenshot(path=OUT / "live_globe.png")

        # The Earth turning under the catalogue at 600x.
        page.locator("#view-globe .rate[data-rate='600']").click()
        page.wait_for_timeout(2500)
        gif(page, OUT / "live_sky.gif", 48, 120)
        page.locator("#view-globe .reset").click()

        # Re-entry watch: the first object on the list, with its forecast and ground track.
        page.locator("#reentry tr.click").first.click()
        page.wait_for_timeout(3500)
        page.screenshot(path=OUT / "live_reentry.png")

        # Riding along with the ISS; wait for daylight at 60x, then film at 10x.
        page.locator("#assets .asset[data-id='25544']").click()
        page.wait_for_selector("#follow", timeout=30000)
        page.wait_for_timeout(1500)
        page.locator("#follow").click()
        page.locator("#view-globe .rate[data-rate='60']").click()
        page.wait_for_function("document.querySelector('#live')?.textContent.includes('Sunlit')", timeout=240000)
        page.wait_for_timeout(9000)
        page.locator("#view-globe .rate[data-rate='10']").click()
        page.wait_for_timeout(2000)
        page.screenshot(path=OUT / "live_iss.png")
        gif(page, OUT / "live_iss.gif", 40, 150, size=(640, 480), crop=(160, 40, 1000, 670))
        page.locator("#view-globe .reset").click()

        page.get_by_role("button", name="Asteroids").click()
        page.wait_for_selector("#neo-cad tr.click", timeout=120000)
        page.locator("#neo-cad tr.click").nth(2).click()
        page.wait_for_timeout(6000)
        page.screenshot(path=OUT / "live_asteroids.png")

        page.get_by_role("button", name="Sky tonight").click()
        page.wait_for_selector("#sky-bodies table", timeout=180000)
        page.wait_for_selector("#sky-now table", timeout=60000)
        page.mouse.move(420, 200)
        page.mouse.down()
        page.mouse.move(420, 420, steps=12)
        page.mouse.up()
        page.wait_for_timeout(3000)
        page.screenshot(path=OUT / "live_sky_tonight.png")
        browser.close()
    for f in ("live_globe.png", "live_sky.gif", "live_reentry.png", "live_iss.png", "live_iss.gif",
              "live_asteroids.png", "live_sky_tonight.png"):
        print("wrote", OUT / f)


if __name__ == "__main__":
    main()
