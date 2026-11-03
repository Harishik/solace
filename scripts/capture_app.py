"""Screenshots and an animated GIF of the Live Sky app for the README (written to assets/).

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


def main():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", args=["--enable-gpu", "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport=SIZE, device_scale_factor=1)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(9000)
        page.screenshot(path=OUT / "live_globe.png")
        page.get_by_role("button", name="600×").click()
        page.wait_for_timeout(2500)
        frames = []
        for _ in range(36):
            frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB").resize((720, 428)))
            page.wait_for_timeout(120)
        frames = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in frames]
        frames[0].save(OUT / "live_sky.gif", save_all=True, append_images=frames[1:], duration=90, loop=0,
                       optimize=True)
        for name, shot in (("Asteroids", "live_asteroids.png"), ("Sky tonight", "live_sky_tonight.png")):
            page.get_by_role("button", name=name).click()
            if name == "Asteroids":
                page.wait_for_selector("#neo-cad tr.click", timeout=120000)
                page.locator("#neo-cad tr.click").nth(2).click()
                page.wait_for_timeout(2000)
            if name == "Sky tonight":  # ISS passes take a while on a cold server
                page.wait_for_selector("#sky-bodies table", timeout=180000)
            page.wait_for_timeout(5000)
            page.screenshot(path=OUT / shot)
        browser.close()
    for f in ("live_globe.png", "live_sky.gif", "live_asteroids.png", "live_sky_tonight.png"):
        print("wrote", OUT / f)


if __name__ == "__main__":
    main()
