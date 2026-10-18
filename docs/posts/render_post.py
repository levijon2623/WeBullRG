"""
render_post.py -- turn a share-card HTML page into a PNG (operator, 2026-10-18).

    python docs/posts/render_post.py docs/posts/2026-10-16-qqq-1030-puts.html
    python docs/posts/render_post.py CARD.html --size 1080x1350 --scale 2 --out CARD.png

Uses the machine's own headless Microsoft Edge or Google Chrome -- no
packages, no network. The card page lays itself out at --size CSS pixels; the
PNG is --scale times that (2 = crisp on phones). The HTML's inline script runs
before the shot (--virtual-time-budget), so cards may draw their SVG in JS.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def browser():
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    for name in ("msedge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    sys.exit("no Edge/Chrome found -- install one or add it to CANDIDATES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("html")
    ap.add_argument("--out")
    ap.add_argument("--size", default="1080x1350", help="CSS px, WxH (4:5 portrait default)")
    ap.add_argument("--scale", type=float, default=2.0)
    a = ap.parse_args()
    src = pathlib.Path(a.html).resolve()
    out = pathlib.Path(a.out).resolve() if a.out else src.with_suffix(".png")
    w, h = (int(v) for v in a.size.lower().split("x"))
    if out.exists():
        out.unlink()
    prof = tempfile.mkdtemp(prefix="render_post_")
    cmd = [browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
           f"--user-data-dir={prof}", f"--window-size={w},{h}",
           f"--force-device-scale-factor={a.scale}", "--virtual-time-budget=5000",
           f"--screenshot={out}", src.as_uri()]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
    # 🚨 Edge on Windows returns from its launcher at once and writes the PNG a
    # few seconds later from a child process: wait for the file to settle.
    last, t0 = -1, time.time()
    while time.time() - t0 < 60:
        size = out.stat().st_size if out.exists() else -1
        if size > 0 and size == last:
            break
        last = size
        time.sleep(0.75)
    shutil.rmtree(prof, ignore_errors=True)
    print(f"{out}  ({int(w * a.scale)}x{int(h * a.scale)})" if out.exists() else "render FAILED")


if __name__ == "__main__":
    main()
