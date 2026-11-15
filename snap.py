"""
Take one baseline photo from every camera, for mapping the room in trace-sim/index.html.

Uses the same cameras as multicam.py (cameras.json, or --cam "Name=source", or webcam 0):
    python snap.py
    python snap.py --cam "Sofa=http://192.168.1.31:4747/video" --cam "Desk=http://192.168.1.23:8080/video"

Saves baseline/<camera-id>.jpg (camera-id = the name, lower-case and hyphenated, as in
multicam.py). Then open trace-sim/index.html, "Map a real room from camera photos",
"Add phone photos" and pick the files in baseline/. baseline/ holds photos of the room and
maybe people: it is git-ignored, never commit it.

Only needs OpenCV (no torch / detector), so it starts in a second or two.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2

CAMERAS_FILE = Path("cameras.json")
OUT = Path("baseline")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "cam"


def camera_specs(args):
    if args.cam:
        specs = []
        for c in args.cam:
            name, _, src = c.partition("=")
            if not src:
                name, src = f"Camera {len(specs) + 1}", name
            specs.append({"name": name.strip(), "source": src.strip()})
        return specs
    if CAMERAS_FILE.exists():
        return json.loads(CAMERAS_FILE.read_text())
    print("No cameras given and no cameras.json - using webcam 0.")
    return [{"name": "Laptop", "source": 0}]


def open_cam(src):
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    if isinstance(src, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(src)


def grab(src, settle):
    """Open the stream, let exposure settle for `settle` seconds, return the newest frame."""
    cap = open_cam(src)
    if not cap.isOpened():
        return None
    frame, t0 = None, time.time()
    while time.time() - t0 < settle or frame is None:
        ok, f = cap.read()
        if ok:
            frame = f
        elif time.time() - t0 > settle + 5:
            break
    cap.release()
    return frame


def main():
    ap = argparse.ArgumentParser(description="Save one baseline photo per camera for room mapping")
    ap.add_argument("--cam", action="append", help='"Name=source", source = URL or webcam index (repeatable)')
    ap.add_argument("--settle", type=float, default=1.5, help="seconds to let exposure settle (default 1.5)")
    ap.add_argument("--out", default=str(OUT), help="folder to save into (default baseline)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    seen, saved = set(), 0
    for s in camera_specs(args):
        cid = s.get("id") or slug(s["name"])
        while cid in seen:
            cid += "-2"
        seen.add(cid)
        print(f"{s['name']} ({s['source']}) ... ", end="", flush=True)
        frame = grab(s["source"], args.settle)
        if frame is None:
            print("no picture. Is the phone app running and on the same Wi-Fi?")
            continue
        path = out / f"{cid}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1
        print(f"saved {path} ({frame.shape[1]}x{frame.shape[0]})")
    print(f"\n{saved} photo(s) in {out}/. Next: `py map_room.py` (Gemini maps the room and "
          "the zones), or by hand in trace-sim/index.html.")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
