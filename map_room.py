"""map_room.py -- Gemini maps the room from the phones' photos. No clicking.

    py snap.py                       # 1. one photo per phone -> baseline/<camera-id>.jpg
    py map_room.py                   # 2. Gemini -> room.json (page) + zones.json (multicam)
    py map_room.py --width 6 --depth 4.5      # if you've measured the room (better scale)
    py map_room.py --photos a.jpg b.jpg       # other photos (camera id = file name)
    py map_room.py --response baseline/gemini-map.json   # re-check a saved answer, no API call

What it writes:
  room.json   the zone map for trace-sim/index.html ("Load a different room" -> "Open
              room.json"), in metres, with a landmark sentence per zone for guidance.
  zones.json  multicam.py's zones: a box per piece of furniture on each phone's picture,
              named exactly as in room.json, so events match the map. The old file is kept
              as zones.json.bak.
  baseline/<camera>-zones.jpg   each photo with the boxes drawn, to check by eye.

All photos go to Gemini (Google) in one request. Take them with nobody in the room.
Gemini only describes what it sees; the code checks every value (types, sizes, boxes,
names) and fills in cover / search time from the furniture type, never from the LLM.
Needs GEMINI_API_KEY in .env (see .env.example). GEMINI_MAP_MODEL picks the model
(default gemini-3.5-flash: a few seconds; gemini-3.8-flash took minutes).
"""
import argparse
import base64
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
BASELINE = Path("baseline")
VIEW_W = 960    # multicam.py resizes every frame to this width; zones use these pixels
TIMEOUT = 300   # s to wait for Gemini's answer (several photos can take a minute or two)
TYPES = ["sofa", "bed", "chair", "counter", "shelf", "hook", "desk", "table"]
# per type, as in the page: chance an item put there ends up hidden, seconds to search it
COVER = {"sofa": 0.45, "bed": 0.5, "desk": 0.2, "table": 0.15, "chair": 0.2, "counter": 0.25,
         "shelf": 0.2, "hook": 0.6}
CHECK = {"sofa": 20, "bed": 20, "desk": 10, "table": 8, "chair": 6, "counter": 15, "shelf": 10,
         "hook": 12}
WORDS = [("sofa", "sofa|couch|settee"), ("bed", "bed"), ("chair", "chair|stool|armchair"),
         ("counter", "counter|worktop|kitchen|sideboard|dresser"),
         ("shelf", "shelf|shelves|bookcase|cabinet|drawer|chest|mantel|windowsill|sill"),
         ("hook", "hook|coat|peg|rack|hanger"), ("desk", "desk"),
         ("table", "table|nightstand|bedside")]

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "room_name": {"type": "STRING"},
        "width_m": {"type": "NUMBER", "description": "along the back wall"},
        "depth_m": {"type": "NUMBER", "description": "from the back wall to the front"},
        "back_wall": {"type": "STRING", "description": "which wall you took as the back wall"},
        "problems": {"type": "ARRAY", "items": {"type": "STRING"},
                     "description": "reasons these photos are poor for mapping, if any"},
        "people_in_view": {"type": "BOOLEAN"},
        "zones": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "type": {"type": "STRING", "enum": TYPES},
                "x_m": {"type": "NUMBER"}, "y_m": {"type": "NUMBER"},
                "w_m": {"type": "NUMBER"}, "d_m": {"type": "NUMBER"},
                "landmark": {"type": "STRING"},
                "seen_in": {"type": "ARRAY", "items": {
                    "type": "OBJECT",
                    "properties": {
                        "camera": {"type": "STRING"},
                        "box_2d": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                    },
                    "required": ["camera", "box_2d"]}},
            },
            "required": ["name", "type", "x_m", "y_m", "w_m", "d_m", "seen_in"]}},
    },
    "required": ["width_m", "depth_m", "zones"],
}

PROMPT = """You are mapping a room for Trace, an app that helps blind people and people with
dementia find their keys, wallet and glasses. You get {n} photo(s) of ONE room, each taken
by a phone standing where it will watch the room. Each photo is preceded by its camera id.

List every surface where someone might put down keys, a wallet or glasses: sofas, beds,
chairs, counters and worktops, shelves and cabinets, coat hooks, desks, tables (coffee,
dining, bedside). Not the floor, walls, windows, lamps, TVs or plants.

For each one:
- name: short everyday words a person would say ("Sofa", "Coffee table", "Coat hook"). No
  colours (blind users). Unique: two chairs are "Chair by the window" and "Desk chair".
- type: one of {types}.
- its footprint on a top-down plan of the room, in metres: x_m, y_m = its back-left corner,
  w_m = width along the back wall, d_m = depth towards the front. The origin is the room's
  back-left corner; x runs along the back wall, y towards the front. Take as the back wall
  the wall the first photo faces most directly. Judge scale from typical sizes (sofa about
  2 m wide and 0.9 m deep, door 0.8 m wide, dining table 0.8 m deep, chair 0.5 m).
- landmark: where it is, for someone who cannot see: "against the back wall, left of the
  window", "by the door". Positions only; no colours.
- seen_in: EVERY photo it appears in, with box_2d = [ymin, xmin, ymax, xmax] normalised
  0-1000 around the top surface where things would be put down (the seat of a sofa, the
  table top), not the legs or the wall above. The same furniture seen by two phones is ONE
  entry with two seen_in items, not two entries.
Also give the room's width_m and depth_m{sizes}.
Set people_in_view if any person is visible in any photo. In problems, list anything that
makes these photos poor for mapping (a face or body filling the frame, a close-up of one
object, too dark, the photos aren't of one room). Don't invent furniture you can't see.
"""


# ------------------------------------------------------------ gemini
def load_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    env = HERE / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip() == "GEMINI_API_KEY":
                key = value.strip().strip('"').strip("'")
    return key


MAP_MODEL = "gemini-3.5-flash"   # 6 s per map in testing; gemini-3.8-flash took 161 s


def default_model():
    if os.environ.get("GEMINI_MAP_MODEL"):
        return os.environ["GEMINI_MAP_MODEL"]
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "GEMINI_MAP_MODEL" and v.strip():
                return v.strip()
    return MAP_MODEL


def ask_gemini(photos, key, model, width=None, depth=None):
    sizes = ""
    if width and depth:
        sizes = " (measured: %.2f m wide along the back wall, %.2f m deep -- use these)" % (width, depth)
    parts = [{"text": PROMPT.format(n=len(photos), types=", ".join(TYPES), sizes=sizes)}]
    for cid, img in photos.items():
        h, w = img.shape[:2]
        s = min(1.0, 1280 / max(h, w))
        small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        parts.append({"text": "Camera id: %s" % cid})
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": base64.b64encode(buf.tobytes()).decode()}})
    config = {"temperature": 0.2, "responseMimeType": "application/json", "responseSchema": SCHEMA,
              "thinkingConfig": {"thinkingLevel": "low"}}   # mapping needs looking, not long thought
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
    data, attempt = None, 0
    while data is None:
        attempt += 1
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.loads(r.read())
            print("  Gemini answered in %.0f s" % (time.time() - t0))
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code == 400 and "thinking" in msg.lower() and "thinkingConfig" in config:
                config.pop("thinkingConfig")      # this model doesn't take the setting: drop it
                attempt -= 1
                continue
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                print("  Gemini busy (%s), trying again in %d s ..." % (e.code, 10 * attempt))
                time.sleep(10 * attempt)
                continue
            sys.exit("Gemini said %s: %s" % (e.code, msg[:500]))
        except (TimeoutError, socket.timeout, urllib.error.URLError) as e:
            if attempt < 2:
                print("  no answer after %.0f s (%s), trying once more ..." % (time.time() - t0, e))
                continue
            sys.exit("Gemini didn't answer in time (%s). Check the internet connection, or try a "
                     "different model: py map_room.py --model gemini-3.5-flash-lite" % e)
    try:
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        return json.loads(text)
    except Exception:
        sys.exit("Gemini's answer wasn't the JSON asked for:\n%s" % json.dumps(data)[:1500])


# ---------------------------------------------------------- checking
def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "zone"


def num(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v)) if v == v else default


def furniture_type(t, name):
    t = str(t or "").lower().strip()
    if t in TYPES:
        return t
    for k, pat in WORDS:
        if re.search(pat, str(name).lower()):
            return k
    return "table"


def build(answer, photos, width=None, depth=None, name=None):
    """Gemini's answer -> (room map, multicam zones, problems). Every value is checked."""
    problems = []
    W = width or num(answer.get("width_m"), 1.5, 30, 6.0)
    H = depth or num(answer.get("depth_m"), 1.5, 30, 5.0)
    room = {"name": name or str(answer.get("room_name") or "Mapped room")[:40],
            "width": round(W, 2), "height": round(H, 2), "zones": []}
    cam_zones, ids, names = [], set(), set()
    for z in answer.get("zones") or []:
        nm = re.sub(r"\s+", " ", str(z.get("name") or "").strip())[:40]
        if not nm:
            problems.append("a zone with no name: skipped")
            continue
        base, k = nm, 2
        while nm.lower() in names:
            nm = "%s %d" % (base, k)
            k += 1
        names.add(nm.lower())
        zid = slug(nm)
        while zid in ids:
            zid += "-2"
        ids.add(zid)
        t = furniture_type(z.get("type"), nm)
        w = num(z.get("w_m"), 0.3, W, 1.0)
        h = num(z.get("d_m"), 0.3, H, 0.6)
        x = num(z.get("x_m"), 0.0, W - w, 0.0)
        y = num(z.get("y_m"), 0.0, H - h, 0.0)
        seen = []
        for s in z.get("seen_in") or []:
            cam, b = str(s.get("camera", "")), s.get("box_2d") or []
            if cam not in photos:
                problems.append("%s: camera %r isn't one of the photos: ignored" % (nm, cam))
                continue
            if len(b) != 4:
                problems.append("%s on %s: box isn't 4 numbers: ignored" % (nm, cam))
                continue
            y1, x1, y2, x2 = [num(v, 0, 1000, 0) for v in b]
            if x2 - x1 < 5 or y2 - y1 < 5:
                problems.append("%s on %s: empty box: ignored" % (nm, cam))
                continue
            ih, iw = photos[cam].shape[:2]
            vh = ih * VIEW_W / iw
            rect = [int(x1 / 1000 * VIEW_W), int(y1 / 1000 * vh), int(x2 / 1000 * VIEW_W), int(y2 / 1000 * vh)]
            cam_zones.append({"name": nm, "id": zid, "rect": rect, "camera": cam})
            seen.append(cam)
        zone = {"id": zid, "name": nm, "type": t, "camera": bool(seen), "cover": COVER[t],
                "check": CHECK[t], "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)}
        lm = str(z.get("landmark") or "").strip()
        if lm:
            zone["landmark"] = lm[:120]
        if seen:
            zone["seen_by"] = seen
        room["zones"].append(zone)
    if not room["zones"]:
        problems.append("no furniture found")
    return room, cam_zones, problems


def draw(photos, cam_zones, out_dir):
    for cid, img in photos.items():
        v = cv2.resize(img, (VIEW_W, int(img.shape[0] * VIEW_W / img.shape[1])))
        for z in cam_zones:
            if z["camera"] == cid:
                x1, y1, x2, y2 = z["rect"]
                cv2.rectangle(v, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(v, z["name"], (x1 + 4, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / ("%s-zones.jpg" % cid)), v)


def main():
    ap = argparse.ArgumentParser(description="Gemini maps the room from the phones' photos")
    ap.add_argument("--photos", nargs="+", help="photos (default: baseline/*.jpg from snap.py)")
    ap.add_argument("--width", type=float, help="room width along the back wall, metres, if measured")
    ap.add_argument("--depth", type=float, help="room depth, metres, if measured")
    ap.add_argument("--name", help="room name")
    ap.add_argument("--model", default=None, help="Gemini model (default GEMINI_MAP_MODEL / GEMINI_MODEL)")
    ap.add_argument("--response", help="use this saved Gemini answer instead of calling the API")
    ap.add_argument("--force", action="store_true", help="write the map even if Gemini says the photos are poor")
    ap.add_argument("--room", default="room.json")
    ap.add_argument("--zones", default="zones.json")
    a = ap.parse_args()

    paths = [Path(p) for p in a.photos] if a.photos else sorted(
        p for p in BASELINE.glob("*.jpg") if not p.stem.endswith("-zones"))
    photos = {}
    for pth in paths:
        img = cv2.imread(str(pth))
        if img is None:
            print("can't read %s: skipped" % pth)
        else:
            photos[slug(pth.stem)] = img
    if not photos:
        sys.exit("no photos: run `py snap.py` first (saves baseline/<camera>.jpg), or pass --photos")
    BASELINE.mkdir(exist_ok=True)

    if a.response:
        answer = json.loads(Path(a.response).read_text(encoding="utf-8"))
    else:
        key = load_key()
        if not key:
            sys.exit("no GEMINI_API_KEY (environment or .env -- see .env.example)")
        model = a.model or default_model()
        print("Sending %d photo(s) (%s) to Gemini %s ..." % (len(photos), ", ".join(photos), model))
        answer = ask_gemini(photos, key, model, a.width, a.depth)
        Path(BASELINE / "gemini-map.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")

    room, cam_zones, problems = build(answer, photos, a.width, a.depth, a.name)
    for pr in problems:
        print("  note: " + pr)
    bad = [str(x) for x in answer.get("problems") or [] if str(x).strip()]
    if answer.get("people_in_view"):
        bad.insert(0, "people are in view: take the photos with nobody in the room "
                      "(they were sent to Gemini; snap.py again when it's empty)")
    if bad and not a.force:
        print("\nThese photos aren't good for mapping, so nothing was written:")
        for b in bad:
            print("  - " + b)
        sys.exit("Fix that and run snap.py + map_room.py again, or add --force to use the result anyway.")
    if not room["zones"]:
        sys.exit("Gemini found no furniture. Are the photos of the room, with surfaces in view?")
    Path(a.room).write_text(json.dumps(room, indent=2), encoding="utf-8")
    zp = Path(a.zones)
    if zp.exists():
        zp.with_suffix(".json.bak").write_text(zp.read_text(encoding="utf-8"), encoding="utf-8")
    zp.write_text(json.dumps(cam_zones, indent=2), encoding="utf-8")
    draw(photos, cam_zones, BASELINE)

    print("\n%s: %.1f x %.1f m%s" % (room["name"], room["width"], room["height"],
                                     " (back wall: %s)" % answer["back_wall"] if answer.get("back_wall") else ""))
    for z in room["zones"]:
        print("  %-22s %-8s %4.1f x %.1f m at (%.1f, %.1f)  %s" % (
            z["name"], z["type"], z["w"], z["h"], z["x"], z["y"],
            ("watched by " + ", ".join(z["seen_by"])) if z.get("seen_by") else "no phone sees it"))
    print("\nWrote %s (open it in the page: Load a different room -> Open room.json),\n"
          "      %s (%d boxes for multicam.py), and baseline/<camera>-zones.jpg to check by eye."
          % (a.room, a.zones, len(cam_zones)))
    print("Gemini's layout is an estimate: fix sizes in the page's top-view editor if needed;\n"
          "renaming there means renaming in multicam too, so prefer re-running this.")


if __name__ == "__main__":
    main()
