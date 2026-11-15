"""events_adapter.py -- turn a camera program's raw detections into SPEC 3.2 events.

    python live_scene.py --jsonl --no-window | python events_adapter.py
    python events_adapter.py --demo            # a fake ~45 s session, no camera needed
    python events_adapter.py --demo --fast     # the same, instantly
    python events_adapter.py --zones zones_map.json --out events.jsonl

Input: one JSON object per line on stdin, as live_scene.py --jsonl writes them
(see the top of scene.py):
    {"time": "2026-11-14T12:00:00.000+00:00", "camera_id": "lounge",
     "image_w": 1280, "image_h": 854, "item": "keys", "box": [x1, y1, x2, y2],
     "det_conf": 0.41, "support": {"rel": "on", "anchor": "table_1"}, ...}
Every line from one frame shares the same "time"; a frame where nothing was seen
prints nothing, so silence also counts as missed frames (see SILENCE below).

Output: SPEC 3.2 events, appended to events.jsonl and printed to stdout:
    {"time": "14:05:10", "t": 1794696414.7, "item": "keys-1", "category": "keys",
     "zone": "table-1", "event": "appeared", "confidence": 0.82, "camera": "lounge"}

Rules (SPEC 3.2):
  appeared     seen in 2 of the last 3 frames in a zone where it wasn't
  disappeared  missing from its zone for 5 frames in a row
  moved        appeared in another zone within 60 s of disappearing (or seen in
               2 of the last 3 frames in a different zone while still present)
Boxes bigger than 15% of the frame are dropped (a detector "seeing keys" in half
the picture is a false alarm). With several boxes of one item in a frame, the
most confident one counts.

Zone ids: the slug of the furniture anchor id (table_1 -> table-1), or whatever
--zones maps it to: a JSON object {"table_1": "desk", "sofa_1": "sofa"}. The room
map in the page must use the same ids. Items on the floor or "unknown" (no
anchor) get "zone": null.

Standard library only. Status messages go to stderr; stdout carries events only.
"""
import argparse
import collections
import datetime as dt
import json
import queue
import random
import re
import sys
import threading
import time

SEEN_OF = (2, 3)        # appeared: seen in 2 of the last 3 frames
MISS_FRAMES = 5         # disappeared: missing this many frames in a row
MOVE_WINDOW = 60.0      # seconds: disappear then appear elsewhere = moved
MAX_BOX_FRAC = 0.15     # drop boxes bigger than this fraction of the frame
DEFAULT_GAP = 0.5       # seconds between frames until we have measured it
FRAME_IDLE = 0.4        # seconds without new lines before a frame is closed
ITEMS = ("keys", "wallet", "glasses")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "zone"


def to_unix(value):
    """ISO time (live_scene), unix seconds, or None -> unix seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def clock(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


class Track:
    """One item as seen by one camera."""
    def __init__(self):
        self.hist = collections.deque(maxlen=SEEN_OF[1])  # per frame: (zone, conf) or None
        self.zone = None          # zone it's present in (None = not present)
        self.present = False
        self.missing = 0
        self.gone = None          # (zone, t) of the last "disappeared"


class Adapter:
    def __init__(self, out_path, zone_map=None, echo=True):
        self.out_path = out_path
        self.zone_map = zone_map or {}
        self.echo = echo
        self.tracks = {}          # (camera, category) -> Track
        self.cams = {}            # camera -> {"t", "wall", "gaps", "silent"}
        self.count = collections.Counter()

    # ---------- output ----------
    def emit(self, ev):
        line = json.dumps(ev)
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)
        self.count[ev["event"]] += 1

    def item_event(self, event, cat, zone, t, conf, cam):
        ev = {"time": clock(t), "t": round(t, 3), "item": cat + "-1", "category": cat,
              "zone": zone, "event": event, "camera": cam}
        if conf is not None:
            ev["confidence"] = round(conf, 2)
        self.emit(ev)

    def person_event(self, event, zone, t, cam):
        self.emit({"time": clock(t), "t": round(t, 3), "zone": zone, "event": event, "camera": cam})

    # ---------- input ----------
    def zone_of(self, det):
        anchor = (det.get("support") or {}).get("anchor")
        if anchor is None:
            return None
        return self.zone_map.get(anchor) or slug(anchor)

    def frame(self, cam, t, dets):
        """One analysed frame from one camera: dets = list of raw detection dicts."""
        best = {}
        for d in dets:
            cat = str(d.get("item") or d.get("category") or "").lower().strip()
            if not cat:
                continue
            box, w, h = d.get("box"), d.get("image_w"), d.get("image_h")
            if box and w and h and (box[2] - box[0]) * (box[3] - box[1]) > MAX_BOX_FRAC * w * h:
                self.count["dropped_big_box"] += 1
                continue
            conf = d.get("det_conf", d.get("confidence"))
            conf = float(conf) if conf is not None else None
            if cat not in best or (conf or 0) > (best[cat][1] or 0):
                best[cat] = (self.zone_of(d), conf)
        info = self.cams.setdefault(cam, {"t": None, "wall": None, "gaps": collections.deque(maxlen=20), "silent": 0})
        if info["t"] is not None and t > info["t"]:
            info["gaps"].append(t - info["t"])
        info["t"], info["wall"], info["silent"] = t, time.time(), 0
        cats = set(best) | {c for (k, c) in self.tracks if k == cam}
        for cat in cats:
            self.step(cam, cat, best.get(cat), t)

    def gap(self, cam):
        g = sorted(self.cams[cam]["gaps"])
        return min(5.0, max(0.05, g[len(g) // 2])) if g else DEFAULT_GAP

    def silence(self):
        """Cameras that print nothing still had frames: count them as empty ones."""
        now = time.time()
        for cam, info in self.cams.items():
            if info["wall"] is None:
                continue
            g = self.gap(cam)
            due = int((now - info["wall"]) / g) - info["silent"]
            for _ in range(max(0, min(due, MISS_FRAMES + 1))):
                info["silent"] += 1
                t = info["t"] + g * info["silent"]
                for (k, cat) in list(self.tracks):
                    if k == cam:
                        self.step(cam, cat, None, t)

    # ---------- the hysteresis ----------
    def step(self, cam, cat, seen, t):
        tr = self.tracks.setdefault((cam, cat), Track())
        tr.hist.append(seen)
        votes = collections.Counter(s[0] for s in tr.hist if s is not None)
        conf_in = lambda z: max((s[1] or 0) for s in tr.hist if s is not None and s[0] == z) or None
        if tr.present:
            if seen is not None and seen[0] == tr.zone:
                tr.missing = 0
                return
            # seen somewhere else in 2 of the last 3 frames: it moved without vanishing
            for z, n in votes.items():
                if z != tr.zone and n >= SEEN_OF[0]:
                    tr.zone, tr.missing = z, 0
                    self.item_event("moved", cat, z, t, conf_in(z), cam)
                    return
            tr.missing += 1
            if tr.missing >= MISS_FRAMES:
                self.item_event("disappeared", cat, tr.zone, t, None, cam)
                tr.gone, tr.present, tr.zone, tr.missing = (tr.zone, t), False, None, 0
            return
        for z, n in votes.most_common():
            if n >= SEEN_OF[0]:
                moved = tr.gone is not None and tr.gone[0] != z and t - tr.gone[1] <= MOVE_WINDOW
                tr.present, tr.zone, tr.missing = True, z, 0
                self.item_event("moved" if moved else "appeared", cat, z, t, conf_in(z), cam)
                return


# ---------------------------------------------------------------- live mode ---
def run_stdin(ad, default_cam):
    q = queue.Queue()

    def reader():
        for line in sys.stdin:
            q.put(line)
        q.put(None)
    threading.Thread(target=reader, daemon=True).start()

    pending = {}   # camera -> {"time": raw time, "t": unix, "dets": [...], "wall": arrival}
    bad = 0

    def close(cam):
        p = pending.pop(cam)
        ad.frame(cam, p["t"], p["dets"])

    while True:
        try:
            line = q.get(timeout=0.1)
        except queue.Empty:
            line = ""
        if line is None:
            break
        line = line.strip() if line else ""
        if line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            cam = str(d.get("camera_id") or d.get("camera") or default_cam)
            key = d.get("time")
            p = pending.get(cam)
            if p is not None and p["time"] != key:
                close(cam)
                p = None
            if p is None:
                p = pending[cam] = {"time": key, "t": to_unix(key), "dets": [], "wall": time.time()}
            p["dets"].append(d)
            p["wall"] = time.time()
        for cam in [c for c, p in pending.items() if time.time() - p["wall"] > FRAME_IDLE]:
            close(cam)
        ad.silence()
    for cam in list(pending):
        close(cam)
    if bad:
        log("skipped %d lines that weren't JSON" % bad)


# ---------------------------------------------------------------- demo mode ---
def run_demo(ad, fast, seed=None):
    """A ~45 s session for the page's default room (zones desk, sofa, shelf, ...):
    keys and wallet on the desk, a weak false alarm on the shelf, you pick the keys
    up, walk to the sofa and put them down there."""
    rnd = random.Random(seed)
    cam, fps = "demo-phone", 2.0
    t0 = time.time()

    def det(item, anchor, conf):
        x = rnd.uniform(300, 900)
        return {"camera_id": cam, "image_w": 1280, "image_h": 720, "item": item,
                "box": [x, 400, x + 60, 440], "det_conf": round(min(0.99, max(0.05, conf + rnd.gauss(0, 0.05))), 3),
                "support": {"rel": "on", "anchor": anchor}}

    def frame(i, dets):
        t = t0 + i / fps
        if not fast:
            time.sleep(max(0.0, t - time.time()))
        ad.frame(cam, t, dets)
        return t

    ad.person_event("person_here", "desk", t0, cam)
    i = 0
    for _ in range(30):                       # 15 s: keys and wallet on the desk, keys blink out now and then
        dets = [det("wallet", "desk", 0.7)]
        if rnd.random() > 0.15:
            dets.append(det("keys", "desk", 0.75))
        if i == 20:
            dets.append(det("keys", "shelf", 0.25))          # one-frame false alarm: ignored
        if i == 22:
            dets.append(det("keys", "desk", 0.7) | {"box": [0, 0, 900, 500]})   # huge box: dropped
        frame(i, dets); i += 1
    for _ in range(8):                        # 4 s: keys picked up, only the wallet is seen
        frame(i, [det("wallet", "desk", 0.7)]); i += 1
    t = t0 + i / fps
    ad.person_event("person_left", "desk", t, cam)
    ad.person_event("person_here", "sofa", t + 0.5, cam)
    for _ in range(10):                       # 5 s: walking, nothing new
        frame(i, [det("wallet", "desk", 0.7)]); i += 1
    for _ in range(20):                       # 10 s: keys put down on the sofa
        dets = [det("wallet", "desk", 0.7)]
        if rnd.random() > 0.1:
            dets.append(det("keys", "sofa", 0.65))
        frame(i, dets); i += 1
    t = t0 + i / fps
    ad.person_event("person_left", "sofa", t, cam)
    for _ in range(8):                        # the wallet is still there; the keys stay on the sofa
        frame(i, [det("wallet", "desk", 0.7), det("keys", "sofa", 0.6)]); i += 1


def main():
    ap = argparse.ArgumentParser(description="Raw detections (JSON lines on stdin) -> SPEC 3.2 events")
    ap.add_argument("--out", default="events.jsonl", help="file to append events to (default events.jsonl)")
    ap.add_argument("--zones", help='JSON file mapping anchor ids to zone ids, e.g. {"table_1": "desk"}')
    ap.add_argument("--camera", default="camera-1", help="camera id for lines that have none")
    ap.add_argument("--demo", action="store_true", help="fake a short session instead of reading stdin")
    ap.add_argument("--fast", action="store_true", help="with --demo: don't wait in real time")
    ap.add_argument("--seed", type=int, help="with --demo: repeatable randomness")
    args = ap.parse_args()
    zone_map = {}
    if args.zones:
        with open(args.zones, encoding="utf-8") as f:
            zone_map = {str(k): slug(v) for k, v in json.load(f).items()}
    ad = Adapter(args.out, zone_map)
    log("writing events to %s" % args.out)
    try:
        if args.demo:
            run_demo(ad, args.fast, args.seed)
        else:
            run_stdin(ad, args.camera)
    except KeyboardInterrupt:
        pass
    log("done: " + ", ".join("%s %d" % kv for kv in sorted(ad.count.items())) if ad.count else "done: no events")


if __name__ == "__main__":
    main()
