"""
FindIt multi-camera tracker - several phones (and/or webcams) watching one room,
sharing ONE memory of items. Keys seen by the hall camera and later by the sofa
camera stay "Dad's car keys".

Setup
-----
  1. On each phone install a camera-streaming app, e.g.
       Android: "IP Webcam"  -> Start server -> URL like http://192.168.1.23:8080  (use .../video)
       iPhone/Android: "DroidCam" -> shows an IP; stream is usually http://<ip>:4747/video
     Use LANDSCAPE, 1280x720 is plenty. Keep phones on chargers, auto-lock off.
  2. Tell the tracker about them, either on the command line:
       python multicam.py --cam "Laptop=0" --cam "Hall=http://192.168.1.23:8080/video" --cam "Sofa=http://192.168.1.31:4747/video"
     or in cameras.json (see cameras.example.json) and just run:
       python multicam.py
  3. Open Trace at http://localhost:8000/ - multicam serves it, with live video of every
     camera (trace_api.py). --api-host 0.0.0.0 lets phones on your Wi-Fi open it too,
     --api-port 0 turns the server off.

What it looks for (prompts, threshold, model) is set in detect.py, which must sit in the
same folder. --conf / --imgsz here override detect.py's defaults.

Controls (click the window first) - same as memory_demo.py:
  drag on a camera tile    draw a zone for THAT camera (e.g. "Hall table")
  n / m / x + click        name / merge / forget an item (click its box or its panel row)
  f                        find by name (type in terminal)
  1-9, 0                   highlight an item / clear
  d                        debug scores     c  clear zones of the camera under the mouse
  h                        heatmap: where people walked, the places, and how long you've been at one
  Esc                      cancel mode      q  save and quit
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# detection lives in detect.py (prompts, thresholds, model) - see its docstring
import detect
import room as roommod
import trace_api
import whereabouts

try:                    # turns detections into SPEC 3.2 events for the Trace model page
    import events_adapter
except Exception as e:  # keep the tracker usable without it
    events_adapter = None
    print(f"events_adapter.py not loaded ({e}): no trace-events.jsonl")

# ---- tuning ----
SAMPLE_SECS = 0.5      # each camera is analysed at most every this many seconds
# s after last sighting (any camera) an item counts as "in view"
VISIBLE_FOR = 1.6
CONT_SECS = 1.6        # nearby box on the same camera within this = same object
CONT_IOU = 0.3
MATCH_THR = 0.86
SOLO_THR = 0.78
# sightings (on one camera) before a new object is created
NEW_HITS = 4
TENT_TTL = 1.6
PROMOTE_THR = 0.82
SWITCH_SECS = 1.2      # when two cameras see an item, switch "where it is" only if the
# current camera lost it this long ago, or the new view is clearly better
GALLERY_MAX = 32
GALLERY_NOVELTY = 0.94
GALLERY_EVERY = 10

# every camera frame is resized to this width (zones use these coords)
VIEW_W = 960
GRID_W = 1100          # on-screen width of the camera grid
PANEL_W = 340
MEM_DIR = Path("memory")
ZONES_FILE, EVENTS_FILE, CAMERAS_FILE = Path(
    "zones.json"), Path("events.jsonl"), Path("cameras.json")
LIVE_FILE, EDITS_FILE = MEM_DIR / "live.json", MEM_DIR / "edits.jsonl"
GREEN, ORANGE, GREY, WHITE, YELLOW, CYAN = (
    80, 200, 80), (0, 165, 255), (150, 150, 150), (255, 255, 255), (0, 230, 255), (255, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================ helpers ====

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(
        a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def crop(frame, box, pad=0.08):
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    c = frame[y1:y2, x1:x2]
    return c if c.size else frame[0:2, 0:2]


def ago(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    return f"{s // 3600}h {s % 3600 // 60}m ago"


def greedy(pairs):
    out, used_d, used_o = {}, set(), set()
    for score, d, o in sorted(pairs, reverse=True):
        if d not in used_d and o not in used_o:
            out[d] = (o, score)
            used_d.add(d)
            used_o.add(o)
    return out


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:      # Windows: the server is reading the file right now
            time.sleep(0.05)
    print(f"  couldn't update {path} (file busy); will retry")


# ---- feeding the Trace model (trace-sim page) ----
TRACE_EVENTS_FILE = Path("trace-events.jsonl")   # SPEC 3.2 events; bridge.py streams it to the page
PERSON_CONF = 0.35      # YOLO-World "person" boxes below this are ignored
PERSON_AT = 0.25        # a person box covering this much of a zone (or their feet in it) = at that zone
MOTION_DIFF = 18        # grey-level change (0-255) that counts as movement (--no-people only)
MOTION_FRAC = 0.02      # share of a zone that has to move for "someone is here" (--no-people only)
PERSON_QUIET = 4.0      # s without a person at a zone before "person_left"


def _area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def person_zone(box, zones):
    """The one zone a person is at: the most covered zone, or the zone their feet are in.
    The model keeps a single place for the person, so this picks one, never several."""
    fx, fy = (box[0] + box[2]) / 2, box[3]
    best, best_s = None, 0.0
    for z in zones:
        r = z["rect"]
        inter = _area([max(r[0], box[0]), max(r[1], box[1]), min(r[2], box[2]), min(r[3], box[3])])
        s = inter / max(1.0, min(_area(r), _area(box)))
        if r[0] <= fx <= r[2] and r[1] <= fy <= r[3]:
            s = max(s, PERSON_AT)
        if s >= PERSON_AT and s > best_s:
            best, best_s = z, s
    return best


class ZoneWatch:
    """person_here / person_left per named zone. From YOLO-World "person" boxes found in the
    same pass as the items (so a TV, a lamp or a shadow doesn't count as someone being there),
    or from movement inside the zone when people aren't detected (--no-people).
    No identity either way: only "a person is at this zone"."""

    def __init__(self, feed):
        self.feed, self.prev, self.state = feed, {}, {}

    def zone_id(self, name):
        return self.feed.zone_map.get(name) or events_adapter.slug(name)

    def update(self, cam, frame, zones, now, people=None):
        """people: [(box, conf)] from the detector, or None to fall back to movement."""
        if people is None:
            return self._motion(cam, frame, zones, now)
        found = [person_zone(b, zones) for b, c in people if c >= PERSON_CONF]
        at = {z["name"] for z in found if z is not None}
        for z in zones:
            self._mark(cam, z["name"], z["name"] in at, now)

    def _mark(self, cam, name, present, now):
        st = self.state.setdefault((cam, name), {"here": False, "last": 0.0})
        if present:
            st["last"] = now
            if not st["here"]:
                st["here"] = True
                self.feed.person_event("person_here", self.zone_id(name), now, cam)
        elif st["here"] and now - st["last"] > PERSON_QUIET:
            st["here"] = False
            self.feed.person_event("person_left", self.zone_id(name), now, cam)

    def _motion(self, cam, frame, zones, now):
        w = 240
        small = cv2.resize(frame, (w, max(1, int(w * frame.shape[0] / frame.shape[1]))))
        g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        prev, self.prev[cam] = self.prev.get(cam), g
        if prev is None or prev.shape != g.shape:
            return
        moved = cv2.absdiff(g, prev) > MOTION_DIFF
        k = w / float(frame.shape[1])
        for z in zones:
            x1, y1, x2, y2 = [int(v * k) for v in z["rect"]]
            roi = moved[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if roi.size == 0:
                continue
            self._mark(cam, z["name"], roi.mean() > MOTION_FRAC, now)


def feed_frame(feed, cam, frame, dets, zones, now):
    """One analysed frame -> the adapter, which applies SPEC 3.2's hysteresis
    (appeared = 2 of 3 frames, disappeared = 5 missed, moved, boxes > 15% dropped).
    Every analysed frame goes in, empty ones too, so misses are counted."""
    h, w = frame.shape[:2]
    rows = []
    for cat, box, conf in dets:
        zone = zone_for(box, zones)
        rows.append({"item": cat, "box": [float(v) for v in box], "det_conf": float(conf),
                     "image_w": w, "image_h": h, "camera_id": cam,
                     "support": {"rel": "in", "anchor": zone} if zone else None})
    feed.frame(cam, now, rows)


def zone_for(box, zones):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for z in reversed(zones):
        x1, y1, x2, y2 = z["rect"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return z["name"]
    return None


def view_score(box, zone, frame_shape):
    """How good a view is: bigger in frame is better, being inside a named zone is better."""
    h, w = frame_shape[:2]
    area = (box[2] - box[0]) * (box[3] - box[1]) / float(w * h)
    return min(area * 20, 1.0) + (0.5 if zone else 0.0)


class Embedder:
    def __init__(self):
        # the same CLIP ViT-B/32 the detector's verifier uses: loaded once, not twice
        self.model, self.pre, self.device = detect.load_clip()

    @torch.no_grad()
    def __call__(self, crops_bgr):
        if not crops_bgr:
            return np.zeros((0, 512), np.float32)
        x = torch.stack([self.pre(Image.fromarray(
            cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr])
        f = self.model.encode_image(x.to(self.device)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()


# ============================================================ cameras ====

class Camera(threading.Thread):
    """Reads one stream in the background and always holds the LATEST frame
    (network streams lag badly if you read them slower than they send)."""

    def __init__(self, cid, name, source):
        super().__init__(daemon=True)
        self.id, self.name, self.source = cid, name, source
        self.is_file = isinstance(source, str) and not re.match(
            r"^(https?|rtsp)://", source)
        self.lock = threading.Lock()
        self.frame, self.frame_time, self.ok = None, 0.0, False
        self.stopped = False
        # results of the last analysis, for drawing
        self.dets, self.assigned, self.debug = [], {}, {}
        self.last_processed, self.processed_frame_time, self.processed_at = 0.0, 0.0, 0.0

    def _open(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        while not self.stopped:
            cap = self._open()
            if not cap.isOpened():
                self.ok = False
                print(
                    f"  [{self.name}] can't open {self.source} - retrying in 3s")
                time.sleep(3)
                continue
            print(f"  [{self.name}] connected")
            while not self.stopped:
                ok, f = cap.read()
                if not ok:
                    if self.is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                v = cv2.resize(
                    f, (VIEW_W, int(f.shape[0] * VIEW_W / f.shape[1])))
                with self.lock:
                    self.frame, self.frame_time, self.ok = v, time.time(), True
                if self.is_file:
                    time.sleep(1 / 25)  # play files at roughly real speed
            self.ok = False
            cap.release()
            print(f"  [{self.name}] stream lost - reconnecting")
            time.sleep(1)

    def latest(self):
        with self.lock:
            return self.frame, self.frame_time


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "cam"


def load_cameras(args):
    specs = []
    if args.cam:
        for c in args.cam:
            name, _, src = c.partition("=")
            if not src:
                name, src = f"Camera {len(specs) + 1}", name
            specs.append({"name": name.strip(), "source": src.strip()})
    elif CAMERAS_FILE.exists():
        specs = json.loads(CAMERAS_FILE.read_text())
    else:
        print("No cameras given and no cameras.json - using webcam 0. See the top of this file for phones.")
        specs = [{"name": "Laptop", "source": 0}]
    cams, seen = [], set()
    for s in specs:
        src = s["source"]
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cid = s.get("id") or slug(s["name"])
        while cid in seen:
            cid += "-2"
        seen.add(cid)
        cams.append(Camera(cid, s["name"], src))
    return cams


# ============================================================= memory ====

class Obj:
    def __init__(self, oid, category, gallery):
        self.id, self.category, self.name = oid, category, None
        self.gallery = list(gallery)
        # "primary" sighting = where we tell the user it is
        self.cam, self.box, self.last_time, self.zone, self.thumb = None, None, None, None, None
        self.score = 0.0
        self.seen = {}            # cam id -> {"box", "time", "zone"}: latest sighting per camera
        self.was_visible = False
        self.extra = {}
        self.frame, self.snapshot_dirty = None, False

    @property
    def label(self):
        return self.name or f"{self.category.title()} #{self.id}"

    def visible(self, now):
        return self.last_time is not None and now - self.last_time < VISIBLE_FOR

    def sim(self, e):
        return float(np.max(np.stack(self.gallery) @ e)) if self.gallery else 0.0

    def add_view(self, e):
        if e is None:
            return
        if not self.gallery or self.sim(e) < GALLERY_NOVELTY:
            if len(self.gallery) < GALLERY_MAX:
                self.gallery.append(e)
            else:
                self.gallery[random.randrange(1, GALLERY_MAX)] = e

    def observe(self, cam, frame, box, now, zones, e=None):
        zone = zone_for(box, zones)
        self.seen[cam] = {"box": list(box), "time": now, "zone": zone}
        self.add_view(e)
        sc = view_score(box, zone, frame.shape)
        prim = self.seen.get(self.cam) if self.cam else None
        stale = prim is None or now - prim["time"] > SWITCH_SECS
        if self.cam == cam or stale or sc > self.score + 0.25:
            self.cam, self.box, self.zone, self.score = cam, list(
                box), zone, sc
            self.thumb = crop(frame, box, pad=0.15).copy()
            self.frame, self.snapshot_dirty = frame, True
        self.last_time = now


class Tentative:
    def __init__(self, cam, category, box, e, now, frame):
        self.cam, self.category, self.box, self.embs, self.hits, self.last = cam, category, box, [
            e], 1, now
        self.frame = frame


class Memory:
    def __init__(self, embed, cam_names):
        self.embed, self.cam_names = embed, cam_names
        self.objs, self.next_id, self.tents, self.frame_i = {}, 1, [], 0
        self.debug_scores = {}
        self.load()
        self.edits_pos = EDITS_FILE.stat().st_size if EDITS_FILE.exists() else 0

    def update(self, cam, frame, dets, now, zones):
        """One camera's detections. dets: [(category, box, conf)]. Returns {det_idx: obj_id}."""
        self.frame_i += 1
        self.debug_scores = {}
        assigned = {}

        # 1) continuity on THIS camera
        pairs = []
        for i, (cat, box, _) in enumerate(dets):
            for o in self.objs.values():
                s = o.seen.get(cam)
                if o.category == cat and s and now - s["time"] < CONT_SECS:
                    u = iou(s["box"], box)
                    if u > CONT_IOU:
                        pairs.append((1 + u, i, o.id))
        for d, (oid, s) in greedy(pairs).items():
            assigned[d] = oid
            self.debug_scores[d] = f"track {s - 1:.2f}"

        # 2) fingerprints
        refresh = self.frame_i % GALLERY_EVERY == 0
        need = [i for i in range(len(dets)) if i not in assigned or refresh]
        embs = dict(zip(need, self.embed(
            [crop(frame, dets[i][1]) for i in need])))

        # 3) re-identify by appearance - including objects another camera sees right now
        #    (overlapping views: the same keys can be in two cameras at once)
        pairs, taken = [], set(assigned.values())
        for i, (cat, box, _) in enumerate(dets):
            if i in assigned:
                continue
            same_cat = [o for o in self.objs.values() if o.category ==
                        cat and o.gallery]
            thr = SOLO_THR if len(same_cat) == 1 else MATCH_THR
            best = 0.0
            for o in same_cat:
                if o.id in taken:
                    continue
                s = o.sim(embs[i])
                best = max(best, s)
                if s >= thr:
                    pairs.append((s, i, o.id))
            self.debug_scores[i] = f"best {best:.2f} (need {thr:.2f})"
        for d, (oid, s) in greedy(pairs).items():
            assigned[d] = oid
            self.debug_scores[d] = f"re-id {s:.2f}"

        for d, oid in assigned.items():
            self.objs[oid].observe(
                cam, frame, dets[d][1], now, zones, embs.get(d))

        # 4) unknown -> tentative (per camera) -> promoted after NEW_HITS
        taken = set(assigned.values())
        for i, (cat, box, _) in enumerate(dets):
            if i in assigned:
                continue
            e = embs[i]
            cands = [t for t in self.tents if t.cam ==
                     cam and t.category == cat]
            t = max(cands, key=lambda t: max(iou(t.box, box), float(
                np.mean(t.embs, 0) @ e) - 0.5), default=None)
            if t and (iou(t.box, box) > 0.2 or float(np.mean(t.embs, 0) @ e) > 0.88):
                t.box, t.last, t.hits, t.frame = box, now, t.hits + 1, frame
                t.embs.append(e)
                if t.hits >= NEW_HITS:
                    self.promote(t, now, zones, taken)
            else:
                self.tents.append(Tentative(cam, cat, box, e, now, frame))
        self.tents = [t for t in self.tents if now -
                      t.last < TENT_TTL and t.hits < NEW_HITS]
        return assigned

    def promote(self, t, now, zones, taken):
        m = np.mean(t.embs, 0)
        m /= np.linalg.norm(m)
        best = max((o for o in self.objs.values() if o.category == t.category and o.id not in taken and o.gallery),
                   key=lambda o: o.sim(m), default=None)
        if best is not None and best.sim(m) >= PROMOTE_THR:
            o = best
            print(
                f"  [{self.cam_names.get(t.cam, t.cam)}] re-identified {o.label} ({o.sim(m):.2f})")
        else:
            o = Obj(self.next_id, t.category, [])
            self.objs[o.id] = o
            self.next_id += 1
            print(f"  [{self.cam_names.get(t.cam, t.cam)}] new object: {o.label}")
        for e in t.embs[:: max(1, len(t.embs) // 6)]:
            o.add_view(e)
        o.observe(t.cam, t.frame, t.box, now, zones)
        taken.add(o.id)

    # ---------------------------------------------------------- edits
    def merge(self, keep, drop):
        a, b = self.objs[keep], self.objs.pop(drop)
        for e in b.gallery:
            a.add_view(e)
        a.name = a.name or b.name
        for cam, s in b.seen.items():
            if cam not in a.seen or s["time"] > a.seen[cam]["time"]:
                a.seen[cam] = s
        if b.last_time and (not a.last_time or b.last_time > a.last_time):
            a.cam, a.box, a.last_time, a.zone, a.thumb, a.frame = b.cam, b.box, b.last_time, b.zone, b.thumb, b.frame
            a.snapshot_dirty = True
        self.save()

    def forget(self, oid):
        self.objs.pop(oid, None)
        for folder in ("thumbs", "snapshots"):
            (MEM_DIR / folder / f"{oid}.jpg").unlink(missing_ok=True)
        self.save()

    def find(self, query):
        q = query.lower().strip()
        hits = [o for o in self.objs.values() if q in o.label.lower()
                or o.category in q or q in o.category]
        return sorted(hits, key=lambda o: o.last_time or 0, reverse=True)

    def apply_edits(self):
        if not EDITS_FILE.exists() or EDITS_FILE.stat().st_size <= self.edits_pos:
            return
        with EDITS_FILE.open() as f:
            f.seek(self.edits_pos)
            lines = f.read().splitlines()
            self.edits_pos = f.tell()
        for line in lines:
            try:
                op = json.loads(line)
            except Exception:
                continue
            o = self.objs.get(op.get("id"))
            if op["op"] == "rename" and o:
                o.name = op["name"]
                print(f"  [app] renamed #{o.id} to '{o.name}'")
            elif op["op"] == "delete" and o:
                self.objs.pop(o.id)
                print(f"  [app] deleted #{o.id}")
            elif op["op"] == "create" and not o:
                n = Obj(op["id"], op["category"], [])
                n.name, n.extra = op["name"], op.get("extra", {})
                self.objs[n.id] = n
                print(f"  [app] added '{n.name}'")
            elif op["op"] == "sighting" and o:
                o.extra["manual"] = {"zone": op.get(
                    "zone"), "note": op.get("note"), "time": op["t"]}

    # ---------------------------------------------------- persistence
    def save(self):
        self.apply_edits()
        (MEM_DIR / "thumbs").mkdir(parents=True, exist_ok=True)
        data = {"next_id": self.next_id, "objects": []}
        for o in self.objs.values():
            fh, fw = o.frame.shape[:2] if o.frame is not None else (None, None)
            data["objects"].append({
                "id": o.id, "category": o.category, "name": o.name, "box": o.box,
                "last_time": o.last_time, "zone": o.zone, "camera": o.cam,
                "camera_name": self.cam_names.get(o.cam, o.cam) if o.cam else None,
                "frame_w": fw, "frame_h": fh, "extra": o.extra,
                "gallery": [[round(float(v), 4) for v in e] for e in o.gallery]})
            if o.thumb is not None:
                cv2.imwrite(str(MEM_DIR / "thumbs" / f"{o.id}.jpg"), o.thumb)
            self.save_snapshot(o)
        atomic_write(MEM_DIR / "memory.json", json.dumps(data))

    def save_snapshot(self, o):
        if o.frame is not None and o.snapshot_dirty:
            (MEM_DIR / "snapshots").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(MEM_DIR / "snapshots" /
                        f"{o.id}.jpg"), o.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            o.snapshot_dirty = False

    def save_live(self):
        live = {str(o.id): {"last_time": o.last_time, "box": o.box, "zone": o.zone, "camera": o.cam,
                            "camera_name": self.cam_names.get(o.cam, o.cam) if o.cam else None}
                for o in self.objs.values() if o.last_time}
        atomic_write(LIVE_FILE, json.dumps(live))

    def load(self):
        try:
            data = json.loads((MEM_DIR / "memory.json").read_text())
        except Exception:
            return
        self.next_id = data["next_id"]
        for d in data["objects"]:
            o = Obj(d["id"], d["category"], [np.array(e, np.float32)
                    for e in d.get("gallery", [])])
            o.name, o.box, o.last_time, o.zone = d.get("name"), d.get(
                "box"), d.get("last_time"), d.get("zone")
            o.cam, o.extra = d.get("camera"), d.get("extra") or {}
            if o.cam and o.box and o.last_time:
                o.seen[o.cam] = {"box": o.box,
                                 "time": o.last_time, "zone": o.zone}
            tp = MEM_DIR / "thumbs" / f"{o.id}.jpg"
            o.thumb = cv2.imread(str(tp)) if tp.exists() else None
            self.objs[o.id] = o
        print(f"Loaded {len(self.objs)} remembered objects.")


def api_item(o, now, cam_names):
    """One remembered object in the shape trace.html reads from /api/items."""
    fh, fw = o.frame.shape[:2] if o.frame is not None else (None, None)
    return {"id": o.id, "name": o.label, "category": o.category, "camera": o.cam,
            "camera_name": cam_names.get(o.cam, o.cam) if o.cam else None, "zone": o.zone,
            "last_time": o.last_time, "box": [round(float(v), 1) for v in o.box] if o.box else None,
            "frame_w": fw, "frame_h": fh, "sentence": None, "confidence": None,
            "guess": False, "flag": None, "in_view": o.visible(now)}


def log_event(o, now, cam_names):
    rec = {"id": o.id, "item": o.label, "category": o.category, "zone": o.zone, "source": "camera",
           "camera": o.cam, "camera_name": cam_names.get(o.cam, o.cam),
           "box": [int(v) for v in o.box] if o.box else None, "t": o.last_time,
           "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(o.last_time or now))}
    with EVENTS_FILE.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ============================================================== zones ====

def load_zones(first_cam):
    try:
        zones = json.loads(ZONES_FILE.read_text())
    except Exception:
        return []
    for z in zones:  # zones from the single-camera demo belong to the first camera
        z.setdefault("camera", first_cam)
    return zones


def save_zones(zones):
    atomic_write(ZONES_FILE, json.dumps(zones, indent=2))


# ============================================================ drawing ====

def dashed_rect(img, p1, p2, color, thickness=2, dash=12):
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def tag(img, text, org, color, scale=0.55):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    y = max(y, h + 6)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT,
                scale, (0, 0, 0), 1, cv2.LINE_AA)


def ibox(b):
    return (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))


def draw_tile(view, cam, mem, zones, now, focus, drag, debug, brain=None):
    """Draw one camera's frame (at VIEW_W coordinates)."""
    if brain is not None:          # 'h': the heatmap + places replace the zone drawing
        brain.overlay(cam.id, view, now)
        zones = []
    for z in zones:
        x1, y1, x2, y2 = z["rect"]
        ov = view.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), CYAN, -1)
        cv2.addWeighted(ov, 0.12, view, 0.88, 0, view)
        cv2.rectangle(view, (x1, y1), (x2, y2), CYAN, 1)
        cv2.putText(view, z["name"], (x1 + 5, y2 - 6),
                    FONT, 0.6, CYAN, 1, cv2.LINE_AA)
    if drag:
        cv2.rectangle(view, drag[0], drag[1], CYAN, 2)

    # ghosts: items whose last known place is in THIS camera
    for o in mem.objs.values():
        if o.cam != cam.id or o.box is None or o.visible(now) or (focus is not None and focus != o.id):
            continue
        p1, p2 = ibox(o.box)
        col = YELLOW if focus == o.id else ORANGE
        dashed_rect(view, p1, p2, col, 3 if focus == o.id else 2)
        tag(view, f"{o.label} - {ago(now - o.last_time)}",
            (p1[0], p1[1] - 4), col, 0.5)

    fresh = now - cam.processed_at < VISIBLE_FOR
    for i, (cat, box, conf) in enumerate(cam.dets if fresh else []):
        p1, p2 = ibox(box)
        if i in cam.assigned:
            o = mem.objs.get(cam.assigned[i])
            if o is None or (focus is not None and focus != o.id):
                continue
            cv2.rectangle(view, p1, p2, GREEN, 3)
            tag(view, o.label, (p1[0], p1[1] - 4), GREEN, 0.6)
        elif focus is None:
            cv2.rectangle(view, p1, p2, GREY, 1)
            tag(view, f"new {cat}?", (p1[0], p1[1] - 4), GREY, 0.5)
        if debug and i in cam.debug and focus is None:
            cv2.putText(
                view, cam.debug[i], (p1[0], p2[1] + 18), FONT, 0.5, YELLOW, 1, cv2.LINE_AA)

    if focus is not None and focus in mem.objs:
        o = mem.objs[focus]
        if o.cam == cam.id and o.box is not None:
            p1, p2 = ibox(o.box)
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            r = int(max(p2[0] - p1[0], p2[1] - p1[1]) *
                    (0.7 + 0.15 * (1 + np.sin(now * 6)))) + 12
            cv2.circle(view, (cx, cy), r, YELLOW, 3, cv2.LINE_AA)
            txt = f"{o.label} - here now" if o.visible(
                now) else f"{o.label} - last seen {ago(now - o.last_time)}"
            tag(view, txt + (f" ({o.zone})" if o.zone else ""),
                (p1[0], p1[1] - 4), YELLOW, 0.6)


class Layout:
    """Places camera tiles in a grid (letterboxed) and maps mouse <-> camera coords."""

    def __init__(self, n):
        self.cols = 1 if n == 1 else 2 if n <= 4 else 3
        self.rows = (n + self.cols - 1) // self.cols
        self.tw = GRID_W // self.cols
        self.th = int(self.tw * 9 / 16)
        self.h = self.th * self.rows
        self.fit = {}  # tile idx -> (ox, oy, scale)

    def tile_origin(self, k):
        return (k % self.cols) * self.tw, (k // self.cols) * self.th

    def place(self, canvas, k, img):
        h, w = img.shape[:2]
        s = min(self.tw / w, self.th / h)
        nw, nh = int(w * s), int(h * s)
        ox, oy = (self.tw - nw) // 2, (self.th - nh) // 2
        tx, ty = self.tile_origin(k)
        canvas[ty + oy:ty + oy + nh, tx + ox:tx +
               ox + nw] = cv2.resize(img, (nw, nh))
        self.fit[k] = (ox, oy, s)

    def to_cam(self, x, y):
        """Screen point -> (tile idx, x, y in camera coords) or None."""
        if x >= GRID_W or y >= self.h:
            return None
        k = (y // self.th) * self.cols + x // self.tw
        if k not in self.fit:
            return None
        tx, ty = self.tile_origin(k)
        ox, oy, s = self.fit[k]
        return k, (x - tx - ox) / s, (y - ty - oy) / s


PANEL_TOP, ROW_H = 74, 64


def panel_order(mem):
    return sorted(mem.objs.values(), key=lambda o: (o.name is None, -(o.last_time or 0)))


def draw_panel(h, mem, now, focus, fps, cam_names):
    panel = np.full((h, PANEL_W, 3), 30, np.uint8)
    cv2.putText(panel, "Where did I leave it?", (14, 32),
                FONT, 0.75, WHITE, 2, cv2.LINE_AA)
    cv2.putText(panel, f"{fps:4.1f} analysed/s  n name m merge x forget f find",
                (14, 54), FONT, 0.38, GREY, 1, cv2.LINE_AA)
    y, order = PANEL_TOP, panel_order(mem)
    if not order:
        cv2.putText(panel, "Nothing remembered yet",
                    (14, y + 24), FONT, 0.5, GREY, 1, cv2.LINE_AA)
    for k, o in enumerate(order):
        if y + ROW_H > h:
            cv2.putText(panel, f"+{len(order) - k} more",
                        (14, h - 8), FONT, 0.45, GREY, 1, cv2.LINE_AA)
            break
        if focus == o.id:
            cv2.rectangle(panel, (4, y - 2), (PANEL_W - 4,
                          y + ROW_H - 6), (70, 70, 0), -1)
        s = ROW_H - 12
        if o.thumb is not None:
            panel[y + 2:y + 2 + s, 10:10 + s] = cv2.resize(o.thumb, (s, s))
        tx = 20 + s
        cv2.putText(panel, f"{str(k + 1) + '  ' if k < 9 else ''}{o.label}"[:26], (tx, y + 18), FONT, 0.55,
                    WHITE if o.name else (200, 200, 200), 1, cv2.LINE_AA)
        if o.last_time is None:
            st, col = "not seen yet", GREY
        elif o.visible(now):
            st, col = "in view now", GREEN
        else:
            st, col = f"last seen {ago(now - o.last_time)}", ORANGE
        cv2.putText(panel, st, (tx, y + 37), FONT, 0.45, col, 1, cv2.LINE_AA)
        cn = cam_names.get(o.cam) if o.cam else None
        where = " · ".join(x for x in (
            o.zone, cn if cn != o.zone else None) if x)
        if where:
            cv2.putText(panel, where[:32], (tx, y + 54),
                        FONT, 0.42, CYAN, 1, cv2.LINE_AA)
        y += ROW_H
    return panel


# =============================================================== main ====

def hit_test(mem, cams, layout, x, y, now):
    if x >= GRID_W:
        k = (y - PANEL_TOP) // ROW_H
        order = panel_order(mem)
        return order[k] if y >= PANEL_TOP and 0 <= k < len(order) else None
    m = layout.to_cam(x, y)
    if not m:
        return None
    cam = cams[m[0]]
    cx, cy = m[1], m[2]
    hits = []
    for o in mem.objs.values():
        s = o.seen.get(cam.id)
        if s and (o.cam == cam.id or now - s["time"] < VISIBLE_FOR):
            b = s["box"]
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                hits.append((now - s["time"] > VISIBLE_FOR,
                            (b[2] - b[0]) * (b[3] - b[1]), o))
    return min(hits, key=lambda h: h[:2])[2] if hits else None


def main():
    global PERSON_CONF
    ap = argparse.ArgumentParser(description="FindIt multi-camera tracker")
    ap.add_argument("--cam", action="append",
                    help='NAME=SOURCE, e.g. "Hall=http://192.168.1.23:8080/video" or "Laptop=0"')
    ap.add_argument("--imgsz", type=int, default=detect.IMGSZ,
                    help="detector input size (default from detect.py)")
    ap.add_argument("--conf", type=float, default=None,
                    help="YOLO-World threshold (default from detect.py: PROPOSE_CONF with verifier.json, "
                         "the same value the verifier was trained on; CONF without it)")
    ap.add_argument("--min-prob", action="append", default=[], metavar="ITEM=P",
                    help="override verifier.json's threshold for one item, e.g. --min-prob keys=0.7 "
                         "(keys default 0.86 is strict: lower it if the real keys never get a box)")
    ap.add_argument("--person-conf", type=float, default=PERSON_CONF,
                    help=f"YOLO-World 'person' boxes below this are ignored (default {PERSON_CONF}; "
                         "a phone looking straight down sees heads and arms, so try 0.15 if "
                         "person_here never fires)")
    ap.add_argument("--zone-map", default=None,
                    help='JSON file renaming zones to the Trace room map ids, e.g. {"Hall table": "desk"}; '
                         'without it a zone "Sofa" becomes "sofa"')
    ap.add_argument("--no-people", action="store_true",
                    help="don't look for people: someone at a zone comes from movement there, no heatmap, "
                         "and Trace can't tell picked-up from covered")
    ap.add_argument("--no-bridge", action="store_true",
                    help="don't start bridge.py (which streams trace-events.jsonl to the Trace page)")
    ap.add_argument("--api-port", type=int, default=8000,
                    help="port for Trace (trace.html + live video); 0 turns it off")
    ap.add_argument("--api-host", default="127.0.0.1",
                    help="127.0.0.1 = this computer only; 0.0.0.0 = phones on your Wi-Fi too")
    args = ap.parse_args()

    cams = load_cameras(args)
    cam_names = {c.id: c.name for c in cams}
    print("Cameras: " + ", ".join(f"{c.name} ({c.source})" for c in cams))
    for c in cams:
        c.start()

    print("Loading models (first run downloads weights)...")
    # people ride along in the same YOLO-World pass at no extra cost; they feed
    # person_here / person_left, the heatmap and whereabouts.py, never the item memory
    prompts = dict(detect.PROMPTS) if args.no_people else {**detect.PROMPTS, "person": ["person"]}
    detector = detect.Detector(prompts=prompts, conf=args.conf, imgsz=args.imgsz)
    PERSON_CONF = args.person_conf
    for spec in args.min_prob:
        item, _, p = spec.partition("=")
        if detector.verifier is None:
            print(f"  --min-prob {spec} ignored: no verifier.json")
            break
        try:
            detector.verifier.thresholds[item.strip()] = float(p)
            print(f"  verifier threshold for {item.strip()}: {float(p):.2f}")
        except ValueError:
            print(f"  --min-prob expects ITEM=P, e.g. keys=0.7 (got {spec})")
    mem = Memory(Embedder(), cam_names)
    zones = load_zones(cams[0].id)
    brain = whereabouts.Whereabouts(roommod.Room(), cam_names)
    brain.room.set_drawn(zones, cam_names)
    brain.room.save()
    show_heat = False
    layout = Layout(len(cams))
    feed = watch = bridge = None
    if events_adapter is not None:
        zmap = {}
        if args.zone_map:
            try:
                zmap = json.loads(Path(args.zone_map).read_text())
            except Exception as e:
                print(f"  couldn't read --zone-map {args.zone_map}: {e}")
        feed = events_adapter.Adapter(str(TRACE_EVENTS_FILE), zmap, echo=False)
        watch = ZoneWatch(feed)
        print(f"Trace model events -> {TRACE_EVENTS_FILE} (zones must match the room map in the page)")
        if not args.no_bridge:
            import subprocess
            try:
                bridge = subprocess.Popen([sys.executable, str(Path(__file__).with_name("bridge.py")),
                                           "--file", str(TRACE_EVENTS_FILE),
                                           "--host", "0.0.0.0" if args.api_host == "0.0.0.0" else "localhost"])
                print("  bridge.py started: open trace-sim/index.html (or /model) and press Connect live")
            except Exception as e:
                print(f"  couldn't start bridge.py ({e}); run it yourself: python bridge.py --file {TRACE_EVENTS_FILE}")
    api = trace_api.TraceAPI(cams, MEM_DIR, EVENTS_FILE).start(
        args.api_host, args.api_port) if args.api_port else None
    if api:
        api.brain = brain
    last_pub = 0.0

    win = "FindIt - all cameras"
    cv2.namedWindow(win)
    ms = {"start": None, "cur": None,
          "rect": None, "click": None, "pos": (0, 0)}

    def on_mouse(event, x, y, flags, _):
        ms["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            ms["start"] = ms["cur"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and ms["start"]:
            ms["cur"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and ms["start"]:
            (x1, y1) = ms["start"]
            if abs(x - x1) < 8 and abs(y - y1) < 8:
                ms["click"] = (x1, y1)
            else:
                a, b = layout.to_cam(x1, y1), layout.to_cam(
                    min(x, GRID_W - 1), min(y, layout.h - 1))
                if a and b and a[0] == b[0] and abs(b[1] - a[1]) > 15 and abs(b[2] - a[2]) > 15:
                    ms["rect"] = (a[0], [int(min(a[1], b[1])), int(min(a[2], b[2])),
                                         int(max(a[1], b[1])), int(max(a[2], b[2]))])
            ms["start"] = ms["cur"] = None

    cv2.setMouseCallback(win, on_mouse)
    print(__doc__.split("Controls")[1])

    focus, mode, merge_first, debug = None, None, None, False
    processed, rate, rate_t = 0, 0.0, time.time()
    last_save, last_live = time.time(), 0.0
    MODE_MSG = {"name": "NAME: click the item (or its row)", "merge": "MERGE: click the first item",
                "merge2": "MERGE: click the duplicate to fold into it", "forget": "FORGET: click the item"}

    while True:
        now = time.time()

        # ---- pick cameras due for analysis and run them as ONE batch
        batch = []
        for c in cams:
            f, ft = c.latest()
            if f is not None and ft > c.processed_frame_time and now - c.last_processed >= SAMPLE_SECS:
                batch.append((c, f, ft))
        if batch:
            per_frame = detector.detect([f for _, f, _ in batch])
            for (c, f, ft), dets in zip(batch, per_frame):
                people = None if args.no_people else [(b, s) for cat, b, s in dets if cat == "person"]
                dets = [d for d in dets if d[0] != "person"]
                cz = [z for z in zones if z["camera"] == c.id]
                c.assigned = mem.update(c.id, f, dets, now, cz)
                if feed is not None:
                    try:
                        feed_frame(feed, c.id, f, dets, cz, now)
                        watch.update(c.id, f, cz, now, people)
                    except Exception as e:      # never let the model feed stop the tracker
                        print(f"  [{c.name}] trace event error: {e}")
                c.dets, c.debug = dets, dict(mem.debug_scores)
                brain.frame(c.id, now, f.shape, people or [],
                            [(oid, mem.objs[oid].category, mem.objs[oid].label, dets[d][1])
                             for d, oid in c.assigned.items() if oid in mem.objs])
                c.last_processed, c.processed_frame_time, c.processed_at = now, ft, now
            processed += len(batch)
        if now - rate_t >= 2:
            rate, processed, rate_t = processed / (now - rate_t), 0, now
        brain.tick(now)

        # ---- out-of-sight events (no camera sees it any more)
        for o in mem.objs.values():
            vis = o.visible(now)
            if o.was_visible and not vis:
                log_event(o, now, cam_names)
                mem.save_snapshot(o)
                print(f"[{time.strftime('%H:%M:%S')}] {o.label} out of sight - last seen"
                      + (f" in {o.zone}" if o.zone else "") + f" ({cam_names.get(o.cam, o.cam)})")
            o.was_visible = vis

        # ---- draw grid + panel
        canvas = np.zeros((layout.h, GRID_W, 3), np.uint8)
        for k, c in enumerate(cams):
            f, _ = c.latest()
            tx, ty = layout.tile_origin(k)
            if f is None or not c.ok:
                cv2.putText(canvas, f"{c.name}: no signal - reconnecting", (tx + 20, ty + layout.th // 2),
                            FONT, 0.6, GREY, 1, cv2.LINE_AA)
                if f is None:
                    continue
            view = f.copy()
            drag = None
            if ms["start"] and not mode:
                a, b = layout.to_cam(
                    *ms["start"]), layout.to_cam(*ms["cur"]) if ms["cur"] else None
                if a and b and a[0] == k == b[0]:
                    drag = ((int(a[1]), int(a[2])), (int(b[1]), int(b[2])))
            draw_tile(view, c, mem, [
                      z for z in zones if z["camera"] == c.id], now, focus, drag, debug,
                      brain if show_heat else None)
            layout.place(canvas, k, view)
            cv2.rectangle(canvas, (tx, ty), (tx + layout.tw - 1,
                          ty + layout.th - 1), (60, 60, 60), 1)
            cv2.rectangle(canvas, (tx, ty), (tx + 12 + 11 *
                          len(c.name), ty + 26), (0, 0, 0), -1)
            cv2.putText(canvas, c.name, (tx + 6, ty + 19), FONT,
                        0.6, WHITE if c.ok else GREY, 1, cv2.LINE_AA)
        if mode:
            cv2.rectangle(canvas, (0, layout.h - 34),
                          (GRID_W, layout.h), (0, 0, 0), -1)
            cv2.putText(
                canvas, MODE_MSG[mode], (12, layout.h - 10), FONT, 0.65, YELLOW, 2, cv2.LINE_AA)
        cv2.imshow(win, np.hstack(
            [canvas, draw_panel(layout.h, mem, now, focus, rate, cam_names)]))

        # ---- clicks
        if ms["click"]:
            x, y = ms["click"]
            ms["click"] = None
            o = hit_test(mem, cams, layout, x, y, now)
            if mode and o is None:
                print(
                    "  no remembered item there - click inside its box or its panel row")
            elif mode == "name":
                name = input(f"Name for {o.label} (blank = keep): ").strip()
                if name:
                    o.name = name
                    mem.save()
                mode = None
            elif mode == "forget":
                print(f"  forgot {o.label}")
                mem.forget(o.id)
                brain.forget(o.id)
                focus, mode = (None if focus == o.id else focus), None
            elif mode == "merge":
                merge_first, mode = o.id, "merge2"
            elif mode == "merge2":
                if o.id == merge_first:
                    print("  that's the same one - click a different item")
                elif o.category != mem.objs[merge_first].category:
                    print("  can't merge different kinds of item")
                    mode = None
                else:
                    print(
                        f"  merged {o.label} into {mem.objs[merge_first].label}")
                    mem.merge(merge_first, o.id)
                    brain.forget(o.id)
                    focus, mode = (None if focus == o.id else focus), None
            elif mode is None and o is not None and x >= GRID_W:
                focus = o.id

        # ---- new zone
        if ms["rect"]:
            (k, rect), ms["rect"] = ms["rect"], None
            name = input(
                f"Name for this zone on {cams[k].name} (e.g. Hall table), blank to cancel: ").strip()
            if name:
                zones.append(
                    {"name": name, "rect": rect, "camera": cams[k].id})
                save_zones(zones)
                brain.room.set_drawn(zones, cam_names)
                brain.room.save()

        # ---- keys
        key = cv2.waitKey(1) & 0xFF
        if key == 27 and mode:
            mode = None
        elif key in (ord("q"), 27):
            break
        elif key == ord("n"):
            mode = "name"
        elif key == ord("m"):
            mode = "merge"
        elif key == ord("x"):
            mode = "forget"
        elif key == ord("d"):
            debug = not debug
        elif key == ord("h"):
            show_heat = not show_heat
        elif key == ord("c"):
            m = layout.to_cam(*ms["pos"])
            if m:
                zones = [z for z in zones if z["camera"] != cams[m[0]].id]
                save_zones(zones)
                brain.room.set_drawn(zones, cam_names)
                brain.room.save()
                print(f"  cleared zones on {cams[m[0]].name}")
        elif key == ord("0"):
            focus, mode = None, None
        elif ord("1") <= key <= ord("9"):
            order = panel_order(mem)
            if key - ord("1") < len(order):
                focus = order[key - ord("1")].id
        elif key == ord("f"):
            q = input("Find what? ").strip()
            hits = mem.find(q) if q else []
            if not hits:
                print("  nothing remembered matching that")
            else:
                o = hits[0]
                focus = o.id
                where = "in view now" if o.visible(now) else (
                    f"last seen {ago(now - o.last_time)}" if o.last_time else "not seen yet")
                print(f"  {o.label}: {where}" + (f" in {o.zone}" if o.zone else "")
                      + (f" ({cam_names.get(o.cam)})" if o.cam else ""))

        if api and now - last_pub > 0.5:
            api.items = [api_item(o, now, cam_names)
                         for o in list(mem.objs.values()) if o.last_time]
            last_pub = now
        if now - last_live > 1:
            mem.apply_edits()
            mem.save_live()
            focus = focus if focus in mem.objs else None
            last_live = now
        if now - last_save > 10:
            mem.save()
            brain.save()
            last_save = now
        if not batch:
            time.sleep(0.01)

    mem.save()
    brain.save()
    print("Memory saved.")
    if bridge is not None:
        bridge.terminate()
    if api:
        api.stop()
    for c in cams:
        c.stopped = True
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
