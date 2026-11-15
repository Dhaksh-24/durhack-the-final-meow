"""Turn the tracker's per-frame sightings into Trace events (SPEC.md 3.2), one JSON per line.

multicam.py calls EventWriter.frame() once per analysed frame of each camera with what that
camera saw. This module decides when something has really changed and writes events.jsonl:

  appeared     item seen in 2 of the last 3 analysed frames in a zone where it wasn't
  disappeared  item not seen for 4 analysed frames in its zone (on every camera watching it)
  moved        item appeared in a zone within 60 s of leaving another (or while still
               listed in another: that one gets a "disappeared" first). Has "from".
  person_here  motion in a zone in 2 of the last 3 analysed frames
  person_left  no motion in that zone for PERSON_HOLD seconds

Zone ids are slug(zone name), the same as the page's zone map ids. Items are
"<category>-<tracker id>", e.g. "keys-3". "floor" (metres) is added when
baseline/calibration.json (exported by the page) has the camera.

No faces, no identities: person events are motion in a zone, nothing more.
Pure Python apart from Motion (OpenCV), so the logic can be tested without cameras or torch.
"""
import json
import re
import time
from collections import deque
from pathlib import Path

APPEAR_OF, APPEAR_HITS = 3, 2   # appeared: seen in APPEAR_HITS of the last APPEAR_OF frames
GONE_FRAMES = 4                 # disappeared: missed this many analysed frames in a row
MOVE_SECS = 60                  # appeared elsewhere within this long of disappearing = moved
MOTION_FRAC = 0.03              # share of a zone's pixels that must change to count as motion
LIGHTING_FRAC = 0.5             # more than this of the whole frame changed: lighting, ignore
PERSON_HOLD = 15.0              # s without motion before person_left
CALIBRATION_FILE = Path("baseline") / "calibration.json"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "zone"


def zone_id(z):
    """Zone id for a multicam zone dict ({"name", "rect", "camera"}): its "id" or slug(name)."""
    return z.get("id") or slug(z["name"])


def load_calibration(path=CALIBRATION_FILE):
    """{camera_id: {"w", "h", "H"}} from the page's "Camera calibration for the tracker" output."""
    try:
        data = json.loads(Path(path).read_text())
    except Exception:
        return {}
    out = {}
    for c in data if isinstance(data, list) else data.get("cameras", []):
        try:
            H = [float(v) for v in c["image_to_floor"]]
            if len(H) == 9:
                out[c.get("camera_id") or slug(c["camera"])] = {
                    "w": float(c["image_w"]), "h": float(c["image_h"]), "H": H}
        except Exception:
            continue
    return out


def floor_point(cal, box, frame_w, frame_h):
    """Floor metres of a box's bottom-centre (SPEC 3.3), scaling the box from the analysed
    frame's size to the calibration photo's size. None if it maps behind the camera."""
    if not cal or not frame_w or not frame_h:
        return None
    x = (box[0] + box[2]) / 2 * cal["w"] / frame_w
    y = box[3] * cal["h"] / frame_h
    H = cal["H"]
    w = H[6] * x + H[7] * y + H[8]
    if abs(w) < 1e-12:
        return None
    fx, fy = (H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w
    if not (abs(fx) < 1e4 and abs(fy) < 1e4):
        return None
    return [round(fx, 2), round(fy, 2)]


class _Track:
    """One item in one zone as seen by one camera."""

    def __init__(self):
        self.hist = deque(maxlen=APPEAR_OF)   # confidence per analysed frame, None = not seen
        self.miss, self.present = 0, False
        self.conf, self.floor = None, None


class _Person:
    """Motion in one zone as seen by one camera."""

    def __init__(self):
        self.hist = deque(maxlen=APPEAR_OF)
        self.present, self.last_motion = False, 0.0


class EventWriter:
    def __init__(self, path="events.jsonl", calibration=None, echo=True):
        self.path = Path(path)
        self.cal = calibration or {}
        self.echo = echo
        self.tracks = {}      # (item, zone, cam) -> _Track
        self.where = {}       # item -> set of zones we've announced it in
        self.gone = {}        # item -> (zone, t) of its last disappearance
        self.info = {}        # item -> {"category", "name"}
        self.people = {}      # (zone, cam) -> _Person
        self.person_zones = set()
        self.written = []     # every record written (handy for tests)

    # ------------------------------------------------------------ output
    def _emit(self, now, event, cam, zone, item=None, **extra):
        rec = {"time": time.strftime("%H:%M:%S", time.localtime(now)), "t": round(now, 2)}
        if item is not None:
            rec["item"] = item
            rec["category"] = self.info[item]["category"]
            if self.info[item].get("name"):
                rec["name"] = self.info[item]["name"]
        rec.update({"zone": zone, "event": event, "camera": cam})
        rec.update({k: v for k, v in extra.items() if v is not None})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self.written.append(rec)
        if self.echo:
            what = f"{item} " if item else ""
            conf = f" ({rec['confidence']:.2f})" if "confidence" in rec else ""
            frm = f" from {rec['from']}" if "from" in rec else ""
            print(f"[{rec['time']}] {what}{event} {zone}{frm}{conf} [{cam}]")
        return rec

    # ------------------------------------------------------------- input
    def frame(self, cam, now, sightings, zones, motion=None, frame_shape=None):
        """One analysed frame of camera `cam`.

        sightings: [{"item": "keys-3", "category": "keys", "name": "Keys #3",
                     "zone": zone id or None, "conf": 0.7, "box": [x1, y1, x2, y2]}]
        zones:     ids of the zones this camera watches
        motion:    {zone id: fraction of the zone that changed}, or None to skip person events
        frame_shape: (h, w, ...) of the analysed frame, for floor positions
        """
        zones = set(zones)
        fh, fw = (frame_shape[0], frame_shape[1]) if frame_shape is not None else (None, None)
        cal = self.cal.get(cam)

        seen = {}   # (item, zone) -> best sighting this frame
        for s in sightings:
            self.info[s["item"]] = {"category": s["category"], "name": s.get("name")}
            z = s.get("zone")
            if z is None or z not in zones:
                continue
            k = (s["item"], z)
            if k not in seen or s["conf"] > seen[k]["conf"]:
                seen[k] = s

        keys = {k for k in self.tracks if k[2] == cam and k[1] in zones}
        keys |= {(item, z, cam) for item, z in seen}
        changed = set()
        for k in keys:
            tr = self.tracks.setdefault(k, _Track())
            s = seen.get((k[0], k[1]))
            tr.hist.append(s["conf"] if s else None)
            if s:
                tr.miss = 0
                if s.get("box") is not None:
                    tr.floor = floor_point(cal, s["box"], fw, fh) or tr.floor
            else:
                tr.miss += 1
            hits = [c for c in tr.hist if c is not None]
            if not tr.present and len(hits) >= APPEAR_HITS:
                tr.present, tr.conf = True, sum(hits) / len(hits)
                changed.add(k[:2])
            elif tr.present and tr.miss >= GONE_FRAMES:
                tr.present = False
                changed.add(k[:2])
            elif not tr.present and not hits:
                del self.tracks[k]    # nothing pending: forget it

        # disappearances first, so a move within this frame reads disappeared -> moved
        for item, z in sorted(changed, key=lambda k: self._present(*k)):
            here = self._present(item, z)
            announced = z in self.where.setdefault(item, set())
            if announced and not here:
                self._disappear(item, z, now, cam)
            elif here and not announced:
                for oz in sorted(self.where[item] - {z}):   # still listed elsewhere: it left
                    for k, tr in self.tracks.items():
                        if k[0] == item and k[1] == oz:   # clear, or old hits re-appear it
                            tr.present, tr.miss = False, GONE_FRAMES
                            tr.hist.clear()
                    self._disappear(item, oz, now, cam)
                tr = self._best(item, z)
                g = self.gone.get(item)
                moved = g is not None and g[0] != z and now - g[1] <= MOVE_SECS
                self.where[item].add(z)
                self._emit(now, "moved" if moved else "appeared", tr[1], z, item,
                           confidence=round(tr[0].conf, 3), floor=tr[0].floor,
                           **({"from": g[0]} if moved else {}))

        if motion is not None:
            self._people(cam, now, zones, motion)

    def _present(self, item, z):
        return any(tr.present for k, tr in self.tracks.items() if k[0] == item and k[1] == z)

    def _best(self, item, z):
        return max(((tr, k[2]) for k, tr in self.tracks.items()
                    if k[0] == item and k[1] == z and tr.present), key=lambda p: p[0].conf)

    def _disappear(self, item, z, now, cam):
        self.where[item].discard(z)
        self.gone[item] = (z, now)
        self._emit(now, "disappeared", cam, z, item)

    def _people(self, cam, now, zones, motion):
        for z in zones:
            p = self.people.setdefault((z, cam), _Person())
            moving = motion.get(z, 0.0) >= MOTION_FRAC
            p.hist.append(moving)
            if moving:
                p.last_motion = now
            if not p.present and sum(p.hist) >= APPEAR_HITS:
                p.present = True
            elif p.present and now - p.last_motion > PERSON_HOLD:
                p.present = False
            here = any(q.present for (zz, _), q in self.people.items() if zz == z)
            if here and z not in self.person_zones:
                self.person_zones.add(z)
                self._emit(now, "person_here", cam, z)
            elif not here and z in self.person_zones:
                self.person_zones.discard(z)
                self._emit(now, "person_left", cam, z)


class Motion:
    """Per-camera background subtraction: how much of each zone is moving right now."""

    def __init__(self, scale=0.25):
        import cv2
        self.cv2, self.scale = cv2, scale
        self.sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def zones(self, frame, zones):
        """frame: BGR at multicam's view size; zones: multicam zone dicts. {zone id: fraction}."""
        cv2, s = self.cv2, self.scale
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = self.sub.apply(cv2.GaussianBlur(small, (5, 5), 0))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) > 0
        if mask.mean() > LIGHTING_FRAC:   # lights on/off or the phone moved: not a person
            return {}
        out = {}
        for z in zones:
            x1, y1, x2, y2 = [int(v * s) for v in z["rect"]]
            m = mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            out[zone_id(z)] = float(m.mean()) if m.size else 0.0
        return out
