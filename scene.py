"""scene.py -- work out WHERE a detected item is, in words a person would use.

The idea: furniture is a fixed frame of reference. Detect the furniture once
(the "anchors"), cache it, then describe any item relative to those anchors.

    mapper = SceneMapper()
    anchors = mapper.anchors(frame)                 # cached per camera
    items = mapper.detect_items(frame, ["keys"])
    desc, sentence = mapper.describe(items[0], anchors, frame.shape)

    sentence -> "Your keys are on the table, just left of the laptop."

On the floor: YOLO-World cannot box a floor reliably, so "floor" is the bottom
band of the image minus every furniture box. FastSAM/SAM are both importable
here, but they return class-agnostic masks -- picking which mask is the floor
needs its own heuristic, so it buys little over the band rule and costs a
weights download plus ~1s per frame on CPU. Left out on purpose.

Gemini is optional and off by default; see gemini_review().

Handing the result to another program (an LLM, say): payload() packs a whole
frame into one JSON-ready dict and write_json() saves it.

    data = mapper.payload(frame.shape, anchors, items, descs)
    write_json("scene.json", data)

    {"version": 1, "timestamp": "2026-11-14T12:00:00.000+00:00",
     "camera": "default", "image": {"width": 1280, "height": 854},
     "anchors": [{"id": "table_1", "cls": "table", "name": "table",
                  "box": [403.7, 511.5, 1017.8, 852.3], "conf": 0.384}],
     "items": [{"id": "mug_1", "item": "mug",
                "box": [593.8, 465.5, 701.8, 549.3], "det_conf": 0.979,
                "support": {"rel": "on", "anchor": "table_1"},
                "near": [{"anchor": "sofa_2", "dir": "next to", "dist": 0.055}],
                "confidence": 0.78,
                "sentence": "Your mug is on the table, next to the sofa."}]}

Boxes are [x1, y1, x2, y2] in pixels of an image.width x image.height frame,
origin top-left. "det_conf" is how sure the detector was that the thing is
there; "confidence" is how sure we are about WHERE it is. "guess" is true
when det_conf is under CONFIG["guess_below"] -- treat those as maybes.

For a stream instead of a snapshot, detections() gives one flat dict per
detected item and emit() writes them as JSON lines:

    out = claim_stdout()                 # stdout now carries JSON only
    emit(out, mapper.detections(frame.shape, items, descs, "lounge"))

    {"time": "2026-11-14T12:00:00.000+00:00", "camera_id": "lounge",
     "image_w": 1280, "image_h": 854, "item": "keys",
     "box": [593.8, 465.5, 701.8, 549.3], "det_conf": 0.412,
     "support": {"rel": "on", "anchor": "table_1"},
     "near": [{"anchor": "sofa_1", "dir": "next to", "dist": 0.055}],
     "confidence": 0.71, "guess": false}

Every line from one frame shares the same "time". The anchor ids it refers
to are resolved against the fixed furniture list -- save_anchors() and
load_anchors() -- which live_scene.py --setup writes once per room.
"""
import datetime
import hashlib
import json
import logging
import math
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLOWorld

# --------------------------------------------------------------------------
# Every threshold lives here so it can be tuned in one place. Fractions are
# relative to the box or the image they describe, never absolute pixels, so
# the same numbers work at any resolution.
# --------------------------------------------------------------------------
CONFIG = {
    # --- detection ---
    "weights": "yolov8s-worldv2.pt",
    "anchor_conf": 0.05,     # furniture is big -- raising this to ~0.15 cuts noise
    "item_conf": 0.05,       # items are small -- keep low so we do not miss them
    "imgsz": 1280,           # default input size for both passes
    "item_imgsz": None,      # live mode drops this to 640 to keep the loop fast


    # --- anchor cache ---
    "anchor_ttl": 300.0,     # seconds before cached furniture is re-detected
    "merge_iou": 0.60,       # same-class boxes above this IoU are one object
    "item_merge_iou": 0.55,  # cross-class: "mug" and "cup" on one object

    # --- "on" (item resting on a surface) ---
    "on_band": 0.35,         # item bottom must land in the top 35% of the anchor
    "on_slack": 0.08,        # ... plus this much above its top edge
    "on_overlap": 0.45,      # fraction of the item's WIDTH over the anchor

    # --- "in" (item inside a container) ---
    "in_overlap": 0.55,      # fraction of the item's AREA inside the container

    # --- "under" ---
    "under_overlap": 0.40,   # fraction of the item's AREA inside the anchor box

    # --- floor ---
    "floor_top": 0.55,       # floor band starts this far down the image

    # --- nearness ---
    "near_max": 0.40,        # max centre distance, as a fraction of the diagonal
    "near_touching": 0.12,   # closer than this reads as "next to" not "left of"
    "near_count": 2,         # how many neighbours to report

    # --- covering ---
    "cover_overlap": 0.15,   # cushion/coat overlap that raises the "under" flag
    "cover_classes": ("cushion", "coat"),

    # --- confidence ---
    "det_conf_full": 0.30,   # detector conf at which we stop discounting

    # --- guesses ---
    "guess_below": 0.30,     # item detections under this are marked "guess"
    "drop_guesses": False,   # True: discard them instead of marking them

    # --- gemini (only used if you call gemini_review explicitly) ---
    # Confirmed against models.list() on 2026-11-14. There is no
    # "gemini-flash-latest" alias; run detect.py --list-models to see yours.
    "gemini_model": os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash",
    "gemini_override_below": 0.60,   # only accept its correction below this conf
}

# Prompts sent to YOLO-World (keys) mapped onto the name we actually use
# (values), which merges synonyms: a "couch" and a "sofa" are one thing.
ANCHOR_SYNONYMS = {
    "table": "table", "coffee table": "table", "desk": "desk",
    "sofa": "sofa", "couch": "sofa",
    "cushion": "cushion", "pillow": "cushion",
    "chair": "chair", "armchair": "chair",
    "bed": "bed",
    "shelf": "shelf", "bookshelf": "shelf",
    "cabinet": "cabinet", "drawer": "drawer",
    "counter": "counter", "countertop": "counter",
    "laptop": "laptop", "lamp": "lamp",
    "bag": "bag", "handbag": "bag", "backpack": "bag",
    "coat": "coat", "jacket": "coat",
    "door": "door", "rug": "rug", "carpet": "rug",
}

# Things an item can sit ON.
SURFACES = {"table", "desk", "sofa", "chair", "bed", "shelf", "cabinet",
            "counter", "rug", "cushion", "laptop"}
# Things an item can be IN.
CONTAINERS = {"bag", "drawer", "cabinet"}
# Things with a gap underneath.
UNDERSIDES = {"table", "desk", "sofa", "bed", "chair", "counter", "shelf",
              "cabinet", "rug"}


# ---------------------------------------------------------------- geometry

def _w(b):
    return max(0.0, b[2] - b[0])


def _h(b):
    return max(0.0, b[3] - b[1])


def _cx(b):
    return (b[0] + b[2]) / 2.0


def _cy(b):
    return (b[1] + b[3]) / 2.0


def _area(b):
    return _w(b) * _h(b)


def _inter_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _iou(a, b):
    union = _area(a) + _area(b) - _inter_area(a, b)
    return _inter_area(a, b) / union if union > 0 else 0.0


def _inside_frac(a, b):
    """How much of box a's area sits inside box b (0..1)."""
    return _inter_area(a, b) / _area(a) if _area(a) > 0 else 0.0


def _x_overlap_frac(a, b):
    """How much of a's width sits within b's horizontal span (0..1)."""
    span = min(a[2], b[2]) - max(a[0], b[0])
    return max(0.0, span) / _w(a) if _w(a) > 0 else 0.0


def _plural(name):
    """Crude but good enough for demo sentences: keys/glasses are, phone is."""
    return name.endswith("glasses") or (name.endswith("s") and not name.endswith("ss"))


# ---------------------------------------------------------------- the mapper

class SceneMapper:
    """Detects furniture anchors, then places items relative to them."""

    def __init__(self, config=None):
        self.cfg = dict(CONFIG)
        self.cfg.update(config or {})
        self._anchor_model = None
        self._item_model = None
        self._cache = {}        # camera -> {"t", "anchors", "shape"}
        self._names = {}        # anchor id -> friendly name from Gemini
        self._gemini_cache = {}  # frame hash -> raw Gemini reply
        self._gemini = None     # the client must outlive the call that uses it

    # -- models ------------------------------------------------------------
    # Two model instances, because set_classes costs ~5s of CLIP text
    # encoding. Keeping anchors and items on separate instances means that
    # cost is paid once each per process instead of on every switch.

    def _load(self, prompts):
        model = YOLOWorld(self.cfg["weights"])
        model.set_classes(prompts)
        return model

    def _detect(self, model, frame, conf, imgsz=None):
        r = model.predict(frame, conf=conf,
                          imgsz=imgsz or self.cfg["imgsz"], verbose=False)[0]
        out = []
        for box in r.boxes:
            out.append({
                "cls": r.names[int(box.cls)],
                "box": [float(v) for v in box.xyxy[0]],
                "conf": float(box.conf),
            })
        return out

    # -- anchors -----------------------------------------------------------

    def anchors(self, frame, camera="default", refresh=False):
        """Furniture in this frame, cached per camera.

        Furniture does not move, so this only re-runs after anchor_ttl
        seconds or when refresh=True.
        """
        hit = self._cache.get(camera)
        if hit and not refresh and time.time() - hit["t"] < self.cfg["anchor_ttl"]:
            return hit["anchors"]

        if self._anchor_model is None:
            self._anchor_model = self._load(list(ANCHOR_SYNONYMS))

        raw = self._detect(self._anchor_model, frame, self.cfg["anchor_conf"])
        for d in raw:                                    # couch -> sofa
            d["cls"] = ANCHOR_SYNONYMS.get(d["cls"], d["cls"])
        anchors = self._assign_ids(self._merge(raw))
        self._cache[camera] = {"t": time.time(), "anchors": anchors,
                               "shape": frame.shape}
        return anchors

    def _merge(self, dets):
        """Drop duplicate boxes of the same class, keeping the confident one."""
        kept = []
        for d in sorted(dets, key=lambda d: -d["conf"]):
            if any(k["cls"] == d["cls"] and _iou(k["box"], d["box"]) > self.cfg["merge_iou"]
                   for k in kept):
                continue
            kept.append(d)
        return kept

    def _assign_ids(self, dets):
        """table_1, table_2, ... numbered left to right so ids are stable."""
        out = []
        for cls in sorted({d["cls"] for d in dets}):
            group = sorted([d for d in dets if d["cls"] == cls],
                           key=lambda d: _cx(d["box"]))
            for i, d in enumerate(group, 1):
                out.append({"id": "%s_%d" % (cls, i), **d})
        return out

    # -- floor -------------------------------------------------------------

    def floor_mask(self, shape, anchors):
        """Bottom band of the image minus every furniture box.

        Note a table's box includes the space under it, so that space is not
        floor -- which is what we want, since "under the table" is a better
        answer than "on the floor".
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[int(h * self.cfg["floor_top"]):, :] = True
        for a in anchors:
            x1, y1, x2, y2 = (int(max(0, v)) for v in a["box"])
            mask[y1:y2, x1:x2] = False
        return mask

    def _on_floor(self, box, shape, floor):
        """Is the item's footprint (bottom centre) on floor?"""
        h, w = shape[:2]
        x = int(min(max(_cx(box), 0), w - 1))
        y = int(min(max(box[3], 0), h - 1))
        if floor is not None:
            return bool(floor[y, x])
        return box[3] > h * self.cfg["floor_top"]

    # -- items -------------------------------------------------------------

    def detect_items(self, frame, names):
        """Detect the things we are actually looking for (keys, phone, ...)."""
        if self._item_model is None or self._item_prompts != list(names):
            self._item_model = self._load(list(names))
            self._item_prompts = list(names)
        found = self._merge_items(self._detect(self._item_model, frame,
                                               self.cfg["item_conf"],
                                               self.cfg["item_imgsz"]))
        if self.cfg["drop_guesses"]:
            found = [d for d in found if d["conf"] >= self.cfg["guess_below"]]
        return found

    _item_prompts = None

    def _merge_items(self, dets):
        """One object often fires several prompts at once -- a single cup comes
        back as "mug" AND "cup" -- and YOLO-World stacks near-duplicate boxes.
        Keep the most confident box per cluster, ignoring class, so each real
        object yields one answer instead of four.
        """
        kept = []
        for d in sorted(dets, key=lambda d: -d["conf"]):
            if any(_iou(k["box"], d["box"]) > self.cfg["item_merge_iou"] for k in kept):
                continue
            kept.append(d)
        return kept

    # -- the interesting part ---------------------------------------------

    def describe(self, item, anchors, shape, floor=None):
        """Place one item. Returns (json-ready dict, plain sentence)."""
        box, name = item["box"], item.get("cls") or item.get("item")
        support, score = self._support(box, anchors, shape, floor)
        near = self._near(box, anchors, shape, skip=support.get("anchor"))
        covered = self._covered_by(box, anchors)

        # Discount the geometry score by how sure the detector was: a 0.06
        # box in the right place is still a guess.
        det = float(item.get("conf", 1.0))
        conf = score * (0.5 + 0.5 * min(1.0, det / self.cfg["det_conf_full"]))

        out = {"item": name, "support": support, "near": near,
               "confidence": round(conf, 2),
               "guess": det < self.cfg["guess_below"]}
        if covered:
            out["flag"] = "possibly under the %s" % self.friendly(covered)
        return out, self.sentence(out)

    def _support(self, box, anchors, shape, floor):
        """Pick the best support relation. Returns (dict, score 0..1).

        Order matters: "in" beats "on" beats "under", because the tests get
        looser in that order and the first match is the most specific claim.
        """
        # --- in ---
        best, best_score = None, 0.0
        for a in anchors:
            if a["cls"] not in CONTAINERS:
                continue
            frac = _inside_frac(box, a["box"])
            if frac >= self.cfg["in_overlap"] and frac > best_score:
                best, best_score = a, frac
        if best is not None:
            return {"rel": "in", "anchor": best["id"]}, min(1.0, best_score)

        # --- on ---
        for a in anchors:
            if a["cls"] not in SURFACES:
                continue
            ov = _x_overlap_frac(box, a["box"])
            if ov < self.cfg["on_overlap"]:
                continue
            top, height = a["box"][1], _h(a["box"])
            band, slack = self.cfg["on_band"] * height, self.cfg["on_slack"] * height
            bottom = box[3]
            if not (top - slack <= bottom <= top + band):
                continue
            # Score: how squarely it overlaps, and how close the item's bottom
            # edge is to the anchor's top edge (a lower edge is less certain).
            depth = (bottom - (top - slack)) / (band + slack)      # 0 at top
            score = 0.5 * min(1.0, ov / 0.8) + 0.5 * (1.0 - min(1.0, depth))
            if score > best_score:
                best, best_score = a, score
        if best is not None:
            return {"rel": "on", "anchor": best["id"]}, min(1.0, best_score)

        # --- under ---
        for a in anchors:
            if a["cls"] not in UNDERSIDES:
                continue
            frac = _inside_frac(box, a["box"])
            top, height = a["box"][1], _h(a["box"])
            if frac >= self.cfg["under_overlap"] and box[3] > top + self.cfg["on_band"] * height:
                if frac > best_score:
                    best, best_score = a, frac
        if best is not None:
            return {"rel": "under", "anchor": best["id"]}, min(1.0, best_score * 0.9)

        # --- floor ---
        if self._on_floor(box, shape, floor):
            return {"rel": "on the floor", "anchor": None}, 0.6

        return {"rel": "unknown", "anchor": None}, 0.3

    def _near(self, box, anchors, shape, skip=None):
        """Nearest 1-2 anchors, each with a direction word."""
        h, w = shape[:2]
        diag = math.hypot(w, h)
        found = []
        for a in anchors:
            if a["id"] == skip:
                continue
            dx = _cx(box) - _cx(a["box"])
            dy = _cy(box) - _cy(a["box"])
            dist = math.hypot(dx, dy) / diag
            if dist > self.cfg["near_max"]:
                continue
            if dist < self.cfg["near_touching"]:
                direction = "next to"
            elif abs(dx) >= abs(dy):
                direction = "left of" if dx < 0 else "right of"
            else:
                direction = "in front of" if dy > 0 else "behind"
            found.append({"anchor": a["id"], "dir": direction,
                          "dist": round(dist, 3)})
        found.sort(key=lambda f: f["dist"])
        return found[:self.cfg["near_count"]]

    def _covered_by(self, box, anchors):
        """Anchor id of a cushion/coat lying over the item, if any."""
        for a in anchors:
            if a["cls"] not in self.cfg["cover_classes"]:
                continue
            if _inside_frac(box, a["box"]) >= self.cfg["cover_overlap"]:
                return a["id"]
        return None

    # -- words -------------------------------------------------------------

    def friendly(self, anchor_id):
        """"table_1" -> "table", unless Gemini gave us something nicer."""
        if anchor_id is None:
            return ""
        return self._names.get(anchor_id) or anchor_id.rsplit("_", 1)[0]

    def sentence(self, desc):
        """One plain sentence a person would actually say."""
        name = desc["item"]
        # A weak detection is worded as a possibility, not a fact.
        verb = "might be" if desc.get("guess") else ("are" if _plural(name) else "is")
        rel, anchor = desc["support"]["rel"], desc["support"]["anchor"]

        if rel == "on the floor":
            where = "on the floor"
        elif rel == "unknown":
            where = "somewhere in view"
        else:
            article = "" if self.friendly(anchor).startswith("the ") else "the "
            where = "%s %s%s" % ("inside" if rel == "in" else rel,
                                 article, self.friendly(anchor))

        text = "Your %s %s %s" % (name, verb, where)
        if desc["near"]:
            n = desc["near"][0]
            lead = "" if n["dir"] == "next to" else "just "
            friendly = self.friendly(n["anchor"])
            article = "" if friendly.startswith("the ") else "the "
            text += ", %s%s %s%s" % (lead, n["dir"], article, friendly)
        if desc.get("flag"):
            text += " -- %s" % desc["flag"]
        return text + "."

    # -- export ------------------------------------------------------------

    def payload(self, shape, anchors, items, descs, camera="default"):
        """Everything known about one frame, as a single JSON-ready dict.

        items and descs must line up: descs[i] is describe(items[i], ...)[0].
        Each item carries its own box and id, so two mugs on one table stay
        two distinguishable entries. Ids are numbered left to right per
        class, like anchors -- stable while nothing moves, but not tracked:
        swap two mugs and their ids swap too.
        """
        h, w = shape[:2]

        ids = {}
        for cls in {it.get("cls") or it.get("item") for it in items}:
            group = sorted((i for i, it in enumerate(items)
                            if (it.get("cls") or it.get("item")) == cls),
                           key=lambda i: _cx(items[i]["box"]))
            for n, i in enumerate(group, 1):
                ids[i] = "%s_%d" % (str(cls).replace(" ", "_"), n)

        out_items = []
        for i, (it, desc) in enumerate(zip(items, descs)):
            entry = {"id": ids[i], "item": desc["item"],
                     "box": [round(v, 1) for v in it["box"]],
                     "det_conf": round(float(it.get("conf", 1.0)), 3)}
            entry.update({k: v for k, v in desc.items() if k != "item"})
            entry["sentence"] = self.sentence(desc)
            out_items.append(entry)

        return {
            "version": 1,
            "timestamp": _now(),
            "camera": camera,
            "image": {"width": int(w), "height": int(h)},
            "anchors": [{"id": a["id"], "cls": a["cls"],
                         "name": self.friendly(a["id"]),
                         "box": [round(v, 1) for v in a["box"]],
                         "conf": round(a["conf"], 3)} for a in anchors],
            "items": out_items,
        }

    def detections(self, shape, items, descs, camera_id="default"):
        """One flat JSON-ready dict per detected item, for a line-per-
        detection stream. items and descs line up as in payload().

        The same facts as payload() minus the anchor list: that is fixed, so
        a reader loads it once from the anchors file rather than getting it
        again on every line. All lines from one frame share one "time".
        """
        h, w = shape[:2]
        now = _now()
        out = []
        for it, desc in zip(items, descs):
            line = {"time": now, "camera_id": camera_id,
                    "image_w": int(w), "image_h": int(h),
                    "item": desc["item"],
                    "box": [round(v, 1) for v in it["box"]],
                    "det_conf": round(float(it.get("conf", 1.0)), 3)}
            line.update({k: v for k, v in desc.items() if k != "item"})
            out.append(line)
        return out

    def renumber(self, anchors):
        """Fresh ids for a trimmed list, so it reads table_1, table_2 with
        no gaps left by the pieces that were deleted."""
        return self._assign_ids([{k: v for k, v in a.items() if k != "id"}
                                 for a in anchors])

    # -- optional Gemini pass ---------------------------------------------

    def gemini_review(self, frame, anchors, descs, api_key=None):
        """OFF BY DEFAULT. Ask Gemini for friendly anchor names and a sanity
        check on each support relation.

        Sends ONE annotated frame (anchor boxes drawn on, so the model can see
        which box is which) and caches by frame content, so the same frame is
        never sent twice. Only overrides our answer when our own confidence
        was below gemini_override_below -- the geometry is usually right when
        it is sure, and we do not want a chatty model undoing it.
        """
        from google import genai
        from google.genai import types

        key = api_key or load_key()
        if not key:
            raise RuntimeError("no GEMINI_API_KEY (environment or .env)")
        # Keep one client on the instance. A throwaway genai.Client() gets
        # collected while the request is still in flight, and the call dies
        # with "the client has been closed".
        if self._gemini is None:
            self._gemini = genai.Client(api_key=key)

        canvas = annotate(frame, anchors, [])
        ok, buf = cv2.imencode(".jpg", canvas)
        if not ok:
            raise RuntimeError("could not encode frame")
        jpeg = buf.tobytes()

        digest = hashlib.md5(jpeg).hexdigest()
        cache_key = digest + "|" + ",".join(sorted(a["id"] for a in anchors))
        if cache_key in self._gemini_cache:
            reply = self._gemini_cache[cache_key]
        else:
            listing = ", ".join("%s" % a["id"] for a in anchors)
            guesses = "; ".join(
                "%s: %s %s" % (d["item"], d["support"]["rel"],
                               d["support"]["anchor"] or "")
                for d in descs) or "none"
            prompt = (
                "The image has furniture boxes drawn on it with ids: %s.\n"
                "A geometry-only system guessed where some items are: %s.\n"
                "Reply with JSON only:\n"
                '{"anchors": {"<id>": "<friendly name, e.g. the grey sofa>"},\n'
                ' "items": [{"item": "<name>", "rel": "on|in|under|on the floor",'
                ' "anchor": "<id or null>"}]}\n'
                "Keep names short. Only include items you can actually see."
            ) % (listing, guesses)

            response = self._gemini.models.generate_content(
                model=self.cfg["gemini_model"],
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                          prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"),
            )
            reply = json.loads(response.text)
            self._gemini_cache[cache_key] = reply

        self._names.update({k: str(v) for k, v in (reply.get("anchors") or {}).items()})

        fixes = {d.get("item"): d for d in (reply.get("items") or [])}
        for desc in descs:
            fix = fixes.get(desc["item"])
            if not fix or desc["confidence"] >= self.cfg["gemini_override_below"]:
                continue
            if fix.get("rel") and fix["rel"] != desc["support"]["rel"]:
                desc["support"] = {"rel": fix["rel"], "anchor": fix.get("anchor")}
                desc["corrected_by"] = "gemini"
        return descs


# ---------------------------------------------------------------- helpers

def load_key():
    """GEMINI_API_KEY from the environment, else from .env beside this file."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == "GEMINI_API_KEY":
                    return value.strip().strip('"').strip("'")
    return ""


def write_json(path, data):
    """Save a payload. Returns False if the file was busy and nothing changed.

    Written to a temp file and swapped in, so a program polling the file
    never reads half of one. On Windows the swap fails while a reader has
    the file open -- in a live loop the next frame simply tries again.
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    try:
        os.replace(tmp, path)
    except PermissionError:
        return False
    return True


def _now():
    """UTC, ISO 8601, to the millisecond."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def claim_stdout():
    """Reserve stdout for JSON lines. Returns the stream to write them to.

    Call it before anything prints. From then on print() goes to stderr, and
    so does ultralytics' logger (which writes to stdout by default), so a
    program reading the pipe sees JSON and nothing else.
    """
    out = sys.stdout
    sys.stdout = sys.stderr
    from ultralytics.utils import LOGGER
    for handler in LOGGER.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stderr)
    return out


def emit(stream, records):
    """Write one JSON object per line and flush, so a reader gets each frame
    as it happens. Returns False once the reader has gone away."""
    try:
        for record in records:
            stream.write(json.dumps(record) + "\n")
        stream.flush()
    except OSError:              # BrokenPipeError, or EINVAL on Windows
        return False
    return True


def save_anchors(path, anchors, shape, camera_id="default"):
    """Save the confirmed furniture as the fixed list for this camera."""
    h, w = shape[:2]
    return write_json(path, {
        "version": 1,
        "saved": _now(),
        "camera_id": camera_id,
        "image_w": int(w), "image_h": int(h),
        "anchors": [{"id": a["id"], "cls": a["cls"],
                     "box": [round(v, 1) for v in a["box"]],
                     "conf": round(a["conf"], 3)} for a in anchors],
    })


def load_anchors(path, shape=None):
    """The fixed furniture list written by save_anchors().

    Boxes are in pixels of the frame they were confirmed on. Pass the
    current frame's shape and they are rescaled if the camera now delivers a
    different resolution.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    anchors = data["anchors"]
    if shape is not None and data.get("image_w") and data.get("image_h"):
        sx, sy = shape[1] / data["image_w"], shape[0] / data["image_h"]
        if (sx, sy) != (1.0, 1.0):
            for a in anchors:
                x1, y1, x2, y2 = a["box"]
                a["box"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
    return anchors


ANCHOR_COLOR = (180, 180, 180)
ITEM_COLOR = (0, 215, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate(frame, anchors, items, sentences=(), floor=None):
    """Anchor boxes, item boxes and the sentences, drawn on a copy."""
    canvas = frame.copy()

    if floor is not None:                      # tint the floor to tune floor_top
        tint = np.zeros_like(canvas)
        tint[floor] = (60, 30, 0)
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.5, 0)

    for a in anchors:
        x1, y1, x2, y2 = (int(v) for v in a["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), ANCHOR_COLOR, 1)
        cv2.putText(canvas, "%s %.2f" % (a["id"], a["conf"]), (x1 + 3, y1 + 16),
                    FONT, 0.45, ANCHOR_COLOR, 1)

    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), ITEM_COLOR, 3)
        label = "%s %.2f" % (it.get("cls") or it.get("item"), it.get("conf", 0.0))
        cv2.putText(canvas, label, (x1, max(14, y1 - 6)), FONT, 0.6, ITEM_COLOR, 2)

    # Sentences go on a darkened band so they stay readable over a bright
    # room. More than six and the picture is no longer worth looking at.
    lines = _wrap(sentences, canvas.shape[1])[:6]
    if lines:
        band = 12 + 26 * len(lines)
        strip = canvas[:band]
        canvas[:band] = cv2.addWeighted(strip, 0.25, np.zeros_like(strip), 0.75, 0)
        y = 28
        for line in lines:
            cv2.putText(canvas, line, (10, y), FONT, 0.62, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y += 26
    return canvas


def _wrap(sentences, width, scale=0.62):
    """Break sentences so they fit the image width."""
    chars = max(20, int(width / (scale * 17)))
    lines = []
    for s in sentences:
        words, current = s.split(), ""
        for word in words:
            if len(current) + len(word) + 1 > chars:
                lines.append(current)
                current = word
            else:
                current = (current + " " + word).strip()
        if current:
            lines.append(current)
    return lines
