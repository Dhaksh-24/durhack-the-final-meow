"""room.py -- Trace's model of the room, built once and then kept in room.json.

The tracker (multicam.py) sees boxes in pixels. People think in places: "on the
coffee table", "down the side of the sofa". This file turns every camera's view
into a list of named PLACES (zones), and records for each one:

    kind     table / sofa / bed / shelf / door / floor / container / chair / other
    hold     0..1, how likely someone is to put a small thing down there
    hiding   where a small thing goes missing at a place like this
    near     places beside it (things knocked off a table land on the floor below)

Coordinates are pixels of the frame multicam.py works on (every camera is
resized to VIEW_W wide), origin top-left, rect = [x1, y1, x2, y2].

Where the places come from, best first:
    1. zones you drag out in multicam.py (zones.json, "Hall table")
    2. furniture YOLO-World found in a photo of the room (scene.py anchors)
    3. a plain left / middle / right / floor split, so nothing is ever unplaced

Build it before the demo (each camera needs one photo, or a scene.py JSON):

    python room.py build --photo "living-room=memory/snapshots/1.jpg"
    python room.py build --scene "living-room=test_out/room1.json"
    python room.py build --grab "living-room=0"        # take a photo from webcam 0
    python room.py show                                # print the model
    python room.py draw living-room photo.jpg out.jpg  # draw the places on a photo

The camera name before = is the camera id multicam.py uses (the camera name in
lower case with dashes, e.g. "Living room" -> "living-room"). Running multicam.py
without room.json still works: it falls back to your drawn zones plus the
plain split, and writes room.json as it goes.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

VIEW_W = 960           # multicam.py resizes every camera frame to this width
ROOM_FILE = Path("room.json")
ZONES_FILE = Path("zones.json")
FLOOR_TOP = 0.78       # the plain floor band starts this far down the frame
NEAR_GAP = 0.08        # places closer than this (fraction of frame width) are "near"
MIN_ANCHOR_CONF = 0.08

# kind -> (words in a name that mean this kind, hold, where small things hide there)
KINDS = {
    "table": (["table", "desk", "counter", "worktop", "console", "dresser", "nightstand",
               "bedside", "sideboard", "island", "bench"],
              1.0, ["under post, papers or a magazine", "behind a mug, lamp or laptop",
                    "on the floor just underneath"]),
    "sofa": (["sofa", "couch", "settee", "armchair", "cushion", "pillow"],
             0.7, ["down the side between the cushions and the arm", "under a cushion or throw",
                   "on the floor under the front edge"]),
    "bed": (["bed", "duvet", "mattress"],
            0.6, ["under the duvet or a pillow", "down the gap by the wall",
                  "on the floor under the bed"]),
    "shelf": (["shelf", "shelves", "bookcase", "cabinet", "cupboard", "unit", "mantel", "windowsill"],
              0.8, ["behind something at the front of the shelf", "on a lower shelf"]),
    "door": (["door", "hall", "entrance", "porch", "doorway", "hook", "coat"],
             0.8, ["in the bowl or on the hook by the door", "in a coat pocket on the hook",
                   "on the floor by the mat"]),
    "container": (["bag", "basket", "box", "drawer", "tray", "bowl", "backpack", "handbag"],
                  0.9, ["inside it, at the bottom", "in a side pocket"]),
    "chair": (["chair", "stool", "seat"],
              0.5, ["on the seat under a jacket", "on the floor under the chair"]),
    "floor": (["floor", "rug", "carpet", "mat"],
              0.25, ["under the edge of the rug", "under the sofa or the table"]),
    "other": ([], 0.4, ["under or behind whatever is there"]),
}

# scene.py anchor class -> kind (things that aren't places, like a lamp, are left out)
ANCHOR_KIND = {"table": "table", "desk": "table", "counter": "table", "sofa": "sofa",
               "cushion": "sofa", "chair": "chair", "bed": "bed", "shelf": "shelf",
               "cabinet": "shelf", "drawer": "container", "bag": "container",
               "door": "door", "rug": "floor"}
NICE = {"table": "table", "desk": "desk", "counter": "counter", "sofa": "sofa",
        "cushion": "cushions", "chair": "chair", "bed": "bed", "shelf": "shelf",
        "cabinet": "cabinet", "drawer": "drawer", "bag": "bag", "door": "door", "rug": "rug"}


def kind_of(name):
    """The last word that names a kind wins: "hall floor" is a floor, "hall table" a table."""
    words = re.findall(r"[a-z]+", str(name).lower())
    for w in reversed(words):
        for kind, (keys, _, _) in KINDS.items():
            if w in keys or (w.endswith("s") and w[:-1] in keys) or (w.endswith("es") and w[:-2] in keys):
                return kind
    return "other"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "place"


def area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def gap(a, b):
    """Distance between two rects (0 when they touch or overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def make_zone(name, camera, rect, kind=None, source="drawn"):
    kind = kind or kind_of(name)
    _, hold, hiding = KINDS[kind]
    return {"id": f"{camera}/{slug(name)}", "name": name, "camera": camera,
            "rect": [round(float(v), 1) for v in rect], "kind": kind, "hold": hold,
            "hiding": list(hiding), "source": source, "near": []}


class Room:
    """Named places for every camera. Thread-safe enough for one writer (multicam's
    loop) and many readers, because zone lists are only ever replaced, never edited."""

    def __init__(self, path=ROOM_FILE):
        self.path = Path(path) if path else None
        self.cameras = {}        # cam id -> {"name", "w", "h"}
        self.zones = []
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.cameras = data.get("cameras", {})
                self.zones = data.get("zones", [])
            except ValueError:
                print(f"room: {self.path} is not valid JSON - starting a fresh room model")
        self.by_id = {z["id"]: z for z in self.zones}

    # -------------------------------------------------------------- building
    def set_camera(self, cam, name=None, w=None, h=None):
        """Remember a camera and give it the plain split if it has no places yet."""
        c = self.cameras.setdefault(cam, {"name": name or cam, "w": w or VIEW_W, "h": h or 540})
        changed = False
        if name and c.get("name") != name:
            c["name"], changed = name, True
        if w and h and (c.get("w"), c.get("h")) != (int(w), int(h)):
            c["w"], c["h"], changed = int(w), int(h), True
        only_auto = not any(z["camera"] == cam and z["source"] != "auto" for z in self.zones)
        if only_auto and (changed or not any(z["camera"] == cam for z in self.zones)):
            self._replace(cam, "auto", self._plain_split(cam))
            changed = True
        return changed

    def _plain_split(self, cam):
        c = self.cameras[cam]
        w, h, nm = c["w"], c["h"], c["name"]
        top = h * FLOOR_TOP
        out = [make_zone(f"{side} of the {nm} view", cam,
                         [w * i / 3, 0, w * (i + 1) / 3, top], "other", "auto")
               for i, side in enumerate(("left side", "middle", "right side"))]
        out.append(make_zone(f"floor in the {nm} view", cam, [0, top, w, h], "floor", "auto"))
        return out

    def set_drawn(self, zones, cam_names=None):
        """Mirror multicam's hand-drawn zones (zones.json). They win over everything else."""
        cam_names = cam_names or {}
        for cam in {z.get("camera") for z in zones} | {c for c in self.cameras}:
            if cam is None:
                continue
            mine = [make_zone(z["name"], cam, z["rect"], z.get("kind"), "drawn")
                    for z in zones if z.get("camera") == cam]
            self.cameras.setdefault(cam, {"name": cam_names.get(cam, cam), "w": VIEW_W, "h": 540})
            self._replace(cam, "drawn", mine)
            has_real = any(z["camera"] == cam and z["source"] in ("drawn", "scene") for z in self.zones)
            self._replace(cam, "auto", [] if has_real else self._plain_split(cam))

    def add_anchors(self, cam, anchors, image_w, image_h, cam_name=None):
        """Furniture from scene.py -> places, scaled to multicam's frame size."""
        s = VIEW_W / float(image_w)
        self.cameras[cam] = {"name": cam_name or self.cameras.get(cam, {}).get("name", cam),
                             "w": VIEW_W, "h": int(round(image_h * s))}
        kept = []
        for a in sorted(anchors, key=lambda a: -a.get("conf", 0)):
            cls = a.get("cls") or a.get("name")
            if cls not in ANCHOR_KIND or a.get("conf", 1) < MIN_ANCHOR_CONF:
                continue
            box = [v * s for v in a["box"]]
            if area(box) > 0.6 * VIEW_W * self.cameras[cam]["h"] and cls != "rug":
                continue          # a "sofa" filling the whole frame is not a place you can point at
            if any(k[0] == cls and iou(k[1], box) > 0.35 for k in kept):
                continue
            kept.append((cls, box))
        counts, zones = {}, []
        for cls, box in sorted(kept, key=lambda k: (k[1][0] + k[1][2]) / 2):
            counts[cls] = counts.get(cls, 0) + 1
        seen = {}
        for cls, box in sorted(kept, key=lambda k: (k[1][0] + k[1][2]) / 2):
            seen[cls] = seen.get(cls, 0) + 1
            name = NICE.get(cls, cls)
            if counts[cls] in (2, 3):
                sides = ("left", "right") if counts[cls] == 2 else ("left", "middle", "right")
                name = f"{sides[seen[cls] - 1]} {name}"
            elif counts[cls] > 3:
                name = f"{name} {seen[cls]}"
            zones.append(make_zone(name, cam, box, ANCHOR_KIND[cls], "scene"))
        h = self.cameras[cam]["h"]       # rugs are smaller than the floor; zone_at picks the rug first
        zones.append(make_zone("floor", cam, [0, h * FLOOR_TOP, VIEW_W, h], "floor", "scene"))
        self._replace(cam, "scene", zones)
        self._replace(cam, "auto", [])
        return zones

    def _replace(self, cam, source, new):
        self.zones = [z for z in self.zones if not (z["camera"] == cam and z["source"] == source)] + new
        self._link()

    def _link(self):
        """near = same camera and within NEAR_GAP of each other. Ids made unique."""
        seen = set()
        for z in self.zones:
            base, n = z["id"], 2
            while z["id"] in seen:
                z["id"], n = f"{base}-{n}", n + 1
            seen.add(z["id"])
        for z in self.zones:
            w = self.cameras.get(z["camera"], {}).get("w", VIEW_W)
            z["near"] = [o["id"] for o in self.zones if o is not z and o["camera"] == z["camera"]
                         and gap(z["rect"], o["rect"]) < NEAR_GAP * w]
        self.by_id = {z["id"]: z for z in self.zones}

    # -------------------------------------------------------------- lookups
    def zone_at(self, cam, box):
        """The most specific place holding this box's centre (smallest zone wins)."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        hits = [z for z in self.zones if z["camera"] == cam
                and z["rect"][0] <= cx <= z["rect"][2] and z["rect"][1] <= cy <= z["rect"][3]]
        if not hits:
            # outside every place: the nearest one on this camera
            cands = [z for z in self.zones if z["camera"] == cam]
            return min(cands, key=lambda z: gap(z["rect"], [cx, cy, cx, cy]), default=None)
        def resting_on(z):          # its bottom edge in the top half of the place = sitting ON it
            r = z["rect"]
            return r[1] - 0.05 * (r[3] - r[1]) <= box[3] <= r[1] + 0.5 * (r[3] - r[1])
        return min(hits, key=lambda z: (z["source"] != "drawn", not resting_on(z), area(z["rect"])))

    def zones_touched(self, cam, person_box, min_frac=0.25):
        """Places a person is at: their box covers enough of the place, or their feet are in it."""
        fx, fy = (person_box[0] + person_box[2]) / 2, person_box[3]
        out = []
        for z in self.zones:
            if z["camera"] != cam:
                continue
            r = z["rect"]
            covered = inter(r, person_box) / max(1.0, min(area(r), area(person_box)))
            feet = r[0] <= fx <= r[2] and r[1] <= fy <= r[3]
            if covered >= min_frac or feet:
                out.append(z)
        # a person standing at the coffee table also overlaps the floor; keep the floor
        # only if it is the only thing they're at
        real = [z for z in out if z["kind"] != "floor"]
        return real or out

    def floor_below(self, zone):
        """The floor place under/next to a zone (where knocked things land)."""
        for zid in zone.get("near", []):
            z = self.by_id.get(zid)
            if z and z["kind"] == "floor":
                return z
        cands = [z for z in self.zones if z["camera"] == zone["camera"] and z["kind"] == "floor"]
        return cands[0] if cands else None

    def cam_name(self, cam):
        return self.cameras.get(cam, {}).get("name", cam)

    def describe(self):
        """A compact, LLM-friendly view of the room (no pixels)."""
        return [{"id": z["id"], "place": z["name"], "camera": self.cam_name(z["camera"]),
                 "kind": z["kind"], "hiding_spots": z["hiding"][:3],
                 "next_to": [self.by_id[n]["name"] for n in z["near"] if n in self.by_id][:4]}
                for z in self.zones]

    def save(self):
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "view_w": VIEW_W, "cameras": self.cameras,
                                   "zones": self.zones}, indent=2))
        os.replace(tmp, self.path)


# ============================================================== CLI ====

def _pairs(specs):
    for s in specs or []:
        cam, _, src = s.partition("=")
        if not src:
            sys.exit(f'expected CAMERA=SOURCE, got "{s}"')
        yield slug(cam), cam, src


def _anchors_from_image(img):
    import scene
    mapper = scene.SceneMapper()
    return mapper.anchors(img, refresh=True)


def build(args):
    room = Room(args.out)
    if ZONES_FILE.exists():
        room.set_drawn(json.loads(ZONES_FILE.read_text()))
    for cam, label, src in _pairs(args.scene):
        data = json.loads(Path(src).read_text())
        w = data.get("image", {}).get("width") or data.get("image_w")
        h = data.get("image", {}).get("height") or data.get("image_h")
        zs = room.add_anchors(cam, data["anchors"], w, h, label)
        print(f"{cam}: {len(zs)} places from {src}")
    grabs = [(p, "photo") for p in _pairs(args.photo)] + [(p, "grab") for p in _pairs(args.grab)]
    if grabs:
        import cv2
    for (cam, label, src), how in grabs:
        if how == "photo":
            img = cv2.imread(src)
        else:
            cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
            for _ in range(10):          # let auto-exposure settle
                ok, img = cap.read()
            cap.release()
            img = img if ok else None
        if img is None:
            sys.exit(f"could not read a picture from {src}")
        zs = room.add_anchors(cam, _anchors_from_image(img), img.shape[1], img.shape[0], label)
        print(f"{cam}: {len(zs)} places found in the picture from {src}")
    room._link()
    room.save()
    print(f"wrote {room.path} - {len(room.zones)} places on {len(room.cameras)} camera(s)")
    show(room)


def show(room):
    for cam, c in room.cameras.items():
        print(f"\n{c['name']} ({cam}, {c['w']}x{c['h']})")
        for z in [z for z in room.zones if z["camera"] == cam]:
            near = ", ".join(room.by_id[n]["name"] for n in z["near"] if n in room.by_id)
            print(f"  {z['name']:<28} {z['kind']:<9} [{z['source']}]  near: {near or '-'}")


def draw(room, cam, src, out):
    import cv2
    img = cv2.imread(src)
    if img is None:
        sys.exit(f"can't read {src}")
    img = cv2.resize(img, (VIEW_W, int(img.shape[0] * VIEW_W / img.shape[1])))
    for z in [z for z in room.zones if z["camera"] == cam]:
        x1, y1, x2, y2 = map(int, z["rect"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(img, f"{z['name']} ({z['kind']})", (x1 + 4, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out, img)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Build Trace's room model (room.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build room.json from zones.json + photos / scene JSON")
    b.add_argument("--scene", action="append", help="CAMERA=scene.json (from scene.py / live_scene.py)")
    b.add_argument("--photo", action="append", help="CAMERA=photo.jpg (finds the furniture in it)")
    b.add_argument("--grab", action="append", help="CAMERA=webcam index or stream URL")
    b.add_argument("--out", default=str(ROOM_FILE))
    s = sub.add_parser("show", help="print room.json")
    s.add_argument("--room", default=str(ROOM_FILE))
    d = sub.add_parser("draw", help="draw one camera's places on a photo")
    d.add_argument("camera")
    d.add_argument("photo")
    d.add_argument("out")
    d.add_argument("--room", default=str(ROOM_FILE))
    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif not Path(args.room).exists():
        sys.exit(f"no {args.room} yet - run: python room.py build ...")
    elif args.cmd == "show":
        show(Room(args.room))
    else:
        draw(Room(args.room), args.camera, args.photo, args.out)


if __name__ == "__main__":
    main()
