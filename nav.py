"""nav.py -- live "turn left / turn right / stop, reach down" from a top-down camera.

multicam.py calls update() after every analysed frame with the person boxes, the verified item
boxes and the camera's zones. trace_api.py serves compute() at GET /api/nav, aimed at where the
guide is sending the person (guide.TARGET). trace.html polls it while a search is on and speaks
the instruction when it changes.

How it works (all from the camera, nothing invented):
  where you are   centre of the most confident "person" box (from above: your head and shoulders)
  which way       the direction you moved over the last ~2 s; standing still keeps the last one
  where to go     the item's own box if the camera sees it now (glasses on the desk), otherwise the
                  centre of the zone Trace sent you to (keys hidden under the cloth on the sofa)
  left or right   the angle between your heading and the target, seen from above
  how far         pixels -> metres with VIEW_WIDTH_M (the width of floor the camera sees; set it with
                  multicam.py --view-width-m), spoken as steps of about 0.6 m

A camera that mirrors the image (a phone's front camera) swaps left and right:
multicam.py --nav-mirror fixes that. Test once: walk forward, step to your right, and check it says right.
"""
import math
import re
import threading
import time
from collections import deque

VIEW_WIDTH_M = 3.0     # metres of floor across the camera image (multicam.py --view-width-m)
MIRROR = False         # True if the camera image is mirrored (multicam.py --nav-mirror)
STEP_M = 0.6           # one walking step
ARRIVE_M = 0.45        # this close to the target = "stop, reach down"
MOVE_M = 0.25          # movement needed before we trust a heading
HEADING_KEEP_S = 30.0  # how long a heading is trusted after you stop moving
STALE_S = 3.0          # a view older than this = the camera stopped
ENABLED = True         # multicam.py --no-nav: compute() always says no_target, so trace.html stays quiet

_lock = threading.Lock()
_view = {}             # the latest analysed frame
_trail = deque(maxlen=60)   # (t, x, y) person positions in pixels
_heading = {"v": None, "t": 0.0}
_last = {}             # the latest compute() result, drawn on the camera window by multicam.py


def update(cam, w, h, people, items, zones, now=None):
    """people: [(box, conf)], items: [(category, box, conf)], zones: {zone_id: (name, rect)}."""
    now = now or time.time()
    best = {}
    for cat, box, conf in items:
        if cat not in best or conf > best[cat][1]:
            best[cat] = (list(box), conf)
    person = max(people, key=lambda p: p[1])[0] if people else None
    with _lock:
        _view.update({"cam": cam, "w": w, "h": h, "t": now, "items": best, "zones": zones,
                      "person": list(person) if person is not None else None})
        if person is not None:
            _trail.append((now, (person[0] + person[2]) / 2, (person[1] + person[3]) / 2))
            _update_heading(now)


def _update_heading(now):
    t, x, y = _trail[-1]
    m_per_px = VIEW_WIDTH_M / max(1, _view["w"])
    for t0, x0, y0 in reversed(_trail):
        if now - t0 > 2.5:
            break
        if math.hypot(x - x0, y - y0) * m_per_px >= MOVE_M:
            _heading.update(v=(x - x0, y - y0), t=now)
            return


def _steps(d):
    n = max(1, round(d / STEP_M))
    return "one step" if n == 1 else f"about {n} steps"


def compute(target):
    r = _compute(target) if ENABLED else {"available": True, "state": "no_target", "say": ""}
    with _lock:
        _last.clear()
        _last.update(r, at=time.time())
    return r


def last(cam, max_age=2.0):
    """The latest directions for this camera, if Trace asked for them in the last couple of seconds."""
    with _lock:
        r = dict(_last)
    if r.get("cam") == cam and time.time() - r.get("at", 0) <= max_age and "tx" in r:
        return r
    return None


def _compute(target):
    """target: guide.TARGET ({"item": "keys-1", "zone": "sofa", "name": "the sofa", ...})."""
    now = time.time()
    with _lock:
        v = dict(_view)
        trail = list(_trail)
        heading, h_t = _heading["v"], _heading["t"]
    if not v or now - v.get("t", 0) > STALE_S:
        return {"available": False, "state": "no_camera", "say": ""}
    zone = (target or {}).get("zone")
    item = (target or {}).get("item") or ""
    if not zone or (target or {}).get("state") == "found":
        return {"available": True, "state": "no_target", "say": ""}
    if zone not in v["zones"]:      # "with you", "outside", or a zone this camera doesn't watch
        return {"available": True, "state": "not_here", "say": "", "zone": zone}
    if not v["person"] or not trail or now - trail[-1][0] > 2.0:
        return {"available": True, "state": "no_person", "say": "I can't see you on the camera yet."}

    cat = re.sub(r"-\d+$", "", item)
    seen = v["items"].get(cat)
    zname, rect = v["zones"][zone]
    if seen:
        b = seen[0]
        tx, ty, what = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, f"your {cat}"
    else:
        tx, ty, what = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2, f"the {zname.lower()}"
    _, px, py = trail[-1]
    m_per_px = VIEW_WIDTH_M / max(1, v["w"])
    dist = math.hypot(tx - px, ty - py) * m_per_px
    out = {"available": True, "target": what, "zone": zone, "seen": bool(seen), "dist_m": round(dist, 2),
           "cam": v["cam"], "px": round(px), "py": round(py), "tx": round(tx), "ty": round(ty)}

    if dist <= ARRIVE_M:
        tip = "Reach down, it's right in front of you." if seen else f"You're at {what}. Feel under things on it."
        return dict(out, state="arrived", say="Stop. " + tip)
    if heading is None or now - h_t > HEADING_KEEP_S:
        return dict(out, state="no_heading", say="Take two small steps forward, so I can see which way you're facing.")

    hx, hy = heading
    gx, gy = tx - px, ty - py
    # image y points down; seen from above, a positive cross product = the target is to your right
    ang = math.degrees(math.atan2(hx * gy - hy * gx, hx * gx + hy * gy))
    if MIRROR:
        ang = -ang
    side = "right" if ang > 0 else "left"
    a = abs(ang)
    if a < 25:
        say, st = f"Keep going straight, {_steps(dist)} to {what}.", "ahead"
    elif a < 60:
        say, st = f"Bear {side} a little, {_steps(dist)} to go.", "bear_" + side
    elif a < 135:
        say, st = f"Turn {side}, then {_steps(dist)}.", "turn_" + side
    else:
        say, st = f"Turn around, {what} is behind you, {_steps(dist)}.", "turn_around"
    return dict(out, state=st, say=say, angle=round(ang))
