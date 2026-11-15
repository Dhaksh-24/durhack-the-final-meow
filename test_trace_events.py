"""Tests for trace_events.py (no cameras, no torch):  py test_trace_events.py [out.jsonl]

Plays scripted frames through EventWriter and checks the events against SPEC.md 3.2.
With an argument, also saves the scripted session there (e.g. to replay it in the page).
"""
import os
import sys
import tempfile

import trace_events as te


def writer(path):
    if os.path.exists(path):
        os.remove(path)
    return te.EventWriter(path, echo=False)


def keys(zone, conf=0.8, box=(100, 100, 140, 130)):
    return [{"item": "keys-1", "category": "keys", "name": None, "zone": zone, "conf": conf, "box": list(box)}]


def events(w, **match):
    return [r for r in w.written if all(r.get(k) == v for k, v in match.items())]


def run(path):
    w = writer(path)
    t = [1794751200.0]

    def step(cam, sightings, zones, motion=None, dt=0.5):
        t[0] += dt
        w.frame(cam, t[0], sightings, zones, motion, (540, 960, 3))

    # one frame is not enough; 2 of 3 is
    step("phone-1", keys("sofa", 0.7), ["sofa"])
    assert not w.written, w.written
    step("phone-1", [], ["sofa"])
    step("phone-1", keys("sofa", 0.9), ["sofa"])
    ev = events(w, event="appeared")
    assert len(ev) == 1 and ev[0]["zone"] == "sofa" and ev[0]["item"] == "keys-1", w.written
    assert abs(ev[0]["confidence"] - 0.8) < 1e-6, ev[0]
    assert ev[0]["category"] == "keys" and ev[0]["camera"] == "phone-1" and ":" in ev[0]["time"]

    # a 1-3 frame dropout is not a disappearance
    for _ in range(3):
        step("phone-1", [], ["sofa"])
    step("phone-1", keys("sofa"), ["sofa"])
    assert not events(w, event="disappeared"), w.written
    # a second camera watching the same zone, but not seeing it, doesn't make it disappear
    for _ in range(6):
        step("phone-2", [], ["sofa"])
    assert not events(w, event="disappeared"), w.written

    # 4 missed frames on the camera that sees it: disappeared
    for _ in range(4):
        step("phone-1", [], ["sofa"])
    assert len(events(w, event="disappeared", zone="sofa")) == 1, w.written

    # appears at the desk 20 s later: moved, from sofa
    t[0] += 20
    step("phone-2", keys("desk"), ["desk"])
    step("phone-2", keys("desk"), ["desk"])
    mv = events(w, event="moved")
    assert len(mv) == 1 and mv[0]["zone"] == "desk" and mv[0]["from"] == "sofa", w.written

    # picked straight up from the desk and seen on the shelf before the desk timed out:
    # disappeared desk, then moved to shelf, in that order
    n = len(w.written)
    step("phone-3", keys("shelf"), ["shelf"])
    step("phone-3", keys("shelf"), ["shelf"])
    tail = [(r["event"], r["zone"]) for r in w.written[n:]]
    assert tail == [("disappeared", "desk"), ("moved", "shelf")], tail
    # the desk doesn't come back from stale frames while the shelf keeps seeing it
    for _ in range(3):
        step("phone-2", [], ["desk"])
        step("phone-3", keys("shelf"), ["shelf"])
    assert len(w.written) == n + 2, w.written[n:]

    # seen again more than 60 s after disappearing: appeared, not moved
    for _ in range(4):
        step("phone-3", [], ["shelf"])
    t[0] += 120
    step("phone-1", keys("sofa"), ["sofa"])
    step("phone-1", keys("sofa"), ["sofa"])
    assert w.written[-1]["event"] == "appeared" and w.written[-1]["zone"] == "sofa", w.written[-1]

    # sightings outside every zone, or in a zone this camera doesn't watch: ignored
    n = len(w.written)
    for _ in range(3):
        step("phone-2", keys(None), ["desk"])
        step("phone-2", keys("sofa"), ["desk"])
    assert len(w.written) == n, w.written[n:]

    # people: motion in 2 of 3 frames -> person_here; quiet for PERSON_HOLD -> person_left
    step("phone-2", [], ["desk"], {"desk": 0.2})
    assert not events(w, event="person_here")
    step("phone-2", [], ["desk"], {"desk": 0.2})
    assert events(w, event="person_here", zone="desk"), w.written[-1]
    for _ in range(int(te.PERSON_HOLD / 0.5) - 2):
        step("phone-2", [], ["desk"], {"desk": 0.0})
    assert not events(w, event="person_left"), w.written[-1]
    for _ in range(4):
        step("phone-2", [], ["desk"], {"desk": 0.0})
    pl = events(w, event="person_left")
    assert len(pl) == 1 and pl[0]["zone"] == "desk" and "item" not in pl[0], pl
    return w


def test_floor():
    # identity homography for a 1920x1080 photo, analysed at 960x540: pixels double
    cal = {"w": 1920, "h": 1080, "H": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    assert te.floor_point(cal, [100, 100, 140, 130], 960, 540) == [240.0, 260.0]
    # a metre per 100 px, offset
    cal = {"w": 960, "h": 540, "H": [0.01, 0, 1, 0, 0.01, 2, 0, 0, 1]}
    assert te.floor_point(cal, [100, 100, 140, 130], 960, 540) == [2.2, 3.3]
    assert te.floor_point(None, [0, 0, 1, 1], 960, 540) is None
    w = te.EventWriter(os.path.join(tempfile.mkdtemp(), "e.jsonl"), {"phone-1": cal}, echo=False)
    for k in range(2):
        w.frame("phone-1", 1000.0 + k, keys("sofa"), ["sofa"], None, (540, 960, 3))
    assert w.written[0]["floor"] == [2.2, 3.3], w.written


def test_calibration_file():
    import json
    p = os.path.join(tempfile.mkdtemp(), "calibration.json")
    with open(p, "w") as f:
        json.dump([{"camera": "Sofa cam", "camera_id": "sofa-cam", "image_w": 1280, "image_h": 720,
                    "image_to_floor": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
                   {"camera": "Broken", "image_w": 1, "image_h": 1, "image_to_floor": [1, 2]}], f)
    c = te.load_calibration(p)
    assert list(c) == ["sofa-cam"] and c["sofa-cam"]["w"] == 1280, c
    assert te.load_calibration(p + ".missing") == {}


def test_slug():
    assert te.slug("Coat hook") == "coat-hook"
    assert te.slug("Kitchen  counter!") == "kitchen-counter"
    assert te.zone_id({"name": "Desk", "rect": [0, 0, 1, 1]}) == "desk"


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.mkdtemp(), "events.jsonl")
    w = run(out)
    test_floor()
    test_calibration_file()
    test_slug()
    print(f"ok: {len(w.written)} events, written to {out}")
    for r in w.written:
        print("  ", r["time"], r["event"], r["zone"], r.get("item", ""), r.get("from", ""))
