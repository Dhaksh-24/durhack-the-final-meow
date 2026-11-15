"""Quick tests of events_adapter.py's hysteresis: python3 test_events_adapter.py"""
import os
import tempfile

import events_adapter as ea

ok = True


def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and bool(cond)


def adapter():
    ad = ea.Adapter(os.path.join(tempfile.mkdtemp(), "ev.jsonl"), echo=False)
    ad.out = []
    ad.emit = lambda ev: ad.out.append(ev)
    return ad


def det(item, zone, conf, cam="a"):
    return {"camera_id": cam, "item": item, "det_conf": conf, "support": {"anchor": zone},
            "image_w": 1000, "image_h": 1000, "box": [100, 100, 150, 150]}


def evs(ad, kind=None):
    return [(e["event"], e["zone"], e["camera"]) for e in ad.out if kind is None or e["event"] == kind]


print("events_adapter hysteresis")
ad = adapter()   # two cameras on one zone: each keeps its own track
for i in range(6):
    ad.frame("a", i, [det("keys", "sofa", 0.8, "a")])
    ad.frame("b", i + 0.1, [det("keys", "sofa", 0.7, "b")])
for i in range(6, 12):
    ad.frame("a", i, [det("keys", "sofa", 0.8, "a")])
    ad.frame("b", i + 0.1, [])
check(evs(ad) == [("appeared", "sofa", "a"), ("appeared", "sofa", "b"), ("disappeared", "sofa", "b")],
      "two cameras on one zone: one losing sight says so for its camera only " + str(evs(ad)))

ad = adapter()   # the item in two zones in the same frame: the most confident box counts
for i in range(10):
    ad.frame("a", i, [det("keys", "desk", 0.8), det("keys", "shelf", 0.3)])
check(evs(ad) == [("appeared", "desk", "a")], "keys on the desk and a weaker false box on the shelf: one appeared, on the desk")

ad = adapter()   # a stronger false box every other frame doesn't make it move
for i in range(12):
    ad.frame("a", i, [det("keys", "desk", 0.6)] + ([det("keys", "shelf", 0.9)] if i % 2 else []))
check(evs(ad) == [("appeared", "desk", "a")], "a stronger false box in every other frame isn't 2 of 3: no move " + str(evs(ad)))

ad = adapter()   # the detector misses the real keys every other frame and sees false ones elsewhere then
for i in range(12):
    ad.frame("a", i, [det("keys", "shelf", 0.9)] if i % 2 else [det("keys", "desk", 0.6)])
check(evs(ad) == [("appeared", "desk", "a")], "real and false boxes in alternate frames: no flip-flopping " + str(evs(ad)))

ad = adapter()   # a real move without vanishing: desk, desk, sofa, sofa
for i in range(8):
    ad.frame("a", i, [det("keys", "desk" if i < 4 else "sofa", 0.8)])
check(evs(ad) == [("appeared", "desk", "a"), ("moved", "sofa", "a")], "a real move is still reported after 2 frames")

ad = adapter()   # one-frame blips and a huge box are ignored
for i in range(10):
    ad.frame("a", i, [det("keys", "shelf", 0.4)] if i == 3 else [])
ad.frame("a", 11, [dict(det("keys", "desk", 0.9), box=[0, 0, 900, 900])])
ad.frame("a", 12, [dict(det("keys", "desk", 0.9), box=[0, 0, 900, 900])])
check(evs(ad) == [] and ad.count["dropped_big_box"] == 2, "a one-frame blip and boxes over 15% of the frame make no events")

ad = adapter()   # moved: disappears from the desk, appears on the sofa within 60 s
for i in range(4):
    ad.frame("a", i, [det("keys", "desk", 0.8)])
for i in range(4, 12):
    ad.frame("a", i, [])
for i in range(12, 15):
    ad.frame("a", i, [det("keys", "sofa", 0.8)])
check([e[0] for e in evs(ad)] == ["appeared", "disappeared", "moved"] and evs(ad)[-1][1] == "sofa", "desk → gone → sofa within 60 s is 'moved'")

ad = adapter()   # silence: a camera that prints nothing still had frames
now = [1000.0]
ea.time.time = lambda: now[0]
for i in range(4):
    ad.frame("a", 1000 + i * 0.5, [det("keys", "desk", 0.8)])
    now[0] = 1000 + i * 0.5
for step in range(20):
    now[0] += 0.5
    ad.silence()
check(evs(ad) == [("appeared", "desk", "a"), ("disappeared", "desk", "a")], "silence counts as missed frames: disappeared after ~5 " + str(evs(ad)))
d = [e for e in ad.out if e["event"] == "disappeared"][0]
check(1001.5 < d["t"] < 1005, "the disappeared time is the frame's time, not the wall clock (t %.1f)" % d["t"])
check(ea.Adapter.__init__.__defaults__ is not None and "trace-events.jsonl" in open(ea.__file__).read(), "default output is trace-events.jsonl")

print("all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
