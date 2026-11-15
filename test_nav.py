"""test_nav.py -- nav.py's directions on made-up camera frames (no camera needed)."""
import nav

ZONES = {"sofa": ("Sofa", [400, 100, 600, 200]), "desk": ("Desk", [50, 100, 200, 200])}
fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    fails += not ok


def walk(points, items=(), t0=1000.0):
    nav._trail.clear(); nav._heading.update(v=None, t=0.0); nav._view.clear()
    for i, (x, y) in enumerate(points):
        nav.update("top", 960, 720, [([x - 40, y - 40, x + 40, y + 40], 0.9)], list(items), ZONES, now=t0 + 0.5 * i)


import time
T = time.time()
target = {"item": "keys-1", "zone": "sofa", "state": "searching"}
# walking up the image (towards the back), sofa up and to the right
walk([(300, 600), (300, 560), (300, 520), (300, 480)], t0=T - 2)
r = nav.compute(target); check(r["state"] in ("turn_right", "bear_right"), f"sofa ahead-right -> {r['say']}")
walk([(700, 600), (700, 560), (700, 520), (700, 480)], t0=T - 2)
r = nav.compute(target); check(r["state"] in ("turn_left", "bear_left"), f"sofa ahead-left -> {r['say']}")
walk([(500, 600), (500, 560), (500, 520), (500, 480)], t0=T - 2)
r = nav.compute(target); check(r["state"] == "ahead", f"sofa straight ahead -> {r['say']}")
walk([(500, 300), (500, 340), (500, 380), (500, 420)], t0=T - 2)
r = nav.compute(target); check(r["state"] == "turn_around", f"walking away -> {r['say']}")
walk([(500, 300), (500, 250), (500, 200), (500, 160)], t0=T - 2)
r = nav.compute(target); check(r["state"] == "arrived", f"at the sofa -> {r['say']}")
walk([(300, 480)] * 4, t0=T - 2)
r = nav.compute(target); check(r["state"] == "no_heading", f"standing still, never moved -> {r['say']}")
# glasses seen on the desk: aims at the glasses' own box
walk([(500, 600), (500, 560), (500, 520), (500, 480)], items=[("glasses", [100, 120, 140, 150], 0.8)], t0=T - 2)
r = nav.compute({"item": "glasses-1", "zone": "desk", "state": "searching"})
check(r["state"] in ("turn_left", "bear_left") and r["seen"], f"glasses on the desk, left -> {r['say']}")
nav.MIRROR = True
r = nav.compute({"item": "glasses-1", "zone": "desk", "state": "searching"})
check(r["state"] in ("turn_right", "bear_right"), f"mirrored camera swaps -> {r['say']}")
nav.MIRROR = False
r = nav.compute({"item": "keys-1", "zone": "with-person", "state": "searching"})
check(r["state"] == "not_here", "pockets: no walking directions")
print("all checks passed" if not fails else f"{fails} FAILED")
