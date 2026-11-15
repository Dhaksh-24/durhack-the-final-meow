"""simulate.py -- every way to lose your keys in a pretend flat, played through Trace's reasoning.

A scripted person walks around a two-camera flat (living room + hall) and loses
the keys in one of seven ways. The script produces what multicam.py would:
noisy item and person boxes every 0.5 s per camera (items flicker, people block
the view, nothing is perfect), and feeds them to whereabouts.py. Then we ask
"where are my keys?" and check the answer against where the script put them.

    python simulate.py                 # 100 runs of each way, accuracy table
    python simulate.py --runs 300 --seed 7
    python simulate.py --show          # one story + answer per way of losing them
    python simulate.py --show --llm    # ... with Gemini writing the answers
    python simulate.py --room places.json  # your own room (needs >= 2 surfaces per camera)

The ways of losing things:
    covered    a jacket or post lands on them; nobody touched them
    moved      picked up, carried, put down somewhere out of sight, then you walked on
    two_stops  picked up, two stops, put down at one of them (we can't know which)
    knocked    knocked onto the floor by someone walking past
    pocket     picked up and carried out of the front door
    slipped    you sat on the sofa and they slid down the cushions
    put_back   picked up, wandered, put back where they were but hidden
"""
import argparse
import random
from collections import defaultdict

import room as roommod
import whereabouts as wb

W, H, DT = 960, 540, 0.5
SPEED = 260.0            # px per second, walking
P_ITEM_MISS = 0.2        # the detector misses the keys this often even in plain view
P_PERSON_MISS = 0.08
CAMS = {"living-room": "Living room", "hall": "Hall"}
ZONES = [   # the same shape multicam.py keeps in zones.json
    {"name": "sofa", "camera": "living-room", "rect": [40, 200, 400, 400]},
    {"name": "coffee table", "camera": "living-room", "rect": [420, 330, 640, 430]},
    {"name": "side table", "camera": "living-room", "rect": [700, 250, 830, 360]},
    {"name": "living room doorway", "camera": "living-room", "rect": [860, 100, 960, 430]},
    {"name": "living room floor", "camera": "living-room", "rect": [0, 430, 960, 540]},
    {"name": "hall console", "camera": "hall", "rect": [80, 220, 320, 330]},
    {"name": "coat hooks", "camera": "hall", "rect": [400, 80, 560, 260]},
    {"name": "front door", "camera": "hall", "rect": [740, 60, 920, 450]},
    {"name": "hall floor", "camera": "hall", "rect": [0, 450, 960, 540]},
]
# where you walk in and out of each camera's view: (from cam, x) -> (to cam, x)
DOORS = {"living-room": 910, "hall": 40}
EXIT = ("hall", 830)


def demo_room():
    r = roommod.Room(path=None)
    for cam, name in CAMS.items():
        r.set_camera(cam, name, W, H)
    r.set_drawn(ZONES, CAMS)
    return r


class Sim:
    def __init__(self, rng, room):
        self.rng, self.room = rng, room
        self.t = 1_800_000_000.0
        self.brain = wb.Whereabouts(room, {c: room.cam_name(c) for c in room.cameras}, state_file=None,
                                    timeline=None, use_llm=False, verbose=False)
        self.pos = None                 # (cam, x) or None when out of every camera
        self.item = None                # {"zone", "box", "visible", "carried"}
        self.surfaces = [z for z in room.zones if z["kind"] not in ("floor", "door") or z["name"] == "coat hooks"]

    # ---------------------------------------------------------------- world
    def person_box(self):
        cam, x = self.pos
        return [x - 80, 140, x + 80, 525]

    def put(self, zone, visible=True, x=None):
        r = zone["rect"]
        cx = x if x is not None else (r[0] + r[2]) / 2 + self.rng.uniform(-0.25, 0.25) * (r[2] - r[0])
        cy = (r[1] + r[3]) / 2 + self.rng.uniform(-0.2, 0.2) * (r[3] - r[1])
        self.item = {"zone": zone["id"], "cam": zone["camera"], "visible": visible, "carried": False,
                     "box": [cx - 18, cy - 11, cx + 18, cy + 11]}

    def step(self):
        b = self.brain
        for cam in self.room.cameras:
            people, items = [], []
            here = self.pos is not None and self.pos[0] == cam
            if here and self.rng.random() > P_PERSON_MISS:
                people = [(self.person_box(), 0.85)]
            it = self.item
            if it and it["cam"] == cam and it["visible"] and not it["carried"]:
                ib = it["box"]
                cx, cy = (ib[0] + ib[2]) / 2, (ib[1] + ib[3]) / 2
                pb = self.person_box() if here else None
                blocked = pb and pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3] and self.rng.random() < 0.85
                if not blocked and self.rng.random() > P_ITEM_MISS:
                    items = [(1, "keys", "Keys #1", ib)]
            b.frame(cam, self.t, (H, W), people, items)
        b.tick(self.t)
        self.t += DT

    def wait(self, secs):
        for _ in range(max(1, int(secs / DT))):
            self.step()

    def _glide(self, x):
        cam, x0 = self.pos
        n = max(1, int(abs(x - x0) / (SPEED * DT)))
        for i in range(1, n + 1):
            self.pos = (cam, x0 + (x - x0) * i / n)
            self.step()

    def walk_to(self, zone):
        cam, x = zone["camera"], (zone["rect"][0] + zone["rect"][2]) / 2
        if self.pos is None:
            self.pos = (cam, DOORS[cam])
        if self.pos[0] != cam:                       # through the doorway into the other room
            self._glide(DOORS[self.pos[0]])
            self.pos = None
            self.wait(1.5)
            self.pos = (cam, DOORS[cam])
        self._glide(x)

    def leave(self):
        if self.pos and self.pos[0] != EXIT[0]:
            self.walk_to(self.room.by_id[f"{EXIT[0]}/front-door"])
        if self.pos:
            self._glide(EXIT[1])
        self.pos = None

    def pick(self, exclude=(), cam=None):
        opts = [z for z in self.surfaces if z["id"] not in exclude and (cam is None or z["camera"] == cam)]
        return self.rng.choice(opts)

    def pick_up(self):
        self.item["carried"] = True

    def drop_hidden(self, zone):
        self.put(zone, visible=False, x=self.pos[1] if self.pos else None)

    def ask(self):
        self.wait(self.rng.uniform(5, 90))
        return self.brain.predict(1, self.t)


# ======================================================== the ways to lose them

def covered(s):
    a = s.pick()
    s.put(a)
    s.wait(s.rng.uniform(10, 30))
    s.walk_to(s.pick(exclude=[a["id"]]))
    s.wait(s.rng.uniform(2, 8))
    s.item["visible"] = False                         # the jacket lands on them
    for _ in range(s.rng.randint(0, 2)):
        s.walk_to(s.pick(exclude=[a["id"]]))
        s.wait(s.rng.uniform(2, 10))
    return {a["id"]}


def moved(s):
    a = s.pick()
    s.put(a)
    s.wait(s.rng.uniform(10, 30))
    s.walk_to(a)
    s.wait(s.rng.uniform(1.5, 3.5))
    s.pick_up()
    b = s.pick(exclude=[a["id"]])
    s.walk_to(b)
    s.wait(s.rng.uniform(3, 10))
    s.drop_hidden(b)
    s.walk_to(s.pick(exclude=[b["id"]]))
    s.wait(s.rng.uniform(2, 6))
    if s.rng.random() < 0.5:
        s.leave()
    return {b["id"]}


def two_stops(s):
    a = s.pick()
    s.put(a)
    s.wait(s.rng.uniform(10, 30))
    s.walk_to(a)
    s.wait(s.rng.uniform(1.5, 3.5))
    s.pick_up()
    b = s.pick(exclude=[a["id"]])
    c = s.pick(exclude=[a["id"], b["id"]])
    drop = s.rng.choice([b, c])
    for z in (b, c):
        s.walk_to(z)
        s.wait(s.rng.uniform(3, 10))
        if z is drop:
            s.drop_hidden(z)
    return {drop["id"]}


def knocked(s):
    a = s.pick(cam="living-room")
    a = s.room.by_id[a["id"]]
    floor = s.room.floor_below(a)
    s.put(a)
    s.wait(s.rng.uniform(10, 30))
    r = a["rect"]
    left, right = max(60, r[0] - 120), min(850, r[2] + 120)
    start, end = (left, right) if s.rng.random() < 0.5 else (right, left)
    s.walk_to(s.room.by_id["living-room/living-room-doorway"])
    s._glide(start)
    cam = s.pos[0]
    ix = (s.item["box"][0] + s.item["box"][2]) / 2
    n = max(1, int(abs(end - start) / (SPEED * DT)))
    knocked_yet = False
    for i in range(1, n + 1):
        s.pos = (cam, start + (end - start) * i / n)
        if not knocked_yet and abs(s.pos[1] - ix) < SPEED * DT:
            s.put(floor, visible=False, x=ix)        # off the edge, under the sofa / table
            knocked_yet = True
        s.step()
    s.walk_to(s.pick(exclude=[a["id"]]))
    s.wait(s.rng.uniform(2, 8))
    return {floor["id"]}


def pocket(s):
    a = s.pick()
    s.put(a)
    s.wait(s.rng.uniform(10, 30))
    s.walk_to(a)
    s.wait(s.rng.uniform(1.5, 3.5))
    s.pick_up()
    s.leave()
    return {wb.WITH_YOU, wb.LEFT_VIEW}


def slipped(s):
    sofa = s.room.by_id["living-room/sofa"]
    s.put(sofa)
    s.wait(s.rng.uniform(10, 30))
    s.walk_to(sofa)
    s.wait(s.rng.uniform(4, 8))
    s.item["visible"] = False
    s.wait(s.rng.uniform(15, 60))                     # still sitting
    s.walk_to(s.pick(exclude=[sofa["id"]]))
    s.wait(s.rng.uniform(2, 8))
    return {sofa["id"]}


def put_back(s):
    a = s.pick()
    s.put(a)
    s.wait(s.rng.uniform(10, 30))
    s.walk_to(a)
    s.wait(s.rng.uniform(1.5, 3.5))
    s.pick_up()
    s.walk_to(s.pick(exclude=[a["id"]]))
    s.wait(s.rng.uniform(1, 4))
    s.walk_to(a)
    s.wait(s.rng.uniform(3, 8))
    s.drop_hidden(a)
    s.walk_to(s.pick(exclude=[a["id"]]))
    s.wait(s.rng.uniform(1, 4))
    return {a["id"]}


WAYS = {"covered": covered, "moved": moved, "two_stops": two_stops, "knocked": knocked,
        "pocket": pocket, "slipped": slipped, "put_back": put_back}


def run_one(way, rng, room_factory):
    s = Sim(rng, room_factory())
    truth = WAYS[way](s)
    pred = s.ask()
    order = [c["zone"] for c in pred.get("candidates", [])]
    rank = next((i for i, z in enumerate(order) if z in truth), None)
    return s, truth, pred, rank


def main():
    ap = argparse.ArgumentParser(description="Simulate ways of losing keys and score Trace's guesses")
    ap.add_argument("--runs", type=int, default=100, help="runs per way of losing them")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", action="store_true", help="print one story and answer per way")
    ap.add_argument("--llm", action="store_true", help="with --show: ask Gemini for the answers")
    ap.add_argument("--room", help="places.json (room.py) to use instead of the built-in flat")
    args = ap.parse_args()
    if args.room:
        # surfaces are taken from the room; knocked/slipped need a living-room sofa and floor
        factory = lambda: roommod.Room(args.room)
    else:
        factory = demo_room

    if args.show:
        rng = random.Random(args.seed)
        for way in WAYS:
            s, truth, pred, rank = run_one(way, rng, factory)
            s.brain.use_llm = args.llm
            print(f"\n=== {way}: really at {', '.join(s.brain._place(t) for t in truth)}")
            for line in pred.get("story", []):
                print("   ", line)
            for c in pred.get("candidates", []):
                print(f"    {c['chance']:.0%}  {c['place']:<22} {c['evidence'][0] if c['evidence'] else ''}")
            ans = s.brain.ask("where are my keys?", now=s.t)
            print(f"  Trace ({ans.get('by')}): {ans['reply']}")
        return

    rng = random.Random(args.seed)
    stats = defaultdict(lambda: [0, 0, 0])     # way -> [n, top1, top3]
    for way in WAYS:
        for _ in range(args.runs):
            _, _, _, rank = run_one(way, rng, factory)
            st = stats[way]
            st[0] += 1
            st[1] += rank == 0
            st[2] += rank is not None and rank < 3
    print(f"\n{'way of losing them':<20}{'runs':>6}{'top-1':>9}{'top-3':>9}")
    tot = [0, 0, 0]
    for way, (n, t1, t3) in stats.items():
        print(f"{way:<20}{n:>6}{t1 / n:>9.0%}{t3 / n:>9.0%}")
        tot = [tot[0] + n, tot[1] + t1, tot[2] + t3]
    print(f"{'all':<20}{tot[0]:>6}{tot[1] / tot[0]:>9.0%}{tot[2] / tot[0]:>9.0%}")
    print("\ntop-1 = Trace's first suggestion was right; top-3 = it was in the first three.")
    print("Chance level is about 1 in", len(factory().zones), "places.")


if __name__ == "__main__":
    main()
