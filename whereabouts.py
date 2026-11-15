"""whereabouts.py -- Trace's reasoning: turns raw sightings into "where did my keys go?"

multicam.py tells it, for every camera frame it analyses, which remembered items
it saw and where the people were. From that it keeps four things:

  items      each item's last place (a named place from room.py), and since when
  visits     where the person went: one entry per stop at a place, with how long
  episodes   an item going missing: where, when, and HOW -- did a person reach for
             that spot as it vanished (picked up / knocked), or did it vanish with
             nobody near (covered, or the camera just lost it)?
  heat       a per-camera heatmap of where people's feet have been

Asked about a missing item, predict() links the episode to the person's trail
after it vanished and scores every place it could be:

  vanished, nobody near           -> still there, covered; or knocked to the floor
                                     if someone brushed past later
  vanished as you reached for it  -> where you STOPPED afterwards (people put
                                     things down where they pause), weighted by
                                     how long you stopped and how good a place it
                                     is to put things; or still on you if you
                                     didn't stop, or left the cameras' view
  vanished and you stayed put     -> it slipped down the cushions / is under you
  plus habits                     -> where this kind of thing turned up before

ask() then hands that evidence (never pixels, never images) to Gemini, which
links the story together, picks the order and says it in one or two spoken
sentences. With no key, no network or a slow reply, a written template answers
instead, from the same scores, so the demo never stalls.

One person at a time: the trail is "whoever the cameras saw". Good enough for a
flat with one person who needs help, which is who this is for.

Try it without cameras:  python tests/simulate.py
"""
import json
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

import room as roommod

# ---- tuning ----
GONE_SECS = 4.0          # unseen on every camera this long = missing (keys flicker in and out)
TOUCH_BEFORE = 3.0       # look for a person this long before ...
TOUCH_AFTER = 2.0        # ... and after the last sighting
TOUCH_FRAC = 0.25        # a person box covering this much of the (padded) item box = reached it
BRUSH_SECS = 0.9         # ... but if they stood still there for less than this, they only brushed past
STILL_SPEED = 60.0       # px/s; a person box moving slower than this is standing (detector jitter included)
VISIT_GAP = 2.5          # a person unseen at a place this long has left it
DWELL_MIN = 1.5          # a stop shorter than this is walking past
STAY_MIN = 6.0           # stayed at the spot this long after it vanished = sat down, not picked up
STILL_FULL = 20.0        # ... and this long = it's there, under you / down the cushions
OUT_OF_VIEW = 15.0       # nobody on any camera this long = you left the room
SWITCH_SECS = 1.5        # an item seen by two cameras keeps its place on the first for this long
PERSON_CONF = 0.35
HEAT_CELL = 16           # px per heatmap cell
HEAT_HALF_LIFE = 1800.0  # s; old footsteps fade so the map shows today's habits
HISTORY_SECS = 3 * 3600
PEOPLE_SECS = 600
TOP_N = 4
LLM_TIMEOUT = 9.0        # s we wait for Gemini; trace.html gives /api/ask 15 s in total
LLM_DEADLINE_MS = 12000  # Gemini refuses deadlines under 10 s; we stop waiting at LLM_TIMEOUT anyway
STATE_FILE = Path("memory") / "whereabouts.json"
TIMELINE_FILE = Path("timeline.jsonl")

WITH_YOU, LEFT_VIEW = "_with_you", "_left_view"


def clock(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def ago(secs):
    s = int(max(0, secs))
    if s < 10:
        return "just now"
    if s < 60:
        return f"{s} seconds ago"
    if s < 120:
        return "a minute ago"
    if s < 3600:
        return f"{s // 60} minutes ago"
    h = s // 3600
    return f"{h} hour{'s' if h > 1 else ''} ago"


def secs_words(s):
    s = int(round(s))
    return f"{s} seconds" if s < 90 else f"{round(s / 60)} minutes"


def pad(box, f=0.5, px=20):
    w, h = box[2] - box[0], box[3] - box[1]
    return [box[0] - w * f - px, box[1] - h * f - px, box[2] + w * f + px, box[3] + h * f + px]


def covered(item_box, person_box):
    """How much of the item's surroundings the person's box covers (0..1)."""
    r = pad(item_box)
    return roommod.inter(r, person_box) / max(1.0, roommod.area(r))


class Whereabouts:
    def __init__(self, room=None, cam_names=None, state_file=STATE_FILE, timeline=TIMELINE_FILE,
                 use_llm=None, verbose=True):
        self.room = room if room is not None else roommod.Room()
        self.cam_names = dict(cam_names or {})
        self.state_file = Path(state_file) if state_file else None
        self.timeline = Path(timeline) if timeline else None
        self.use_llm = (os.environ.get("TRACE_LLM", "1") != "0") if use_llm is None else use_llm
        self.verbose = verbose
        self.lock = threading.RLock()
        self.items = {}            # oid -> {"label", "category", "cam", "box", "zone", "since", "first", "last", "lost"}
        self.people = deque()      # (t, cam, box)
        self.person_seen = {}      # cam -> last time a person was seen there
        self.last_people = {}      # cam -> [boxes] from the latest frame (for drawing)
        self.visits = []           # {"cam", "zone", "start", "end"}
        self.open_visits = {}      # (cam, zone) -> visit
        self.episodes = {}         # oid -> open episode
        self.closed = deque(maxlen=50)
        self.heat = {}             # cam -> np.float32 grid
        self.heat_t = {}
        self.habits = {"rest": {}, "found": {}, "asked": {"total": 0, "top1": 0, "top3": 0}}
        self.recent_log = deque(maxlen=200)
        self.left_logged = True
        self._tick_t = 0.0
        self._llm = None
        self.load()

    # ================================================================ input
    def frame(self, cam, now, shape, people=(), items=()):
        """One analysed camera frame.
        people: [(box, conf)]   items: [(oid, category, label, box)] -- boxes in this frame's pixels."""
        with self.lock:
            h, w = shape[:2]
            if self.room.set_camera(cam, self.cam_names.get(cam), w, h):
                self.room.save()
            boxes = [list(map(float, b)) for b, c in people if c >= PERSON_CONF]
            self.last_people[cam] = (now, boxes)
            for b in boxes:
                self.people.append((now, cam, b))
                self.person_seen[cam] = now
                self._add_heat(cam, w, h, b, now)
                # the floor is where you walk, not where you stop to put things down
                for z in [z for z in self.room.zones_touched(cam, b) if z["kind"] != "floor"]:
                    key = (cam, z["id"])
                    v = self.open_visits.get(key)
                    if v and now - v["end"] <= VISIT_GAP:
                        v["end"] = now
                    else:
                        v = {"cam": cam, "zone": z["id"], "start": now, "end": now}
                        self.visits.append(v)
                        self.open_visits[key] = v
            if boxes and self.left_logged:
                self.left_logged = False
            for oid, category, label, box in items:
                self._item_seen(oid, category, label, cam, list(map(float, box)), now)

    def _item_seen(self, oid, category, label, cam, box, now):
        z = self.room.zone_at(cam, box)
        zid = z["id"] if z else None
        rec = self.items.get(oid)
        if rec is None:
            self.items[oid] = {"label": label, "category": category, "cam": cam, "box": box,
                               "zone": zid, "since": now, "first": now, "last": now, "lost": False}
            self._log(now, "seen", label, zid, f"{label} first seen on the {self._place(zid)}")
            return
        rec["label"], rec["category"] = label, category
        if rec["lost"]:
            self._resolve(oid, rec, cam, box, zid, now)
        elif cam != rec["cam"] and now - rec["last"] < SWITCH_SECS:
            rec["last"] = now           # the other camera still has it; keep that place
            return
        elif zid != rec["zone"]:
            self._log(now, "moved", label, zid,
                      f"{label} moved from the {self._place(rec['zone'])} to the {self._place(zid)}")
            rec["since"] = now
        else:
            rest = self.habits["rest"].setdefault(category, {})
            rest[zid] = rest.get(zid, 0.0) + min(2.0, now - rec["last"])
        rec.update(cam=cam, box=box, zone=zid, last=now, lost=False)

    def forget(self, oid):
        with self.lock:
            self.items.pop(oid, None)
            self.episodes.pop(oid, None)

    def tick(self, now):
        """Call often (every loop). Closes stops, notices items going missing."""
        with self.lock:
            if now - self._tick_t < 0.25:
                return
            self._tick_t = now
            for key, v in list(self.open_visits.items()):
                if now - v["end"] > VISIT_GAP:
                    del self.open_visits[key]
                    if v["end"] - v["start"] >= DWELL_MIN and self._watching(v["start"]):
                        self._log(v["end"], "stop", None, v["zone"],
                                  f"person stopped at the {self._place(v['zone'])} for {secs_words(v['end'] - v['start'])}")
            for oid, rec in self.items.items():
                if not rec["lost"] and now - rec["last"] > GONE_SECS:
                    self._open_episode(oid, rec, now)
            last = max(self.person_seen.values(), default=None)
            if last and not self.left_logged and now - last > OUT_OF_VIEW:
                self.left_logged = True
                self._log(last, "left", None, None, "person left the cameras' view")
            while self.people and now - self.people[0][0] > PEOPLE_SECS:
                self.people.popleft()
            keep = min([e["lost_at"] for e in self.episodes.values()] + [now - HISTORY_SECS])
            if self.visits and self.visits[0]["end"] < keep:
                self.visits = [v for v in self.visits if v["end"] >= keep]

    def _watching(self, t):
        return any(t - e["lost_at"] > -TOUCH_BEFORE for e in self.episodes.values())

    # ============================================================ episodes
    def _open_episode(self, oid, rec, now):
        t0 = rec["last"]
        touch, near, samples = 0.0, False, []
        for t, cam, b in self.people:
            if cam == rec["cam"] and t0 - TOUCH_BEFORE <= t <= t0 + TOUCH_AFTER + 1:
                near = True
                c = covered(rec["box"], b)
                touch = max(touch, c)
                samples.append((t, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, c >= TOUCH_FRAC))
        # reaching for something means standing still at it; brushing past it doesn't.
        # Only back-to-back sightings count, so walking past twice isn't "standing".
        still = 0.0
        for (ta, xa, ya, ra), (tb, xb, yb, rb) in zip(samples, samples[1:]):
            dt = tb - ta
            if ra and rb and 0 < dt <= 1.5 and math.hypot(xb - xa, yb - ya) / dt < STILL_SPEED:
                still += dt
        cause = "unseen" if touch < TOUCH_FRAC else "handled" if still >= BRUSH_SECS else "brushed"
        rec["lost"] = True
        ep = {"oid": oid, "label": rec["label"], "category": rec["category"], "cam": rec["cam"],
              "box": rec["box"], "zone": rec["zone"], "since": rec["since"], "lost_at": t0,
              "cause": cause, "touch": round(touch, 2), "person_near": near, "asked": None}
        self.episodes[oid] = ep
        how = ("as someone reached for that spot" if cause == "handled" else
               "as someone walked past it" if cause == "brushed" else
               "with somebody nearby" if near else "with nobody near it")
        self._log(t0, "vanished", rec["label"], rec["zone"],
                  f"{rec['label']} vanished from the {self._place(rec['zone'])} {how}")

    def _resolve(self, oid, rec, cam, box, zid, now):
        ep = self.episodes.pop(oid, None)
        rec["since"] = now
        if not ep:
            return
        ep.update(found_at=now, found_zone=zid)
        self.closed.append(ep)
        same = zid == ep["zone"]
        self._log(now, "found", rec["label"], zid,
                  f"{rec['label']} seen again on the {self._place(zid)}"
                  + (" (same place)" if same else f", {secs_words(now - ep['lost_at'])} after vanishing "
                     f"from the {self._place(ep['zone'])}"))
        found = self.habits["found"].setdefault(ep["category"], {})
        found[zid] = found.get(zid, 0) + 1
        if ep.get("asked"):
            a = self.habits["asked"]
            a["total"] += 1
            a["top1"] += int(ep["asked"][:1] == [zid])
            a["top3"] += int(zid in ep["asked"][:3])

    # ============================================================ predict
    def find_item(self, query):
        """Item id for words like 'my keys' or "Dad's glasses" (most recently seen wins)."""
        try:
            from targets import TARGETS
        except Exception:
            TARGETS = {}
        text = " " + re.sub(r"[^a-z0-9 ]+", " ", str(query).lower().replace("'", "")) + " "
        cats = {c for c, t in TARGETS.items() for a in t["aliases"] + [c] if f" {a} " in text}
        with self.lock:
            pool = list(self.items.items())
        named = [(oid, r) for oid, r in pool
                 if any(len(w) > 2 and f" {w} " in text for w in re.findall(r"[a-z0-9]+", r["label"].lower())
                        if w not in (r["category"], "the", "my"))]
        hits = named or [(oid, r) for oid, r in pool if r["category"] in cats or f" {r['category']} " in text]
        if not hits:
            return None
        return max(hits, key=lambda x: x[1]["last"])[0]

    def predict(self, oid, now=None):
        now = time.time() if now is None else now
        with self.lock:
            rec = self.items.get(oid)
            if rec is None:
                return {"status": "unknown"}
            base = {"item_id": oid, "item": rec["label"], "category": rec["category"],
                    "last_seen": {"place": self._place(rec["zone"]), "zone": rec["zone"],
                                  "camera": self._cam(rec["cam"]), "at": clock(rec["last"]),
                                  "ago": ago(now - rec["last"])}}
            if not rec["lost"] and now - rec["last"] <= GONE_SECS:
                return {**base, "status": "in_view", "candidates": [
                    self._cand(rec["zone"], 1.0, ["in view of the camera right now"])]}
            ep = self.episodes.get(oid)
            if ep is None:                     # out of sight, but tick() hasn't run yet
                self._open_episode(oid, rec, now)
                ep = self.episodes[oid]
            scores, why, story = self._score(ep, now)
            total = sum(scores.values()) or 1.0
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:TOP_N]
            return {**base, "status": "missing", "cause": ep["cause"],
                    "vanished": {"place": self._place(ep["zone"]), "at": clock(ep["lost_at"]),
                                 "ago": ago(now - ep["lost_at"])},
                    "story": story,
                    "candidates": [self._cand(z, s / total, why[z]) for z, s in ranked]}

    def _score(self, ep, now):
        R = self.room
        z0 = ep["zone"]
        zone0 = R.by_id.get(z0)
        t0 = ep["lost_at"]
        S, W = defaultdict(float), defaultdict(dict)

        def add(z, s, reason):
            if z is None or s <= 0:
                return
            S[z] += s
            W[z][reason] = W[z].get(reason, 0.0) + s

        # the person's stops after it vanished, in order
        after = sorted([v for v in self.visits if v["end"] >= t0], key=lambda v: v["start"])
        stay = [v for v in after if v["zone"] == z0 and v["start"] <= t0 + TOUCH_AFTER]
        still = max([v["end"] - t0 for v in stay], default=0.0)
        trail = [v for v in after if v not in stay and v["start"] >= t0 - 0.5]
        stops = [v for v in trail if v["end"] - v["start"] >= DWELL_MIN]
        last_person = max(self.person_seen.values(), default=0.0)
        left = last_person > t0 and now - last_person > OUT_OF_VIEW
        # where you're standing now: you'd likely have spotted it, so it counts for less,
        # and the "last place you stopped" bonus goes to the last stop you've finished
        ongoing = [v for v in stops if now - v["end"] <= VISIT_GAP]
        last_stop = max([v for v in stops if v not in ongoing], key=lambda v: v["end"], default=None)

        if ep["cause"] == "handled":
            if still >= STAY_MIN:
                add(z0, 0.15 + 1.35 * min(1.0, still / STILL_FULL),
                    f"you stayed at the {self._place(z0)} for {secs_words(still)} after it vanished, "
                    "so it may have slipped under something there")
            else:
                add(z0, 0.08, "it was last seen here")
            for v in stops:
                zone = R.by_id.get(v["zone"])
                d = v["end"] - v["start"]
                s = (zone["hold"] if zone else 0.4) * min(1.0, d / 8.0) ** 0.7
                if v is last_stop:
                    s *= 1.3
                if v in ongoing:
                    s *= 0.6
                add(v["zone"], s, f"you stopped here for {secs_words(d)} after picking it up")
            for v in trail:
                if v not in stops:
                    zone = R.by_id.get(v["zone"])
                    add(v["zone"], 0.05 * (zone["hold"] if zone else 0.4), "you walked past here with it")
            if left:
                add(LEFT_VIEW, 0.55, "you left the cameras' view afterwards, maybe still holding it")
                add(WITH_YOU, 0.3, "you may still be carrying it")
            else:
                add(WITH_YOU, 0.6 if not stops else 0.22,
                    "you didn't stop anywhere after picking it up" if not stops else
                    "you may still be carrying it")
            if zone0:
                fb = R.floor_below(zone0)
                if fb and fb["id"] != z0:
                    add(fb["id"], 0.06, "it could have dropped while being picked up")
        elif ep["cause"] == "brushed":
            fb = R.floor_below(zone0) if zone0 else None
            if fb and fb["id"] != z0:
                add(fb["id"], 0.7, "it vanished as someone walked past without stopping, so it was "
                                   "probably knocked off onto the floor")
            add(z0, 0.35, "it may only have been pushed along or covered where it was")
            for n in (zone0["near"] if zone0 else []):
                nz = R.by_id.get(n)
                if nz and nz["kind"] != "floor":
                    add(n, 0.06, f"right next to the {self._place(z0)}")
            add(WITH_YOU, 0.1, "it could have caught on your clothes or bag")
            for v in stops:
                zone = R.by_id.get(v["zone"])
                add(v["zone"], 0.1 * (zone["hold"] if zone else 0.4) * min(1.0, (v["end"] - v["start"]) / 8.0),
                    "you stopped here afterwards")
        else:
            add(z0, 1.0, "it vanished with nobody touching it, so it is most likely still there, covered")
            passes = [v for v in after if v["zone"] == z0 or (zone0 and v["zone"] in zone0["near"])]
            if zone0:
                fb = R.floor_below(zone0)
                if fb and fb["id"] != z0:
                    add(fb["id"], 0.12 + 0.2 * min(3, len(passes)),
                        f"someone went past {len(passes)} time(s) since, so it may have been knocked off"
                        if passes else "small things often fall off onto the floor")
                for n in zone0["near"]:
                    nz = R.by_id.get(n)
                    if nz and nz["kind"] != "floor":
                        add(n, 0.04, f"right next to the {self._place(z0)}")
            for v in stops:              # in case the camera missed the hand reaching in
                zone = R.by_id.get(v["zone"])
                add(v["zone"], 0.12 * (zone["hold"] if zone else 0.4) * min(1.0, (v["end"] - v["start"]) / 8.0),
                    "a person stopped here after it vanished")

        found = self.habits["found"].get(ep["category"], {})
        tot = sum(n for z, n in found.items() if z in R.by_id)
        for z, n in found.items():
            if z in R.by_id and tot:
                add(z, 0.25 * n / tot, f"your {ep['category']} turned up here {n} time(s) before")

        story = self._story(ep, after, stay, still, stops, trail, left, last_person, now)
        why = {z: [r for r, _ in sorted(rs.items(), key=lambda kv: -kv[1])] for z, rs in W.items()}
        return S, why, story

    def _story(self, ep, after, stay, still, stops, trail, left, last_person, now):
        p0 = self._place(ep["zone"])
        out = [f"{clock(ep['since'])}: {ep['label']} on the {p0} ({self._cam(ep['cam'])} camera)"]
        how = {"handled": "as a person reached for that spot",
               "brushed": "as a person walked past it without stopping",
               "unseen": "with somebody nearby but not touching it" if ep["person_near"] else "with nobody near it"}
        out.append(f"{clock(ep['lost_at'])} ({ago(now - ep['lost_at'])}): it vanished {how[ep['cause']]}")
        if still >= STAY_MIN:
            out.append(f"the person stayed at the {p0} for {secs_words(still)} after that")
        steps = trail if len(trail) <= 8 else stops      # long walks: keep only the stops
        for v in steps[:10]:
            d = v["end"] - v["start"]
            verb = f"stopped at the {self._place(v['zone'])} for {secs_words(d)}" if v in stops \
                else f"walked past the {self._place(v['zone'])}"
            now_there = " (still there)" if v in stops and now - v["end"] <= VISIT_GAP else ""
            out.append(f"{clock(v['start'])}: then {verb}{now_there}")
        if left:
            out.append(f"{clock(last_person)}: then left the cameras' view, nobody seen since")
        elif not after:
            out.append("no one has been seen since")
        return out

    def _cand(self, z, p, why):
        zone = self.room.by_id.get(z)
        if z == WITH_YOU:
            place, cam, look = "with you", None, ["in your pockets", "in your hand or bag"]
        elif z == LEFT_VIEW:
            place, cam = "out of the cameras' view", None
            look = ["in the room you went to next", "in your coat or bag"]
        else:
            place = zone["name"] if zone else str(z)
            cam = self._cam(zone["camera"]) if zone else None
            look = zone["hiding"][:3] if zone else []
        return {"zone": z, "place": place, "camera": cam, "chance": round(p, 2),
                "evidence": why[:3], "look": look}

    # ================================================================ ask
    def ask(self, message, history=None, now=None):
        """The /api/ask answer: {"reply", "item_id", "state", "prediction"}. An empty reply
        means 'not a question about a tracked item' (trace.html then answers itself)."""
        now = time.time() if now is None else now
        oid = self.find_item(message)
        if oid is None:
            return {"reply": "", "item_id": None, "state": "idle"}
        pred = self.predict(oid, now)
        if pred["status"] == "in_view":
            ls = pred["last_seen"]
            reply = (f"I can see your {self._noun(pred)} right now on the {ls['camera']} camera, "
                     f"on the {ls['place']}.")
            return {"reply": reply, "item_id": oid, "state": "found", "prediction": pred}
        out = None
        if self.use_llm:
            out = self._ask_llm(message, history or [], pred, now)
        if out is None:
            out = {"reply": self.template(pred), "order": [c["zone"] for c in pred["candidates"]], "by": "rules"}
        with self.lock:
            ep = self.episodes.get(oid)
            if ep is not None:
                ep["asked"] = out["order"]
        top = out["order"][0] if out["order"] else None
        state = "found" if top == pred["last_seen"]["zone"] else "idle"
        self._log(now, "asked", pred["item"], top, f"asked about {pred['item']}: {out['reply']}")
        return {"reply": out["reply"], "item_id": oid, "state": state, "by": out["by"], "prediction": pred}

    def _noun(self, pred):
        return pred["category"] if pred["item"].lower().startswith(pred["category"]) else pred["item"]

    def template(self, pred):
        c = pred["candidates"]
        if not c:
            return f"I haven't got a good idea where your {self._noun(pred)} went, sorry."
        noun, best = self._noun(pred), c[0]
        plural = noun.endswith("s")
        them, they_are, were = ("them", "they're", "were") if plural else ("it", "it's", "was")
        v, last = pred["vanished"], pred["last_seen"]
        spot = f", especially {best['look'][0]}" if best["look"] else ""
        if best["zone"] == last["zone"] and pred["cause"] == "unseen":
            first = (f"Your {noun} {were} on the {best['place']} {v['ago']} and nobody touched {them}, "
                     f"so {they_are} probably still there{spot}.")
        elif best["zone"] == last["zone"] and best["evidence"] and best["evidence"][0].startswith("you stopped here"):
            first = (f"You picked up your {noun} from the {best['place']} {v['ago']}, but then came back "
                     f"and stopped there again, so check there first{spot}.")
        elif best["zone"] == last["zone"]:
            first = (f"Your {noun} went out of sight on the {best['place']} {v['ago']} while you were "
                     f"there, so check there first{spot}.")
        elif best["zone"] == WITH_YOU:
            first = (f"You picked up your {noun} {v['ago']} and I didn't see you put {them} down. "
                     "Check your pockets or your hand.")
        elif best["zone"] == LEFT_VIEW:
            first = (f"You picked up your {noun} {v['ago']} and then left the cameras' view, so "
                     f"{they_are} probably in the next room, or in a pocket.")
        elif pred["cause"] == "brushed":
            first = (f"Your {noun} {were} on the {last['place']} until someone walked past {v['ago']}. "
                     f"{they_are.capitalize()} probably knocked off, so check the {best['place']}{spot}.")
        elif pred["cause"] == "handled":
            first = (f"You picked up your {noun} from the {last['place']} {v['ago']}, then stopped at "
                     f"the {best['place']}. Look there first{spot}.")
        else:
            first = f"Your {noun} {were} last on the {last['place']}. Try the {best['place']} first{spot}."
        second = ""
        if len(c) > 1 and c[1]["chance"] >= 0.12 and not (best["zone"] == LEFT_VIEW and c[1]["zone"] == WITH_YOU):
            nxt = c[1]
            second = (" If not, check your pockets." if nxt["zone"] == WITH_YOU else
                      f" If not, try the {nxt['place']}.")
        return first + second

    SYSTEM = (
        "You are the reasoning step of Trace, a home helper that finds lost things for blind and "
        "low-vision people and people living with dementia. Cameras log sightings; you get the item's "
        "story (when and where it was seen, how it vanished, where the person went afterwards) and a "
        "scored list of candidate places. Link the evidence and decide the search order. Rules: only "
        "use places from the candidate list; prefer the scores unless the story clearly says otherwise; "
        "the reply is spoken aloud, so: at most 40 words, calm and warm, no percentages, no camera "
        "jargon, no 'zone'; lead with the single best place and one concrete spot to check there, "
        "then at most one fallback; never claim certainty. Answer as JSON only: "
        '{"order": [candidate zone ids, best first], "reply": "<what Trace says>"}')

    def _ask_llm(self, message, history, pred, now):
        payload = {"question": message, "now": clock(now), "item": pred["item"],
                   "last_seen": pred["last_seen"], "how_it_vanished": pred["cause"],
                   "story": pred["story"],
                   "candidates": [{k: c[k] for k in ("zone", "place", "camera", "chance", "evidence", "look")}
                                  for c in pred["candidates"]],
                   "recent_chat": history[-4:]}
        box = {}

        def run():
            try:
                box["out"] = self._gemini(json.dumps(payload))
            except Exception as exc:
                box["err"] = f"{type(exc).__name__}: {str(exc)[:120]}"

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(LLM_TIMEOUT)
        if "out" not in box:
            if self.verbose:
                print("whereabouts: LLM", box.get("err", "timed out"), "- using the rules answer")
            return None
        try:
            data = json.loads(box["out"])
            ids = [c["zone"] for c in pred["candidates"]]
            order = [z for z in data.get("order", []) if z in ids]
            order += [z for z in ids if z not in order]
            reply = str(data.get("reply", "")).strip()
            if not reply:
                return None
            return {"reply": reply[:400], "order": order, "by": "llm"}
        except (ValueError, AttributeError, TypeError):
            return None

    def _gemini(self, text):
        if self._llm is None:
            key = gemini_key()
            if not key:
                raise RuntimeError("no GEMINI_API_KEY")
            from google import genai
            from google.genai import types
            self._llm = (genai.Client(api_key=key, http_options=types.HttpOptions(timeout=LLM_DEADLINE_MS)),
                         types)
        client, types = self._llm
        r = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash", contents=text,
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               system_instruction=self.SYSTEM, temperature=0.3))
        return r.text

    # ============================================================ heatmap
    def _add_heat(self, cam, w, h, box, now):
        g = self.heat.get(cam)
        gh, gw = int(math.ceil(h / HEAT_CELL)), int(math.ceil(w / HEAT_CELL))
        if g is None or g.shape != (gh, gw):
            g = self.heat[cam] = np.zeros((gh, gw), np.float32)
            self.heat_t[cam] = now
        dt = now - self.heat_t[cam]
        if dt > 5:
            g *= 0.5 ** (dt / HEAT_HALF_LIFE)
            self.heat_t[cam] = now
        fx = int(min(gw - 1, max(0, (box[0] + box[2]) / 2 / HEAT_CELL)))
        fy = int(min(gh - 1, max(0, box[3] / HEAT_CELL)))
        g[max(0, fy - 1):fy + 2, max(0, fx - 1):fx + 2] += 0.5
        g[fy, fx] += 1.0

    def overlay(self, cam, view, now=None):
        """Draw the heatmap, people and the places onto a camera frame (BGR, in place)."""
        import cv2
        now = time.time() if now is None else now
        with self.lock:
            g = self.heat.get(cam)
            zones = [z for z in self.room.zones if z["camera"] == cam]
            pt, boxes = self.last_people.get(cam, (0, []))
            open_v = {v["zone"]: v for (c, _), v in self.open_visits.items() if c == cam}
        h, w = view.shape[:2]
        if g is not None and g.max() > 0:
            m = cv2.GaussianBlur(g, (0, 0), 1.2)
            m = np.sqrt(m / m.max())
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            col = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
            a = (np.clip(m * 1.4, 0, 1) * 0.55)[..., None]
            view[:] = (view * (1 - a) + col * a).astype(np.uint8)
        for z in zones:
            x1, y1, x2, y2 = map(int, z["rect"])
            v = open_v.get(z["id"])
            col = (0, 230, 255) if v else (255, 200, 0)
            cv2.rectangle(view, (x1, y1), (x2, y2), col, 1)
            label = z["name"] + (f"  {v['end'] - v['start']:.0f}s" if v else "")
            cv2.putText(view, label, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        if now - pt < 2:
            for b in boxes:
                cv2.rectangle(view, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)
                cv2.circle(view, (int((b[0] + b[2]) / 2), int(b[3])), 5, (255, 255, 255), -1)
        return view

    # ============================================================ state
    def snapshot(self, now=None):
        """Everything worth showing a judge, JSON-ready (GET /api/whereabouts)."""
        now = time.time() if now is None else now
        with self.lock:
            ids = list(self.items)
        return {"items": [self.predict(i, now) for i in ids],
                "accuracy": dict(self.habits["asked"]),
                "timeline": list(self.recent_log)[-30:]}

    def _place(self, zid):
        if zid == WITH_YOU:
            return "you"
        z = self.room.by_id.get(zid)
        return z["name"] if z else "unknown place"

    def _cam(self, cam):
        return self.cam_names.get(cam) or self.room.cam_name(cam)

    def _log(self, t, kind, item, zid, text):
        rec = {"t": round(t, 2), "time": clock(t), "type": kind, "item": item, "zone": zid,
               "place": self._place(zid) if zid else None, "text": text}
        self.recent_log.append(rec)
        if self.verbose:
            print(f"[{rec['time']}] {text}")
        if self.timeline:
            try:
                with self.timeline.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                pass

    def save(self):
        if not self.state_file:
            return
        with self.lock:
            data = {"items": {str(k): v for k, v in self.items.items()},
                    "episodes": {str(k): v for k, v in self.episodes.items()},
                    "visits": self.visits[-2000:], "person_seen": self.person_seen,
                    "habits": self.habits,
                    "heat": {c: g.round(3).tolist() for c, g in self.heat.items()},
                    "heat_t": self.heat_t}
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self.state_file)

    def load(self):
        if not self.state_file or not self.state_file.exists():
            return
        try:
            d = json.loads(self.state_file.read_text())
        except ValueError:
            return
        key = lambda k: int(k) if str(k).isdigit() else k
        self.items = {key(k): v for k, v in d.get("items", {}).items()}
        self.episodes = {key(k): v for k, v in d.get("episodes", {}).items()}
        self.visits = d.get("visits", [])
        self.person_seen = d.get("person_seen", {})
        self.habits.update(d.get("habits", {}))
        self.heat = {c: np.array(g, np.float32) for c, g in d.get("heat", {}).items()}
        self.heat_t = d.get("heat_t", {})
        for rec in self.items.values():     # tick() marks anything unseen since as missing
            rec.setdefault("since", rec.get("first", rec["last"]))


def gemini_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    p = Path(__file__).resolve().parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            if name.strip() == "GEMINI_API_KEY":
                return value.strip().strip('"').strip("'")
    return ""
