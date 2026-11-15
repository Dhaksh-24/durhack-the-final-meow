"""Tests for guide.py, no key or network needed:  python test_guide.py"""
import json

import guide

S = {"vis": 0.95, "cov": 0.75, "person": 0.95, "out": 0.8}
PLACES = [
    {"id": "sofa", "name": "Sofa", "kind": "zone", "pv": 0.30, "pc": 0.20, "t": 20, "phone": True},
    {"id": "desk", "name": "Desk", "kind": "zone", "pv": 0.20, "pc": 0.05, "t": 10, "phone": True},
    {"id": "kitchen", "name": "Kitchen counter", "kind": "zone", "pv": 0.08, "pc": 0.02, "t": 15, "phone": False},
    {"id": "person", "name": "With you (hands or pockets)", "kind": "person", "pv": 0.12, "pc": 0.0, "t": 5},
    {"id": "outside", "name": "Outside the room (car or bag)", "kind": "outside", "pv": 0.03, "pc": 0.0, "t": 90},
]
BUNDLE = {"item": "keys", "item_name": "keys", "places": PLACES, "search": S}
ok = True


def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg)
    ok = ok and cond


def total(ps):
    return sum(p["pv"] + p["pc"] for p in ps)


print("plan update (SPEC 4.4)")
after = guide.apply_evidence(PLACES, [{"zone": "sofa", "part": "all", "found": False, "sure": "high"}], S)
sofa = next(p for p in after if p["id"] == "sofa")
raw = 0.30 * 0.05 + 0.20 * 0.25
check(abs(total(after) - 1) < 1e-9, "probabilities still add up to 1")
check(abs((sofa["pv"] + sofa["pc"]) - raw / (1 - 0.5 + raw)) < 1e-9, "sofa: visible x0.05, covered x0.25, renormalised")
half = guide.apply_evidence(PLACES, [{"zone": "sofa", "part": "all", "found": False, "sure": "medium"}], S)
check(next(p for p in half if p["id"] == "sofa")["pv"] > sofa["pv"], "'I think I looked' counts for less than a proper check")
under = guide.apply_evidence(PLACES, [{"zone": "sofa", "part": "covered", "found": False, "sure": "high"}], S)
u = next(p for p in under if p["id"] == "sofa")
check(u["pc"] / u["pv"] < 0.20 / 0.30, "looking under the cushions only lowers the covered part")

print("words from numbers (SPEC 5.2)")
check([guide.prob_word(p) for p in (0.85, 0.7, 0.5, 0.2)] == ["very likely", "probably", "might be", "possibly"], "very likely / probably / might be / possibly")

print("rules parser (no key)")
r = guide.rules_parse("where are my glasses?", PLACES, "keys")
check(r["intent"] == "where" and r["item"] == "glasses" and not r["evidence"], "a question gives no evidence, item glasses")
r = guide.rules_parse("I had a quick look under the sofa cushions, nothing", PLACES)
check(r["evidence"] == [{"zone": "sofa", "part": "covered", "found": False, "sure": "low"}], "quick look under the cushions -> sofa covered, low")
r = guide.rules_parse("I had a quick look under the sofa cushions", PLACES)
check(r["evidence"] == [{"zone": "sofa", "part": "covered", "found": False, "sure": "low"}], "'had a look' alone counts as checked")
r = guide.rules_parse("I think I checked the kitchen counter and my pockets", PLACES)
check({e["zone"] for e in r["evidence"]} == {"kitchen", "person"} and all(e["sure"] == "medium" for e in r["evidence"]), "two places, medium")
r = guide.rules_parse("found them on the desk!", PLACES)
check(r["intent"] == "found" and r["evidence"][0]["zone"] == "desk" and r["evidence"][0]["found"], "found on the desk")
r = guide.rules_parse("they're not on the counter", PLACES)
check(r["evidence"] and r["evidence"][0]["zone"] == "kitchen", "'counter' matches Kitchen counter")

print("ask() with a fake LLM: two calls, the second sees the updated plan")
calls = []


def fake(system, user):
    calls.append((system[:30], json.loads(user)))
    if system is guide.PARSE_SYSTEM:
        return {"intent": "checked", "item": "keys", "evidence": [
            {"zone": "sofa", "part": "all", "found": False, "sure": "high"},
            {"zone": "garage", "part": "all", "found": False, "sure": "high"}]}
    plan = json.loads(user)["PLAN"]
    return {"say": "Not on the sofa. Try the desk next: it's %s there." % plan[0]["word"],
            "places": [{"zone": plan[0]["zone"], "label": plan[0]["word"]}], "chirp": "desk"}


res = guide.ask(BUNDLE, "not on the sofa, I checked properly", "default", llm=fake)
check(len(calls) == 2, "exactly two LLM calls")
check(res["evidence"] == [{"zone": "sofa", "part": "all", "found": False, "sure": "high"}], "unknown zone 'garage' dropped")
before = [r["zone"] for r in guide.rank(PLACES, S)]
seen = [p["zone"] for p in calls[1][1]["PLAN"]]
check(before.index("sofa") < 3 and seen.index("sofa") == 3, "wording sees the plan after the sofa was ruled out (sofa drops to 4th)")
check(res["say"].startswith("Not on the sofa") and res["chirp"] == "desk" and res["source"] == "llm", "say and chirp come back")

calls.clear()
res = guide.ask(BUNDLE, "found them on the desk", "default", llm=lambda s, u: (calls.append(1), {"intent": "found", "evidence": [{"zone": "desk", "found": True}]})[1])
check(len(calls) == 1 and res["say"] is None and res["evidence"][0]["found"], "found it: no second call, the page says the template")


def broken(system, user):
    raise TimeoutError("no network")


res = guide.ask(BUNDLE, "I looked under the sofa", "default", llm=broken)
check(res["source"] == "rules" and res["say"] is None and res["evidence"][0]["zone"] == "sofa" and res["errors"], "LLM down: rules parse, template sentence")

print("trace.html conversation (guide.chat)")
KEYS = dict(BUNDLE, item_plural=True, visible_now=None, last_seen={"zone": "desk", "time": "14:05", "ago": "10 minutes ago"})
MODEL = {"items": {"keys": KEYS}}
try:
    guide.chat("where are my keys", [], {}, {})
    check(False, "no model posted -> NoModel")
except guide.NoModel:
    check(True, "no model posted -> NoModel (the server answers 503, trace.html falls back)")
ses = {}
say = lambda m, **kw: guide.chat(m, [], MODEL, ses, llm=kw.get("llm", broken), now=kw.get("now", 1000))
r1 = say("Where are my keys?")
first = guide.rank(PLACES, S)[0]
check(r1["state"] == "searching" and r1["item"] == "keys" and r1["reply"].startswith("Your keys were last seen on the desk 10 minutes ago.")
      and ("first: %s there" % first["word"]) in r1["reply"], "where: template from the posted model, word from code")
r2 = say("not there")
check(r2["reply"].startswith("Not there.") and r2["evidence"][0]["zone"] == first["zone"] and ses["keys"]["tried"] == [first["name"]], "'not there' rules out the top place, no LLM")
check(say("again")["reply"] == r2["reply"], "'again' repeats the last line")
r4 = say("I had a quick look under the sofa cushions")
check(r4["evidence"] == [{"zone": "sofa", "part": "covered", "found": False, "sure": "low"}] and "Sofa (quick look)" in ses["keys"]["tried"], "described check -> evidence, kept in the session")
r5 = say("found it")
check(r5["state"] == "found" and r5["reply"].startswith("Great, you found them"), "'found it' ends the search")
r6 = say("where are my keys?")
check(r6["reply"].startswith("Your keys were last seen") and ses["keys"]["tried"] == [], "a new question starts a fresh search")
check(guide.chat("where is it?", [], MODEL, {}, llm=broken)["reply"].startswith("Which one"), "no item named and nothing to go on: asks which one")
check(guide.chat("is it in the car?", [{"role": "user", "content": "where are my keys"}], MODEL, {}, llm=broken)["item"] == "keys", "item taken from the history")


def llm_says(text, places=None, chirp=None):
    def f(system, user):
        if system is guide.PARSE_SYSTEM:
            return {"intent": "where", "item": "keys", "evidence": []}
        return {"say": text, "places": places or [], "chirp": chirp}
    return f


ses.clear()
w = first["word"]
good = guide.chat("where are my keys", [], MODEL, ses, llm=llm_says("They were last seen on the desk 10 minutes ago. Try %s next: %s there." % (guide.where_text(first), w)), now=2000)
check(good["source"] == "llm" and good["reply"].startswith("They were last seen") and not good["why"], "a valid LLM answer is used as it is")
ses.clear()
bad = guide.chat("where are my keys", [], MODEL, ses, llm=llm_says("They are definitely on the desk."), now=2000)
check(bad["reply"].startswith("Your keys were last seen") and "certain" in bad["why"], "'definitely' is rejected, template used")
ses.clear()
wrong = guide.chat("where are my keys", [], MODEL, ses, llm=llm_says("Try %s, very likely." % guide.where_text(first) if w != "very likely" else "Try %s, possibly." % guide.where_text(first)), now=2000)
check("doesn't match the plan" in wrong["why"], "a probability word that doesn't match the plan is rejected")
print("'not there' when the top place is suggested again")
SEEN = [dict(p, pv=0.9996 if p["id"] == "sofa" else 0.0001, pc=0.0) for p in PLACES]   # a phone sees them now
VIS = {"items": {"keys": dict(BUNDLE, places=SEEN, item_plural=True, visible_now="sofa",
                              last_seen={"zone": "sofa", "time": "14:05", "ago": "just now"})}}
vs = {}
guide.chat("where are my keys", [], VIS, vs, llm=broken, now=3000)
again = guide.chat("not there", [], VIS, vs, llm=broken, now=3001)["reply"]
check("still sees something that looks like your keys" in again and "the sofa again" in again and "next" not in again,
      "phone still sees them: says so and asks for a careful second look, not 'next'")
again2 = guide.chat("not there", [], VIS, vs, llm=broken, now=3002)["reply"]
check(again2 != again and "may be something else" in again2 and "once more." in again2,
      "second 'not there' at the same place: doesn't repeat itself, says the phone may be wrong")
check("Check %s next" % guide.where_text(guide.rank(guide.apply_evidence(PLACES, [{"zone": first["zone"], "part": "all", "found": False, "sure": "high"}], S), S)[0]) in r2["reply"],
      "a normal 'not there' still moves on to the next place")
ses.clear()
late = guide.chat("where are my keys", [], MODEL, ses, llm=llm_says("You had them on the desk about 3 minutes ago."), now=2000)
check("time" in late["why"], "a time that isn't from the model is rejected")


# ======================================================================= the demo (HANDOVER 5)
# Bundles shaped exactly as bundleFor() in trace-sim/index.html posts them. The keys one was
# captured from /sim replaying the demo recording (keys on the sofa, slid under a cloth, person
# walked to the desk); the glasses sit in view of the sofa phone.
import threading
import time
import urllib.request

ZONES = [("sofa", "Sofa", 20, True), ("desk", "Desk", 10, True), ("shelf", "Shelf", 8, True),
         ("kitchen", "Kitchen counter", 15, False), ("hook", "Coat hook", 12, False)]


def demo_places(mass):
    """mass: {id: (pv, pc)}; zones, then person and outside, as the page lists PLACES."""
    out = [{"id": z, "name": n, "kind": "zone", "t": t, "pv": mass.get(z, (0, 0))[0], "pc": mass.get(z, (0, 0))[1], "phone": ph}
           for z, n, t, ph in ZONES]
    out.append({"id": "person", "name": "With you (hands or pockets)", "kind": "person", "t": 5, "pv": mass.get("person", (0, 0))[0], "pc": 0, "phone": False})
    out.append({"id": "outside", "name": "Outside the room (car or bag)", "kind": "outside", "t": 90, "pv": mass.get("outside", (0, 0))[0], "pc": 0, "phone": False})
    return out


def demo_bundle(item, name, plural, mass, **kw):
    b = {"task": "Tell the user where to look...", "item": item + "-1",
         "zones": [{"id": z, "name": n, "phone": ph} for z, n, _, ph in ZONES],
         "last_seen": None, "recent_events": [], "already_checked": [], "visible_now": None, "item_plural": plural,
         "search_plan": [], "story": [], "story_text": "", "best_question": None, "trail_since_last_seen": [],
         "item_name": name, "places": demo_places(mass), "search": dict(S)}
    b.update(kw)
    return b


DEMO_KEYS = demo_bundle(
    "keys", "keys", True,
    {"sofa": (1e-05, 0.42889), "desk": (0.00028, 0.05328), "shelf": (0, 0.00012), "kitchen": (0.00169, 0.00056),
     "hook": (0.00225, 0.00338), "person": (0.50896, 0), "outside": (0.00057, 0)},
    last_seen={"zone": "sofa", "time": "09:24", "ago": "3 minutes ago"},
    story=[{"where": "on the sofa", "from": "09:23", "via": []}, {"where": "with you", "from": "09:24", "via": ["sofa", "desk"]}],
    story_text="Your keys were on the sofa at 09:23, then most likely you picked them up at 09:24 and carried them to the desk. "
               "They might also be under things on the sofa.",
    trail_since_last_seen=[{"place": "sofa", "from": "09:24", "minutes": 1}, {"place": "desk", "from": "09:25", "minutes": 2}],
    best_question={"id": "cover-sofa", "text": "Did you put anything down on top of things on the sofa?", "gain_s": -3.2,
                   "worth_asking": False, "yes": [{"zone": "sofa", "fv": 0.15, "fc": 0.75}], "no": [{"zone": "sofa", "fv": 0.85, "fc": 0.25}]})
DEMO_GLASSES = demo_bundle("glasses", "glasses", True, {"sofa": (0.985, 0.005), "person": (0.005, 0), "desk": (0.005, 0)},
                           visible_now="sofa", last_seen={"zone": "sofa", "time": "09:27", "ago": "just now"})
DEMO_WALLET = demo_bundle("wallet", "wallet", False, {z: (0.1, 0.04) for z, *_ in ZONES} | {"person": (0.2, 0), "outside": (0.1, 0)})
DEMO = {"room": "Demo flat", "items": {"keys": DEMO_KEYS, "wallet": DEMO_WALLET, "glasses": DEMO_GLASSES}}

llm_calls = []


def counting(system, user):      # a fake LLM that's always down, and counts how often it was asked
    llm_calls.append(system[:12])
    raise ConnectionError("no network in the tests")


def talk(ses, msg, model=DEMO, **kw):
    return guide.chat(msg, kw.pop("history", []), model, ses, llm=kw.pop("llm", counting), now=kw.pop("now", 5000), **kw)


print("the blindfold demo, rules only (LLM down)")
ses = {}
llm_calls.clear()
d1 = talk(ses, "Hey Trace, where are my keys?")
check(d1["reply"].startswith("Your keys were last seen on the sofa 3 minutes ago.") and "Check your pockets and hands first" in d1["reply"]
      and "about 5 seconds" in d1["reply"] and "under things on the sofa" in d1["reply"], "where: pockets first (5 s), then under the sofa")
check(guide.TARGET.get("zone") == "person" and guide.TARGET.get("state") == "searching", "/api/target points the sonar at the pockets")
n_llm = len(llm_calls)
d2 = talk(ses, "Not there.")
check(d2["reply"].startswith("Not there. Check under things on the sofa next: very likely there") and len(llm_calls) == n_llm,
      "'not there' -> under the sofa, very likely; no LLM call")
check(guide.TARGET.get("zone") == "sofa", "sonar moves to the sofa")
d3 = talk(ses, "Found them!")
check(d3["state"] == "found" and d3["reply"] == "Great, you found them on the sofa. I'll remember that." and len(llm_calls) == n_llm,
      "'found them!' ends the search at the sofa, no LLM")
check(guide.TARGET.get("state") == "found" and guide.TARGET.get("zone") is None, "sonar stops")
d4 = talk(ses, "Where are my glasses?")
check(d4["item"] == "glasses" and d4["reply"] == "The sofa phone can see your glasses right now." and len(llm_calls) == n_llm,
      "glasses: 'the phone can see them right now', instantly, no LLM")
before_target = dict(guide.TARGET)
d5 = talk(ses, "What happened to my keys?")
check(d5["item"] == "keys" and d5["reply"] == DEMO_KEYS["story_text"] and guide.TARGET == before_target,
      "'what happened to my keys?' tells the story; the sonar isn't moved")

print("found variants and found elsewhere")
for said in ("found my keys", "yes I found them", "here they are", "got them", "I've found them", "found it thanks"):
    s2 = {}
    talk(s2, "where are my keys")
    r = talk(s2, said)
    check(r["state"] == "found", "%r counts as found" % said)
s2 = {}
talk(s2, "where are my keys")
r = talk(s2, "found them on the desk")
check(r["state"] == "found" and r["evidence"] == [{"zone": "desk", "part": "all", "found": True, "sure": "high"}]
      and "on the desk" in r["reply"], "'found them on the desk' (not the top suggestion): found at the desk")
s2 = {}
talk(s2, "where are my keys")
r = talk(s2, "they were in my pocket, I found them")
check(r["state"] == "found" and "with you" in r["reply"], "found in a pocket: 'found them with you'")
r = talk({}, "found them")
check(r["state"] == "idle" and r["reply"].startswith("Which one"), "'found them' with no item and no search: asks which one")
s2 = {}
r = talk(s2, "found my keys")
check(r["state"] == "found" and r["reply"] == "Great, you found them. I'll remember that.", "found before any suggestion: no made-up place")
s2 = {}
talk(s2, "where are my keys")
r = talk(s2, "I haven't found them")
check(r["state"] == "searching", "'I haven't found them' is not found")

print("switching items, resets, odd utterances")
s3 = {}
talk(s3, "where are my keys?")
talk(s3, "not there")
r = talk(s3, "and my glasses?", now=5001)
check(r["item"] == "glasses" and "can see your glasses right now" in r["reply"], "'and my glasses?' switches item")
r = talk(s3, "ok, where are my keys", now=5002)
check(r["item"] == "keys" and r["reply"].startswith("Your keys were last seen") and s3["keys"]["tried"] == [], "back to keys with 'where': a fresh search")
talk(s3, "not there", now=5003)
r = talk(s3, "my keys again?", now=5004)
check(r["item"] == "keys" and s3["keys"]["tried"], "naming the item without 'where' keeps the search going")
r = talk(s3, "where are my keys", now=5005)
r2 = talk(s3, "where are my keys", now=5006)
check(r["reply"] == r2["reply"] and s3["keys"]["tried"] == [], "asking 'where' twice: same fresh answer")
r = talk({}, "not there")
check(r["state"] == "idle" and r["reply"].startswith("Which one"), "'not there' before any question: asks which item")
s4 = {}
r = talk(s4, "not there", history=[{"role": "user", "content": "where are my keys"}])
check(r["item"] == "keys" and r["state"] == "searching" and not s4["keys"]["tried"], "'not there' with nothing suggested yet doesn't rule anything out")
for junk in ("", "   ", "blah blah", "banana", "?!", "x" * 600):
    r = talk(s3, junk)
    check(isinstance(r["reply"], str) and r["reply"], "unknown utterance %r: a reply, no crash" % junk[:12])
r = talk(s3, "again")
check(r["reply"] == s3["keys"]["last"], "'again' repeats")

print("Trace's question (yes / no)")
asky = {"items": {"keys": dict(DEMO_KEYS, best_question=dict(DEMO_KEYS["best_question"], worth_asking=True))}}
s5 = {}
r = talk(s5, "where are my keys", model=asky)
check(r["reply"].startswith("Before you look: Did you put anything"), "worth asking: the question comes first")
r = talk(s5, "no", model=asky)
check(r["reply"].startswith("Thanks. ") and s5["keys"]["evidence"], "'no' applies the question's factors")
r = talk({}, "help me remember my keys", model=DEMO)
check(r["reply"] == DEMO_KEYS["best_question"]["text"], "'help me remember' asks the question")

print("empty / missing fields in the bundle")
for name, b in [("empty bundle", {}), ("no places", {"item_name": "keys", "places": []}),
                ("places missing numbers", {"item_name": "keys", "places": [{"id": "sofa", "name": "Sofa"}, {"id": "person", "kind": "person"}, {"name": "no id"}, "junk"]}),
                ("bad last_seen", dict(DEMO_KEYS, last_seen={}, story_text=None, best_question=None, search=None)),
                ("strings for numbers", dict(DEMO_KEYS, places=[dict(p, pv=str(p["pv"]), t=str(p["t"])) for p in DEMO_KEYS["places"]])),
                ("NaN", dict(DEMO_KEYS, places=[dict(p, pv=float("nan")) for p in DEMO_KEYS["places"]]))]:
    m = {"items": {"keys": b}}
    s6 = {}
    try:
        outs = [talk(s6, u, model=m)["reply"] for u in ("where are my keys", "not there", "what happened to my keys", "help me remember", "found them")]
        check(all(isinstance(o, str) and o for o in outs), name + ": every turn answers")
    except Exception as exc:
        check(False, "%s: %s: %s" % (name, type(exc).__name__, exc))
try:
    guide.chat("where are my keys", [], {"items": {}}, {})
    check(False, "empty items -> NoModel")
except guide.NoModel:
    check(True, "empty items -> NoModel")

print("the Gemini path (fake LLM; never the network)")


def fake_llm(word, parse=None, delay=0.0):
    calls = []

    def f(system, user):
        calls.append((system is guide.PARSE_SYSTEM and "parse" or "word", json.loads(user)))
        time.sleep(delay)
        if system is guide.PARSE_SYSTEM:
            return parse if parse is not None else {"intent": "other", "item": None, "evidence": []}
        return word(json.loads(user)) if callable(word) else word
    f.calls = calls
    return f


top = guide.rank(guide.places_of(DEMO_KEYS), S)
good = fake_llm(lambda f: {"say": "Your keys were last seen on the sofa 3 minutes ago. Check your pockets first: %s there." % f["PLAN"][0]["word"],
                           "places": [{"zone": f["PLAN"][0]["zone"], "label": f["PLAN"][0]["word"]}], "chirp": None})
r = talk({}, "where are my keys?", llm=good)
check(r["source"] == "llm" and r["reply"].startswith("Your keys were last seen") and "might be there" in r["reply"], "valid LLM wording is used")
check([c[0] for c in good.calls] == ["word"], "a plain 'where' question skips the parse call (one LLM call)")
check(good.calls[0][1]["PLAN"][0]["zone"] == "person" and good.calls[0][1]["LAST_SEEN"]["zone"] == "sofa", "the LLM sees the plan and last seen")
for mode in ("blind", "dementia", "nonsense"):
    g = fake_llm({"say": "Check your pockets first.", "places": []})
    guide.chat("where are my keys", [], DEMO, {}, mode=mode, llm=g, now=5000)
    check(g.calls and g.calls[-1][1]["MODE"] == (mode if mode != "nonsense" else "default"), "mode %s reaches the wording call" % mode)
bad = fake_llm({"say": "Your keys are definitely in your pocket.", "places": []})
r = talk({}, "where are my keys?", llm=bad)
check(r["reply"].startswith("Your keys were last seen on the sofa") and "template" in r["why"], "invalid LLM answer -> template")
bad2 = fake_llm({"say": "Check the coat hook, very likely.", "places": []})
r = talk({}, "where are my keys?", llm=bad2)
check("Check your pockets" in r["reply"] and r["why"], "LLM names a place / word not in the plan -> template")
bad3 = fake_llm(["not", "a", "dict"])
r = talk({}, "where are my keys?", llm=bad3)
check(r["reply"].startswith("Your keys were last seen"), "LLM returns a list -> template")
r = talk({}, "I checked the shelf for my keys", llm=fake_llm({"say": "x"}, parse=["junk"]))
check(isinstance(r["reply"], str) and r["evidence"] and r["evidence"][0]["zone"] == "shelf", "parse returns junk -> rules parser")
r = talk({}, "I checked the desk for my keys", llm=fake_llm({"say": None}, parse={"intent": "checked", "evidence": [{"zone": "desk", "found": "false"}]}))
check(r["state"] == "searching" and r["evidence"][0]["found"] is False, "'found': \"false\" from the LLM is not found")
r = talk({}, "where are my keys", llm=counting)
check(r["reply"].startswith("Your keys were last seen"), "LLM raises -> template")
slow = fake_llm({"say": "Check your pockets first.", "places": []}, delay=2.0)
t0 = time.time()
r = guide.chat("where are my keys", [], DEMO, {}, llm=slow, now=5000, budget=0.4)
el = time.time() - t0
check(el < 1.0 and r["reply"].startswith("Your keys were last seen") and "Check your pockets" in r["reply"],
      "slow LLM: the template answer comes within the budget (%.2f s)" % el)
slow2 = fake_llm({"say": "Not in the shelf."}, parse={"intent": "checked", "evidence": []}, delay=2.0)
t0 = time.time()
r = guide.chat("I looked on the shelf for my keys", [], DEMO, {}, llm=slow2, now=5000, budget=0.4)
check(time.time() - t0 < 1.0 and r["evidence"] and r["evidence"][0]["zone"] == "shelf", "slow parse: rules parser takes over within the budget")
check(guide.BUDGET <= 8 and guide.TIMEOUT >= 10, "turn budget %.0f s (what the person waits), Gemini deadline %.0f s (its minimum)" % (guide.BUDGET, guide.TIMEOUT))
fast = fake_llm({"say": "unused"})
s7 = {}
talk(s7, "where are my keys", llm=fast)
n0 = len(fast.calls)
for u in ("not there", "again", "what happened to my keys", "found them"):
    talk(s7, u, llm=fast)
check(len(fast.calls) == n0, "'not there', 'again', the story and 'found them' never wait on the LLM")
r = talk({}, "where are my glasses", llm=fast)
check(len(fast.calls) == n0 and "right now" in r["reply"], "item in view of a phone: answered without the LLM")

moved = {"items": {"keys": dict(DEMO_KEYS, last_seen={"zone": "shelf", "time": "09:20", "ago": "7 minutes ago"})}}
ls_llm = fake_llm(lambda f: {"say": "Your keys were last seen on the shelf 7 minutes ago. Check your pockets first: %s there." % f["PLAN"][0]["word"], "places": []})
r = talk({}, "where are my keys", model=moved, llm=ls_llm)
check(r["source"] == "llm" and "shelf" in r["reply"] and not r["why"], "'last seen on the shelf' is allowed even when the shelf isn't in the plan")

print("the server: /api/ask, /api/model, /api/target")
import socket
import trace_api
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()
srv = trace_api.TraceAPI([], "/nonexistent-memory", "/nonexistent-events.jsonl").start("127.0.0.1", port)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(path, body=None):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


try:
    guide_key, guide.load_key = guide.load_key, lambda: ""      # never reach Gemini from the tests
    code, j = http("/api/ask", {"message": "where are my keys", "history": []})
    check(code == 503 and "error" in j, "no model posted: /api/ask answers 503 (trace.html falls back to its own reply)")
    code, j = http("/api/model", {"room": "x"})
    check(code == 400, "/api/model without items: 400")
    big = dict(DEMO, items=dict(DEMO["items"], keys=dict(DEMO_KEYS, recent_events=[{"pad": "x" * 200}] * 400)))
    code, j = http("/api/model", big)
    check(code == 200, "/api/model takes a large bundle (%d KB)" % (len(json.dumps(big)) // 1024))
    code, j = http("/api/model", DEMO)
    check(code == 200 and j.get("ok"), "/api/model accepts the page's bundle")
    code, j = http("/api/ask", {"message": "where are my keys", "history": [], "mode": "blind"})
    check(code == 200 and "pockets" in j.get("reply", "") and j.get("state") == "searching", "/api/ask answers from the posted model")
    code, t = http("/api/target")
    check(code == 200 and t.get("item") == "keys" and t.get("zone") == "person", "/api/target says where Trace is sending them")
    code, j = http("/api/ask", {"message": "not there", "history": []})
    check(code == 200 and "under things on the sofa" in j["reply"], "'not there' over HTTP")
    code, j = http("/api/ask", {"message": "found them", "history": []})
    check(code == 200 and j["state"] == "found", "'found them' over HTTP")
    code, j = http("/api/ask", {"message": None, "history": "junk", "mode": 7})
    check(code == 200 and j.get("reply"), "junk fields: still a reply")
finally:
    guide.load_key = guide_key
    srv.stop()

print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
