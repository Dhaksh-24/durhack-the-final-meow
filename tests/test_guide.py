"""Tests for guide.py, no key or network needed:  python tests/test_guide.py"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo folder
sys.path.insert(0, ROOT)   # so the top-level modules (detect, targets, guide...) import
import guide  # noqa: E402

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
check(r5["state"] == "found" and r5["reply"].startswith("Found them:"), "'found it' ends the search")
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

print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
