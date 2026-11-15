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

print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
