"""guide.py -- the LLM half of "Where are my keys?" (SPEC 5).

The rule: the LLM decides what was said and how to say the answer; the model in the page
decides what's true. So the LLM is called at most twice per question:

  1. parse  utterance -> evidence  [{"zone", "part": visible|covered|all, "found", "sure": high|medium|low}]
  2. word   the plan AFTER that evidence (probabilities, words and times all come from code)
            -> {"say", "places": [{"zone", "label"}], "chirp": zone|null}

ask(bundle, utterance, mode) does both and returns
  {"say", "places", "evidence", "chirp", "intent", "item", "source", "plan"}
"say" is None when there's nothing worth an LLM sentence (no key, the LLM failed, or the item
was found): the page then speaks its own template sentence. The page also validates every "say"
(SPEC 5.2) and applies "evidence" to its own model; "plan" here is only what the wording used.

The bundle comes from the page (bundleFor in trace-sim/index.html). For the plan update it
carries, per place, the masses Trace believes are visible / covered (pv, pc) and the search
success rates S, so step 2 sees exactly the plan the page will show (SPEC 4.4):
  a check that didn't find it multiplies visible mass by 1 - w*S_vis and covered mass by
  1 - w*S_cov (w = 1, 0.5, 0.3 for sure high / medium / low), then everything is renormalised;
  places are ranked by chance of finding it per second.

Provider: Gemini through the google-genai SDK, like scene.py and detect.py. GEMINI_API_KEY from
the environment or .env; GEMINI_MODEL picks the model. With no key or no SDK, a small rules
parser stands in for step 1 and step 2 is skipped (the page's template sentence is used).
"""
import json
import os
import re
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"
TIMEOUT = 8.0            # seconds per LLM call
SURE_W = {"high": 1.0, "medium": 0.5, "low": 0.3}
PARTS = ("visible", "covered", "all")
ITEMS = ("keys", "wallet", "glasses")


# ------------------------------------------------------------------ words from numbers
def prob_word(p):
    """SPEC 5.2: probability words are decided in code."""
    return "very likely" if p >= 0.8 else "probably" if p >= 0.6 else "might be" if p >= 0.4 else "possibly"


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


# ------------------------------------------------------------------ the plan (exact, SPEC 4.4)
def places_of(bundle):
    return [dict(p) for p in bundle.get("places") or []]


def apply_evidence(places, evidence, S):
    """Places after "checked here, not found" evidence. Each place: {id, pv, pc, t, ...}."""
    out = [dict(p) for p in places]
    by_id = {p["id"]: p for p in out}
    for ev in evidence:
        if ev.get("found"):
            continue
        p = by_id.get(ev["zone"])
        if not p:
            continue
        w = SURE_W.get(ev.get("sure"), 1.0)
        kind = p.get("kind", "zone")
        if kind == "zone":
            if ev.get("part") in ("visible", "all"):
                p["pv"] *= 1 - w * S["vis"]
            if ev.get("part") in ("covered", "all"):
                p["pc"] *= 1 - w * S["cov"]
        else:
            s = S["person"] if kind == "person" else S["out"]
            p["pv"] *= 1 - w * s
    tot = sum(p["pv"] + p["pc"] for p in out) or 1.0
    for p in out:
        p["pv"] /= tot
        p["pc"] /= tot
    return out


def rank(places, S):
    rows = []
    for p in places:
        prob = p["pv"] + p["pc"]
        if prob <= 0.003:
            continue
        kind = p.get("kind", "zone")
        if kind == "zone":
            f = p["pv"] * S["vis"] + p["pc"] * S["cov"]
        else:
            f = prob * (S["person"] if kind == "person" else S["out"])
        rows.append({"zone": p["id"], "name": p["name"], "p": prob, "word": prob_word(prob),
                     "seconds": p["t"], "covered": p["pc"] > p["pv"], "phone": bool(p.get("phone")),
                     "score": f / max(p["t"], 1e-9)})
    rows.sort(key=lambda r: -r["score"])
    return rows


# ------------------------------------------------------------------ step 1 without an LLM
SURE_LOW = re.compile(r"\b(quick(ly)?|glance[d]?|brief(ly)?|a peek|didn.t look properly)\b")
SURE_MED = re.compile(r"\b(i think|maybe|probably|not sure|i guess|might have)\b")
COVERED = re.compile(r"\b(under(neath)?|beneath|inside|behind|cushions?|pillows?|in the pocket of|lift(ed)?)\b")
NOT_FOUND = re.compile(r"\b(not|no|nothing|isn.t|aren.t|wasn.t|weren.t|didn.t find|couldn.t find|can.t see|check(ed)?|look(ed)?|search(ed)?|tried)\b")
FOUND = re.compile(r"\b(found|got (it|them)|here (it is|they are)|have (it|them))\b")


def zone_aliases(places):
    out = []
    for p in places:
        names = {_norm(p["name"]), _norm(p["id"].replace("-", " "))}
        if p.get("kind") == "person":
            names |= {"pocket", "pockets", "hand", "hands", "my coat", "on me", "with me"}
        elif p.get("kind") == "outside":
            names |= {"car", "bag", "outside", "backpack", "other room"}
        else:
            last = _norm(p["name"]).split(" ")[-1]
            if len(last) > 2:
                names.add(last)
        for n in names:
            if n:
                out.append((n, p["id"]))
    out.sort(key=lambda a: -len(a[0]))      # longest name first: "kitchen counter" before "counter"
    return out


def rules_parse(utterance, places, item=None):
    u = _norm(utterance)
    want = next((i for i in ITEMS if re.search(r"\b%s\b" % i, u)), item)
    if re.search(r"\bwhere\b", u) and not NOT_FOUND.search(u):
        return {"intent": "where", "item": want, "evidence": []}
    hits, taken = [], u
    for name, zid in zone_aliases(places):
        if re.search(r"\b%s\b" % re.escape(name), taken):
            if zid not in [h for h, _ in hits]:
                hits.append((zid, name))
            taken = re.sub(r"\b%s\b" % re.escape(name), " ", taken)
    if not hits:
        return {"intent": "other", "item": want, "evidence": []}
    found = bool(FOUND.search(u)) and not re.search(r"\b(not|didn.t|couldn.t|haven.t)\b", u)
    sure = "low" if SURE_LOW.search(u) else "medium" if SURE_MED.search(u) else "high"
    part = "covered" if COVERED.search(u) else "all"
    if not found and not NOT_FOUND.search(u):
        return {"intent": "other", "item": want, "evidence": []}
    ev = [{"zone": z, "part": part, "found": found, "sure": sure} for z, _ in hits]
    return {"intent": "found" if found else "checked", "item": want, "evidence": ev}


# ------------------------------------------------------------------ the LLM
def load_key():
    """GEMINI_API_KEY from the environment, else from .env beside this file (as scene.py)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                name, _, value = line.partition("=")
                if name.strip() == "GEMINI_API_KEY":
                    return value.strip().strip('"').strip("'")
    return ""


_client = None


def gemini_json(system, user, timeout=TIMEOUT):
    """One Gemini call that must return a JSON object. Raises on any failure."""
    global _client
    key = load_key()
    if not key:
        raise RuntimeError("no GEMINI_API_KEY (environment or .env)")
    from google import genai
    from google.genai import types
    if _client is None:
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
    r = _client.models.generate_content(
        model=MODEL, contents=user,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0.2,
                                           response_mime_type="application/json"))
    out = json.loads(r.text)
    if not isinstance(out, dict):
        raise ValueError("LLM did not return a JSON object")
    return out


PARSE_SYSTEM = """You turn what a person said while searching for a lost item into evidence for a search model.
Return JSON only: {"intent": "where"|"checked"|"found"|"question"|"other", "item": "keys"|"wallet"|"glasses"|null,
"evidence": [{"zone": <id from PLACES>, "part": "visible"|"covered"|"all", "found": true|false, "sure": "high"|"medium"|"low"}]}
- Only use zone ids listed in PLACES. "pockets"/"hands" is "person"; "car"/"bag" is "outside".
- "part": "covered" if they looked under/inside things (cushions, coats), "visible" if only on top, else "all".
- "sure": "high" if they checked properly, "medium" for "I think I looked", "low" for "had a quick look".
- found=true only if they say they found it there. No evidence for questions like "where are my keys?".
Never add places they did not mention."""

WORD_SYSTEM = """You are Trace, a calm voice helping someone find a lost item. Write what Trace says next.
Return JSON only: {"say": "...", "places": [{"zone": <id>, "label": "<the word given for it>"}], "chirp": <zone id or null>}
Rules (they are checked by code; break one and your answer is thrown away):
- Mention only places from PLAN, at most two, best first. For each, use exactly its "word"
  ("very likely", "probably", "might be", "possibly") if you give a likelihood. No percentages.
- Use the time phrases given (e.g. "about twenty minutes ago"); never invent times.
- Never say "definitely", "certainly", "it is there", "they are there" or "is on the X" as fact; it was LAST SEEN somewhere.
- Never say or guess who moved anything. Never mention cameras watching people.
- If the person just said a place was empty, acknowledge it in a few words first ("Not on the sofa.").
- chirp: the first place's zone if it has "phone": true and it's very likely or probably there, else null.
MODE default: one or two short sentences.
MODE blind: name where it is relative to big furniture and the person, step by step, no colours, never "over there".
MODE dementia: one calm step only, short words, you may add one memory prompt from TRAIL ("you were at the desk around two")."""


def _sanitize_evidence(raw, places):
    ids = {p["id"] for p in places}
    out = []
    for ev in raw or []:
        if not isinstance(ev, dict) or ev.get("zone") not in ids:
            continue
        out.append({"zone": ev["zone"],
                    "part": ev.get("part") if ev.get("part") in PARTS else "all",
                    "found": bool(ev.get("found")),
                    "sure": ev.get("sure") if ev.get("sure") in SURE_W else "high"})
    return out


def ask(bundle, utterance, mode="default", llm=None):
    """See the module docstring. llm(system, user) -> dict; defaults to Gemini, or rules if no key."""
    t0 = time.time()
    places, S = places_of(bundle), bundle.get("search") or {"vis": 0.95, "cov": 0.75, "person": 0.95, "out": 0.8}
    item = bundle.get("item_name") or bundle.get("item")
    use_llm = llm or (gemini_json if load_key() else None)
    source, errors = "rules", []
    parsed = None
    if use_llm:
        try:
            listing = [{"zone": p["id"], "name": p["name"]} for p in places]
            parsed = use_llm(PARSE_SYSTEM, json.dumps({"PLACES": listing, "ITEM": item, "SAID": utterance}))
            source = "llm"
        except Exception as exc:
            errors.append("parse: %s: %s" % (type(exc).__name__, exc))
    if parsed is None:
        parsed = rules_parse(utterance, places, bundle.get("item"))
    evidence = _sanitize_evidence(parsed.get("evidence"), places)
    intent = parsed.get("intent") if parsed.get("intent") in ("where", "checked", "found", "question", "other") else "other"
    want = parsed.get("item") if parsed.get("item") in ITEMS else None
    plan = rank(apply_evidence(places, evidence, S), S)
    res = {"say": None, "places": [], "chirp": None, "evidence": evidence, "intent": intent, "item": want,
           "source": source, "plan": plan[:4], "errors": errors}
    if any(e["found"] for e in evidence) or not use_llm or source != "llm":
        res["ms"] = int((time.time() - t0) * 1000)
        return res
    try:
        checked = [e["zone"] for e in evidence if not e["found"]]
        facts = {"ITEM": item, "MODE": mode, "SAID": utterance,
                 "JUST_CHECKED_EMPTY": [p["name"] for p in places if p["id"] in checked],
                 "ALREADY_CHECKED": bundle.get("already_checked") or [],
                 "LAST_SEEN": bundle.get("last_seen"),
                 "TRAIL": bundle.get("trail_since_last_seen") or [],
                 "PLAN": [{k: r[k] for k in ("zone", "name", "word", "seconds", "covered", "phone")} for r in plan[:4]]}
        out = llm_out = use_llm(WORD_SYSTEM, json.dumps(facts))
        say = str(llm_out.get("say") or "").strip()
        res["say"] = say or None
        res["places"] = [p for p in (out.get("places") or []) if isinstance(p, dict) and p.get("zone")]
        ch = out.get("chirp")
        res["chirp"] = ch if any(r["zone"] == ch and r["phone"] for r in plan[:2]) else None
    except Exception as exc:
        errors.append("word: %s: %s" % (type(exc).__name__, exc))
    res["ms"] = int((time.time() - t0) * 1000)
    return res
