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
import concurrent.futures
import json
import os
import re
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "gemini-3.8-flash"   # default; GEMINI_MODEL in the environment or .env overrides it
TIMEOUT = 10.0           # seconds per LLM call (Gemini rejects deadlines under 10 s)
BUDGET = 6.0             # seconds the person waits for the LLM per turn (both calls); then the template
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm")
S_DEFAULT = {"vis": 0.95, "cov": 0.75, "person": 0.95, "out": 0.8}
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
def _num(v, default=0.0):
    try:
        v = float(v)
        return v if v == v and v >= 0 else default     # NaN or negative: treat as missing
    except (TypeError, ValueError):
        return default


def places_of(bundle):
    """The bundle's places, with missing fields filled in (a place without an id is dropped)."""
    out = []
    for p in (bundle or {}).get("places") or []:
        if not isinstance(p, dict) or not p.get("id"):
            continue
        q = dict(p)
        q["id"] = str(q["id"])
        q["name"] = str(q.get("name") or q["id"])
        q["kind"] = q.get("kind") if q.get("kind") in ("zone", "person", "outside") else "zone"
        q["pv"], q["pc"], q["t"] = _num(q.get("pv")), _num(q.get("pc")), _num(q.get("t"), 10.0) or 10.0
        out.append(q)
    return out


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
        if "fv" in ev:   # an answer to Trace's question: Bayes factors for this place
            p["pv"] *= float(ev["fv"])
            p["pc"] *= float(ev.get("fc", 1.0))
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
                     "seconds": int(round(p["t"])), "covered": p["pc"] > p["pv"], "phone": bool(p.get("phone")),
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
def env_setting(var):
    """var from the environment, else from .env beside this file (as scene.py)."""
    val = os.environ.get(var, "").strip()
    if val:
        return val
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                name, _, value = line.partition("=")
                if name.strip() == var:
                    return value.strip().strip('"').strip("'")
    return ""


def load_key():
    return env_setting("GEMINI_API_KEY")


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
        model=env_setting("GEMINI_MODEL") or MODEL, contents=user,
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
- If PHONE_SEES_IT_NOW_AT is set, say first that the phone there can see it right now.
- If the person just said a place was empty, acknowledge it in a few words first ("Not on the sofa.").
- chirp: the first place's zone if it has "phone": true and it's very likely or probably there, else null.
MODE default: one or two short sentences.
MODE blind: name where it is relative to big furniture and the person, step by step, no colours, never "over there".
MODE dementia: one calm step only, short words, you may add one memory prompt from TRAIL ("you were at the desk around two").
STORY is the model's most likely account of what happened to the item (times from it are fine). If the
person asks what happened, retell it in one or two sentences, saying "most likely"; never present it as certain."""


def _sanitize_evidence(raw, places):
    ids = {p["id"] for p in places}
    out = []
    for ev in raw or []:
        if not isinstance(ev, dict) or ev.get("zone") not in ids:
            continue
        out.append({"zone": ev["zone"],
                    "part": ev.get("part") if ev.get("part") in PARTS else "all",
                    "found": ev.get("found") is True or str(ev.get("found")).lower() == "true",
                    "sure": ev.get("sure") if ev.get("sure") in SURE_W else "high"})
    return out


def _call(llm, system, user, deadline):
    """llm(system, user) in a worker thread, waiting no later than deadline (time.time()).
    A call that's too slow is abandoned (it finishes in the background); raises TimeoutError."""
    left = deadline - time.time()
    if left <= 0.05:
        raise TimeoutError("no time left for the LLM")
    fut = _POOL.submit(llm, system, user)
    done, _ = concurrent.futures.wait([fut], timeout=left)
    if not done:
        raise TimeoutError("LLM took longer than %.1f s" % left)
    return fut.result()


def ask(bundle, utterance, mode="default", llm=None, budget=None, parsed=None):
    """See the module docstring. llm(system, user) -> dict; defaults to Gemini, or rules if no key.
    budget: seconds for both LLM calls together (default BUDGET); past it, the rules / template
    answer is used, so a slow network never leaves the person waiting. parsed: the parse already
    done by the caller (skips LLM call 1)."""
    t0 = time.time()
    deadline = t0 + (BUDGET if budget is None else budget)
    places, S = places_of(bundle), bundle.get("search") or dict(S_DEFAULT)
    item = bundle.get("item_name") or bundle.get("item")
    use_llm = llm or (gemini_json if load_key() else None)
    source, errors = "rules", []
    if parsed is not None:
        source = "llm" if use_llm else "rules"
    elif use_llm:
        try:
            listing = [{"zone": p["id"], "name": p["name"]} for p in places]
            parsed = _call(use_llm, PARSE_SYSTEM, json.dumps({"PLACES": listing, "ITEM": item, "SAID": utterance}), deadline)
            if not isinstance(parsed, dict):
                raise ValueError("parse did not return a JSON object")
            source = "llm"
        except Exception as exc:
            parsed = None
            errors.append("parse: %s: %s" % (type(exc).__name__, exc))
    if parsed is None:
        parsed = rules_parse(utterance, places, bundle.get("item"))
    evidence = _sanitize_evidence(parsed.get("evidence"), places)
    intent = parsed.get("intent") if parsed.get("intent") in ("where", "checked", "found", "question", "other") else "other"
    want = parsed.get("item") if parsed.get("item") in ITEMS else None
    plan = rank(apply_evidence(places, evidence, S), S)
    res = {"say": None, "places": [], "chirp": None, "evidence": evidence, "intent": intent, "item": want,
           "source": source, "plan": plan[:4], "errors": errors}
    if any(e["found"] for e in evidence) or intent == "found" or not use_llm or source != "llm" or not plan:
        res["ms"] = int((time.time() - t0) * 1000)
        return res
    try:
        checked = [e["zone"] for e in evidence if not e["found"]]
        names = {p["id"]: p["name"] for p in places}
        facts = {"ITEM": item, "MODE": mode if mode in ("default", "blind", "dementia") else "default", "SAID": utterance,
                 "JUST_CHECKED_EMPTY": [p["name"] for p in places if p["id"] in checked],
                 "ALREADY_CHECKED": bundle.get("already_checked") or [],
                 "LAST_SEEN": bundle.get("last_seen"),
                 "PHONE_SEES_IT_NOW_AT": names.get(bundle.get("visible_now")) if bundle.get("visible_now") else None,
                 "TRAIL": bundle.get("trail_since_last_seen") or [],
                 "STORY": bundle.get("story_text") or bundle.get("story") or [],
                 "PLAN": [{k: r[k] for k in ("zone", "name", "word", "seconds", "covered", "phone")} for r in plan[:4]]}
        out = _call(use_llm, WORD_SYSTEM, json.dumps(facts), deadline)
        if not isinstance(out, dict):
            raise ValueError("wording did not return a JSON object")
        say = str(out.get("say") or "").strip()
        res["say"] = say or None
        res["places"] = [p for p in (out.get("places") or []) if isinstance(p, dict) and p.get("zone")]
        ch = out.get("chirp")
        res["chirp"] = ch if any(r["zone"] == ch and r["phone"] for r in plan[:2]) else None
    except Exception as exc:
        errors.append("word: %s: %s" % (type(exc).__name__, exc))
    res["ms"] = int((time.time() - t0) * 1000)
    return res


# ================================================================== the proper front end
# trace.html sends {"message", "history"} and wants {"reply", "item_id", "state"}. It has no model
# of its own: the analytics page (trace-sim, in Real events mode) posts the model's belief for each
# item to /api/model, and the conversation's checks live here, per item, applied with the same
# exact update (SPEC 4.4) on top of whatever the model believes now.
SESSION_IDLE = 15 * 60       # forget a search after this long without a word about it
ITEM_ALIASES = {"keys": r"\b(keys?|keyring)\b", "wallet": r"\b(wallet|purse|cards?)\b",
                "glasses": r"\b(glasses|specs|spectacles)\b"}
YES = re.compile(r"^(yes|yeah|yep|yup|i did|i think so|maybe yes|probably)\b")
NO = re.compile(r"^(no|nope|nah|i didn.?t|don.?t think so|not that i know)\b")
WHAT_HAPPENED = re.compile(r"\b(what happened|how did i lose|where did i (put|leave)|what did i do with)\b")
AGAIN = re.compile(r"^(again|repeat|repeat that|say (that|it) again|what|pardon|sorry what)$")
NOT_THERE = re.compile(r"^(no |nope )?(it's |its |they're |theyre |it is |they are )?not (there|here)( either)?$|^(nope|no|nothing)$")
FOUND_IT = re.compile(r"^((yes|yeah|yay|great|ok|okay|oh|ah|got it|thanks|thank you) )*(i )?(i've |ive |i have |we )?"
                      r"(found|got|have) (it|them|my (keys|wallet|glasses|specs|purse)|the (keys|wallet|glasses))"
                      r"( now| here| thanks| thank you)?$|^(found|got (it|them))$|^here (it is|they are)$")


class NoModel(Exception):
    """No model state has been posted (open the analytics page in Real events mode)."""


def item_in(text):
    t = _norm(text)
    return next((k for k, rx in ITEM_ALIASES.items() if re.search(rx, t)), None)


def where_text(r):
    if r["zone"] == "person":
        return "your pockets and hands"
    if r["zone"] == "outside":
        return "outside the room, in the car or your bag"
    return ("under things on the " if r["covered"] else "the ") + r["name"].lower()


def again_text(bundle, r0, tried, names):
    """The top place was already checked (a phone still sees it there, or nothing else comes close):
    say so, instead of "Check the sofa next" as if it were new. None if r0 wasn't checked."""
    times = sum(str(t).lower().startswith(r0["name"].lower()) for t in tried or [])
    if not times:
        return None
    s, noun = "", bundle.get("item_name") or "item"
    if bundle.get("visible_now") == r0["zone"]:
        where = names.get(r0["zone"], r0["zone"])
        if times >= 2:      # checked properly twice: the phone may be seeing something else
            return ("The %s phone still sees something that looks like your %s, but it may be something else. "
                    "Feel around the %s once more." % (where, noun, where))
        s = "The %s phone still sees something that looks like your %s. " % (where, noun)
    careful = ", more carefully, under and around things" if r0["zone"] not in ("person", "outside") and not r0["covered"] else ", more carefully"
    return s + "Check %s again%s: %s there, about %s seconds." % (where_text(r0), careful, r0["word"], r0["seconds"])


def template(bundle, plan, tries, tried=()):
    """The page's own sentence (trace-sim sentence()), in Python: what Trace says without the LLM.
    tries: how many checks so far; tried: their place names (to spot a place suggested again)."""
    name, plural = bundle.get("item_name") or "item", bool(bundle.get("item_plural"))
    names = {p["id"]: p["name"].lower() for p in places_of(bundle)}
    if not plan:
        return "I don't have a good idea where %s." % ("they are" if plural else "it is")
    if not tries and bundle.get("visible_now"):
        return "The %s phone can see your %s right now." % (names.get(bundle["visible_now"], bundle["visible_now"]), name)
    s = "Not there. " if tries else ""
    ls = bundle.get("last_seen") if isinstance(bundle.get("last_seen"), dict) and bundle["last_seen"].get("zone") else None
    if not tries:
        s += ("Your %s %s last seen on the %s %s. " % (name, "were" if plural else "was", names.get(ls["zone"], ls["zone"]), ls.get("ago", ""))
              if ls else "No phone has seen your %s today. " % name)
    r0 = plan[0]
    s += again_text(bundle, r0, tried, names) or "Check %s %s: %s there, about %s seconds." % (
        where_text(r0), "next" if tries else "first", r0["word"], r0["seconds"])
    if len(plan) > 1:
        s += " If not, try %s (%s)." % (where_text(plan[1]), plan[1]["word"])
    return s


WORDS = ("very likely", "probably", "might be", "possibly")
TIME_RX = re.compile(r"\b(\d{1,2}[:.]\d{2}|\d{1,2} ?(?:am|pm)|(?:about |around )?(?:a|an|one|two|three|four|five|ten|fifteen|twenty|thirty|forty|\d+) (?:minute|minutes|hour|hours) ago|half an hour ago)\b")


def validate_say(say, places_said, plan, all_places, tried, facts):
    """SPEC 5.2, the same rules as validateSay in trace-sim. Returns (ok, why)."""
    s = str(say or "").strip()
    low, fx = s.lower(), str(facts or "").lower()
    if not s:
        return False, "empty"
    if len(s) > 400:
        return False, "too long"
    if re.search(r"\b(definitely|certainly|for sure|guaranteed?|no doubt|without a doubt)\b", low):
        return False, "says it is certain"
    if re.search(r"\b(it is|it's|its|they are|they're) (right |still |now )?(there|here|on|in|under|inside|at|behind)\b", low):
        return False, "says where it is as a fact"
    if re.search(r"\d+(\.\d+)? ?(%|percent|per cent)", low):
        return False, "gives a number instead of a word"
    if (re.search(r"\b(someone|somebody|he|she|your (mum|mom|dad|husband|wife|son|daughter|partner|carer|flatmate|roommate))\b[^.!?]*\b(moved|took|put|left|hid|dropped|grabbed)\b", low)
            or re.search(r"\b(mum|mom|dad|husband|wife|partner|carer|flatmate|roommate) (moved|took|put|left|hid|dropped|grabbed)\b", low)):
        return False, "says who moved it"
    in_plan = {r["zone"]: r for r in plan}
    done = [str(t).lower() for t in tried or []]
    aliases = zone_aliases(all_places)
    for sentence in re.split(r"(?<=[.!?])\s+", low):
        rest, said = " " + re.sub(r"[^a-z' ]+", " ", sentence) + " ", []
        for name, zid in aliases:
            rx = r"\b%s\b" % re.escape(name)
            if re.search(rx, rest):
                if zid not in said:
                    said.append(zid)
                rest = re.sub(rx, " ", rest)
        for zid in said:
            nm = next((p["name"].lower() for p in all_places if p["id"] == zid), zid)
            if zid not in in_plan and not any(t.startswith(nm) for t in done):
                return False, "mentions %s, which isn't in the plan" % nm
        w, used = " " + sentence + " ", []
        for word in WORDS:
            while word in w:
                used.append(word)
                w = w.replace(word, " ", 1)
        if re.search(r"\blikely\b", re.sub(r"\bunlikely\b", "", w)):
            return False, "says 'likely' instead of the word from the plan"
        allowed = {in_plan[z]["word"] for z in said if z in in_plan}
        for word in used:
            if word not in allowed:
                return False, "'%s' doesn't match the plan" % word
    for pc in places_said or []:
        r = in_plan.get(pc.get("zone"))
        if not r:
            return False, "place %s isn't in the plan" % pc.get("zone")
        if pc.get("label") in WORDS and pc["label"] != r["word"]:
            return False, "label for %s doesn't match the plan" % r["name"]
    for m in TIME_RX.finditer(low):
        t = re.sub(r"^(about|around) ", "", m.group(0))
        if t not in fx:
            return False, "gives a time ('%s') that isn't from the model" % m.group(0)
    return True, ""


# Where Trace is sending the person right now, for the sonar on /sim: {item, zone, name, state, t}.
TARGET = {}


def live_target(model, sessions):
    """Re-plan the walk from the newest model state (the /sim page posts it every few seconds).
    Called on every GET /api/nav. Re-routes only on a clear change: the camera sees the item somewhere
    else, or the place Trace sent you to has collapsed (< 5%) while another untried place is now the
    favourite (>= 40%). Places already checked are skipped. Same ranking and evidence as the
    conversation; the words come from prob_word(); no LLM, no new maths."""
    t = dict(TARGET)
    item, zone = t.get("item"), t.get("zone")
    if not item or not zone or t.get("state") != "searching":
        return None
    bundle = ((model or {}).get("items") or {}).get(item)
    if not bundle:
        return None
    S = bundle.get("search") or dict(S_DEFAULT)
    ses = (sessions or {}).get(item) or {"evidence": [], "tried": []}
    tried = {re.sub(r" \((quick look|not sure)\)$", "", n) for n in ses.get("tried") or []}
    seen = bundle.get("visible_now")
    plan = rank(apply_evidence(places_of(bundle), ses.get("evidence") or [], S), S)
    cand = [r for r in plan if r["name"] not in tried or r["zone"] == seen]
    if not cand:
        return None
    top, cur = cand[0], next((r for r in plan if r["zone"] == zone), None)
    name = bundle.get("item_name") or re.sub(r"-\d+$", "", item)
    if seen and seen != zone:
        top = next((r for r in plan if r["zone"] == seen), top)
        why = "Change of plan: the camera can see your %s on the %s now." % (name, top["name"].lower())
    elif top["zone"] != zone and (cur is None or cur["p"] < 0.05) and top["p"] >= 0.4:
        why = "Change of plan: %s now looks %s." % (top["name"], top["word"])
    else:
        return None
    TARGET.update({"zone": top["zone"], "name": top["name"], "t": time.time(), "replanned": True})
    if sessions is not None and item in sessions:
        sessions[item]["suggested"] = top["zone"]
    return why


def chat(message, history, model, sessions, mode="default", llm=None, now=None, budget=None):
    """One turn of trace.html's conversation. model: {"items": {id: bundle}} posted by the analytics
    page. sessions: a dict the server keeps between turns. Returns
    {"reply", "item", "state": "searching"|"found"|"idle", "chirp", "evidence", "source", "why"}."""
    now = time.time() if now is None else now
    bundles = (model or {}).get("items") or {}
    if not bundles:
        raise NoModel()
    for k in [k for k, s in sessions.items() if now - s["used"] > SESSION_IDLE or s.get("found")]:
        del sessions[k]
    said = str(message or "").strip()
    n = re.sub(r"\s+", " ", re.sub(r"[^a-z' ]+", " ", said.lower())).strip()
    current = max(sessions, key=lambda k: sessions[k]["used"]) if sessions else None
    want = item_in(said) or current
    if not want:
        for h in reversed(history or []):
            if h.get("role") == "user" and item_in(h.get("content", "")):
                want = item_in(h["content"])
                break
    if not want or want not in bundles:
        return {"reply": "Which one should I look for: your keys, your wallet or your glasses?", "item": None,
                "state": "idle", "chirp": None, "evidence": [], "source": "page", "why": ""}
    bundle = bundles[want]
    S = bundle.get("search") or dict(S_DEFAULT)
    ses = sessions.get(want)
    if ses is None or (item_in(said) and re.search(r"\bwhere\b", n)):
        ses = sessions[want] = {"evidence": [], "tried": [], "used": now, "last": ""}
    ses["used"] = now
    base = places_of(bundle)

    def plan_now():   # as rank(), but the place they just checked isn't suggested again straight away,
        # unless a phone sees the item there now: then again_text() asks for a careful second look
        p = rank(apply_evidence(base, ses["evidence"], S), S)
        last = re.sub(r" \((quick look|not sure)\)$", "", ses["tried"][-1]) if ses["tried"] else None
        if last and len(p) > 1 and p[0]["name"] == last and p[0]["zone"] != bundle.get("visible_now"):
            p[0], p[1] = p[1], p[0]
        return p
    plan = plan_now()

    def turn(reply, state="searching", evidence=(), source="page", why="", chirp=None, target=True):
        ses["last"] = reply
        top = plan[0] if plan and state != "found" else None
        ses["suggested"] = top["zone"] if (top and target) else ses.get("suggested")
        if target:          # the sonar on /sim follows where Trace sends the person
            TARGET.clear()
            TARGET.update({"item": want, "zone": top["zone"] if top else None, "name": top["name"] if top else None,
                           "state": state, "t": time.time()})
        if state == "found":
            ses["found"] = True
        return {"reply": reply, "item": want, "state": state, "chirp": chirp,
                "evidence": list(evidence), "source": source, "why": why}

    def found_at(zone, name):
        """The person has it. zone None: we don't know where they found it."""
        them = "them" if bundle.get("item_plural") else "it"
        kind = next((p["kind"] for p in base if p["id"] == zone), "zone")
        where = "" if zone is None else " with you" if kind == "person" else "" if kind == "outside" else " on the " + name.lower()
        ev = [{"zone": zone, "part": "all", "found": True, "sure": "high"}] if zone else []
        return turn("Great, you found %s%s. I'll remember that." % (them, where), "found", ev)

    # Trace's one best question (the page's model picks it by expected value of information)
    bq = bundle.get("best_question")
    if ses.get("q") and (YES.match(n) or NO.match(n)):
        q, yes = ses.pop("q"), bool(YES.match(n))
        ses["evidence"].extend(q["yes"] if yes else q["no"])
        plan = plan_now()
        return turn("Thanks. " + template(bundle, plan, 0))
    ses.pop("q", None)
    asked_for = re.search(r"\b(help me remember|ask me|any questions?|question)\b", n)
    if bq and asked_for:
        ses["asked"] = True
        ses["q"] = bq
        return turn(bq["text"], why="best question: about %s s saved on average" % bq.get("gain_s"))
    if False and bq and bq.get("worth_asking"):   # unprompted questions off for the demo: only "help me remember" asks
        ses["asked"] = True
        ses["q"] = bq
        return turn("Before you look: " + bq["text"], why="best question, saves about %s s on average" % bq.get("gain_s"))
    # the three simple commands never need the LLM
    if AGAIN.match(n) and ses["last"]:
        return turn(ses["last"])
    if NOT_THERE.match(n) and plan and ses["last"]:
        sug = next((r for r in plan if r["zone"] == ses.get("suggested")), plan[0])   # the place Trace last sent them to
        ev = {"zone": sug["zone"], "part": "all", "found": False, "sure": "high"}
        ses["evidence"].append(ev)
        ses["tried"].append(sug["name"])
        plan = plan_now()
        return turn(template(bundle, plan, len(ses["tried"]), ses["tried"]), evidence=[ev])
    if WHAT_HAPPENED.search(n) and bundle.get("story_text"):
        return turn(bundle["story_text"], target=False)
    if FOUND_IT.match(n):
        if plan and ses["last"]:
            return found_at(plan[0]["zone"], plan[0]["name"])
        return found_at(None, "")
    # "where are my keys?": no LLM needed to understand it. A phone sees it now: the template says so
    # at once; otherwise only the wording call is made
    where_q = bool(re.search(r"\bwhere\b", n)) and not NOT_FOUND.search(n) and not ses["tried"]
    if where_q and bundle.get("visible_now"):
        return turn(template(bundle, plan, 0))

    places = apply_evidence(base, ses["evidence"], S)
    res = ask(dict(bundle, places=places, already_checked=list(ses["tried"])), said, mode, llm=llm, budget=budget,
              parsed={"intent": "where", "item": want, "evidence": []} if where_q else None)
    for e in res.get("errors") or []:
        print("guide:", e)
    for ev in res["evidence"]:
        if ev["found"]:
            return found_at(ev["zone"], next((p["name"] for p in base if p["id"] == ev["zone"]), ev["zone"]))
    if res["intent"] == "found":      # "found them!" in other words, no place: where Trace sent them
        return found_at(plan[0]["zone"], plan[0]["name"]) if plan and ses["last"] else found_at(None, "")
    for ev in res["evidence"]:
        ses["evidence"].append(ev)
        nm = next((p["name"] for p in base if p["id"] == ev["zone"]), ev["zone"])
        ses["tried"].append(nm + {"low": " (quick look)", "medium": " (not sure)"}.get(ev["sure"], ""))
    plan = plan_now()
    fallback = template(bundle, plan, len(ses["tried"]), ses["tried"])
    if not res.get("say"):
        return turn(fallback, evidence=res["evidence"], source=res["source"])
    # places it may name besides the plan: those already checked, and where it was last seen / is seen now
    seen = [z for z in ((bundle.get("last_seen") or {}).get("zone") if isinstance(bundle.get("last_seen"), dict) else None,
                        bundle.get("visible_now")) if z]
    mentionable = list(ses["tried"]) + [p["name"] for p in base if p["id"] in seen]
    ok, why = validate_say(res["say"], res.get("places"), plan, base, mentionable, json.dumps(bundle) + " " + fallback)
    if not ok:
        return turn(fallback, evidence=res["evidence"], source=res["source"], why="answer rejected (%s), template used" % why)
    return turn(res["say"], evidence=res["evidence"], source=res["source"], chirp=res.get("chirp"))
