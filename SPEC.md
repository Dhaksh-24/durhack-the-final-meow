# Trace: project spec

Read this with `CLAUDE.md`. `CLAUDE.md` is the short brief; this is the full spec: what we're
building, how each part works, the exact data formats, the maths, and what "done" means for
each remaining task.

---

## 1. Product

**One line:** AirTags without the AirTag. Phones you already own watch a room; Trace works out
where your keys, wallet and glasses probably are, and talks you to them.

**The frustrating process we're reimagining:** looking for lost things. Everyone accepts it as
"just how it is"; people search the same three places over and over. For blind people and
people with dementia it costs independence, not just minutes.

**Users**
- Blind and low-vision people: guidance by voice with landmarks, no colours, no "over there".
- People with dementia and their carers: calm, one step at a time, memory prompts ("you were
  at the desk around two, then the kitchen").
- Everyone else: one-line answer.

**What it is not:** not a medical device, not surveillance. It never says who moved something,
never stores video, and only keeps where items and people were by zone.

**The demo (the thing everything serves):**
1. Four phones on stands watch the demo room. The 3D model of the room is on the big screen.
2. The presenter puts their keys down in view, walks around, and the keys end up hidden
   (pocketed and dropped, or covered) while the phones don't see.
3. The presenter is blindfolded (a teammate stays beside them). They ask "Where are my keys?"
4. Trace answers by voice with the most likely place; the 3D room shows the probability
   columns; the presenter says "not there" and the columns move; Trace guides them to the keys.
5. Close on the numbers: simulated days, search time vs the "last seen" rule, calibration.

**Judging fit**
- Accenture: real pain, LLM used where it's genuinely good (language, common sense, conversation),
  working prototype, responsible-AI story (privacy, honesty, no overclaiming).
- Marshall Wace ("New Worlds"): an exact probabilistic model (HMM), Bayesian search theory,
  Monte Carlo evaluation, calibration, information-gain questions. Elegant because each piece
  has one job and the maths is exact rather than approximated.

---

## 2. Architecture

```
 phones (DroidCam / IP Webcam)            laptop                                browser
 ─────────────────────────────    ───────────────────────────────    ─────────────────────────────
 video ─────────────────────────▶ multicam.py                         trace-sim/index.html
                                   ├ baseline photo per camera ──────▶  map the room (homography)
                                   ├ detect.py (YOLO-World/YOLOE)      ├ zone map ─▶ 3D room
                                   ├ CLIP identity across cameras      ├ HMM per item (model)
                                   ├ zones (named, per camera)         ├ search planner
                                   └ events.jsonl ──▶ bridge.py ──ws──▶ ├ LLM agent + tools
                                                                        ├ voice in / out
                                                                        └ live metrics
```

- **Cameras / mapping stream** owns everything left of `events.jsonl`.
- **Model / LLM stream** owns the page.
- `events.jsonl` and the zone map are the only contracts between them (section 3).
- For the demo the page runs **locally** (opened from the repo) so it can reach the bridge on
  `ws://localhost:8765`. The hosted copy is for sharing and simulation only.

---

## 3. Contracts

### 3.1 Zone map
```json
{
  "name": "Demo room", "width": 6.0, "height": 5.0,
  "zones": [
    {"id": "sofa", "name": "Sofa", "type": "sofa", "camera": true,
     "cover": 0.45, "check": 20, "x": 0.8, "y": 0.6, "w": 2.2, "h": 0.9}
  ]
}
```
- Metres. Origin is the back-left corner of the room; x along the back wall, y towards the front.
- `type`: one of `sofa bed chair counter shelf hook desk table` (drives the 3D model and habits).
- `camera`: a phone watches this zone. `cover`: chance an item put here ends up hidden
  (under a cushion, in a coat). `check`: seconds a person needs to search it.
- `id` must equal the zone name `multicam.py` uses, lower-case and hyphenated (`slug(name)`).

### 3.2 Events (`events.jsonl`, one object per line)

(The running system writes these to `trace-events.jsonl`: `multicam.py` keeps its own older log in `events.jsonl`.)
```json
{"time": "14:05:10", "t": 1794696414.7, "item": "keys-1", "category": "keys",
 "zone": "sofa", "event": "appeared", "confidence": 0.82, "camera": "phone-1",
 "floor": [1.5, 2.1]}
```
| field | required | meaning |
|---|---|---|
| `time` / `t` | yes | local clock string and unix seconds |
| `item` | for item events | stable id, e.g. `keys-1`; `category` is keys / wallet / glasses |
| `zone` | yes | zone id from the map, or `null` if outside every zone |
| `event` | yes | `appeared`, `moved`, `disappeared`, `person_here`, `person_left`, `something_changed` |
| `confidence` | for item events | detector confidence 0–1 (calibrated later, see 4.3) |
| `camera` | yes | camera id |
| `floor` | optional | metres, from `baseline/calibration.json` |

Rules for producing them (cameras stream):
- `appeared`: seen in 2 of the last 3 analysed frames in a zone where it wasn't.
- `disappeared`: not seen for 4–5 analysed frames in its zone (or its spot matches the
  baseline again, once change detection exists).
- `moved`: disappeared from one zone and appeared in another within 60 s.
- `person_here` / `person_left`: a person arrives at / leaves a zone (no face or identity, ever).
  multicam finds people with a `person` prompt in the same YOLO-World pass as the items; a
  person is at the one zone their box covers most (or their feet are in), and has left it after
  4 s without. `--no-people` falls back to movement inside the zone.
- `something_changed`: a change in a zone that the detector couldn't name (e.g. a cushion
  moved while someone was there). Useful evidence that something may now be covered.
- Worn glasses seen on a face are logged as `{"event": "appeared", "zone": "person"}`.

### 3.3 Calibration (`baseline/calibration.json`)
Exported by the page. Per camera: `camera_id`, `image_w`, `image_h`, `image_to_floor` (3×3,
row-major). Floor point of a box = `H · [cx, y2, 1]` divided by the third component.

---

## 4. The model (in `trace-sim/index.html`, between `CORE START` and `CORE END`)

### 4.1 State
Per item, a probability vector over `2Z + 2` states: for each zone *z*, `z-visible` and
`z-covered`; plus `person` (in hand or pocket) and `outside` (car, bag, another room).

### 4.2 Predict (each 10 s tick)
Items only move when a person moves them.
- Person at zone *i*: mass in `i-visible` / `i-covered` moves to `person` at rate `pick`.
  Mass in `person` drops into zone *i* at rate `drop × habit(item, i) / max habit`, split
  `1 − cover` visible and `cover` covered. Covered becomes visible at a small rate.
- Person leaves the room: all `person` mass → `outside`. Returns: `returnWith` of `outside` → `person`.
- Mix in 0.05% of the prior each tick (the model admits it can be wrong).

### 4.3 Update on what the phones saw
Today (clean simulator): for each watched zone *j*, a sighting there multiplies `j-visible`
by `D_vis` (0.9), `j-covered` by `D_cov` (0.03), every other state by ~0; no sighting
multiplies `j-visible` by `1 − D_vis` and `j-covered` by `1 − D_cov`.

**Task 1 replaces the sighting update with a likelihood that allows false alarms:**
`P(sighting at j with confidence c | item in j-visible) = D_vis · g(c)` and
`P(sighting at j | item not at j) = f · h(c)`, where `f` is the false-alarm rate per check
and `g`, `h` come from the maths teammate's calibration (confidence histograms for true vs
false detections). A 0.9 sighting then moves the belief a lot, a 0.1 sighting barely.

### 4.4 Search
- Chance of finding it if you check place *k*: `P(k-visible) · S_vis + P(k-covered) · S_cov`
  (`S_vis` 0.95, `S_cov` 0.75); pockets 0.95; outside 0.8.
- Rank places by that chance divided by `check` seconds (Bayesian search theory; optimal
  order for a stationary target with independent checks).
- "Not there": multiply the checked place's states by `1 − S` and renormalise.
- Evidence from speech carries a confidence: "definitely checked" uses `S` as is,
  "I think I looked" uses `0.5 · S`, "had a quick look" `0.3 · S`.

### 4.5 Evaluation (Monte Carlo)
`runEpisode()` plays a random day, loses one item at a moment no phone sees it, then searches
with Trace and with the baseline (last seen, then habits, then pockets, then outside).
Report: found on first check, within three checks, mean checks, mean seconds, and calibration
(top-guess probability vs hit rate in bins). Always label results as simulated.

### 4.6 Most likely story (task 4)
Viterbi over the same HMM from the start of the day to now gives the single most likely
sequence of states, e.g. `desk 13:58 → person 14:05 → counter 14:06`. Turn it into at most
three steps with times for the LLM to narrate.

---

## 5. The LLM agent

**Rule: the LLM decides what to do and how to say it; the model decides what's true.**

### 5.1 Tools
| tool | input | returns |
|---|---|---|
| `where_is` | `{item}` | last seen, top 4 places with probability and seconds, already checked |
| `record_check` | `{item, place, part: "visible"\|"covered"\|"all", found: bool, sure: "high"\|"medium"\|"low"}` | the updated plan |
| `best_question` | `{item}` | the yes/no question with the largest expected drop in search time, and what each answer would do |
| `story` | `{item}` | up to three timed steps (Viterbi) |
| `landmarks` | `{zone}` | e.g. "by the window, left of the lamp" |
| `chirp` | `{zone}` | plays a sound on that zone's phone |

At most 2 tool calls per reply (latency). Temperature low.

### 5.2 Output
```json
{"say": "Not on the sofa. Try the coat hook next: you were there about twenty minutes ago.",
 "places": [{"zone": "hook", "label": "LIKELY"}], "chirp": null}
```
Validation in code before speaking: every place mentioned exists in the current plan; the
probability word matches the number (below); no "it is there", no "definitely", no person
named as having moved anything. On failure, use the template sentence the page already builds.

Probability words, decided in code: ≥ 0.8 "very likely", 0.6–0.8 "probably", 0.4–0.6 "might be",
< 0.4 "possibly". Times come from code ("about twenty minutes ago"), never from the LLM.

### 5.3 Modes
Same facts, different delivery: `blind` (landmarks, steps, no colours), `dementia` (one step,
calm, repeat on request, memory prompts from `story`), `default` (one line).

### 5.4 Voice
Browser Web Speech API (`SpeechRecognition` in, `speechSynthesis` out), Chrome. Push-to-talk
button plus a typed fallback. Speak short sentences.

### 5.5 Provider
Not decided yet: whichever API key the team has. Keep the key in `.env` / a local config,
never in the repo or the hosted page.

---

## 6. Detection pipeline (cameras stream)

Target pipeline (see the "optimised pipeline" notes):
1. Fixed phones, 1–1.5 m from the surface at ~45°; a lamp for light; bright keyring.
2. Baseline photo per camera at start-up (done in `multicam.py`).
3. Change detection against the baseline once motion settles (1.2 s); ignore blobs < tiny or
   > 30% of the zone.
4. Crop each changed blob from the full-resolution frame, run the detector on the crop.
5. YOLOE with visual prompts of the user's own keys / wallet / glasses (register by showing).
6. Drop any box > 15% of the frame; per-object thresholds from the benchmark (`targets.py`).
7. Emit events per 3.2.

---

## 7. Responsible AI (say this in the pitch, build it in the code)
- Privacy: positions by zone, not video. Baseline photos and memory stay on the laptop and are
  git-ignored. The repo is private. No faces, no identities, no "who moved it".
- Honesty: calibrated probabilities, words chosen by code, "last seen" not "is", visible
  uncertainty. Simulated results always labelled.
- Safety in the demo: a teammate beside the blindfolded presenter; clear floor.
- Limits we state: small objects are hard for cameras; the model copes with misses but false
  alarms need calibration; it's an aid, not a guarantee.

---

## 8. Remaining work and what "done" means

Times are Sunday morning unless noted. Owners: **M** = model/LLM (Dhaksh), **C** = cameras,
**Q** = maths.

| # | task | owner | done when |
|---|---|---|---|
| 1 | False alarms in simulator + likelihood update (4.3) | M + Q | simulator emits false sightings at a set rate; with them on, Trace still beats the baseline and calibration bins stay within ±10 points; a single false sighting no longer collapses the belief |
| 2 | `multicam.py` events per 3.2 | C | a session produces appeared / moved / disappeared / person_here lines with zone ids matching the map and a confidence |
| 3a | Replay `events.jsonl` in the page | M | uploading a recorded file drives the HMM and 3D room through the session in order |
| 3b | Live bridge (`bridge.py`, websockets, tails the file) | M or C | page opened locally updates within 1 s of a new event |
| 4 | Viterbi story; LLM agent with tools; voice | M | "Where are my keys?" spoken → validated spoken answer in < 3 s; "not there" updates and re-answers |
| 5 | Detection improvements (section 6) | C | 20 placements of the real keys in the demo room: report hits and false alarms per placement |
| 6 | Confidence calibration | Q | function `confidence → (g, h)` from ≥ 40 labelled placements, plus one chart |
| 7 | Stress test | Q | slide: results when habits, miss rate and pocketing are wrong |
| 8 | Map the demo room | all | baseline photos → page → zone map in metres; `calibration.json` saved |
| 9 | Rehearse the demo ×5, timed | all | 5 runs, each finds the keys; fallback (text mode, replay) tested once |
| 10 | Pitch + slides | all | problem → live demo → 3D room → numbers → responsible AI, under the time limit |

**Order:** 2 and 1 tonight (they unblock everything), 3a, 8, 4, then 3b, 5–7 in parallel,
9–10 last. If time runs short, cut: best question, chirp, the bedroom demo. Never cut 3, 8 or 9.

---

## 9. How to work on the page
- It's one HTML file; keep it that way (no build step, three.js from cdnjs).
- Test the core in Node:
  ```js
  const vm = require("vm"), fs = require("fs");
  const js = fs.readFileSync("trace-sim/index.html", "utf8").match(/<script>\n\(function([\s\S]*)<\/script>/)[0].slice(8, -9);
  const m = { exports: {} };
  vm.runInNewContext(js, { module: m, Math, console, Float64Array, JSON, Set, Object, String, isFinite, Infinity, RegExp, Array });
  const C = m.exports; C.setupMap(JSON.parse(JSON.stringify(C.DEFAULT_MAP)));
  const r = Array.from({ length: 500 }, () => C.runEpisode());
  ```
- Change the core first, test it in Node, then the UI.
- Never commit photos, `memory/`, `baseline/`, `events.jsonl`, API keys.
