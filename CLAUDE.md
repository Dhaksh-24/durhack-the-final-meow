# Trace (DurHack 2026)

Full spec: `SPEC.md` (product, architecture, data contracts, the maths, the LLM agent,
and what "done" means for each remaining task). Read it before starting a task.

"AirTags without the AirTag." Phones placed around a room watch zones (sofa, desk, shelf…).
Trace keeps a probability for where each important item (keys, wallet, glasses) is, and an
LLM guides the person to it by voice. For blind people, people with dementia, and everyone.
Demo: the presenter's keys are hidden, they're blindfolded, and Trace talks them to the keys.

Challenges: Accenture (reimagine a frustrating process with AI and LLMs; judged on problem,
innovation, effective use of AI, impact, prototype, responsible AI) and Marshall Wace ("New
Worlds"; quant judges who value complexity, intelligence and elegance).

## Team streams
- **Cameras / mapping** (branch `adelver`): `multicam.py` reads several phone streams
  (DroidCam / IP Webcam), detects items with YOLO-World via `detect.py`, keeps identity across
  cameras with CLIP fingerprints. `trace_events.py` turns its per-frame sightings into
  `events.jsonl` (SPEC 3.2) and adds `floor` metres from `baseline/calibration.json`.
  `snap.py` saves a baseline photo per camera to `baseline/`. Tests: `py test_trace_events.py`.
  Benchmarking work on prompts and thresholds per object is on `main` (`bench.py`, `pull_lvis.py`).
- **Detection:** `detect.Detector` = YOLO-World at a loose threshold proposes boxes (≤ 40% of
  the frame, top 8), then a CLIP ViT-B/32 classifier over {keys, glasses, other}
  (`verifier.json`, built by `train_verifier.py` from LVIS + Wikimedia Commons + YOLO's own
  false alarms, blended with zero-shot) keeps a box (wallets count as look-alikes) only if "other" loses and the class clears
  a precision-tuned threshold. Returned conf = the classifier's probability. Own photos in
  `data/own/<class>/` and `py train_verifier.py --redo=embed` tune it to the demo items.
- **Room mapping:** `py map_room.py` sends the `baseline/` photos to Gemini once and writes
  `room.json` (zone map with landmarks, for the page: "Open room.json") and `zones.json`
  (multicam's per-camera zone boxes, same names). Code validates every value.
- **Model / LLM** (branch `trace-model`): `trace-sim/index.html`, a single self-contained page.
- **Maths**: calibrate detector confidence into evidence strength; stress-test the results.

## trace-sim/index.html
One file, no build step. three.js r128 from cdnjs. The core (everything between
`CORE START` and `CORE END`) has no DOM and runs in Node: when there's no `window`,
the script does `module.exports = CORE`, so you can `vm.runInNewContext` it and call
`setupMap`, `runEpisode`, `filterStep`, `rank` for tests.

- **World simulator:** one person walks between zones and picks up, pockets, drops and covers
  items; leaves the room sometimes (items with them go "outside"). 10 s per tick.
- **Model:** hidden Markov model per item over states {zone visible, zone covered} × zones +
  "with the person" + "outside". Predict (items only move with the person) then update on
  what each phone did and didn't see. Solved exactly with the forward algorithm.
- **Search:** rank places by P(found here) / seconds to check; "not there" multiplies by the
  miss rate and renormalises.
- **Evaluation:** Monte Carlo over simulated days vs "last seen, then habits" baseline,
  plus a calibration table. Last run (10% false alarms): first-check success 52% vs 20%, 2.4 vs 4.7 checks.
- **3D view:** low-poly room built from the zone map (furniture type inferred from the name).
- **Mapping:** open `room.json` from `map_room.py` (Gemini); or draw the room from above;
  or (advanced) click a reference rectangle on the floor in each photo (homography) for
  per-camera image→floor matrices. Zones may carry a `landmark` sentence for guidance.
- **Live metrics** and "where should the next phone go?" (Monte Carlo per blind spot).

## Contracts (keep both sides to these)
Zone map:
```json
{"name": "Demo flat", "width": 10, "height": 7,
 "zones": [{"id": "sofa", "name": "Sofa", "type": "sofa", "camera": true,
            "cover": 0.45, "check": 20, "x": 0.8, "y": 0.6, "w": 3.4, "h": 1.6}]}
```
Metres from the back-left corner; `type` one of sofa, bed, chair, counter, shelf, hook, desk, table.

Event (one JSON object per line):
```json
{"time": "14:05:10", "item": "keys-1", "zone": "sofa", "event": "appeared|moved|disappeared|person_here", "confidence": 0.8}
```

## Rules we decided on
- The LLM never does maths. It parses speech into evidence, supplies common-sense priors,
  and words the answer. Probabilities come from the model. Agent design: one LLM with tools
  (`where_is`, `record_check`, `best_question`, `story`, `landmarks`, `chirp`), 1–2 tool
  calls per reply for latency.
- Honest wording: "last seen", "probably"; never "it is there"; never say who moved
  something. Probability → words is done in code; validate LLM output, fall back to a template.
- Privacy: never commit `memory/`, `baseline/`, `events.jsonl`, `zones.json`, `cameras.json`
  (they contain photos of people). The repo should be private.
- Results from the simulator are simulated; say so whenever they're shown.

## Next, in priority order
1. ~~False alarms~~ done: the simulator emits false sightings (`FALSE_RATE`, per watched zone per
   step) with a confidence; the filter multiplies `z-visible` by `D_vis·L + 1 − D_vis` with
   `L = g(c) / (f·h(c))`. `g`, `h` are placeholder Betas: swap in the maths calibration with
   `setCalibration({g, h, sampleTrue, sampleFalse})`. Tests: `node trace-sim/test-core.js`.
2. ~~`multicam.py` events~~ done (`trace_events.py`): appeared (2 of 3 frames) / disappeared
   (4 missed) / moved (within 60 s, with `from`) per item per zone, person_here / person_left
   from MOG2 motion in a zone, zone id = `slug(name)`, detector confidence, `floor`. Zones drawn
   in multicam must be named as in the page's map. Not yet: `something_changed` (needs task 5).
3. Connect real events to the page: ~~replay an `events.jsonl`~~ done ("Replay a recorded
   events.jsonl" in the Events card; `replayFromEvents` in the core; try
   `trace-sim/sample-events.jsonl`). Next: live via a small local bridge (the hosted page
   can't reach the laptop, so run the page locally).
4. Most likely story (Viterbi) for the LLM to tell; LLM wiring; voice in/out in the browser.
5. Detection: change detection on fixed phones + crop + YOLOE visual prompts of the user's
   own items; drop boxes larger than ~15% of the frame.
