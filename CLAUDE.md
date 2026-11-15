# Trace (DurHack 2026)

Start with `HANDOVER.md` (how to run it, demo set-up, honest status). `python run_trace.py --demo`
checks the page without cameras.

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
  cameras with CLIP fingerprints, logs to `events.jsonl`. Saves a baseline photo per camera to
  `baseline/` at start-up and loads `baseline/calibration.json` to put sightings on the floor
  in metres. Benchmarking work on prompts and thresholds per object is on `main`.
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
- **Mapping from photos:** click a reference rectangle on the floor (homography), then the
  front edge of each piece of furniture; outputs the zone map and per-camera image→floor matrices.
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
Done: false alarms, multicam → trace-events.jsonl, replay, live bridge, camera feeds beside the 3D
room, Ask Trace (voice + LLM wording + validation + beep), run_trace.py, presentation view, room.json.
1. Real-room test: phones on stands, real keys, `python run_trace.py`; fix what the camera can't see.
2. Confidence calibration from ≥40 real placements → `setCalibration(...)` (maths teammate).
3. Rehearse the blindfold demo ×5; record a session as the stage fallback.
4. Nice to have: most likely story (Viterbi) for the LLM to tell; best yes/no question.
