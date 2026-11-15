# Trace: handover

Read this first. It's everything you need to run Trace, set up the demo room, and know what's
left. Detail lives in `SPEC.md` (design) and `RUN_LIVE.md` (the live chain, piece by piece).

**Trace in one line:** phones watch a room; a probability model works out where your keys,
wallet and glasses probably are; Trace talks you to them, and shows it all on a 3D model of the room.

---

## 1. Set up once (5 min)

Branch: **`demo`**. Python 3.10–3.12, in the repo folder (Windows PowerShell shown; on
macOS/Linux use `python3` and `cp` instead of `copy`). Use a laptop where `multicam.py`
already runs if you can (its Python is already set up).

```powershell
git fetch origin
git checkout demo
git pull
pip install -r requirements.txt
```

`--demo` (no cameras) needs only `pip install websockets opencv-python numpy`. The offline voice
is a separate, optional install (`pip install "kokoro>=0.9.4" soundfile sounddevice`); without
it Trace uses Microsoft's online voice, then the browser's.

Optional, for the LLM wording: `copy .env.example .env` and put the key after `GEMINI_API_KEY=`.
Without it, Trace still answers with its built-in sentences. **Never commit `.env`.**

---

## 2. Run it: one command

| You want to… | Run | What opens |
|---|---|---|
| Check the page works, no cameras | `python run_trace.py --demo` | the 3D room; a fake session plays every minute |
| Test with a phone video instead of live phones | `python run_trace.py --video keys_test.mp4` | the camera window + the page |
| Run the real thing | `python run_trace.py` | the camera window + the page |
| Show the judges | add `--present` | a clean page: 3D room, cameras, where things are, Ask Trace |

For the real thing, first `copy cameras.example.json cameras.json` and put in each phone's
DroidCam address (the phone app shows it, e.g. `http://192.168.1.23:4747/video`).
`cameras.json` is git-ignored: never commit it.

`run_trace.py` only takes the options above. The camera options in section 4 (`--no-people`,
`--person-conf`, `--min-prob`, `--zone-map`) belong to `multicam.py`: to use them, run
`python multicam.py <options>` (add `--api-host 0.0.0.0` to share) and open
`http://localhost:8000/sim?live` and `http://localhost:8000/` yourself.

It opens **two tabs**, and both matter:
- **http://localhost:8000/sim**: the model and the 3D room. **Keep it open** (in Real events, which
  `run_trace.py` does for you): it runs the probability model and sends it to the server. Put it on
  the big screen. "Presentation view" / "Show everything" switches between the clean judge view and
  the full page.
- **http://localhost:8000/**: `trace.html`, where you talk to Trace ("Hey Trace", the mascot, voice).
  Its answers come from the model running in the other tab.

**Two laptops (recommended for the demo):** run `python run_trace.py --share --present` on the
laptop with the cameras. It prints two addresses like `http://192.168.1.20:8000/sim?live&view=present`.
Open them on the second laptop (same Wi-Fi or hotspot): `/sim` on the big screen, `/` for talking
to Trace. The camera laptop then only does detection. `--share` makes the cameras' live video
visible to anyone on that network, so use your own hotspot, not the venue Wi-Fi. If the second laptop
can't connect, the camera laptop's firewall is blocking ports 8000 and 8765: allow Python when
Windows asks, or temporarily allow them.

---

## 3. Set up the demo room (15 min, once)

1. **Put the phones on stands**, 1–1.5 m from the furniture they watch, pointing down at about 45°
   (for the one-phone demo in section 5: taped high, pointing straight down). Lamp on. Bright
   keyring on the keys.
2. **Draw the room.** On the page: "Draw the room (top view)". Type the room's width and depth, add a
   phone photo or video frame, click the 4 floor corners (back-left, back-right, front-right,
   front-left), drag a box over each piece of furniture, name it (**Sofa, Desk, Coat hook…**), tick
   the ones a phone watches, press **Build the room**. It's saved for every browser (`room.json`).
3. **Name the camera zones the same way.** In the multicam window, drag a box over each piece of
   furniture on each camera and type its name in the terminal: **exactly the same names** as step 2.
   This is the one thing that breaks everything if it's wrong.
4. **Test:** put the keys on the sofa. Within ~2 s the event log says `appeared keys-1 sofa` and the
   sofa's column rises. Move them: `moved`. Ask "where are my keys?".

---

## 4. If it doesn't work

| Symptom | Cause | Fix |
|---|---|---|
| Page won't open | the server isn't running | look for an error in the `run_trace.py` terminal |
| No green boxes on the keys in the camera window | the detector can't see them, or the CLIP check rejects them (keys need 0.86) | closer, more light, bright keyring; then `python multicam.py --min-prob keys=0.7` |
| No `person_here` when you stand at the box | a phone looking straight down sees a head and arms, not a person | `python multicam.py --person-conf 0.15`, or `--no-people` (movement in the zone instead) |
| Items from an earlier rehearsal show up in `trace.html` | `multicam.py` remembers items in `memory/` | quit multicam, delete `memory/` (it holds photos: never commit it), start again |
| Green boxes, but nothing happens on the page | zone names don't match the room | read the note under the event log ("zones not in the room map") and rename |
| "Live: couldn't connect" | the bridge didn't start | `python bridge.py --file trace-events.jsonl` in another terminal |
| Trace answers but doesn't speak | browser voice blocked | click anywhere on the page once, then ask again |
| Trace's answers are vague / ignore the cameras | the `/sim` tab isn't open or isn't live | open `http://localhost:8000/sim?live`; `http://localhost:8000/api/model` should show items |
| Wrong room on the page | an old room is loaded | "Draw the room" → Build, or "Load a different room" |

---

## 5. The demo, step by step (one top-down phone, keys + glasses)

**Set-up (before judges arrive):** phone taped high, pointing straight down at the taped rectangle;
the box ("Table" or "Sofa", same name in "Draw the room" and in the multicam window) inside it;
a cloth or box flap the keys disappear under completely. Keys and glasses on the box, apart.
Camera laptop: `python run_trace.py --share --present`. Display laptop: the two printed addresses,
`/sim` on the big screen (optionally `&seed=morning` for a labelled simulated heat map).

| Time | Who | Action | What the room / Trace does |
|---|---|---|---|
| 0:00 | Presenter | "Everyone loses things. For blind people and people with dementia it costs independence." | – |
| 0:20 | Presenter | Points at the screen: keys and glasses on the box | both columns tall: "the camera can see them" |
| 0:35 | Presenter | Sits at the box, slides the keys under the cloth, walks away | keys' column splits: **under the box** and **with you**; glasses don't move |
| 0:50 | Teammate | Blindfolds the presenter, stays beside them | – |
| 1:00 | Presenter | "Hey Trace, where are my keys?" | "Last seen on the table a minute ago. Check your pockets first: about 5 seconds. If not, under the table." |
| 1:15 | Presenter | Checks pockets: "Not there" | columns move: "Check under the table next: very likely there." Table phone beeps if set up |
| 1:30 | Presenter | Finds the keys: "Found them!" | done |
| 1:40 | Presenter | "Where are my glasses?" | "The camera can see your glasses on the table right now." |
| 1:50 | Presenter | Back to slides: how it works, results, responsible AI | – |

**Checks in each rehearsal:** `disappeared keys-1` appears in the event log within ~4 s of hiding;
you appear as `person_here` at the box (if not, see section 4: `--person-conf 0.15` or
`--no-people`, both `multicam.py` options); the glasses never disappear.

**Fallback if live fails on stage:** "Real events" → "Load recorded events" → the `backup-run.jsonl`
you recorded during a good rehearsal (it's git-ignored: copy it to the display laptop by USB
or chat, or `git add -f` it). Say "this is our recorded run from this morning".

---

## 6. Which files matter

| File | What it is |
|---|---|
| `run_trace.py` | starts everything |
| `multicam.py` | cameras: reads the phones, detects items, writes `trace-events.jsonl`, serves the page |
| `detect.py` + `verifier.json` | the detector: YOLO-World proposes boxes, a CLIP classifier (`verifier.json`, built by `train_verifier.py` with `pull_lvis.py`) keeps the real keys / glasses |
| `events_adapter.py` | turns raw detections into events (2 of 3 frames in, 5 missed out, giant boxes dropped) |
| `bridge.py` | streams `trace-events.jsonl` to the page |
| `trace_api.py` | the web server: page, live video, voice, `/api/ask`, `/api/room`, phone beeper |
| `guide.py` | the LLM: turns what you say into evidence, words the answer (Gemini, 2 calls max) |
| `trace-sim/index.html` | `/sim`: runs the model, 3D room, cameras, metrics (keep it open) |
| `trace.html` | `/`: the front end you talk to ("Hey Trace", voice, mascot) |
| `tts.py` | the voice: Microsoft neural (edge-tts) online, Kokoro offline |
| `whereabouts.py` + `room.py` | backup answers: multicam's own person trail (where you stopped, how the item went missing). `/api/ask` uses it only when the `/sim` tab isn't posting the model. Its places live in `places.json`; it uses `targets.py` (item names) and `room.py build` uses `scene.py` |
| `snap.py` | optional: one photo per phone into `baseline/`, for drawing the room |
| `simulate.py`, `test_guide.py`, `trace-sim/test-core.js` | tests (no camera, no network) |

`archive/`: earlier experiments, not used by the demo (detector benchmarking `bench.py`,
`pull_data.py`, `results/`; the old one-camera chain `live_scene.py`, `test_scene.py`,
`test_images/`; the Task Tree board). Run them from the repo root with the root on the path
(`PYTHONPATH=.`), since they import `targets.py` / `scene.py`.

Never commit: `memory/`, `baseline/`, `events.jsonl`, `trace-events.jsonl`, recorded runs,
phone videos, `room.json`, `zones.json`, `places.json`, `timeline.jsonl`, `cameras.json`, `.env`.
They're git-ignored (`*.jsonl`, `*.mp4`…); keep it that way: some contain photos of people.
Before `git add`, run `git status` and check nothing from that list appears.

---

## 7. Status (honest), 15 Nov 02:30

| Part | State | Tested how |
|---|---|---|
| Probability model, false alarms, search ranking | **Done** | `node trace-sim/test-core.js`: 1,000 simulated days × 4 false-alarm rates, all pass |
| 3D room, drawing the room from a photo/video, metrics | **Done** | in the browser |
| Events from cameras (`multicam.py` → `trace-events.jsonl`) | **Done, not tried on real phones** | fake camera frames: correct appeared / moved / person events |
| Live link to the page (bridge, Connect live, camera feeds beside the 3D room) | **Done** | end to end with fake cameras and with `--demo` |
| Talking to Trace (`trace.html` + `/sim` model + LLM wording, validation, phone beep) | **Done, not tried with a real key** | `python test_guide.py` (rules path, no network) |
| One-command start, presentation view, shared room | **Done** | `run_trace.py --demo` end to end |
| **Your real keys seen by a phone in the demo room** | **Not tested: the biggest risk** | benchmark said keys are the hardest object; test tonight / first thing |
| Real phones → page → voice, start to finish | **Not tested** | needs the demo room |
| Confidence calibration from real placements | **Not done** (placeholder curves) | maths teammate |
| Most likely story (Viterbi), best question by value of information | **Done** | `node trace-sim/test-core.js` |
| Blindfold rehearsal ×5, slides, pitch | **Not done** | everyone, morning |

**Bottom line:** the software is built and each piece is tested; what's unproven is the physical
part: phones seeing real keys in the real room. Everything above is about 85% of the build; the
remaining 15% is real-room testing, rehearsal and the pitch, which is a morning's work.

## 6. Live walking directions (top-down camera)

While Trace is sending you somewhere, `nav.py` uses the top-down camera to say "bear left",
"keep going straight", "stop, reach down", and draws a yellow arrow from you to the target on the
camera window (green when you arrive). It aims at the item's own box if the camera sees it
(glasses on the desk), otherwise at the zone Trace sent you to (keys under the cloth on the sofa).

- Needs person detection: don't use `--no-people`; if it says "I can't see you", try `--person-conf 0.1`.
- Left/right comes from the way you walked in the last ~2 s: take two steps forward first.
- `--view-width-m 2.5` = metres of floor across the camera image (for "about 2 steps").
- If left and right come out swapped (mirrored camera), add `--nav-mirror`. Test once before judging.
- Speaks through `http://localhost:8000/` (trace.html) after you ask "where are my glasses?".

**The live loop.** Every directions poll (about once a second) re-plans from the newest model state
(`guide.live_target`): if the camera sees the item somewhere else, or the place you were sent to
drops below 5% while another unchecked place reaches 40%, Trace says "Change of plan: …" and walks
you there. After 8 s standing at the target it asks "Anything there? Say found them, or not there",
and your answer updates the model and picks the next place. Places you've checked aren't re-suggested.
