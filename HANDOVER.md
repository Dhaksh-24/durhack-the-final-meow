# Trace: handover

Read this first. It's everything you need to run Trace, set up the demo room, and know what's
left. The design is in `docs/SPEC.md`; section 8 below runs the live chain piece by piece.

**Trace in one line:** phones watch a room; a probability model works out where your keys and
glasses probably are; Trace talks you to them, and shows it all on a 3D model of the room.
(The detector looks for keys and glasses only; the page still has a wallet for the simulator.)

---

## 1. Set up once (5 min)

Branch: **`demo`**. Windows PowerShell, in the repo folder. Use a laptop where `multicam.py`
already runs if you can (its Python is already set up).

```powershell
git fetch origin
git checkout demo
git pull
pip install -r requirements.txt websockets
```

If `kokoro` (the offline voice) fails to install, ignore it: Trace uses the browser's voice instead.

Optional, for the LLM wording: create a file `.env` in the repo folder containing
`GEMINI_API_KEY=your-key`. Without it, Trace still answers with its built-in sentences.
**Never commit `.env`.**

---

## 2. Run it: one command

| You want to… | Run | What opens |
|---|---|---|
| Check the page works, no cameras | `python run_trace.py --demo` | the 3D room; a fake session plays every minute |
| Test with a phone video instead of live phones | `python run_trace.py --video keys_test.mp4` | the camera window + the page |
| Run the real thing | `python run_trace.py` | the camera window + the page |
| Show the judges | add `--present` | a clean page: 3D room, cameras, where things are, Ask Trace |

For the real thing, first copy `cameras.example.json` to `cameras.json` and put in each phone's
DroidCam address (the phone app shows it, e.g. `http://192.168.1.23:4747/video`).

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

1. **Put the phones on stands**, 1–1.5 m from the furniture they watch, pointing down at about 45°.
   Lamp on. Bright keyring on the keys.
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
| No green boxes on the keys in the camera window | the detector can't see them | closer, more light, bright keyring, or use glasses |
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
you appear as `person_here` at the box (if not, `--no-people`); the glasses never disappear.

**Fallback if live fails on stage:** "Real events" → "Load recorded events" → the `backup-run.jsonl`
you recorded during a good rehearsal. Say "this is our recorded run from this morning".

---

## 6. Which files matter

| File | What it is |
|---|---|
| `run_trace.py` | starts everything |
| `multicam.py` | cameras: reads the phones, detects items, writes `trace-events.jsonl`, serves the page |
| `detect.py` | the detector (YOLO-World); swap the model here |
| `events_adapter.py` | turns raw detections into events (2 of 3 frames in, 5 missed out, giant boxes dropped) |
| `bridge.py` | streams `trace-events.jsonl` to the page |
| `trace_api.py` | the web server: page, live video, voice, `/api/ask`, `/api/room`, phone beeper |
| `guide.py` | the LLM: turns what you say into evidence, words the answer (Gemini, 2 calls max) |
| `trace-sim/index.html` | `/sim`: runs the model, 3D room, cameras, metrics (keep it open) |
| `trace.html` | `/`: the front end you talk to ("Hey Trace", voice, mascot) |
| `tts.py` | the voice: Microsoft neural (edge-tts) online, Kokoro offline |
| `whereabouts.py` + `room.py` | backup answers: multicam's own person trail (where you stopped, how the item went missing). `/api/ask` uses it only when the `/sim` tab isn't posting the model. Its places live in `places.json`. See `docs/whereabouts.md` |
| `scene.py`, `targets.py` | used by the two above: furniture found in a photo becomes places (`room.py` imports `scene`), and the item list (`whereabouts.py` imports `targets`). Don't move or delete them |
| `snap.py` | one photo per phone, only for mapping the room from angled photos |

Not needed for the demo: `research/` (detector benchmarking and training the CLIP check),
`tests/`, and `legacy/live_scene.py` (the earlier one-camera version).

Never commit: `memory/`, `baseline/`, `events.jsonl`, `trace-events.jsonl`, `room.json`,
`places.json`, `timeline.jsonl`, `cameras.json`, `.env`. They're git-ignored; keep it that way (some contain photos of people).

---

## 7. Status (honest), 15 Nov 02:30

| Part | State | Tested how |
|---|---|---|
| Probability model, false alarms, search ranking | **Done** | `node trace-sim/test-core.js`: 1,000 simulated days × 4 false-alarm rates, all pass |
| 3D room, drawing the room from a photo/video, metrics | **Done** | in the browser |
| Events from cameras (`multicam.py` → `trace-events.jsonl`) | **Done, not tried on real phones** | fake camera frames: correct appeared / moved / person events |
| Live link to the page (bridge, Connect live, camera feeds beside the 3D room) | **Done** | end to end with fake cameras and with `--demo` |
| Talking to Trace (`trace.html` + `/sim` model + LLM wording, validation, phone beep) | **Done, not tried with a real key** | `python tests/test_guide.py` (rules path, no network) |
| One-command start, presentation view, shared room | **Done** | `run_trace.py --demo` end to end |
| **Your real keys seen by a phone in the demo room** | **Not tested: the biggest risk** | benchmark said keys are the hardest object; test tonight / first thing |
| Real phones → page → voice, start to finish | **Not tested** | needs the demo room |
| Confidence calibration from real placements | **Not done** (placeholder curves) | maths teammate |
| Most likely story (Viterbi) for the LLM to tell | **Not done** | nice to have |
| Blindfold rehearsal ×5, slides, pitch | **Not done** | everyone, morning |

**Bottom line:** the software is built and each piece is tested; what's unproven is the physical
part: phones seeing real keys in the real room. Everything above is about 85% of the build; the
remaining 15% is real-room testing, rehearsal and the pitch, which is a morning's work.

---

## 8. Running the pieces by hand

`run_trace.py` does all of this for you. Use this when one piece breaks and you want to see which.

```
phones -> multicam.py -> trace-events.jsonl -> bridge.py (ws://localhost:8765) -> /sim ("Real events" -> "Connect live")
            └─ also serves / (trace.html) and /sim on http://localhost:8000
```

**Cameras.** One `--cam` per phone, or list them in `cameras.json`:

```powershell
python multicam.py --cam "Sofa=http://<phone1-ip>:4747/video" --cam "Desk=http://<phone2-ip>:4747/video"
```

A zone you drag and name "Sofa" in the multicam window becomes id `sofa` in the events. To use other
names, pass `--zone-map zone_map.json` with e.g. `{"Hall table": "desk"}`. multicam starts
`bridge.py` itself (`--no-bridge` if you run `python bridge.py --file trace-events.jsonl` yourself),
and `--api-host 0.0.0.0` lets other devices on the Wi-Fi open the pages.

**What multicam writes** to `trace-events.jsonl` (format in `docs/SPEC.md` 3.2): `appeared` /
`moved` / `disappeared` per item (2 of 3 frames to appear, 5 missed frames to disappear, boxes over
15% of the frame dropped), and `person_here` / `person_left` per zone from YOLO-World `person` boxes
(no identity; `--no-people` uses movement inside the zone instead, which a TV or lamp can set off).
Delete `trace-events.jsonl` between runs for a clean start (`run_trace.py` does this unless
`--keep-events`).

**No cameras at all:** `python bridge.py` in one terminal, `python events_adapter.py --demo` in
another, then open `/sim` (or `trace-sim\index.html`) and press **Connect live**: keys appear on the
desk, disappear, then show up on the sofa, in the page's built-in "Demo flat".

**The two pages.** `/sim` runs the model and, in Real events mode, posts its belief for every item to
the server about once a second; `/` (trace.html) asks `POST /api/ask` and its answers come from that.
If `/sim` isn't running, the server answers 503 and trace.html falls back to its own replies.
"not there", "found it" and "again" are handled without the LLM, and each item's search is
remembered for 15 minutes. `GET /api/model` shows what the server currently holds.

**Phone beeps.** If Trace sends you to a watched zone, that zone's phone beeps: on each phone open
`http://<laptop-ip>:8000/phone?zone=<zone id>` and tap "turn sound on" (the server must be reachable,
so `--share`). With no phone page open, the laptop beeps.

**Plugging in a different camera program.** Print one JSON object per detection per frame on stdout,
flushed each frame, with `time` (the same for every line of one frame), `camera_id`, `item` (`keys`,
`wallet` or `glasses`), `det_conf` (0–1), `box` [x1,y1,x2,y2], `image_w` / `image_h`, and
`support: {"anchor": "<furniture id>"}` or `null`. Pipe it through the adapter
(`cmd /c "python your_camera.py | python events_adapter.py --zones zones_map.json"`), where
`zones_map.json` renames your furniture ids to the room map's zone ids, or write SPEC 3.2 events to
the events file yourself. Then run `bridge.py`, Connect live, and check the "zones not in the room
map" note under the event log.
