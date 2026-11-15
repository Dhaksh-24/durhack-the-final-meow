# Running the live chain (camera → events → page)

## Four phones: multicam.py (use this for the demo)

```
phones ─▶ multicam.py ─▶ trace-events.jsonl ─▶ bridge.py (started for you) ─▶ ws://localhost:8765 ─▶ trace-sim/index.html ("Connect live")
            └─ also serves Adel's camera console + voice at http://localhost:8000/  and the model page at http://localhost:8000/sim
```

```powershell
git fetch origin; git checkout demo; git pull
pip install websockets
python multicam.py --cam "Sofa=http://<phone1-ip>:4747/video" --cam "Desk=http://<phone2-ip>:4747/video"   # one --cam per phone
```

1. In the multicam window, **drag a zone on each camera tile and name it exactly like the room map**
   in the page ("Sofa", "Desk", "Coat hook"…). A zone named "Sofa" becomes id `sofa`; to use other
   names, pass `--zone-map zone_map.json` with e.g. `{"Hall table": "desk"}`. Zones are saved in `zones.json`.
2. Open `http://localhost:8000/sim` (or `trace-sim\index.html`), make sure the room map has the same
   zone ids (map the room or paste a zone map), press **Real events → Connect live**.
3. Put your keys down in a zone: within ~1.5 s the event log shows `appeared` and the 3D room moves.
   Opened from `http://localhost:8000/sim`, the page also shows every phone's live video under the 3D room
   ("Cameras, live"), so you can watch the cameras and the model side by side.

What multicam now writes to `trace-events.jsonl` (SPEC 3.2): `appeared` / `moved` / `disappeared` for
keys, wallet, glasses (2-of-3 frames in, 5 missed frames out, boxes over 15% of the frame dropped),
and `person_here` / `person_left` per zone from movement inside the zone (no person detector, no identity).
`--no-bridge` if you run `bridge.py --file trace-events.jsonl` yourself. Delete `trace-events.jsonl`
between demo runs for a clean start.

## One camera: live_scene.py

```
live_scene.py --jsonl  →  events_adapter.py  →  events.jsonl  →  bridge.py  →  ws://localhost:8765  →  trace-sim/index.html
(raw detections)         (SPEC 3.2 events)                       (tails it)                           ("Connect live")
```

Branch: `integration`. Windows PowerShell, from the repo folder. Three terminals.

## Once

```powershell
git fetch origin
git checkout integration
git pull
pip install websockets
```

## Test without a camera (demo)

```powershell
# terminal 1
python bridge.py
# terminal 2
python events_adapter.py --demo
```

Open `trace-sim\index.html` in Chrome (`start chrome trace-sim\index.html`), press **Connect live**.
Within a second: keys appear on the desk, disappear, then show up on the sofa. The demo uses the
page's built-in "Demo flat" room (zones `desk`, `sofa`, …); press "Back to the demo flat" under
"Load a different room" if the page remembers another room.

## With the real camera

```powershell
# once per room: confirm the furniture (writes anchors.json; ids like table_1, sofa_1)
python live_scene.py --setup
# terminal 1
python bridge.py
# terminal 2  (cmd /c keeps the pipe as plain text lines)
cmd /c "python live_scene.py --jsonl --no-window | python events_adapter.py --zones zones_map.json"
```

`zones_map.json` renames furniture anchors to the room map's zone ids, e.g.
`{"table_1": "desk", "sofa_1": "sofa"}`. Without it a zone id is the anchor id with `_` → `-`
(`table_1` → `table-1`). Either way, **the room map loaded in the page must have zones with exactly
those ids** (draw the room, or paste a zone map, and make sure the ids match), and at least one
must have `"camera": true`. Anything else shows as "zones not in the room map" under the event log.

`live_scene.py` opens webcam 0. A phone works through the DroidCam Windows client, which shows up
as a webcam; if the laptop's own camera is 0, the phone is usually 1.

Stop with Ctrl+C in each terminal. `events.jsonl` keeps growing across runs (it's git-ignored);
delete it between demo runs if you want a clean start (`Remove-Item events.jsonl`).

## Two pages: the front end and the analytics

- **http://localhost:8000/** is `trace.html`, the proper front end (mascot, "Hey Trace", voice).
  Its questions go to `POST /api/ask` and are answered by the guide below.
- **http://localhost:8000/sim** is `trace-sim/index.html`, the live analytics (3D room, probability
  columns, metrics). It also runs the model: in **Real events** mode it posts its belief for every
  item to the server about once a second, and that is what the front end's answers come from.

So for the demo, keep `/sim` open in Real events mode (Connect live, or a loaded recording), e.g.
on the big screen, and talk to Trace in `/`. If `/sim` isn't running, the server answers 503 and
`trace.html` falls back to its own built-in replies. "not there", "found it" and "again" in
`trace.html` are handled by the server without the LLM; each item's search is remembered there for
15 minutes. `GET /api/model` shows what the server currently holds.

## The guide: ask Trace by voice or text (branch `demo`)

```powershell
copy .env.example .env      # then paste the team's GEMINI_API_KEY into .env (never commit it)
python trace_api.py         # or multicam.py, which starts the same server with its cameras
```

Open **http://localhost:8000/sim** in Chrome (serving it from the same server avoids browser
security prompts). Load events or press Connect live as above, then in **Ask Trace**: type, or
press **🎤 Talk** and speak (allow the microphone). Pick the mode: Everyone, Blind / low vision,
Dementia.

- "not there", "found it" and "again" are handled in the page, no LLM.
- Everything else goes to `POST /api/ask`: the LLM turns what you said into evidence ("I had a
  quick look under the cushions" → sofa, covered, low) and words the answer from the updated plan.
  At most two LLM calls. The page applies the evidence to its own model and checks the answer
  (SPEC 5.2: only places in the plan, probability words that match the code, times from the model,
  no "definitely", nobody named as moving things). If it fails, or there's no key, Trace says its
  own template sentence instead. The note under the box says which happened and why.
- Chirp: if the answer sends you to a watched zone, that zone's phone beeps. On each phone, open
  `http://<laptop-ip>:8000/phone?zone=<zone id>` and tap "turn sound on" (start the server with
  `python trace_api.py --host 0.0.0.0` so phones can reach it). With no phone page open, the
  laptop beeps.
- The orange patches on the floor are where you were since the item was last seen (from
  person_here / person_left); the same trail goes to the LLM for memory prompts.

Tests without a key or network: `python test_guide.py` and `node trace-sim/test-core.js`.

## Plugging in a different camera program

1. Print one JSON object per detection per frame on stdout (stderr for everything else), and flush each frame.
2. Each line needs `time` (same value for every line of one frame), `camera_id`, `item` (`keys`, `wallet` or `glasses`), `det_conf` (0–1), `box` [x1,y1,x2,y2] with `image_w`/`image_h`, and `support: {"anchor": "<furniture id>"}` (or `null`).
3. Use furniture ids that, after `--zones` mapping or slugging, equal the zone ids of the room map in the page.
4. Pipe it: `cmd /c "python your_camera.py | python events_adapter.py"` (or write SPEC 3.2 events to `events.jsonl` yourself and skip the adapter).
5. Run `python bridge.py`, open the page locally, press **Connect live**, and check the event log and "zones not in the room map" note.

Never commit `events.jsonl`, `memory/`, `baseline/` or `.env` (all git-ignored).
