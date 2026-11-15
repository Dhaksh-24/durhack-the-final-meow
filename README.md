# Trace

**AirTags without the AirTag.** Phones placed around a room watch the furniture. Trace keeps a
probability for where each important item (keys, glasses) is, and talks you to it by voice, with a
3D model of the room on screen. Built for blind and low-vision people, people with dementia, and
anyone who loses their keys. DurHack 2026.

## Run it

Windows PowerShell, in this folder, on the `demo` branch:

```powershell
pip install -r requirements.txt websockets
```

| You want to… | Run |
|---|---|
| Check the page works, no cameras | `python run_trace.py --demo` |
| Test with a phone video instead of live phones | `python run_trace.py --video keys_test.mp4` |
| Run it with the phones (list them in `cameras.json`, copied from `cameras.example.json`) | `python run_trace.py` |
| Show the judges | add `--present` |
| Show the pages on a second laptop on the same Wi-Fi | add `--share` |

Two tabs open: **http://localhost:8000/sim** runs the model and shows the 3D room (keep it open),
and **http://localhost:8000/** is where you talk to Trace ("Hey Trace", voice). Optional: put
`GEMINI_API_KEY=...` in a `.env` file for the LLM's wording; without it Trace uses its own sentences.

**Demo day:** read [HANDOVER.md](HANDOVER.md). It has the room set-up, the timed demo script,
troubleshooting and the honest status.

## How it works

```
phones (DroidCam)
  -> multicam.py + detect.py      find keys / glasses / people in each frame (YOLO-World, then a CLIP check)
  -> events_adapter.py            turn detections into events: appeared, moved, disappeared, person_here
  -> trace-events.jsonl
  -> bridge.py                    stream new events to the browser (WebSocket, port 8765)
  -> trace-sim/index.html (/sim)  the model: a hidden Markov model per item, updated on what each
                                  phone did and didn't see; ranks where to look; draws the 3D room
  -> trace_api.py                 the web server (port 8000); the page posts its model here
  -> guide.py                     the LLM turns what you say into evidence and words the answer;
                                  the probabilities always come from the model, never the LLM
  -> trace.html (/) + tts.py      what you see and hear
```

The maths, the data formats and the LLM rules are in [docs/SPEC.md](docs/SPEC.md).

## What's where

| Path | What it is |
|---|---|
| `run_trace.py` | starts everything with one command |
| `multicam.py` | reads the phones, tracks items across cameras, writes events, serves the page |
| `detect.py`, `verifier.json` | the detector: YOLO-World finds candidates, a CLIP classifier rejects look-alikes |
| `events_adapter.py` | raw detections → events (2 of 3 frames to appear, 5 missed to disappear) |
| `bridge.py` | streams `trace-events.jsonl` to the page |
| `trace_api.py` | the web server: both pages, live video, voice, `/api/ask`, `/api/room`, `/phone` beeper |
| `guide.py` | the LLM side: speech → evidence, plan → words, and checks the LLM's answer |
| `tts.py` | the voice: Microsoft neural (online), Kokoro (offline) |
| `trace-sim/` | `index.html`: the model, 3D room, room drawing and metrics, in one file. `test-core.js` tests it |
| `trace.html` | the page you talk to |
| `whereabouts.py`, `room.py`, `scene.py`, `targets.py` | the backup reasoning multicam runs itself (see [docs/whereabouts.md](docs/whereabouts.md)), plus the item list and photo-to-places code it uses |
| `snap.py` | takes one photo per phone, for mapping the room from photos |
| `tests/` | `test_guide.py`, `test_scene.py`, `simulate.py` and their test photos |
| `research/` | detector benchmarking and training: `bench.py`, `pull_lvis.py`, `pull_data.py`, `train_verifier.py` |
| `legacy/` | `live_scene.py`, the earlier one-camera version |
| `docs/` | `SPEC.md` (design), `voice.md` (talking to Trace, voices), `whereabouts.md` (the backup reasoning) |

## Tests

No API key or network needed:

```powershell
python tests/test_guide.py
node trace-sim/test-core.js
python tests/simulate.py
```

## Privacy

Never commit `memory/`, `baseline/`, `events.jsonl`, `trace-events.jsonl`, `zones.json`,
`cameras.json`, `room.json`, `places.json`, `timeline.jsonl` or `.env`. They're git-ignored, and
some contain photos of people. Results from the simulator are simulated; say so wherever they're shown.
