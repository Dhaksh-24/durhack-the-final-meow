# Trace

AirTags without the AirTag: phones watch a room, Trace keeps a probability for where your
keys, wallet and glasses are, and talks you to them. Spec: `SPEC.md`; brief: `CLAUDE.md`.

## Files
| file | what it does |
|---|---|
| `snap.py` | one baseline photo per phone → `baseline/` |
| `map_room.py` | Gemini maps the room from those photos → `room.json` (page) + `zones.json` (tracker) |
| `multicam.py` | reads the phones, tracks items across cameras, writes `events.jsonl` |
| `detect.py` | YOLO-World proposals + CLIP verifier (`verifier.json`) |
| `trace_events.py` | turns sightings into appeared / moved / disappeared / person events |
| `train_verifier.py` | builds `verifier.json` from LVIS + Wikimedia Commons examples |
| `bench.py`, `pull_lvis.py`, `targets.py` | detector benchmark on LVIS (`results/bench.csv`) |
| `trace-sim/index.html` | the model, search, 3D room and simulator (open in a browser) |

## Demo set-up
```
py snap.py                 # phones on stands, room empty of people
py map_room.py             # needs GEMINI_API_KEY in .env
py multicam.py             # events.jsonl
```
Open `trace-sim/index.html` → "Load a different room" → "Open room.json".

Tests: `py test_trace_events.py`, `node trace-sim/test-core.js`.
Never commit `memory/`, `baseline/`, `events.jsonl`, `zones.json`, `cameras.json`, `room.json`.
