# Running the live chain (camera → events → page)

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

## Plugging in a different camera program

1. Print one JSON object per detection per frame on stdout (stderr for everything else), and flush each frame.
2. Each line needs `time` (same value for every line of one frame), `camera_id`, `item` (`keys`, `wallet` or `glasses`), `det_conf` (0–1), `box` [x1,y1,x2,y2] with `image_w`/`image_h`, and `support: {"anchor": "<furniture id>"}` (or `null`).
3. Use furniture ids that, after `--zones` mapping or slugging, equal the zone ids of the room map in the page.
4. Pipe it: `cmd /c "python your_camera.py | python events_adapter.py"` (or write SPEC 3.2 events to `events.jsonl` yourself and skip the adapter).
5. Run `python bridge.py`, open the page locally, press **Connect live**, and check the event log and "zones not in the room map" note.

Never commit `events.jsonl`, `memory/`, `baseline/` or `.env` (all git-ignored).
