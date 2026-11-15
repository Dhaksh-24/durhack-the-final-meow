# Trace: model and room simulator

Open `index.html` in a browser (needs internet once, for three.js).

- **The room:** a 3D model built from a zone map, with a simulated person moving keys, wallet and glasses.
- **Where Trace thinks it is:** a hidden Markov model over zones, "covered", "with the person" and "outside", updated from phone events.
- **Ask Trace:** ranks places by chance of finding the item per second of searching, and updates when you say "not there".
- **Test it on simulated days:** compares Trace with "check where it was last seen" and checks the percentages are honest.
- **Live metrics:** room size and area, what's in it, the phone network (sightings per phone), how often the model's top guess is right, its uncertainty in bits, and a test of where the next phone would help most.
- **Map a real room from camera photos:** click a sheet of known size on the floor in each phone's photo, then the front edge of each piece of furniture; the 3D room is rebuilt in metres. Also outputs each phone's pixel-to-floor matrix for the tracker.

**Model a room the quick way:** "Draw the room (top view)": type the room's width and depth, drag one box per piece of furniture, name each (the type is guessed from the name), tick which ones a phone watches, "Build the room". To trace instead of measuring, add a phone photo or paused video frame taken from any angle and click the room's 4 floor corners (back-left, back-right, front-right, front-left): the page flattens it into a true top view to trace over. Trace where furniture touches the floor; tall things lean away from the camera. The browser remembers the room.

**Advanced, from angled phone photos (only if the tracker needs pixel-to-floor calibration):**
1. Phones on their stands, streaming (IP Webcam or DroidCam), cameras listed in `cameras.json` as for `multicam.py`.
2. `python snap.py` saves one photo per camera to `baseline/` (git-ignored; it only needs OpenCV).
3. Open the page, "Advanced: map from angled phone photos": enter the room's width and depth, "Add phone photos" → the files in `baseline/`.
4. In each photo click the four corners of the sheet on the floor, then the two ends of each piece of furniture's front edge; name it.
5. "Build the room". The browser remembers the room on reload. "Copy the zone map" gives the zone ids `multicam.py` should use, and "Camera calibration for the tracker" gives `baseline/calibration.json`.

Event format the model reads (one JSON object per line, full contract in `docs/SPEC.md` 3.2):

    {"time": "14:05:10", "t": 1794696414.7, "item": "keys-1", "category": "keys", "zone": "sofa", "event": "appeared", "confidence": 0.82, "camera": "phone-1"}

**Real events:** switch the room from "Simulation" to "Real events", then "Load recorded events" plays a session
from the cameras through the same model and 3D room, in time order, at the chosen speed
(`sample-events.jsonl` is a scripted example for the demo room). There's no ground truth in a
recording, so "show where things really are" is off and you answer "Not there" / "Found it"
yourself. Zones must match the room map; unknown zones and items are listed, not crashed on.

**Tests:** `node trace-sim/test-core.js [days]` checks false alarms, calibration and replay.

All results here are simulated: they test the reasoning, not the cameras.
