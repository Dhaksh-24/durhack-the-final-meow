# Trace: model and room simulator

Open `index.html` in a browser (needs internet once, for three.js).

- **The room:** a 3D model built from a zone map, with a simulated person moving keys, wallet and glasses.
- **Where Trace thinks it is:** a hidden Markov model over zones, "covered", "with the person" and "outside", updated from phone events.
- **Ask Trace:** ranks places by chance of finding the item per second of searching, and updates when you say "not there".
- **Test it on simulated days:** compares Trace with "check where it was last seen" and checks the percentages are honest.
- **Live metrics:** room size and area, what's in it, the phone network (sightings per phone), how often the model's top guess is right, its uncertainty in bits, and a test of where the next phone would help most.
- **Map a real room from camera photos:** click a sheet of known size on the floor in each phone's photo, then the front edge of each piece of furniture; the 3D room is rebuilt in metres. Also outputs each phone's pixel-to-floor matrix for the tracker.

Event format the model reads (one JSON object per line, full contract in `SPEC.md` 3.2):

    {"time": "14:05:10", "t": 1794696414.7, "item": "keys-1", "category": "keys", "zone": "sofa", "event": "appeared", "confidence": 0.82, "camera": "phone-1"}

**Replay a recording:** in the Events card, "Replay a recorded events.jsonl" plays a session
from the cameras through the same model and 3D room, in time order, at the chosen speed
(`sample-events.jsonl` is a scripted example for the demo room). There's no ground truth in a
recording, so "show where things really are" is off and you answer "Not there" / "Found it"
yourself. Zones must match the room map; unknown zones and items are listed, not crashed on.

**Tests:** `node trace-sim/test-core.js [days]` checks false alarms, calibration and replay.

All results here are simulated: they test the reasoning, not the cameras.
