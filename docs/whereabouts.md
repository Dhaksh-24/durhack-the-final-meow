# The backup reasoning: room model, person trail, predictions

This is the reasoning `multicam.py` runs on its own. The main model is the one in `trace-sim/index.html` (the `/sim` tab); `/api/ask` falls back to this one only when that tab isn't posting its model.

The detector only knows where an item was last *seen*. Three files turn that into
"where it probably is now":

- **`room.py`: the room model** (`places.json`). Each camera view becomes named places
  ("coffee table", "sofa", "hall floor") with a kind, how likely you are to put
  things down there, where small things hide there, and which places are next to each
  other. Places you drag out in `multicam.py` come first. Then furniture found in a
  photo, and then a plain left / middle / right / floor split, so every item always
  has a place.

  ```bash
  python room.py build --grab "Living room=0"          # photo from webcam 0, find the furniture
  python room.py build --photo "Hall=hall.jpg"         # or from a photo
  python room.py show
  python room.py draw living-room photo.jpg out.jpg    # check the places on a photo
  ```

- **`whereabouts.py`: the reasoning.** `multicam.py` now also looks for `person` in
  the same YOLO-World pass (`--no-people` turns that off) and passes each frame's items
  and people to it. It logs:
  - every stop the person makes at a place, and for how long
  - a heatmap of where their feet have been (press **h** in `multicam.py`)
  - how each item went missing: *nobody near* (it's covered), *someone walked past
    without stopping* (it was knocked onto the floor), or *someone stood at it* (it was
    picked up). Picked up and then you stayed put means it slid down the cushions.

  Ask about a missing item and it links that to where you went next: the places you
  stopped at afterwards, how long you stopped, how good each is for putting things
  down, whether you left the room, and where that item turned up before. Gemini gets
  that evidence as text (never images), puts the places in order and words the
  answer. With no key, no network or a slow reply, a rules answer built from the
  same scores is used instead. Semantic events go to `timeline.jsonl`, and the state
  to `memory/whereabouts.json`.

  New endpoints (`trace_api.py`): `POST /api/ask` (trace.html already calls it, so
  "Hey Trace, where are my keys?" now uses this), `GET /api/predict?item=keys`,
  `GET /api/whereabouts` and `GET /api/heatmap/<camera>.jpg`.

- **`tests/simulate.py`: the "every way to lose it" simulation.** A scripted person loses
  their keys seven ways in a pretend two-camera flat: covered, moved, two stops,
  knocked off, carried out, slipped down the sofa, and put back hidden. It feeds noisy
  detections through the real engine and scores the guesses:

  ```bash
  python tests/simulate.py           # accuracy per way of losing them
  python tests/simulate.py --show    # the story and the spoken answer for each
  ```

  It checks that the reasoning does what it's meant to, under idealised conditions:
  one person, people detected 92% of the time. Real-room accuracy will be lower.
