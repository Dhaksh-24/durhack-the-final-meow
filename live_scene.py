"""live_scene.py -- say where your things are, live from the webcam.

    python live_scene.py --setup               # confirm the furniture once, save it
    python live_scene.py                       # look for keys, wallet, glasses
    python live_scene.py --items=keys
    python live_scene.py --jsonl               # one JSON line per detection on stdout
    python live_scene.py --jsonl --drop-guesses    # ... without the weak detections
    python live_scene.py --json=scene.json     # keep a file updated for another program
    python live_scene.py --camera-id=lounge    # the name written into the JSON
    python live_scene.py --anchors=lounge.json # which furniture file to use
    python live_scene.py --gemini              # name the anchors nicely at startup
    python live_scene.py --anchor-conf=0.20 --item-conf=0.30
    python live_scene.py --no-window --frames=30   # headless benchmark

Setup (--setup) detects the furniture once and shows each piece in turn:
    y / Enter   keep it
    d / n       delete it
    b           go back one
    Esc         cancel, nothing saved
The pieces you keep are numbered left to right (table_1, table_2, ...) and
saved to anchors.json. Every later run loads that file instead of detecting
furniture, so the ids stay the same until you run --setup again. Furniture
detection is deliberately loose; if there are too many phantom pieces to
click through, add --anchor-conf=0.15.

While it runs:
    Esc   quit
    r     re-map the furniture (ignored once a saved list is loaded)
    g     ask Gemini to name the anchors, once, for this frame
    f     toggle the floor tint

With --jsonl, stdout carries JSON lines and nothing else -- every status
message goes to stderr -- so it pipes straight into another program:
    python live_scene.py --jsonl --no-window | python their_program.py
The line format is at the top of scene.py. Detections under
scene.CONFIG["guess_below"] (0.30) carry "guess": true; --drop-guesses
discards them instead.

Why this stays fast: furniture is detected ONCE at full resolution (or not
at all, once anchors.json exists). Per frame we only run the small item
model, at ITEM_IMGSZ, so the loop costs the same as detect.py.
"""
import os
import sys
import time

import cv2

import scene

ITEMS = ["keys", "wallet", "glasses"]
ITEM_IMGSZ = 640     # per-frame pass; anchors still use scene.CONFIG["imgsz"]
HOLD = 1.5           # keep the last sentence on screen this long after the
                     # item drops out, so the overlay does not flicker
WARMUP = 5           # discard frames: the first off a webcam is often black
DARK_MEAN = 40       # below this (of 255) the lens is probably covered
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "where is it"
ANCHORS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "anchors.json")

ESC, ENTER, SPACE = 27, 13, 32
KEEP_KEYS = (ord("y"), ENTER, SPACE)
DELETE_KEYS = (ord("d"), ord("n"))


def wait_key():
    """Block until a key is pressed. Closing the window counts as Esc."""
    while True:
        key = cv2.waitKey(50)
        if key != -1:
            return key & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return ESC


def review_anchors(frame, detected, mapper):
    """Step through the detected furniture; the user keeps or deletes each.

    Returns the kept pieces with their final ids, or None if cancelled.
    """
    keep = [False] * len(detected)
    i = 0
    while True:
        if i < len(detected):
            current = detected[i]
            decided = [a for j, a in enumerate(detected) if j < i and keep[j]]
            canvas = scene.annotate(frame, decided, [current], [
                "%d of %d: %s (detector confidence %.2f)"
                % (i + 1, len(detected), current["id"], current["conf"]),
                "y keep    d delete    b back    Esc cancel",
            ])
            cv2.imshow(WINDOW, canvas)
            key = wait_key()
            if key == ESC:
                return None
            if key in KEEP_KEYS:
                keep[i], i = True, i + 1
            elif key in DELETE_KEYS:
                keep[i], i = False, i + 1
            elif key == ord("b") and i > 0:
                i -= 1
        else:
            # Renumber before the last look, so what is on screen is exactly
            # what gets saved.
            kept = mapper.renumber([a for a, k in zip(detected, keep) if k])
            canvas = scene.annotate(frame, kept, [], [
                "Keeping %d of %d: %s" % (len(kept), len(detected),
                                          ", ".join(a["id"] for a in kept) or "nothing"),
                "Enter save    b back    Esc cancel",
            ])
            cv2.imshow(WINDOW, canvas)
            key = wait_key()
            if key == ESC:
                return None
            if key in KEEP_KEYS or key == ord("s"):
                return kept
            if key == ord("b") and detected:
                i = len(detected) - 1


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    items, overrides = list(ITEMS), {"item_imgsz": ITEM_IMGSZ}
    show = "--no-window" not in flags
    want_gemini = "--gemini" in flags
    setup = "--setup" in flags
    limit = 0
    json_path = None
    anchors_path = ANCHORS_FILE
    camera_id = "webcam"
    if "--drop-guesses" in flags:
        overrides["drop_guesses"] = True

    for flag in flags:
        if flag.startswith("--json="):
            json_path = flag.split("=", 1)[1]
        if flag.startswith("--anchors="):
            anchors_path = flag.split("=", 1)[1]
        if flag.startswith("--camera-id="):
            camera_id = flag.split("=", 1)[1]
        if flag.startswith("--items="):
            items = [s.strip() for s in flag.split("=", 1)[1].split(",") if s.strip()]
        if flag.startswith("--anchor-conf="):
            overrides["anchor_conf"] = float(flag.split("=", 1)[1])
        if flag.startswith("--item-conf="):
            overrides["item_conf"] = float(flag.split("=", 1)[1])
        if flag.startswith("--imgsz="):
            overrides["item_imgsz"] = int(flag.split("=", 1)[1])
        if flag.startswith("--frames="):
            limit = int(flag.split("=", 1)[1])

    # Before the first print: from here every print() lands on stderr and
    # stdout belongs to the JSON lines.
    lines_out = scene.claim_stdout() if "--jsonl" in flags else None

    if setup and not show:
        sys.exit("--setup needs the window to ask you about each piece: drop --no-window")

    mapper = scene.SceneMapper(overrides)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # DSHOW opens fast on Windows
    if not cap.isOpened():
        sys.exit("could not open webcam at index 0")

    # Warm up before mapping. Mapping the room from a black first frame
    # caches an empty scene for anchor_ttl seconds, which looks like a bug
    # for the next five minutes.
    for _ in range(WARMUP):
        ok, frame = cap.read()
    if not ok:
        sys.exit("webcam opened but returned no frame")
    if frame.mean() < DARK_MEAN:
        print("WARNING: the camera is very dark (mean %.0f/255). Privacy "
              "shutter closed, or the lights off? Nothing will be found."
              % frame.mean())

    # --- one-off: encode the item prompts and map the room ---------------
    print("looking for: %s" % ", ".join(items))
    t0 = time.perf_counter()
    mapper.detect_items(frame, items)             # pays the CLIP text encoding
    print("item prompts ready in %.1fs" % (time.perf_counter() - t0))

    # Map from a FRESH frame. The prompt encoding above takes about four
    # seconds, which is plenty for auto-exposure to settle -- the frame we
    # grabbed at startup is often still dark.
    ok, fresh = cap.read()
    if ok:
        frame = fresh

    # Three ways to get the furniture: confirm it now (--setup), load what
    # was confirmed earlier, or -- with no saved list -- trust the detector.
    fixed = False
    if setup:
        detected = mapper.anchors(frame, camera=camera_id, refresh=True)
        print("found %d pieces of furniture -- confirm each one in the window"
              % len(detected))
        anchors = review_anchors(frame, detected, mapper)
        if anchors is None:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit("setup cancelled -- nothing saved")
        if not scene.save_anchors(anchors_path, anchors, frame.shape, camera_id):
            print("WARNING: %s is open in another program -- not saved" % anchors_path)
        else:
            print("saved %d anchors to %s: %s"
                  % (len(anchors), anchors_path, ", ".join(a["id"] for a in anchors) or "none"))
        fixed = True
    elif os.path.exists(anchors_path):
        anchors = scene.load_anchors(anchors_path, frame.shape)
        fixed = True
        print("loaded %d fixed anchors from %s: %s"
              % (len(anchors), anchors_path, ", ".join(a["id"] for a in anchors) or "none"))
    else:
        t0 = time.perf_counter()
        anchors = mapper.anchors(frame, camera=camera_id)
        print("mapped room in %.1fs: %s"
              % (time.perf_counter() - t0, ", ".join(a["id"] for a in anchors) or "nothing"))
        print("these are unconfirmed -- run with --setup to check each one and "
              "save a fixed list")
    floor = mapper.floor_mask(frame.shape, anchors)

    if not anchors:
        print("no furniture -- without anchors every item reads as 'on the "
              "floor' or 'somewhere in view'. Point the camera at the room and "
              "run --setup, or lower --anchor-conf.")

    if want_gemini:
        try:
            mapper.gemini_review(frame, anchors, [])
            print("gemini named: %s" % mapper._names)
        except Exception as exc:
            print("gemini pass failed: %s: %s" % (type(exc).__name__, exc))

    fps, n = 0.0, 0
    held, held_at = [], 0.0
    held_items, held_descs = [], []
    show_floor = True
    if json_path:
        print("writing the scene to %s every frame" % json_path)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = time.perf_counter()
        found = mapper.detect_items(frame, items)
        sentences, descs = [], []
        for item in found:
            desc, line = mapper.describe(item, anchors, frame.shape, floor)
            descs.append(desc)
            sentences.append(line)
        dt = time.perf_counter() - t
        fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

        # The stream reports what THIS frame saw, nothing held over: each
        # line is one detection, stamped with the time of its frame.
        if lines_out is not None:
            if not scene.emit(lines_out, mapper.detections(frame.shape, found,
                                                           descs, camera_id)):
                print("the program reading stdout has gone -- stopping")
                break

        # Hold the last answer briefly: a box that blinks out for one frame
        # should not wipe the sentence off the screen.
        if sentences:
            held, held_at = sentences, time.time()
            held_items, held_descs = found, descs
        elif time.time() - held_at > HOLD:
            held, held_items, held_descs = [], [], []

        # The file mirrors the overlay, hold included: a reader polling it
        # should not see the keys vanish because one frame missed them. So
        # "timestamp" is when the file was written; the items in it can be
        # up to HOLD seconds old.
        if json_path:
            scene.write_json(json_path, mapper.payload(
                frame.shape, anchors, held_items, held_descs, camera=camera_id))

        canvas = scene.annotate(frame, anchors, found, held,
                                floor if show_floor else None)
        cv2.putText(canvas, "%4.1f fps | %d anchors%s | Esc quit  r re-map  g gemini"
                    % (fps, len(anchors), " (fixed)" if fixed else ""),
                    (10, canvas.shape[0] - 10), FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if show:
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ESC:
                break
            if key == ord("r"):
                if fixed:
                    print("the furniture is fixed by %s -- run --setup to redo it"
                          % anchors_path)
                else:
                    anchors = mapper.anchors(frame, camera=camera_id, refresh=True)
                    floor = mapper.floor_mask(frame.shape, anchors)
                    print("re-mapped: %s" % ", ".join(a["id"] for a in anchors))
            if key == ord("g"):
                try:
                    mapper.gemini_review(frame, anchors, [])
                    print("gemini named: %s" % mapper._names)
                except Exception as exc:
                    print("gemini failed: %s: %s" % (type(exc).__name__, exc))
            if key == ord("f"):
                show_floor = not show_floor

        n += 1
        if limit and n >= limit:
            print("%d frames, %.1f fps" % (n, fps))
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
