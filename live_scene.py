"""live_scene.py -- say where your things are, live from the webcam.

    python live_scene.py                       # look for keys, phone, wallet...
    python live_scene.py --items=keys
    python live_scene.py --gemini              # name the anchors nicely at startup
    python live_scene.py --anchor-conf=0.20 --item-conf=0.30
    python live_scene.py --json=scene.json     # keep a file updated for another program
    python live_scene.py --no-window --frames=30   # headless benchmark

While it runs:
    Esc   quit
    r     re-map the furniture (after you move the camera)
    g     ask Gemini to name the anchors, once, for this frame
    f     toggle the floor tint

Why this stays fast: furniture is detected ONCE at full resolution and
cached by scene.CONFIG["anchor_ttl"]. Per frame we only run the small item
model, at ITEM_IMGSZ, so the loop costs the same as detect.py.
"""
import sys
import time

import cv2

import scene

ITEMS = ["keys", "phone", "wallet", "remote control", "glasses"]
ITEM_IMGSZ = 640     # per-frame pass; anchors still use scene.CONFIG["imgsz"]
HOLD = 1.5           # keep the last sentence on screen this long after the
                     # item drops out, so the overlay does not flicker
WARMUP = 5           # discard frames: the first off a webcam is often black
DARK_MEAN = 40       # below this (of 255) the lens is probably covered
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    items, overrides = list(ITEMS), {"item_imgsz": ITEM_IMGSZ}
    show = "--no-window" not in flags
    want_gemini = "--gemini" in flags
    limit = 0
    json_path = None

    for flag in flags:
        if flag.startswith("--json="):
            json_path = flag.split("=", 1)[1]
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

    t0 = time.perf_counter()
    anchors = mapper.anchors(frame, camera="webcam")
    floor = mapper.floor_mask(frame.shape, anchors)
    print("mapped room in %.1fs: %s"
          % (time.perf_counter() - t0, ", ".join(a["id"] for a in anchors) or "nothing"))
    if not anchors:
        print("no furniture found -- without anchors every item reads as "
              "'on the floor' or 'somewhere in view'. Point the camera at the "
              "room and press r to re-map, or lower --anchor-conf.")

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
                frame.shape, anchors, held_items, held_descs, camera="webcam"))

        canvas = scene.annotate(frame, anchors, found, held,
                                floor if show_floor else None)
        cv2.putText(canvas, "%4.1f fps | %d anchors | Esc quit  r re-map  g gemini"
                    % (fps, len(anchors)), (10, canvas.shape[0] - 10),
                    FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if show:
            cv2.imshow("where is it", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("r"):
                anchors = mapper.anchors(frame, camera="webcam", refresh=True)
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
