"""test_scene.py -- run scene.py over the images in .\\test_images

    python test_scene.py                      # every image, geometry only
    python test_scene.py --gemini             # also run the optional Gemini pass
    python test_scene.py --items=keys,phone   # what to look for
    python test_scene.py room1.jpg            # a single image
    python test_scene.py --anchor-conf=0.15   # fewer phantom sofas
    python test_scene.py --item-conf=0.25     # fewer phantom books

On first run, if .\\test_images is empty, three freely licensed room photos are
downloaded from Wikimedia Commons. Annotated copies (anchor boxes, item boxes,
the sentence, and the floor band tinted) land in .\\test_out, each beside a
.json of the same name holding everything found (see scene.payload).
"""
import json
import os
import sys
import time
import urllib.request

import cv2

import scene

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "test_images")
OUTPUT = os.path.join(HERE, "test_out")

# Wikimedia Commons, freely licensed. Rooms with sofas, tables and small
# objects on them -- which is what we need to exercise "on" and "next to".
SAMPLES = {
    "room1.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/"
                 "Cozy_living_room_setting_with_a_yellow_sofa_and_a_coffee_table_"
                 "adorned_with_cups_and_a_flower_pot.jpg/1280px-Cozy_living_room_"
                 "setting_with_a_yellow_sofa_and_a_coffee_table_adorned_with_cups_"
                 "and_a_flower_pot.jpg",
    "room2.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/"
                 "Living_room_with_1960s_objects.jpg/"
                 "1280px-Living_room_with_1960s_objects.jpg",
    "room3.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/"
                 "Cabin_14_living_room_sofa_fireplace_microwave_Douthat_State_Park_"
                 "Virginia_%2827698134543%29.jpg/1280px-Cabin_14_living_room_sofa_"
                 "fireplace_microwave_Douthat_State_Park_Virginia_%2827698134543%29.jpg",
}

ITEMS = ["keys", "phone", "wallet", "remote control", "mug", "cup", "book", "glasses"]


def fetch_samples():
    """Download any missing sample photos.

    Skipped entirely once you drop your own photos in here -- we only ever
    fetch the three files we know about.
    """
    os.makedirs(IMAGES, exist_ok=True)
    present = [f for f in os.listdir(IMAGES)
               if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if [f for f in present if f not in SAMPLES]:
        return                                      # your own images, left alone
    missing = {n: u for n, u in SAMPLES.items() if n not in present}
    if not missing:
        return

    print("downloading %d sample photo(s) into test_images" % len(missing))
    for name, url in missing.items():
        # Wikimedia rejects the default urllib user agent, and rate-limits
        # bursts with HTTP 429 -- hence the backoff and the pause between files.
        request = urllib.request.Request(url, headers={"User-Agent": "scene-test/1.0"})
        for attempt in (1, 2, 3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                with open(os.path.join(IMAGES, name), "wb") as fh:
                    fh.write(data)
                print("  %-10s %d KB" % (name, len(data) // 1024))
                break
            except Exception as exc:
                if attempt == 3:
                    print("  %-10s FAILED: %s" % (name, exc))
                else:
                    time.sleep(2 * attempt)
        time.sleep(1)


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    picked = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_gemini = "--gemini" in flags

    # The two or three knobs worth turning from the command line; everything
    # else lives in scene.CONFIG.
    items, overrides = ITEMS, {}
    for flag in flags:
        if flag.startswith("--items="):
            items = [s.strip() for s in flag.split("=", 1)[1].split(",") if s.strip()]
        if flag.startswith("--anchor-conf="):
            overrides["anchor_conf"] = float(flag.split("=", 1)[1])
        if flag.startswith("--item-conf="):
            overrides["item_conf"] = float(flag.split("=", 1)[1])
        if flag.startswith("--imgsz="):
            overrides["imgsz"] = int(flag.split("=", 1)[1])

    fetch_samples()
    os.makedirs(OUTPUT, exist_ok=True)

    files = picked or sorted(f for f in os.listdir(IMAGES)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        sys.exit("no images in %s" % IMAGES)

    mapper = scene.SceneMapper(overrides)
    print("looking for: %s" % ", ".join(items))
    if overrides:
        print("config overrides: %s" % overrides)

    for name in files:
        path = name if os.path.isabs(name) else os.path.join(IMAGES, name)
        frame = cv2.imread(path)
        if frame is None:
            print("\n== %s -- could not read, skipping" % name)
            continue

        print("\n" + "=" * 70)
        print("%s  %dx%d" % (name, frame.shape[1], frame.shape[0]))

        # camera=name keeps each image's anchors in its own cache slot, so
        # image 2 never inherits image 1's furniture.
        anchors = mapper.anchors(frame, camera=name)
        floor = mapper.floor_mask(frame.shape, anchors)
        print("anchors: %s" % (", ".join(a["id"] for a in anchors) or "none found"))

        # Outputs are named by basename. A picked image may be an absolute
        # path, and os.path.join(OUTPUT, <absolute path>) is just that path --
        # the annotated copy would be saved over the original photo.
        base = os.path.basename(name)
        out_path = os.path.join(OUTPUT, base)
        json_name = os.path.splitext(base)[0] + ".json"
        json_path = os.path.join(OUTPUT, json_name)

        found = mapper.detect_items(frame, items)
        if not found:
            print("no items found -- try --conf or different --items")
            cv2.imwrite(out_path, scene.annotate(frame, anchors, [], (), floor))
            scene.write_json(json_path, mapper.payload(frame.shape, anchors,
                                                       [], [], camera=base))
            continue

        descriptions, sentences = [], []
        for item in found:
            desc, line = mapper.describe(item, anchors, frame.shape, floor)
            descriptions.append(desc)
            sentences.append(line)

        if use_gemini:
            try:
                descriptions = mapper.gemini_review(frame, anchors, descriptions)
                sentences = [mapper.sentence(d) for d in descriptions]
                print("(gemini pass applied)")
            except Exception as exc:
                print("(gemini pass failed: %s: %s)" % (type(exc).__name__, exc))

        for line in sentences:
            print("  " + line)
        print(json.dumps(descriptions, indent=2))

        out = scene.annotate(frame, anchors, found, sentences, floor)
        cv2.imwrite(out_path, out)
        print("annotated -> %s" % os.path.join("test_out", base))

        # After the Gemini pass, so any correction it made is in the file.
        scene.write_json(json_path, mapper.payload(frame.shape, anchors, found,
                                                   descriptions, camera=base))
        print("json      -> %s" % os.path.join("test_out", json_name))


if __name__ == "__main__":
    main()
