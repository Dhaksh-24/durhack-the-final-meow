"""pull_lvis.py -- build labelled test sets from LVIS. No account, no API key.

    python research/pull_lvis.py                  # every target
    python research/pull_lvis.py keys wallet      # just these
    python research/pull_lvis.py --max=150 --neg=75

LVIS labels 1,203 everyday object categories on COCO photos, and everything
is a plain public download. It is also the benchmark YOLO-World reports its
zero-shot numbers on, so the model has not been trained on these labels.

Output is one folder per target, in the layout bench.py reads:

    data\\lvis\\keys\\images\\*.jpg
    data\\lvis\\keys\\labels\\*.txt      (class 0 = the target)
    data\\lvis\\keys\\data.yaml

Why one folder per target. LVIS is "federated": each photo was only checked
for some categories. For a target we can only use photos where its category
was labelled (positives) or explicitly verified absent (negatives). On any
other photo an unlabelled wallet would be scored as a false alarm when the
real problem is a missing label. Photos flagged "not exhaustively labelled"
for the category are dropped for the same reason.

Targets built from ONE LVIS category (keys, wallet, phone, remote, watch)
follow that rule exactly. Targets that merge two (bag, glasses, headphones)
are slightly looser: a photo counts if either category was checked.
"""
import json
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo folder
LVIS = os.path.join(ROOT, "data", "lvis")
CACHE = os.path.join(LVIS, "_cache")                 # each photo downloaded once
ANNOTATIONS = os.path.join(LVIS, "lvis_v1_val.json")
ANNOTATIONS_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"

# our target -> the LVIS categories that count as it
CATEGORIES = {
    "keys": ["key"],
    "wallet": ["wallet"],
    "phone": ["cellular_telephone"],
    "remote": ["remote_control"],
    "glasses": ["spectacles", "sunglasses"],
    "watch": ["watch"],
    "headphones": ["earphone", "headset"],
    "bag": ["backpack", "handbag"],
}

MAX_POSITIVE = 150     # photos containing the target, per target
MAX_NEGATIVE = 75      # photos verified NOT to contain it (these test false alarms)

# LVIS boxes every key on a ring separately. For "where are my keys" the
# bunch is one object, and a detector that boxes the bunch would match none
# of the thin per-key boxes. For these targets, touching boxes are merged
# into one box before the labels are written.
MERGE_TOUCHING = {"keys"}


def merge_touching(found, grow=0.25):
    """Union boxes [x, y, w, h] that touch into single boxes.

    Two boxes touch if they intersect once padded by `grow` of the smaller
    one's longer side -- keys on one ring overlap or nearly do; keys on
    opposite sides of a table do not.
    """
    rects = [[x, y, x + w, y + h] for x, y, w, h in found]
    changed = True
    while changed:
        changed = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                pad = grow * min(max(a[2] - a[0], a[3] - a[1]), max(b[2] - b[0], b[3] - b[1]))
                if (a[0] - pad <= b[2] and b[0] - pad <= a[2]
                        and a[1] - pad <= b[3] and b[1] - pad <= a[3]):
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3])]
                    del rects[j]
                    changed = True
                    break
            if changed:
                break
    return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in rects]


def fetch(url, path, tries=3):
    """Download url to path. Returns True on success."""
    request = urllib.request.Request(url, headers={"User-Agent": "lvis-pull/1.0"})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            with open(path + ".part", "wb") as fh:
                fh.write(data)
            os.replace(path + ".part", path)         # never leave half a file
            return True
        except Exception:
            if attempt == tries:
                return False
            time.sleep(2 * attempt)


def ensure_annotations():
    if os.path.exists(ANNOTATIONS):
        return
    os.makedirs(LVIS, exist_ok=True)
    print("downloading LVIS annotations (64 MB) ...")
    archive = ANNOTATIONS + ".zip"
    if not fetch(ANNOTATIONS_URL, archive):
        sys.exit("could not download %s" % ANNOTATIONS_URL)
    with zipfile.ZipFile(archive) as z:
        z.extractall(LVIS)
    os.remove(archive)


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    targets = [a for a in sys.argv[1:] if not a.startswith("--")] or list(CATEGORIES)
    max_pos, max_neg = MAX_POSITIVE, MAX_NEGATIVE
    for flag in flags:
        if flag.startswith("--max="):
            max_pos = int(flag.split("=", 1)[1])
        if flag.startswith("--neg="):
            max_neg = int(flag.split("=", 1)[1])
    for t in targets:
        if t not in CATEGORIES:
            sys.exit("unknown target '%s' -- choose from: %s" % (t, ", ".join(CATEGORIES)))

    ensure_annotations()
    print("reading annotations (about 15 seconds) ...")
    with open(ANNOTATIONS, encoding="utf-8") as fh:
        lvis = json.load(fh)
    images = {im["id"]: im for im in lvis["images"]}
    cat_id = {c["name"]: c["id"] for c in lvis["categories"]}

    # boxes[category id][image id] -> list of [x, y, w, h] in pixels
    wanted_ids = {cat_id[n] for t in targets for n in CATEGORIES[t]}
    boxes = {cid: {} for cid in wanted_ids}
    for ann in lvis["annotations"]:
        if ann["category_id"] in wanted_ids:
            boxes[ann["category_id"]].setdefault(ann["image_id"], []).append(ann["bbox"])

    # --- choose photos per target -----------------------------------------
    plan = {}
    for t in targets:
        members = [cat_id[n] for n in CATEGORIES[t]]
        positive = set().union(*(boxes[m].keys() for m in members))
        # drop photos where the category is present but not fully labelled
        positive = {i for i in positive
                    if not set(members) & set(images[i]["not_exhaustive_category_ids"])}
        negative = {i for i, im in images.items()
                    if set(members) & set(im["neg_category_ids"])} - positive
        rng = random.Random(0)                       # same photos every run
        use_pos = sorted(rng.sample(sorted(positive), min(max_pos, len(positive))))
        use_neg = sorted(rng.sample(sorted(negative), min(max_neg, len(negative))))
        plan[t] = (members, use_pos, use_neg, len(positive), len(negative))

    # --- download each photo once -----------------------------------------
    needed = sorted({i for _, pos, neg, _, _ in plan.values() for i in pos + neg})
    os.makedirs(CACHE, exist_ok=True)
    missing = [i for i in needed if not os.path.exists(os.path.join(CACHE, "%012d.jpg" % i))]
    print("%d photos needed, %d already cached, downloading %d ..."
          % (len(needed), len(needed) - len(missing), len(missing)))
    started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda i: fetch(images[i]["coco_url"], os.path.join(CACHE, "%012d.jpg" % i)),
            missing))
    failed = {i for i, ok in zip(missing, results) if not ok}
    if missing:
        print("downloaded %d in %.0fs, %d failed"
              % (len(missing) - len(failed), time.time() - started, len(failed)))

    # --- write one dataset folder per target ------------------------------
    print("\n%-11s %10s %7s %10s   %s" % ("target", "positives", "boxes", "negatives", "folder"))
    for t, (members, use_pos, use_neg, have_pos, have_neg) in plan.items():
        folder = os.path.join(LVIS, t)
        if os.path.exists(folder):
            shutil.rmtree(folder)                    # rebuilt from scratch each run
        os.makedirs(os.path.join(folder, "images"))
        os.makedirs(os.path.join(folder, "labels"))

        count = 0
        for i in use_pos + use_neg:
            if i in failed:
                continue
            name = "%012d" % i
            shutil.copyfile(os.path.join(CACHE, name + ".jpg"),
                            os.path.join(folder, "images", name + ".jpg"))
            width, height = images[i]["width"], images[i]["height"]
            found = [b for m in members for b in boxes[m].get(i, [])]
            if t in MERGE_TOUCHING:
                found = merge_touching(found)
            lines = []
            for x, y, w, h in found:
                lines.append("0 %.6f %.6f %.6f %.6f" % (
                    min(1.0, (x + w / 2) / width), min(1.0, (y + h / 2) / height),
                    min(1.0, w / width), min(1.0, h / height)))
            count += len(lines)
            # negatives get an empty label file: "checked, nothing here"
            with open(os.path.join(folder, "labels", name + ".txt"), "w") as fh:
                fh.write("\n".join(lines))

        with open(os.path.join(folder, "data.yaml"), "w") as fh:
            fh.write("# built by pull_lvis.py from LVIS v1 val: %s\n" % ", ".join(CATEGORIES[t]))
            fh.write("path: .\ntrain: images\nval: images\nnames:\n  0: %s\n" % t)

        print("%-11s %4d / %-4d %7d %4d / %-4d   data\\lvis\\%s"
              % (t, len([i for i in use_pos if i not in failed]), have_pos, count,
                 len([i for i in use_neg if i not in failed]), have_neg, t))
    print("\n(positives and negatives show used / available)")


if __name__ == "__main__":
    main()
