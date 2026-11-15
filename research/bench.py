"""bench.py -- score a detector version against labelled images.

Every run appends one row per (weights, prompt) to results\\bench.csv, so
prompt variants, model sizes and later fine-tuned weights all end up in one
table you can sort.

    python research/bench.py data\\coco128 --target=phone
    python research/bench.py data\\keys --target=keys --prompts="keys,bunch of keys"
    python research/bench.py data\\keys --target=keys --prompts="keys+keychain,keys"
    python research/bench.py data\\keys --target=keys --weights=yolov8s-worldv2.pt,yolov8m-worldv2.pt
    python research/bench.py data\\keys --target=keys --limit=0       # every image
    python research/bench.py data\\keys --target=keys --together      # one pass, see below
    python research/bench.py data\\keys --target=keys --label=Key     # dataset label override

How it scores. A prediction is right when it overlaps a ground-truth box of
the target at IoU >= 0.5; each ground-truth box can only be claimed once.
Ranking every prediction by confidence gives:

    AP50   area under the precision/recall curve -- the headline number,
           independent of any threshold
    F1     the best balance of precision and recall on that curve
    conf   the confidence threshold where that best F1 happens. This is the
           value to copy into CONF instead of guessing 0.05.

A version can be a SET of prompts: write them joined with +, as in
"keys+keychain". The set is loaded together and scored as one detector --
any of them firing counts -- which is how detect.py actually runs.

By default each prompt is scored ALONE, and that is the only fair way to
compare similar prompts. With several loaded at once, each box is credited
only to its highest-scoring prompt, so near-synonyms steal each other's
detections: "glasses" scored 0.006 next to "pair of glasses", and 0.479 on
its own. --together is several times faster, but only trust it to pick a
winner among prompts that mean different things.
"""
import csv
import os
import random
import sys
import time

from ultralytics import YOLO, YOLOWorld
# ultralytics ships its own YAML reader; PyYAML is not installed here.
from ultralytics.utils import ROOT as ULTRALYTICS_ROOT, YAML

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo folder
sys.path.insert(0, ROOT)   # so the top-level modules (detect, targets, guide...) import
from targets import TARGETS  # noqa: E402

IOU_MIN = 0.5
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "bench.csv")
COLUMNS = ["time", "weights", "dataset", "target", "prompt", "mode", "imgsz",
           "images", "gt", "ap50", "f1", "precision", "recall", "conf", "ms_per_img"]


# ---------------------------------------------------------------- dataset

def normalise(name):
    return " ".join(str(name).lower().replace("-", " ").replace("_", " ").split())


def load_names(root):
    """Class names, from the dataset's own yaml or ultralytics' bundled one."""
    for candidate in ("data.yaml", "dataset.yaml"):
        path = os.path.join(root, candidate)
        if os.path.exists(path):
            names = YAML.load(path).get("names")
            return list(names.values()) if isinstance(names, dict) else list(names)
    # coco128 / coco8 ship as bare images+labels; ultralytics carries the yaml.
    bundled = ULTRALYTICS_ROOT / "cfg" / "datasets" / (os.path.basename(root.rstrip("\\/")) + ".yaml")
    if bundled.exists():
        names = YAML.load(bundled).get("names")
        return list(names.values()) if isinstance(names, dict) else list(names)
    sys.exit("no data.yaml in %s -- cannot tell what the class ids mean" % root)


def list_images(root):
    """Every image that lives under an 'images' folder."""
    found = []
    for folder, _, files in os.walk(root):
        if "images" not in folder.split(os.sep):
            continue
        found += [os.path.join(folder, f) for f in files if f.lower().endswith(IMG_EXT)]
    return sorted(found)


def label_path(image):
    """.../images/x.jpg -> .../labels/x.txt (the YOLO layout convention)."""
    parts = image.split(os.sep)
    last = len(parts) - 1 - parts[::-1].index("images")
    parts[last] = "labels"
    return os.path.splitext(os.sep.join(parts))[0] + ".txt"


def read_boxes(image, wanted):
    """Ground-truth boxes of the wanted class ids, as normalised xyxy.

    Handles both plain boxes (class cx cy w h) and the polygons Roboflow
    writes for segmentation datasets (class x1 y1 x2 y2 ...).
    """
    path, boxes = label_path(image), []
    if not os.path.exists(path):
        return boxes                                 # no file = no objects
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            vals = line.split()
            if len(vals) < 5 or int(float(vals[0])) not in wanted:
                continue
            nums = [float(v) for v in vals[1:]]
            if len(nums) == 4:
                cx, cy, w, h = nums
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            else:
                xs, ys = nums[0::2], nums[1::2]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return boxes


# ---------------------------------------------------------------- scoring

def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, truths):
    """Greedy matching, most confident prediction first. -> [(conf, hit)]"""
    taken, out = [False] * len(truths), []
    for conf, box in sorted(preds, key=lambda p: -p[0]):
        best, best_iou = -1, IOU_MIN
        for i, truth in enumerate(truths):
            if not taken[i]:
                overlap = iou(box, truth)
                if overlap >= best_iou:
                    best, best_iou = i, overlap
        if best >= 0:
            taken[best] = True
        out.append((conf, best >= 0))
    return out


def summarise(records, total_truths):
    """AP50 plus the best-F1 operating point from ranked (conf, hit) pairs."""
    records = sorted(records, key=lambda r: -r[0])
    hits = misses = 0
    precision, recall = [], []
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "conf": 0.0}
    for conf, hit in records:
        hits += hit
        misses += not hit
        p = hits / (hits + misses)
        r = hits / total_truths if total_truths else 0.0
        precision.append(p)
        recall.append(r)
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best["f1"]:
            best = {"f1": f1, "precision": p, "recall": r, "conf": conf}
    # Area under the precision envelope (precision made non-increasing).
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap, previous = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - previous)
        previous = r
    return ap, best


# ---------------------------------------------------------------- running

def run_pass(model, open_vocab, prompts, aliases, images, truths, imgsz, pool_as=None):
    """One sweep over the images with these prompts loaded.

    pool_as: score the prompts as ONE version under this name -- any of them
    firing counts. Class-agnostic NMS then removes the duplicates you get
    when "keys" and "keychain" both box the same object.

    -> ({name: [(conf, hit), ...]}, milliseconds per image)
    """
    if open_vocab:
        model.set_classes(prompts)
    keys = [pool_as] if pool_as else prompts
    records = {k: [] for k in keys}
    started = time.perf_counter()
    for image in images:
        # conf=0.001 keeps (nearly) everything so we see the whole curve.
        r = model.predict(image, conf=0.001, imgsz=imgsz, max_det=1000,
                          agnostic_nms=bool(pool_as), verbose=False)[0]
        grouped = {k: [] for k in keys}
        for box in r.boxes:
            name = r.names[int(box.cls)]
            if pool_as:
                key = pool_as                        # the whole set is one bucket
            elif open_vocab:
                key = name                           # the prompt text itself
            elif normalise(name) in aliases:
                key = prompts[0]                     # closed-set: one bucket
            else:
                continue
            grouped[key].append((float(box.conf), [float(v) for v in box.xyxyn[0]]))
        for k in keys:
            records[k] += match(grouped[k], truths[image])
    return records, (time.perf_counter() - started) * 1000 / max(1, len(images))


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    root = os.path.abspath(args[0])

    opt = {"target": None, "prompts": None, "weights": "yolov8s-worldv2.pt",
           "limit": "150", "imgsz": "640", "label": None}
    for flag in flags:
        if "=" in flag:
            key, value = flag[2:].split("=", 1)
            opt[key] = value
    together = "--together" in flags
    if opt["target"] not in TARGETS:
        sys.exit("--target must be one of: %s" % ", ".join(TARGETS))

    target = TARGETS[opt["target"]]
    prompts = ([p.strip() for p in opt["prompts"].split(",") if p.strip()]
               if opt["prompts"] else list(target["prompts"]))
    aliases = {normalise(a) for a in target["aliases"]}
    if opt["label"]:
        aliases = {normalise(l) for l in opt["label"].split(",")}
    imgsz, limit = int(opt["imgsz"]), int(opt["limit"])

    # --- which dataset classes are "the target"? ---
    names = load_names(root)
    wanted = {i for i, n in enumerate(names) if normalise(n) in aliases}
    if not wanted:
        sys.exit("no class in this dataset matches target '%s'.\n  dataset classes: %s\n"
                 "  add an alias in targets.py, or pass --label=<class name>"
                 % (opt["target"], ", ".join(map(str, names))))
    print("dataset   %s" % root)
    print("target    %s  <-  dataset label(s): %s"
          % (opt["target"], ", ".join(names[i] for i in sorted(wanted))))

    # --- images and ground truth ---
    images = list_images(root)
    if not images:
        sys.exit("no images found under an 'images' folder in %s" % root)
    if limit and len(images) > limit:
        images = sorted(random.Random(0).sample(images, limit))   # same subset every run
    truths = {image: read_boxes(image, wanted) for image in images}
    total = sum(len(t) for t in truths.values())
    positives = sum(1 for t in truths.values() if t)
    print("images    %d (%d contain the target, %d boxes)" % (len(images), positives, total))
    if total < 30:
        print("          WARNING: under 30 boxes -- treat these numbers as a rough signal")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    new_file = not os.path.exists(RESULTS)
    rows = []

    for weights in [w.strip() for w in opt["weights"].split(",") if w.strip()]:
        open_vocab = "world" in os.path.basename(weights).lower()
        model = YOLOWorld(weights) if open_vocab else YOLO(weights)
        # Each pass is (prompts to load, name to pool them under or None).
        if not open_vocab:
            passes = [(["(model classes)"], None)]   # nothing to prompt
        elif together:
            if any("+" in p for p in prompts):
                sys.exit("--together scores single prompts; drop the a+b sets or drop the flag")
            passes = [(prompts, None)]
        else:
            passes = [([s.strip() for s in p.split("+")], p) if "+" in p else ([p], None)
                      for p in prompts]

        print("\n%s  imgsz=%d  %s" % (weights, imgsz,
                                      "all prompts in one pass" if together and open_vocab else ""))
        print("  %-28s %6s %6s %6s %6s %6s %8s" % ("prompt", "AP50", "F1", "P", "R", "conf", "ms/img"))
        scored = []
        for group, pool in passes:
            records, ms = run_pass(model, open_vocab, group, aliases, images, truths, imgsz, pool)
            for prompt in ([pool] if pool else group):
                ap, best = summarise(records[prompt], total)
                scored.append((ap, prompt, best, ms))
        for ap, prompt, best, ms in sorted(scored, key=lambda s: -s[0]):
            print("  %-28s %6.3f %6.3f %6.2f %6.2f %6.2f %8.0f"
                  % (prompt, ap, best["f1"], best["precision"], best["recall"], best["conf"], ms))
            rows.append([time.strftime("%Y-%m-%d %H:%M:%S"), weights, os.path.basename(root),
                         opt["target"], prompt, "together" if together else "alone", imgsz,
                         len(images), total, round(ap, 4), round(best["f1"], 4),
                         round(best["precision"], 4), round(best["recall"], 4),
                         round(best["conf"], 4), round(ms, 1)])

    with open(RESULTS, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(COLUMNS)
        writer.writerows(rows)
    print("\n%d row(s) appended to %s" % (len(rows), os.path.relpath(RESULTS, HERE)))


if __name__ == "__main__":
    main()
