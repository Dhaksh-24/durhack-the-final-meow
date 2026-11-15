"""train_verifier.py -- build verifier.json, the CLIP check detect.py runs on every box.

    py train_verifier.py              # every stage (cached: re-runs skip finished ones)
    py train_verifier.py --redo=train # redo one stage and the ones after it
    py train_verifier.py --mine-max=1200

Stages (outputs in data/verifier/, git-ignored):
  select  LVIS v1 train+val annotations (data/lvis/*.json.zip, streamed: no 5 GB load):
          every key, wallet and glasses box, plus up to CAP_DISTRACTOR boxes of each
          look-alike (cards, phones, keyboards, drinking glasses, scissors...).
  web     close-up photos from Wikimedia Commons categories (freely licensed): wallets,
          keys, glasses, and decks of cards / bank cards / phones as negatives.
  fetch   download the photos (COCO images for LVIS, Commons thumbnails).
  mine    run YOLO-World (detect.Detector.propose, exactly as at runtime) on the LVIS
          photos. Proposals overlapping a real key/wallet/glasses box become positives
          framed the way the detector frames them; proposals on photos verified NOT to
          contain that thing become hard negatives ("other").
  embed   CLIP ViT-B/32 embedding of every crop (detect.embed, same crop as runtime).
  train   multinomial logistic regression on the embeddings, blended with CLIP zero-shot
          scores (alpha, chosen on validation): the classifier learns from the examples,
          zero-shot covers look-alikes the examples lack. Per-class thresholds are chosen
          on validation proposals for precision (F0.5). Reported on held-out test photos.

Photos are split by photo (70% train, 15% validation, 15% test) so no photo is in two.
Your own photos help most: put crops in data/own/keys/, data/own/glasses/ and
data/own/other/ (look-alikes: a deck of cards, your wallet) and re-run --redo=embed.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the repo folder
sys.path.insert(0, ROOT)   # so the top-level modules (detect, targets, guide...) import
import detect  # noqa: E402
from pull_lvis import merge_touching  # noqa: E402

LVIS_DIR = os.path.join(ROOT, "data", "lvis")
OUT = os.path.join(ROOT, "data", "verifier")
IMAGES = os.path.join(OUT, "images")
OWN = os.path.join(ROOT, "data", "own")
SPLITS = ["lvis_v1_train.json.zip", "lvis_v1_val.json.zip"]
STAGES = ["select", "web", "fetch", "mine", "embed", "train"]

CLASSES = ["keys", "glasses", "other"]   # what the tracker reports; everything else is "other"
# LVIS categories collected. Wallets are still collected, but trained as "other": a look-alike.
TARGETS = {"keys": ["key"], "glasses": ["spectacles", "sunglasses"], "wallet": ["wallet"]}
AS_OTHER = {"wallet"}      # collected labels that are look-alikes, not things we report
# Look-alikes, labelled "other": cards, phones, keyboards, drinking glasses, jewellery...
DISTRACTORS = [
    "card", "business_card", "identity_card", "birthday_card", "postcard", "booklet",
    "notebook", "envelope", "passport", "money", "cigarette_case", "pouch", "calculator",
    "cellular_telephone", "remote_control", "computer_keyboard", "coaster", "handbag",
    "glass_(drink_container)", "wineglass", "goggles", "mask", "earphone",
    "bracelet", "earring", "scissors", "bottle_opener", "lanyard", "tag", "coin",
]
CAP_GLASSES = 900          # glasses boxes are plentiful; keys and wallets: all of them
CAP_DISTRACTOR = 120       # boxes per look-alike category
CAP_DISTRACTOR_IMGS = 60   # photos per look-alike category
N_RANDOM = 300             # extra photos verified free of a target, for mining false alarms
MIN_SIDE, MIN_AREA = 12, 300   # px: smaller ground-truth boxes are too coarse to learn from
OTHER_PER_IMAGE = 4        # hardest false-alarm proposals kept per photo
WEB_PER_CATEGORY = 120
MAX_THRESHOLD = 0.90       # validation has few keys: don't let it pick 0.98 and miss real ones

# Commons categories (missing ones are skipped). Positives are filtered by CLIP zero-shot
# (drops e.g. a guitar in "Keys"); negatives are kept as they are, hard ones included.
WEB = {
    "keys": ["Groups of keys", "Close-up photographs of keys", "Automobile keys", "Keyrings",
             "Key fobs", "Keychains"],
    "wallet": ["Leather wallets", "Wallets in Germany", "Credit card holders", "Wallets"],
    "glasses": ["Eyeglasses", "Glasses", "Reading glasses", "Spectacles (eyewear)", "Sunglasses"],
    "other": ["Playing cards", "Decks of playing cards", "Playing card decks", "Poker",
              "Credit cards", "Payment cards", "Smartphones", "Mobile phones", "Notebooks",
              "Passports", "Remote controls", "Computer keyboards", "Drinking glasses",
              "Business cards", "Banknotes", "Card games"],
}
UA = "trace-dataset-builder/0.1 (DurHack research prototype)"

# Zero-shot prompts per class ("a photo of ..."). "other" names the look-alikes.
PROMPTS = {
    "keys": ["keys", "a bunch of keys on a keyring", "house keys", "car keys", "a key",
             "a keychain with keys"],
    "glasses": ["eyeglasses", "a pair of glasses", "reading glasses", "sunglasses",
                "spectacles", "folded glasses on a table"],
    "other": ["a wallet", "a leather wallet", "a card holder wallet",
              "a deck of playing cards", "a playing card", "a fan of playing cards",
              "a credit card", "a bank card", "a smartphone", "a mobile phone",
              "a TV remote control", "a computer keyboard", "a notebook", "a book",
              "a passport", "an envelope", "a drinking glass", "a wine glass",
              "a pair of scissors", "a pen", "a bracelet", "an earring", "a necklace",
              "a wristwatch", "goggles", "a face mask", "headphones", "a coaster",
              "a cushion", "a table top", "a carpet", "a hand", "a cable", "a bag",
              "a coin", "banknotes", "a calculator", "a bottle opener", "a toy",
              "a box", "a shadow", "a blurry photo"],
}


def p(*a):
    print(*a, flush=True)


def split_of(key):
    h = zlib.crc32(str(key).encode()) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name) + ".part", "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(os.path.join(OUT, name) + ".part", os.path.join(OUT, name))


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as fh:
        return json.load(fh)


def done(name):
    return os.path.exists(os.path.join(OUT, name))


def stream(zpath, prefix):
    import ijson
    with zipfile.ZipFile(zpath) as z, z.open(z.namelist()[0]) as f:
        yield from ijson.items(f, prefix, use_float=True)


# ------------------------------------------------------------- select
def stage_select():
    zips = [os.path.join(LVIS_DIR, s) for s in SPLITS]
    for z in zips:
        if not os.path.exists(z):
            sys.exit("missing %s -- download it from https://dl.fbaipublicfiles.com/LVIS/%s"
                     % (z, os.path.basename(z)))
    p("select: categories ...")
    name_of = {c["id"]: c["name"] for c in stream(zips[1], "categories.item")}
    id_of = {v: k for k, v in name_of.items()}
    target_of = {id_of[n]: t for t, ns in TARGETS.items() for n in ns}
    distractor_ids = {id_of[n] for n in DISTRACTORS if n in id_of}
    rng = random.Random(0)

    boxes = []          # [image_id, label, category name, x, y, w, h]
    for z in zips:
        t0 = time.time()
        for a in stream(z, "annotations.item"):
            c = a["category_id"]
            if c in target_of or c in distractor_ids:
                x, y, w, h = a["bbox"]
                boxes.append([a["image_id"], target_of.get(c, "other"), name_of[c], x, y, w, h])
        p("  %s annotations: %d boxes so far (%.0fs)" % (os.path.basename(z), len(boxes), time.time() - t0))

    # keys on one ring are one object (pull_lvis does the same)
    by_img = {}
    for b in boxes:
        by_img.setdefault((b[0], b[2]), []).append(b)
    merged = []
    for (img, cat), bs in by_img.items():
        if bs[0][1] == "keys":
            for x, y, w, h in merge_touching([b[3:] for b in bs]):
                merged.append([img, "keys", cat, x, y, w, h])
        else:
            merged.extend(bs)
    boxes = [b for b in merged if min(b[5], b[6]) >= MIN_SIDE and b[5] * b[6] >= MIN_AREA]

    # caps: all keys and wallets; some glasses; a few per look-alike
    rng.shuffle(boxes)
    keep, per_cat, imgs_cat = [], {}, {}
    for b in boxes:
        if b[1] in ("keys", "wallet"):
            keep.append(b)
            continue
        cap = CAP_GLASSES if b[1] == "glasses" else CAP_DISTRACTOR
        s = imgs_cat.setdefault(b[2], set())
        if per_cat.get(b[2], 0) >= cap or (b[1] == "other" and b[0] not in s and len(s) >= CAP_DISTRACTOR_IMGS):
            continue
        per_cat[b[2]] = per_cat.get(b[2], 0) + 1
        s.add(b[0])
        keep.append(b)
    wanted = {b[0] for b in keep}

    # photo records: URL, size, which categories were checked (LVIS is federated)
    tid = {t: [id_of[n] for n in ns] for t, ns in TARGETS.items()}
    images, spare = {}, []
    for z in zips:
        for im in stream(z, "images.item"):
            rec = {"url": im["coco_url"], "w": im["width"], "h": im["height"],
                   "neg": [c for c in im.get("neg_category_ids", []) if c in target_of],
                   "notex": [c for c in im.get("not_exhaustive_category_ids", []) if c in target_of]}
            if im["id"] in wanted:
                images[im["id"]] = rec
            elif rec["neg"]:
                spare.append((im["id"], rec))
    for i, rec in rng.sample(spare, min(N_RANDOM, len(spare))):
        images[i] = rec
    # which targets were labelled on each photo
    pos = {}
    for b in keep:
        if b[1] != "other":
            pos.setdefault(b[0], set()).add(b[2])
    for i, rec in images.items():
        rec["pos"] = sorted({id_of[n] for n in pos.get(i, ())})
    counts = {}
    for b in keep:
        counts[b[1]] = counts.get(b[1], 0) + 1
    p("select: %d boxes on %d photos: %s" % (len(keep), len(images), counts))
    save("lvis.json", {"boxes": keep, "images": {str(k): v for k, v in images.items()},
                       "target_ids": tid})


# ---------------------------------------------------------------- web
def commons(category):
    q = {"action": "query", "generator": "categorymembers", "gcmtitle": "Category:" + category,
         "gcmtype": "file", "gcmlimit": str(WEB_PER_CATEGORY), "prop": "imageinfo",
         "iiprop": "url|mime", "iiurlwidth": "448", "format": "json"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            break
        except Exception:
            time.sleep(5 * (attempt + 1))
    else:
        return []
    out = []
    for pg in (data.get("query") or {}).get("pages", {}).values():
        ii = (pg.get("imageinfo") or [{}])[0]
        if ii.get("mime") in ("image/jpeg", "image/png") and ii.get("thumburl"):
            out.append({"title": pg["title"], "url": ii["thumburl"]})
    return out


def stage_web():
    items, seen = [], set()
    for label, cats in WEB.items():
        for c in cats:
            got = commons(c)
            new = [g for g in got if g["title"] not in seen]
            for g in new:
                seen.add(g["title"])
                items.append({"label": label, "category": c, **g})
            p("  web %-8s %-30s %3d photos" % (label, c, len(new)))
            time.sleep(1.5)   # be polite to the API
    save("web.json", items)


# -------------------------------------------------------------- fetch
def fetch(url, path):
    if os.path.exists(path):
        return True
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(path + ".part", "wb") as fh:
                fh.write(data)
            os.replace(path + ".part", path)
            return True
        except Exception:
            time.sleep(2 * (attempt + 1))
    return False


def lvis_path(i):
    return os.path.join(IMAGES, "coco_%s.jpg" % i)


def web_path(w):
    return os.path.join(IMAGES, "web_%08x.jpg" % zlib.crc32(w["title"].encode()))


def stage_fetch():
    os.makedirs(IMAGES, exist_ok=True)
    lv, web = load("lvis.json"), load("web.json")
    jobs = [(r["url"], lvis_path(i)) for i, r in lv["images"].items()]
    t0, ok = time.time(), 0
    with ThreadPoolExecutor(16) as ex:
        for n, r in enumerate(ex.map(lambda j: fetch(*j), jobs)):
            ok += r
            if n % 500 == 499:
                p("  coco %d/%d (%.0fs)" % (n + 1, len(jobs), time.time() - t0))
    p("fetch: %d/%d COCO photos" % (ok, len(jobs)))
    jobs = [(w["url"], web_path(w)) for w in web]
    with ThreadPoolExecutor(3) as ex:     # Commons asks for gentle clients
        ok = sum(ex.map(lambda j: fetch(*j), jobs))
    p("fetch: %d/%d web photos" % (ok, len(jobs)))
    save("fetched.json", {"coco": len(lv["images"]), "web": ok})   # marks the stage done


# --------------------------------------------------------------- mine
def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def checked(rec, ids):
    """Every LVIS category of the target was checked on this photo (labelled or verified absent)."""
    return all((c in rec["pos"] or c in rec["neg"]) and c not in rec["notex"] for c in ids)


def stage_mine(mine_max):
    lv = load("lvis.json")
    gt = {}
    for b in lv["boxes"]:
        if b[1] != "other":
            gt.setdefault(str(b[0]), []).append((b[1], b[3:]))
    ids = list(lv["images"])
    random.Random(1).shuffle(ids)
    ids = sorted(ids[:mine_max], key=lambda i: i not in gt)  # photos with targets first
    det = detect.Detector(verify=False, conf=detect.PROPOSE_CONF)
    props, t0 = [], time.time()
    for n in range(0, len(ids), 8):
        chunk = [(i, cv2.imread(lvis_path(i))) for i in ids[n:n + 8]]
        chunk = [(i, f) for i, f in chunk if f is not None]
        if not chunk:
            continue
        for (i, f), ps in zip(chunk, det.propose([f for _, f in chunk])):
            rec = lv["images"][i]
            for cat, box, score in ps:
                xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
                best = max(((iou_xywh(xywh, g), t) for t, g in gt.get(i, [])), default=(0.0, None))
                if best[0] >= 0.5:
                    label = best[1]
                elif best[0] < 0.2 and checked(rec, lv["target_ids"][cat]):
                    label = "other"
                else:
                    continue    # half-overlap or an unchecked photo: can't say, don't learn
                props.append([i, cat, round(score, 4), label] + [round(v, 1) for v in box])
        if n % 400 == 0:
            p("  mine %d/%d photos, %d proposals (%.0fs)" % (n, len(ids), len(props), time.time() - t0))
    # keep every proposal on a real target; the hardest few false alarms per photo
    out, per = [], {}
    for pr in sorted(props, key=lambda r: -r[2]):
        if pr[3] == "other":
            per[pr[0]] = per.get(pr[0], 0) + 1
            if per[pr[0]] > OTHER_PER_IMAGE:
                continue
        out.append(pr)
    labels = {}
    for pr in out:
        labels[pr[3]] = labels.get(pr[3], 0) + 1
    p("mine: %d proposals from %d photos: %s" % (len(out), len(ids), labels))
    save("proposals.json", {"photos": ids, "proposals": out})


# -------------------------------------------------------------- embed
def stage_embed():
    lv, web, mined = load("lvis.json"), load("web.json"), load("proposals.json")
    recs, crops = [], []   # rec: (source, label, split, photo, yolo_cat, yolo_score, box)

    def add(img, box, label, source, photo, cat=None, score=None, flip=False):
        c = detect.crop_box(img, box)
        if c.shape[0] < 4 or c.shape[1] < 4:
            return
        crops.append(c)
        recs.append((source, label, split_of(photo), str(photo), cat, score, box))
        if flip and split_of(photo) == "train":
            crops.append(cv2.flip(c, 1))
            recs.append((source + "-flip", label, "train", str(photo), cat, score, box))

    by_photo = {}
    for b in lv["boxes"]:
        by_photo.setdefault(str(b[0]), []).append(b)
    for i, bs in by_photo.items():
        img = cv2.imread(lvis_path(i))
        if img is None:
            continue
        for b in bs:
            x, y, w, h = b[3:]
            add(img, [x, y, x + w, y + h], b[1], "lvis-gt", i, cat=b[2], flip=b[1] != "other")
    p("embed: %d ground-truth crops" % len(crops))
    by_photo = {}
    for pr in mined["proposals"]:
        by_photo.setdefault(pr[0], []).append(pr)
    for i, ps in by_photo.items():
        img = cv2.imread(lvis_path(i))
        if img is None:
            continue
        for pr in ps:
            add(img, pr[4:8], pr[3], "proposal", i, pr[1], pr[2])
    p("embed: %d crops with proposals" % len(crops))
    for w in web:
        img = cv2.imread(web_path(w))
        if img is not None:
            h, wd = img.shape[:2]
            add(img, [0, 0, wd, h], w["label"], "web", w["title"], cat=w["category"],
                flip=w["label"] != "other")
    for label in CLASSES:   # your own photos: data/own/<class>/*.jpg (whole photo = the object)
        d = os.path.join(OWN, label)
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            img = cv2.imread(os.path.join(d, fn))
            if img is not None:
                h, wd = img.shape[:2]
                add(img, [0, 0, wd, h], label, "own", "own/" + label + "/" + fn, cat="own", flip=True)
    p("embed: %d crops in all; CLIP ViT-B/32 ..." % len(crops))
    t0, embs = time.time(), []
    for k in range(0, len(crops), 256):
        embs.append(detect.embed(crops[k:k + 256]))
        p("  %d/%d (%.0fs)" % (min(k + 256, len(crops)), len(crops), time.time() - t0))
    np.savez_compressed(os.path.join(OUT, "embeddings.npz"), X=np.concatenate(embs),
                        meta=np.array([json.dumps(r) for r in recs]))


# -------------------------------------------------------------- train
def stage_train():
    from sklearn.linear_model import LogisticRegression
    d = np.load(os.path.join(OUT, "embeddings.npz"))
    X = d["X"]
    meta = [json.loads(m) for m in d["meta"]]
    src = np.array([m[0] for m in meta])
    y = np.array([CLASSES.index("other" if m[1] in AS_OTHER else m[1]) for m in meta])
    split = np.array([m[2] for m in meta])
    is_web = np.array([s.startswith("web") for s in src])
    prompts = [PROMPTS[c] for c in CLASSES]
    Z = detect.zero_shot_scores(X, prompts)

    # web positives: drop photos CLIP can't see the object in (wrong category members)
    zs_pick = Z.argmax(1)
    bad_web = is_web & (y != CLASSES.index("other")) & (zs_pick != y)
    p("train: dropped %d/%d web positives zero-shot can't see the object in"
      % (bad_web.sum(), (is_web & (y != CLASSES.index("other"))).sum()))
    use = ~bad_web
    tr, va, te = use & (split == "train"), use & (split == "val"), use & (split == "test")
    prop = np.array([s == "proposal" for s in src])

    def fit(C):
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        return m.fit(X[tr], y[tr])

    def probs(m, alpha, idx):
        z = m.decision_function(X[idx]) + alpha * Z[idx]
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def nll(P, yy):
        return float(-np.log(P[np.arange(len(yy)), yy] + 1e-9).mean())

    best = None
    for C in (0.3, 1.0, 3.0, 10.0):
        m = fit(C)
        for alpha in (0.0, 0.05, 0.1, 0.2, 0.4):
            s = nll(probs(m, alpha, va), y[va])
            p("  C=%-4s alpha=%-4s val log-loss %.4f" % (C, alpha, s))
            if best is None or s < best[0]:
                best = (s, C, alpha, m)
    _, C, alpha, model = best
    p("train: chose C=%s alpha=%s" % (C, alpha))

    # thresholds on validation PROPOSALS (what the detector really hands over)
    def decide(P, c, t):
        return (P.argmax(1) == c) & (P[:, c] >= t)

    thresholds = {}
    vp = va & prop
    Pv = probs(model, alpha, vp)
    for c, name in enumerate(CLASSES[:-1]):
        pos = y[vp] == c
        bestf = (-1, 0.5)
        for t in np.arange(0.30, MAX_THRESHOLD + 0.001, 0.01):
            k = decide(Pv, c, t)
            tp, fp, fn = (k & pos).sum(), (k & ~pos).sum(), (~k & pos).sum()
            pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            f05 = 1.25 * pr * rc / max(0.25 * pr + rc, 1e-9)
            if f05 > bestf[0] + 1e-9:
                bestf = (f05, round(float(t), 2))
        thresholds[name] = bestf[1]
    p("train: thresholds %s" % thresholds)

    report = evaluate(model, alpha, thresholds, X, Z, y, src, split, use, meta)
    W = model.coef_.astype(float)
    b = model.intercept_.astype(float)
    out = {"about": "CLIP ViT-B/32 crop classifier for detect.py; built by train_verifier.py",
           "clip": "ViT-B/32", "crop_pad": detect.CROP_PAD, "classes": CLASSES,
           "W": [[round(v, 6) for v in row] for row in W], "b": [round(v, 6) for v in b],
           "alpha": alpha, "prompts": PROMPTS, "thresholds": thresholds, "report": report}
    with open(detect.VERIFIER_FILE, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    p("wrote %s" % detect.VERIFIER_FILE)


def evaluate(model, alpha, thresholds, X, Z, y, src, split, use, meta):
    """Held-out test photos. Proposal level = what the tracker would report."""
    te = use & (split == "test")
    P = model.decision_function(X[te]) + alpha * Z[te]
    P = np.exp(P - P.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    yt, st = y[te], src[te]
    mt = [m for m, k in zip(meta, te) if k]
    rep = {"note": "held-out test photos (15%, split by photo); never used for training or thresholds"}

    # 1) end to end on LVIS proposals: YOLO-World alone vs with the verifier
    prop = st == "proposal"
    yolo_cat = np.array([CLASSES.index(m[4]) if m[4] in CLASSES else -1 for m in mt])
    yolo_score = np.array([m[5] if m[5] is not None else 0 for m in mt], float)
    e2e = {}
    for c, name in enumerate(CLASSES[:-1]):
        pos = prop & (yt == c)
        # YOLO alone: best-F1 threshold on its own score, chosen on the test set (generous)
        bestf = None
        for t in np.unique(np.round(yolo_score[prop & (yolo_cat == c)], 3)):
            k = prop & (yolo_cat == c) & (yolo_score >= t)
            tp, fp = (k & pos).sum(), (k & ~pos).sum()
            pr, rc = tp / max(tp + fp, 1), tp / max(pos.sum(), 1)
            f1 = 2 * pr * rc / max(pr + rc, 1e-9)
            if bestf is None or f1 > bestf[0]:
                bestf = (f1, pr, rc, float(t))
        k = prop & (P.argmax(1) == c) & (P[:, c] >= thresholds[name])
        tp, fp = (k & pos).sum(), (k & ~pos).sum()
        pr, rc = tp / max(tp + fp, 1), tp / max(pos.sum(), 1)
        e2e[name] = {"proposals_on_real": int(pos.sum()),
                     "yolo_only": {"precision": round(bestf[1], 3), "recall": round(bestf[2], 3),
                                   "conf": bestf[3]} if bestf else None,
                     "with_verifier": {"precision": round(float(pr), 3), "recall": round(float(rc), 3),
                                       "threshold": thresholds[name]}}
    rep["proposals"] = e2e

    # 2) crop classification: every test crop, by source
    pred = np.where(P[np.arange(len(P)), P.argmax(1)] >= np.array(
        [thresholds.get(CLASSES[k], 0) for k in P.argmax(1)]), P.argmax(1), CLASSES.index("other"))
    acc = {}
    for s in ("lvis-gt", "proposal", "web"):
        k = st == s
        if k.any():
            acc[s] = {"crops": int(k.sum()), "accuracy": round(float((pred[k] == yt[k]).mean()), 3)}
    rep["crops"] = acc

    # 3) look-alikes (ground-truth boxes / web photos of "other"): share called keys,
    #    or glasses, per source category. Proposals give YOLO's view of the same.
    other = CLASSES.index("other")
    look = {}
    for m, pr_, yy in zip(mt, pred, yt):
        if yy == other and m[0] in ("lvis-gt", "web"):
            n = look.setdefault(m[0] + ": " + m[4], [0, 0, {}])
            n[0] += 1
            if pr_ != other:
                n[1] += 1
                n[2][CLASSES[pr_]] = n[2].get(CLASSES[pr_], 0) + 1
    rep["look_alikes_mistaken"] = {k: {"crops": n, "mistaken": e, "as": a}
                                   for k, (n, e, a) in sorted(look.items()) if n >= 3}
    k = prop & (yt == other)
    yolo_fa = {}
    for c, name in enumerate(CLASSES[:-1]):
        t = (e2e[name]["yolo_only"] or {}).get("conf", 1.0)
        yolo_fa[name] = {"yolo_only": int((k & (yolo_cat == c) & (yolo_score >= t)).sum()),
                         "with_verifier": int((k & (P.argmax(1) == c) & (P[:, c] >= thresholds[name])).sum())}
    rep["false_alarms_on_test_photos"] = yolo_fa
    p(json.dumps(rep, indent=2))
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--redo", choices=STAGES, help="redo this stage and the ones after it")
    ap.add_argument("--mine-max", type=int, default=2000, help="photos to run YOLO-World on")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    first = STAGES.index(a.redo) if a.redo else len(STAGES)
    out = {"select": "lvis.json", "web": "web.json", "fetch": "fetched.json", "mine": "proposals.json",
           "embed": "embeddings.npz", "train": None}
    for k, s in enumerate(STAGES):
        if k < first and out[s] and done(out[s]):
            p("%s: cached" % s)
            continue
        t0 = time.time()
        {"select": stage_select, "web": stage_web, "fetch": stage_fetch,
         "mine": lambda: stage_mine(a.mine_max), "embed": stage_embed, "train": stage_train}[s]()
        p("%s: done in %.0fs" % (s, time.time() - t0))


if __name__ == "__main__":
    main()
