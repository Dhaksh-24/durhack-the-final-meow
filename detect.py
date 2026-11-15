"""YOLO-World detector (CPU) -- used standalone AND imported by multicam.py.

Standalone, with optional Gemini verification:

    python detect.py                          # live webcam, Esc to quit
    python detect.py photo.jpg                # one image
    python detect.py --classes=keys,glasses   # override the prompts
    python detect.py --imgsz=320              # smaller input, ~3x faster
    python detect.py --conf=0.02              # surface weaker detections
    python detect.py --no-verify              # detection only, no Gemini
    python detect.py --list-models            # what your key can reach
    python detect.py --no-window --frames=30  # headless benchmark

For the tracker, Detector checks every YOLO-World box with a CLIP classifier
(verifier.json, built by train_verifier.py) -- see "Second stage" below.

Standalone, YOLO-World runs deliberately loose (CONF is low) so it rarely misses, then
Gemini confirms or rejects each candidate crop. The API call happens on a
background thread, so the video loop never waits on the network -- verdicts
land a beat after the box does. Put GEMINI_API_KEY in a .env file beside
this script (see .env.example); with no key it runs unverified.

Used by the tracker (multicam.py):

    from detect import Detector
    det = Detector()
    per_frame = det.detect([frame1, frame2])   # one list per frame of
                                               # (category, [x1, y1, x2, y2], conf)

THE CONTRACT: multicam.py only ever calls Detector(conf=, imgsz=),
Detector.detect(frames), reads CONF and IMGSZ, and may use load_clip() and
Detector.last_embeddings (the CLIP embedding of each returned box, or None). Swap the model inside
freely -- as long as detect() returns the same shape, the tracker keeps
working. Importing this module never opens a camera; the standalone code
only runs under `python detect.py`.
"""
import json
import os
import sys
import threading
import time

import cv2
from ultralytics import YOLOWorld

CLASSES = ["keys", "keychain", "car keys"]
CONF = 0.05          # low on purpose -- Gemini does the filtering
IMGSZ = 640
LIMIT = 0            # webcam: stop after N frames (0 = until Esc)
VERIFY_EVERY = 2.0   # minimum seconds between Gemini calls
MIN_CROP = 32        # px -- smaller crops are too coarse to judge
MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3.8-flash"   # --list-models


# ---------------------------------------------------------------------------
# Tracker interface, as used by multicam.py (from the adelver branch).
# ---------------------------------------------------------------------------

# category -> prompt synonyms. Synonyms are merged into ONE category, so a set
# of keys that matches "keys" AND "keychain" is still one detection.
PROMPTS = {
    "keys": ["keys", "keychain", "car keys"],
    "glasses": ["glasses", "eyeglasses", "reading glasses"],
}
WEIGHTS = "yolov8s-worldv2.pt"
MERGE_IOU = 0.5     # synonym boxes overlapping more than this = the same object


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(
        a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Second stage: a CLIP check on every candidate box (train_verifier.py).
#
# YOLO-World alone is a weak judge of small things (LVIS AP50: keys 0.11,
# glasses 0.48, see targets.BEST) and it calls cards, keyboards and drinking
# glasses keys or glasses. So it runs loose and only PROPOSES boxes; each box is cropped, put
# through CLIP ViT-B/32 and scored by a small classifier over
# {keys, glasses, other}, trained on LVIS photos, close-up web photos
# and YOLO-World's own false alarms, and blended with CLIP's zero-shot
# knowledge of look-alikes (cards, phones, keyboards, drinking glasses...).
# A box survives only if "other" loses and its class clears a threshold
# chosen for precision. The returned conf is then the classifier's
# probability, not YOLO's score.
# ---------------------------------------------------------------------------

VERIFIER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verifier.json")
PROPOSE_CONF = 0.02  # YOLO-World threshold when the verifier is on: recall first
MAX_AREA = 0.40      # drop boxes bigger than this share of the frame (sofas, tables);
                     # the verifier rejects furniture itself, so this can be generous
MIN_SIDE = 8         # px -- smaller boxes can't be judged
TOP_K = 8            # most proposals verified per frame (CPU budget)
CROP_PAD = 0.10      # context around each box for CLIP, as a share of its size

_CLIP = None


def load_clip():
    """CLIP ViT-B/32, loaded once and shared (multicam.py's fingerprints use it too)."""
    global _CLIP
    if _CLIP is None:
        import clip
        import torch
        from ultralytics.utils import WEIGHTS_DIR
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, pre = clip.load("ViT-B/32", device=device, download_root=str(WEIGHTS_DIR / "clip"))
        model.eval()
        _CLIP = (model, pre, device)
    return _CLIP


def crop_box(frame, box, pad=CROP_PAD):
    """BGR crop of box [x1, y1, x2, y2] with `pad` of its size around it."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    c = frame[y1:y2, x1:x2]
    return c if c.size else frame[0:2, 0:2]


def embed(crops_bgr, batch=32):
    """L2-normalised CLIP image embeddings, (n, 512) float32."""
    import numpy as np
    import torch
    from PIL import Image
    if not len(crops_bgr):
        return np.zeros((0, 512), np.float32)
    model, pre, device = load_clip()
    out = []
    with torch.no_grad():
        for i in range(0, len(crops_bgr), batch):
            x = torch.stack([pre(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                             for c in crops_bgr[i:i + batch]])
            f = model.encode_image(x.to(device)).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embed_text(texts):
    import clip
    import torch
    model, _, device = load_clip()
    with torch.no_grad():
        f = model.encode_text(clip.tokenize(texts).to(device)).float()
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()


def zero_shot_scores(embs, class_prompts, text=None):
    """Per class, log-sum-exp over its prompts of CLIP's 100 * cosine: (n, classes)."""
    import numpy as np
    if text is None:
        text = [embed_text(p) for p in class_prompts]
    out = []
    for t in text:
        s = 100.0 * embs @ t.T
        m = s.max(1, keepdims=True)
        out.append((m + np.log(np.exp(s - m).sum(1, keepdims=True)))[:, 0])
    z = np.stack(out, 1)
    return z - z.mean(1, keepdims=True)


class ClipVerifier:
    """Scores crops with the classifier saved by train_verifier.py (verifier.json)."""

    def __init__(self, path=VERIFIER_FILE):
        import numpy as np
        with open(path, encoding="utf-8") as fh:
            v = json.load(fh)
        self.classes = v["classes"]
        self.W, self.b = np.array(v["W"], np.float32), np.array(v["b"], np.float32)
        self.alpha = float(v["alpha"])
        self.thresholds = v["thresholds"]
        self.prompts = [v["prompts"][c] for c in self.classes]
        self.text = [embed_text(p) for p in self.prompts]
        self.report = v.get("report", {})

    def logits(self, embs):
        return embs @ self.W.T + self.b + self.alpha * zero_shot_scores(embs, None, self.text)

    def probs(self, embs):
        import numpy as np
        z = self.logits(embs)
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def judge(self, probs_row):
        """(category, probability) if the crop passes, else None."""
        k = int(probs_row.argmax())
        c = self.classes[k]
        if c == "other" or probs_row[k] < self.thresholds.get(c, 1.0):
            return None
        return c, float(probs_row[k])


class Detector:
    def __init__(self, prompts=None, conf=None, imgsz=IMGSZ, weights=WEIGHTS, verify=True):
        """verify: use the CLIP check from verifier.json if it exists (see above).
        conf: YOLO-World threshold; defaults to PROPOSE_CONF with the check, CONF without."""
        prompts = prompts or PROMPTS
        self.verifier = None
        if verify and os.path.exists(VERIFIER_FILE):
            self.verifier = ClipVerifier()
            print("verifier on: %s (classes %s)" % (os.path.basename(VERIFIER_FILE),
                                                     ", ".join(self.verifier.classes)))
        elif verify:
            print("no verifier.json -- YOLO-World only (run train_verifier.py)")
        if conf is None:
            conf = PROPOSE_CONF if self.verifier else CONF
        self.conf, self.imgsz = conf, imgsz
        self.last_embeddings = []   # per frame, CLIP embedding of each returned detection
        self.categories = list(prompts)
        self.flat, self.cat_of = [], []
        for cat, words in prompts.items():
            for w in words:
                self.flat.append(w)
                self.cat_of.append(cat)
        self.model = YOLOWorld(weights)
        print("encoding prompts:", self.flat)
        t0 = time.perf_counter()
        # ~6s of CLIP text encoding -- ONCE, never per frame
        self.model.set_classes(self.flat)
        print("ready in %.1fs" % (time.perf_counter() - t0))

    def propose(self, frames):
        """YOLO-World boxes per frame, synonyms merged, oversized/tiny boxes dropped,
        at most TOP_K by score: [(category, box, score), ...]."""
        results = self.model.predict(frames, imgsz=self.imgsz, conf=self.conf, verbose=False)
        out = []
        for f, r in zip(frames, results):
            h, w = f.shape[:2]
            keep = [(c, b, s) for c, b, s in self._merge(r)
                    if (b[2] - b[0]) * (b[3] - b[1]) <= MAX_AREA * w * h
                    and min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE]
            out.append(keep[:TOP_K])
        return out

    def detect(self, frames):
        """frames: one image or a list. Returns a list (one per frame) of
        [(category, [x1, y1, x2, y2], conf), ...] with synonyms merged.
        With the verifier, category may be corrected (YOLO said glasses, it's keys)
        and conf is the verifier's probability."""
        single = not isinstance(frames, list)
        frames = [frames] if single else frames
        props = self.propose(frames)
        if self.verifier is None:
            self.last_embeddings = [None] * len(frames)
            return props[0] if single else props
        import numpy as np
        crops = [crop_box(f, b) for f, p in zip(frames, props) for _, b, _ in p]
        embs = embed(crops)
        probs = self.verifier.probs(embs) if len(crops) else []
        out, self.last_embeddings, k = [], [], 0
        for p in props:
            kept, ke = [], []
            for cat, box, score in p:
                # a category the verifier wasn't trained on passes through unchecked
                j = self.verifier.judge(probs[k]) if cat in self.verifier.classes else (cat, score)
                if j is not None:
                    kept.append((j[0], box, j[1], embs[k]))
                k += 1
            kept.sort(key=lambda d: -d[2])
            final = []
            for d in kept:   # relabelling can create duplicates: keep the best per object
                if all(f[0] != d[0] or _iou(f[1], d[1]) < MERGE_IOU for f in final):
                    final.append(d)
            out.append([(c, b, s) for c, b, s, _ in final])
            self.last_embeddings.append(np.stack([e for *_, e in final]) if final else np.zeros((0, 512), np.float32))
        return out[0] if single else out

    def _merge(self, r):
        raw = [(self.cat_of[int(c)], b, float(s)) for b, c, s in
               zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        raw.sort(key=lambda d: -d[2])
        kept = []
        for cat, box, s in raw:  # keep the strongest box per object per category
            if all(k[0] != cat or _iou(k[1], box) < MERGE_IOU for k in kept):
                kept.append((cat, box, s))
        return kept


# ---------------------------------------------------------------------------
# Gemini verification (standalone mode only).
# ---------------------------------------------------------------------------

def load_key():
    """GEMINI_API_KEY from the environment, else from .env beside this file."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() == "GEMINI_API_KEY":
                    return value.strip().strip('"').strip("'")
    return ""


class Verifier:
    """Asks Gemini whether a cropped box really shows what it was labelled."""

    PROMPT = ('Does this cropped photo show "%s"? Reply with JSON only: '
              '{"match": true or false, "what": "<what it actually shows, 4 words max>"}')

    SYSTEM = ("You audit the output of an object detector. Be strict: if the view is "
              "partial, blurred or ambiguous, it is not a match.")

    def __init__(self, key, model):
        from google import genai
        self.client = genai.Client(api_key=key)
        self.model = model
        self.lock = threading.Lock()
        self.verdicts = {}       # label -> (state, text, timestamp)
        self.job = None
        self.last = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def ask(self, jpeg, label):
        """Blocking call. Returns (state, text) with state one of yes/no/err."""
        from google.genai import types
        try:
            reply = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                    self.PROMPT % label,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    system_instruction=self.SYSTEM,
                ),
            )
            data = json.loads(reply.text)
            return ("yes" if data.get("match") else "no", str(data.get("what", ""))[:40])
        except Exception as exc:
            return "err", "%s: %s" % (type(exc).__name__, str(exc)[:70])

    def offer(self, frame, xyxy, label):
        """Queue a crop if the worker is idle and the cooldown has elapsed."""
        if time.time() - self.last < VERIFY_EVERY:
            return
        with self.lock:
            if self.job is not None:
                return
        height, width = frame.shape[:2]
        pad = 12
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        crop = frame[max(0, y1 - pad):min(height, y2 + pad),
                     max(0, x1 - pad):min(width, x2 + pad)]
        if crop.shape[0] < MIN_CROP or crop.shape[1] < MIN_CROP:
            return
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            return
        self.last = time.time()
        with self.lock:
            self.job = (buf.tobytes(), label)

    def _loop(self):
        while True:
            with self.lock:
                job = self.job
            if job is None:
                time.sleep(0.05)
                continue
            state, text = self.ask(job[0], job[1])
            with self.lock:
                self.verdicts[job[1]] = (state, text, time.time())
                self.job = None

    def recent(self, max_age=8.0):
        with self.lock:
            now = time.time()
            return [(label, state, text)
                    for label, (state, text, ts) in self.verdicts.items()
                    if now - ts < max_age]


MARK = {"yes": "OK", "no": "NO", "err": "??"}
COLOR = {"yes": (0, 220, 0), "no": (0, 0, 230), "err": (0, 190, 230)}


def draw_verdicts(img, rows):
    for i, (label, state, text) in enumerate(rows[:5]):
        cv2.putText(img, "%s %s -- %s" % (MARK[state], label, text),
                    (10, img.shape[0] - 12 - i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[state], 2)


# ---------------------------------------------------------------------------
# Standalone mode: python detect.py ...
# ---------------------------------------------------------------------------

def main():
    # The flags tune the module-level settings, exactly as when this was a
    # plain script. Nothing here runs when multicam.py imports the module.
    global CLASSES, CONF, IMGSZ, LIMIT, MODEL

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    show = "--no-window" not in flags
    use_gemini = "--no-verify" not in flags
    source = args[0] if args else None

    for a in flags:
        if a.startswith("--classes="):
            CLASSES = [c.strip() for c in a.split("=", 1)[1].split(",") if c.strip()]
        if a.startswith("--frames="):
            LIMIT = int(a.split("=", 1)[1])
        if a.startswith("--imgsz="):
            IMGSZ = int(a.split("=", 1)[1])
        if a.startswith("--conf="):
            CONF = float(a.split("=", 1)[1])
        if a.startswith("--model="):
            MODEL = a.split("=", 1)[1]

    if "--list-models" in flags:
        api_key = load_key()
        if not api_key:
            sys.exit("no GEMINI_API_KEY found (environment or .env)")
        from google import genai
        # Bind the client to a name: a throwaway genai.Client() is garbage
        # collected mid-request and the call dies with "client has been closed".
        listing_client = genai.Client(api_key=api_key)
        for m in listing_client.models.list():
            actions = getattr(m, "supported_actions", None)
            if not actions or "generateContent" in actions:
                print(m.name)
        sys.exit()

    verifier = None
    if use_gemini:
        api_key = load_key()
        if api_key:
            verifier = Verifier(api_key, MODEL)
            print("gemini verification on (%s)" % MODEL)
        else:
            print("no GEMINI_API_KEY -- running unverified (see .env.example)")

    model = YOLOWorld(WEIGHTS)
    print("encoding prompts:", CLASSES)
    t0 = time.perf_counter()
    model.set_classes(CLASSES)          # ~6s of CLIP text encoding -- ONCE, never per frame
    print("ready in %.1fs" % (time.perf_counter() - t0))

    if source:
        r = model.predict(source, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        print("%d candidate detections in %s" % (len(r.boxes), source))
        canvas = r.plot()
        rows = []
        for box in r.boxes:
            label = r.names[int(box.cls)]
            line = "  %-12s %.2f" % (label, float(box.conf))
            if verifier:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                crop = r.orig_img[y1:y2, x1:x2]
                if min(crop.shape[:2]) < MIN_CROP:
                    line += "   (too small to verify)"
                else:
                    ok, buf = cv2.imencode(".jpg", crop)
                    state, text = (verifier.ask(buf.tobytes(), label) if ok
                                   else ("err", "encode failed"))
                    line += "   -> %-3s %s" % (MARK[state], text)
                    rows.append((label, state, text))
            print(line)
        draw_verdicts(canvas, rows)
        if show:
            cv2.imshow("yolo-world", canvas)
            cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # DSHOW opens fast on Windows
        if not cap.isOpened():
            sys.exit("could not open webcam at index 0")
        fps = 0.0
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter()
            r = model.predict(frame, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
            dt = time.perf_counter() - t
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            if verifier and len(r.boxes):
                best = max(r.boxes, key=lambda b: float(b.conf))
                verifier.offer(frame, best.xyxy[0], r.names[int(best.cls)])
            canvas = r.plot()
            cv2.putText(canvas, "%4.1f fps | %s" % (fps, ", ".join(CLASSES)), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if verifier:
                draw_verdicts(canvas, verifier.recent())
            if show:
                cv2.imshow("yolo-world", canvas)
                if cv2.waitKey(1) == 27:
                    break
            n += 1
            if LIMIT and n >= LIMIT:
                print("%d frames, %.1f fps" % (n, fps))
                break
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
