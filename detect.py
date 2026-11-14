"""YOLO-World detector with Gemini verification (CPU).

    python detect.py                          # live webcam, Esc to quit
    python detect.py bus.jpg                  # one image
    python detect.py --classes=keys,wallet    # override the prompts
    python detect.py --imgsz=320              # smaller input, ~3x faster
    python detect.py --conf=0.02              # surface weaker detections
    python detect.py --no-verify              # detection only, no Gemini
    python detect.py --list-models            # what your key can reach
    python detect.py --no-window --frames=30  # headless benchmark

YOLO-World runs deliberately loose (CONF is low) so it rarely misses, then
Gemini confirms or rejects each candidate crop. The API call happens on a
background thread, so the video loop never waits on the network -- verdicts
land a beat after the box does.

Put GEMINI_API_KEY in a .env file beside this script (see .env.example).
With no key the detector still runs, just unverified.
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

model = YOLOWorld("yolov8s-worldv2.pt")
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
                state, text = verifier.ask(buf.tobytes(), label) if ok else ("err", "encode failed")
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
