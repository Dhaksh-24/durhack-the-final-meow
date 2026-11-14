"""YOLO-World detector (CPU) - used standalone AND by multicam.py.

Standalone (same flags as before):
    python detect.py                          # live webcam, Esc to quit
    python detect.py bus.jpg                  # one image
    python detect.py http://192.168.43.57:8080/video   # a phone stream
    python detect.py --classes=keys,wallet    # override the prompts
    python detect.py --imgsz=320              # smaller input, ~3x faster
    python detect.py --conf=0.02              # surface weaker detections
    python detect.py --no-window --frames=30  # headless benchmark

Open-vocab scores run low, so CONF is deliberately small -- raise it
until the false positives stop.

Used by the tracker:
    from detect import Detector
    det = Detector()
    per_frame = det.detect([frame1, frame2])   # one list per frame of
                                               # (category, [x1, y1, x2, y2], conf)

THE CONTRACT: multicam.py only ever calls Detector(conf=, imgsz=) and
Detector.detect(frames). Swap the model inside (YOLOE visual prompts, a
fine-tuned YOLO26...) freely - as long as detect() returns the same shape,
the tracker, memory, API and phone app keep working untouched.
"""
import sys
import time

import cv2
from ultralytics import YOLOWorld

# category -> prompt synonyms. Synonyms are merged into ONE category, so a set
# of keys that matches "keys" AND "keychain" is still one detection.
PROMPTS = {
    "keys": ["keys", "keychain", "car keys"],
    "glasses": ["glasses", "eyeglasses", "reading glasses"],
    "wallet": ["wallet", "purse"],
}
CONF = 0.05
IMGSZ = 640
WEIGHTS = "yolov8s-worldv2.pt"
MERGE_IOU = 0.5     # synonym boxes overlapping more than this = the same object


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(
        a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Detector:
    def __init__(self, prompts=None, conf=CONF, imgsz=IMGSZ, weights=WEIGHTS):
        prompts = prompts or PROMPTS
        self.conf, self.imgsz = conf, imgsz
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

    def detect(self, frames):
        """frames: one image or a list. Returns a list (one per frame) of
        [(category, [x1, y1, x2, y2], conf), ...] with synonyms merged."""
        single = not isinstance(frames, list)
        results = self.model.predict(frames if not single else [frames],
                                     imgsz=self.imgsz, conf=self.conf, verbose=False)
        out = [self._merge(r) for r in results]
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


def draw(frame, dets):
    out = frame.copy()
    for cat, (x1, y1, x2, y2), s in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        cv2.putText(out, "%s %.2f" % (cat, s), (p1[0], max(16, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def report(dets):
    for cat, _, s in dets:
        print("  %-12s %.2f" % (cat, s))


def main():
    conf, imgsz, limit, prompts = CONF, IMGSZ, 0, None
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--no-window" not in sys.argv
    source = args[0] if args else None

    for a in sys.argv[1:]:
        # --classes=keys,wallet: each word its own category
        if a.startswith("--classes="):
            prompts = {c.strip(): [c.strip()] for c in a.split(
                "=", 1)[1].split(",") if c.strip()}
        if a.startswith("--frames="):       # --frames=20, handy for benchmarking
            limit = int(a.split("=", 1)[1])
        if a.startswith("--imgsz="):        # --imgsz=320 roughly doubles speed
            imgsz = int(a.split("=", 1)[1])
        if a.startswith("--conf="):         # --conf=0.02 to surface weaker hits
            conf = float(a.split("=", 1)[1])

    det = Detector(prompts, conf=conf, imgsz=imgsz)
    label = ", ".join(det.categories)

    is_image = source and source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            sys.exit("could not read %s" % source)
        dets = det.detect(frame)
        print("%d detections in %s" % (len(dets), source))
        report(dets)
        if show:
            cv2.imshow("yolo-world", draw(frame, dets))
            cv2.waitKey(0)
    else:
        if source is None or source.isdigit():
            cap = cv2.VideoCapture(
                int(source or 0), cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        else:
            # phone stream URL or a video file
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            sys.exit("could not open %s" % (source or "webcam at index 0"))
        fps, n = 0.0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.perf_counter()
            dets = det.detect(frame)
            dt = time.perf_counter() - t
            fps = 1 / dt if fps == 0 else 0.9 * fps + 0.1 / dt
            out = draw(frame, dets)
            cv2.putText(out, "%4.1f fps | %s" % (fps, label), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if show:
                cv2.imshow("yolo-world", out)
                if cv2.waitKey(1) == 27:
                    break
            n += 1
            if limit and n >= limit:
                print("%d frames, %.1f fps" % (n, fps))
                break
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
