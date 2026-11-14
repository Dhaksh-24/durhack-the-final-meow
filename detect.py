"""YOLO-World detector (CPU).

    python detect.py                          # live webcam, Esc to quit
    python detect.py bus.jpg                  # one image
    python detect.py --classes=keys,wallet    # override the prompts
    python detect.py --imgsz=320              # smaller input, ~3x faster
    python detect.py --conf=0.02              # surface weaker detections
    python detect.py --no-window --frames=30  # headless benchmark

Open-vocab scores run low, so CONF is deliberately small -- raise it
until the false positives stop.
"""
import sys
import time

import cv2
from ultralytics import YOLOWorld

CLASSES = ["keys", "keychain", "car keys"]
CONF = 0.05
IMGSZ = 640
LIMIT = 0          # webcam: stop after N frames (0 = run until Esc)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
show = "--no-window" not in sys.argv
source = args[0] if args else None

for a in sys.argv[1:]:                      # --classes=keys,wallet,phone
    if a.startswith("--classes="):
        CLASSES = [c.strip() for c in a.split("=", 1)[1].split(",") if c.strip()]
    if a.startswith("--frames="):      # --frames=20, handy for benchmarking
        LIMIT = int(a.split("=", 1)[1])
    if a.startswith("--imgsz="):       # --imgsz=320 roughly doubles speed
        IMGSZ = int(a.split("=", 1)[1])
    if a.startswith("--conf="):        # --conf=0.02 to surface weaker hits
        CONF = float(a.split("=", 1)[1])

model = YOLOWorld("yolov8s-worldv2.pt")
print("encoding prompts:", CLASSES)
t0 = time.perf_counter()
model.set_classes(CLASSES)          # ~6s of CLIP text encoding -- ONCE, never per frame
print("ready in %.1fs" % (time.perf_counter() - t0))


def report(result):
    for box in result.boxes:
        print("  %-12s %.2f" % (result.names[int(box.cls)], float(box.conf)))


if source:
    r = model.predict(source, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    print("%d detections in %s" % (len(r.boxes), source))
    report(r)
    if show:
        cv2.imshow("yolo-world", r.plot())
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
        out = r.plot()
        cv2.putText(out, "%4.1f fps | %s" % (fps, ", ".join(CLASSES)), (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if show:
            cv2.imshow("yolo-world", out)
            if cv2.waitKey(1) == 27:
                break
        n += 1
        if LIMIT and n >= LIMIT:
            print("%d frames, %.1f fps" % (n, fps))
            break
    cap.release()

cv2.destroyAllWindows()
