"""run_trace.py -- start everything Trace needs with one command, then open the page.

    python run_trace.py                         # the phones listed in cameras.json
    python run_trace.py --video keys_test.mp4   # a phone video instead of live phones (for testing)
    python run_trace.py --demo                  # no cameras at all: a fake session, to check the page
    python run_trace.py --present               # open the page in presentation view (for the judges)

What it starts:
  cameras / --video  ->  multicam.py   (detects items; serves the page on :8000; starts bridge.py)
  --demo             ->  trace_api.py (serves the page) + bridge.py + events_adapter.py --demo, looping
Then it opens http://localhost:8000/sim?live in your browser. Ctrl+C stops everything.

cameras.json (see cameras.example.json): [{"name": "Sofa", "source": "http://192.168.1.23:4747/video"}, ...]
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
URL = "http://localhost:8000/sim"
EVENTS = ROOT / "trace-events.jsonl"


def missing(mods):
    return [m for m in mods if importlib.util.find_spec(m) is None]


def wait_for_server(seconds):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            urllib.request.urlopen("http://localhost:8000/api/cameras", timeout=1).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def start(args, **kw):
    print("  starting:", " ".join(Path(a).name if i == 1 else a for i, a in enumerate(args)))
    return subprocess.Popen(args, cwd=ROOT, **kw)


def main():
    ap = argparse.ArgumentParser(description="Start Trace and open the page")
    ap.add_argument("--video", help="a video file to use as the camera (test without phones)")
    ap.add_argument("--demo", action="store_true", help="no cameras: a fake session to check the page works")
    ap.add_argument("--present", action="store_true", help="open the page in presentation view")
    ap.add_argument("--keep-events", action="store_true", help="don't clear trace-events.jsonl first")
    a = ap.parse_args()

    need = ["websockets", "cv2", "numpy"] + ([] if a.demo else ["ultralytics", "torch", "clip"])
    gone = missing(need)
    if gone:
        names = {"cv2": "opencv-python", "clip": '"clip @ git+https://github.com/ultralytics/CLIP.git"'}
        print("Missing Python packages: " + ", ".join(gone))
        print("Install them with:\n  pip install " + " ".join(names.get(m, m) for m in gone))
        sys.exit(1)

    if not a.demo and not a.video and not (ROOT / "cameras.json").exists():
        print("No cameras.json. Either copy cameras.example.json to cameras.json and put your phones'")
        print("DroidCam addresses in it, or test with a video:  python run_trace.py --video keys_test.mp4")
        sys.exit(1)

    if not a.keep_events and EVENTS.exists():
        EVENTS.unlink()          # a clean start, so old events don't replay
    EVENTS.touch()

    procs, url = [], URL + "?live"
    if a.present:
        url += "&view=present"
    try:
        if a.demo:
            url += "&room=demo"
            procs.append(start([PY, str(ROOT / "trace_api.py")]))
            procs.append(start([PY, str(ROOT / "bridge.py"), "--file", str(EVENTS)]))
            if not wait_for_server(20):
                print("The page server didn't start: look for an error above.")
                return
            webbrowser.open(url)
            print(f"\nOpened {url}\nA fake session plays every minute: keys appear on the desk, then move to the sofa.")
            print("Ctrl+C to stop.\n")
            while True:
                subprocess.run([PY, str(ROOT / "events_adapter.py"), "--demo", "--out", str(EVENTS)],
                               cwd=ROOT, stdout=subprocess.DEVNULL)
                time.sleep(15)
        else:
            cmd = [PY, str(ROOT / "multicam.py")]
            if a.video:
                cmd += ["--cam", "Test=" + str(Path(a.video).resolve())]
            procs.append(start(cmd))
            print("Loading the models (the first run downloads them; up to a minute)...")
            if not wait_for_server(180):
                print("multicam.py didn't start its page server: look for an error above.")
                return
            webbrowser.open(url)
            print(f"\nOpened {url}")
            print("1. In the multicam window, drag a box over each piece of furniture a camera watches and")
            print("   type its name in this terminal, exactly as in the room map (Sofa, Desk, Coat hook...).")
            print("2. If the page shows the wrong room, use 'Draw the room' there and press 'Build the room'.")
            print("3. Put your keys down: the event log says 'appeared' and the 3D room reacts.")
            print("Ctrl+C here (or q in the multicam window) to stop.\n")
            procs[0].wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        print("Stopped.")


if __name__ == "__main__":
    main()
