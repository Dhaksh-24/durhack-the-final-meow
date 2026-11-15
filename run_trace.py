"""run_trace.py -- start everything Trace needs with one command, then open the page.

    python run_trace.py                         # the phones listed in cameras.json
    python run_trace.py --video keys_test.mp4   # a phone video instead of live phones (for testing)
    python run_trace.py --demo                  # no cameras at all: a fake session, to check the page
    python run_trace.py --present               # open the page in presentation view (for the judges)
    python run_trace.py --seed                  # preload a simulated morning into the heat map (labelled)
    python run_trace.py --share                 # let a second laptop on the same Wi-Fi open the pages
    python run_trace.py --port 8100             # if 8000 is taken (the live feed then uses 8865)
    python run_trace.py --no-people             # any other option is passed on to multicam.py

What it starts:
  cameras / --video  ->  multicam.py (detects items; serves the page on :8000) + bridge.py (:8765)
  --demo             ->  trace_api.py (serves the page) + bridge.py + events_adapter.py --demo, looping
Then it opens two tabs:
  http://localhost:8000/sim?live  the model + 3D room (keep it open: it runs the model; big screen)
  http://localhost:8000/          trace.html, where you talk to Trace ("Hey Trace", voice)
Ctrl+C stops everything.

cameras.json (see cameras.example.json): [{"name": "Sofa", "source": "http://192.168.1.23:4747/video"}, ...]
"""
import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
PORT = 8000          # the page server; the live event feed (bridge.py) is on PORT + 765 (8765)
EVENTS = ROOT / "trace-events.jsonl"
EXAMPLE = '[\n  {"name": "Table", "source": "http://192.168.1.23:4747/video"}\n]'


def ws_port():
    return PORT + 765


def port_free(port, host):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def missing(mods):
    return [m for m in mods if importlib.util.find_spec(m) is None]


def wait_for_server(seconds, procs):
    """True once the page server answers; False if it times out or a process we started exits."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost: never via a proxy
    t0 = time.time()
    while time.time() - t0 < seconds:
        if any(p.poll() is not None for p in procs):
            return False
        try:
            opener.open("http://localhost:%d/api/cameras" % PORT, timeout=1).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))   # no packet is sent; just picks the Wi-Fi address
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


SHARE = False


def open_pages(query):
    home = "http://localhost:%d/" % PORT
    url = home + "sim?" + query
    if SHARE:
        ip = lan_ip() or "<this-laptop-ip>"
        q = query.replace("&view=present", "")
        print("\nSecond laptop (same Wi-Fi): open")
        print(f"  http://{ip}:{PORT}/sim?{q}&view=present   (big screen)")
        print(f"  http://{ip}:{PORT}/                     (talk to Trace)")
    try:
        webbrowser.open(url)
        time.sleep(1)
        webbrowser.open(home)
    except Exception:
        pass
    print(f"\nOpen these two tabs (they should have opened already):\n  {url}   the model + 3D room: keep it open, it runs the model (big screen)")
    print(f"  {home}   talk to Trace here (\"Hey Trace\", or type)")


def start(args, **kw):
    print("  starting:", " ".join(Path(a).name if i == 1 else a for i, a in enumerate(args)))
    return subprocess.Popen(args, cwd=ROOT, **kw)


def stop_all(procs):
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    ap = argparse.ArgumentParser(description="Start Trace and open the page",
                                 epilog="Any other option (e.g. --no-people) is passed on to multicam.py.")
    ap.add_argument("--video", help="a video file to use as the camera (test without phones)")
    ap.add_argument("--demo", action="store_true", help="no cameras: a fake session to check the page works")
    ap.add_argument("--present", action="store_true", help="open the page in presentation view")
    ap.add_argument("--seed", action="store_true", help="preload a SIMULATED morning into the heat map (labelled as simulated on the page)")
    ap.add_argument("--share", action="store_true",
                    help="open the server and live feed to this Wi-Fi, so a second laptop can show the pages")
    ap.add_argument("--keep-events", action="store_true", help="don't clear trace-events.jsonl first")
    ap.add_argument("--port", type=int, default=8000,
                    help="page server port (default 8000); the live event feed uses this + 765 (8765)")
    a, extra = ap.parse_known_args()   # anything else goes to multicam.py, e.g. --no-people --min-prob keys=0.7
    if a.demo and extra:
        print("Ignoring " + " ".join(extra) + ": those are camera options, and --demo has no cameras.")
    global SHARE, PORT
    SHARE, PORT = a.share, a.port
    if a.video and not Path(a.video).exists():
        print(f"Can't find the video {a.video}. Give the path to a video file, e.g. --video keys_test.mp4")
        sys.exit(1)

    need = ["websockets", "cv2", "numpy"] + ([] if a.demo else ["ultralytics", "torch", "clip"])
    gone = missing(need)
    if gone:
        names = {"cv2": "opencv-python", "clip": '"clip @ git+https://github.com/ultralytics/CLIP.git"'}
        print("Missing Python packages: " + ", ".join(gone))
        print("Install everything with:  pip install -r requirements.txt")
        print("or just these:  pip install " + " ".join(names.get(m, m) for m in gone))
        sys.exit(1)

    cams_file = ROOT / "cameras.json"
    if not a.demo and not a.video:
        if not cams_file.exists():
            print("No cameras.json next to run_trace.py. Make one listing each phone's DroidCam address")
            print("(the DroidCam app shows it), named like the furniture it watches, for example:\n")
            print(EXAMPLE + "\n")
            print("Or check the page without cameras:  python run_trace.py --demo")
            print("Or test with a phone video:         python run_trace.py --video keys_test.mp4")
            sys.exit(1)
        try:
            cams = json.loads(cams_file.read_text())
            assert isinstance(cams, list) and cams and all("name" in c and "source" in c for c in cams)
        except Exception:
            print("cameras.json isn't right. It should be a list of phones like this:\n")
            print(EXAMPLE)
            sys.exit(1)

    host = "0.0.0.0" if a.share else "127.0.0.1"
    for port, what in ((PORT, "the page server"), (ws_port(), "the live event feed")):
        if not port_free(port, host):
            print(f"Port {port} ({what}) is already in use: is Trace already running in another window?")
            print(f"Close it, or use another port:  python run_trace.py --port {PORT + 100} ...")
            sys.exit(1)

    if not a.keep_events and EVENTS.exists():
        EVENTS.unlink()          # a clean start, so old events don't replay
    EVENTS.touch()

    procs, query = [], "live"
    if a.present:
        query += "&view=present"
    if a.seed:
        query += "&seed=morning"
    bridge = [PY, str(ROOT / "bridge.py"), "--file", str(EVENTS), "--port", str(ws_port())] + \
             (["--host", "0.0.0.0"] if a.share else [])
    try:
        if a.demo:
            query += "&room=demo"
            procs.append(start([PY, str(ROOT / "trace_api.py"), "--port", str(PORT)] +
                               (["--host", "0.0.0.0"] if a.share else [])))
            procs.append(start(bridge))
            if not wait_for_server(20, procs):
                print("The page server didn't start: look for an error above.")
                return
            open_pages(query)
            print("A fake session plays every minute: keys appear on the desk, then move to the sofa.")
            print("Ctrl+C to stop.\n")
            while True:
                subprocess.run([PY, str(ROOT / "events_adapter.py"), "--demo", "--out", str(EVENTS)],
                               cwd=ROOT, stdout=subprocess.DEVNULL)
                time.sleep(15)
        else:
            cmd = [PY, str(ROOT / "multicam.py"), "--no-bridge", "--api-port", str(PORT)]
            if a.video:
                cmd += ["--cam", "Test=" + str(Path(a.video).resolve())]
            if a.share:
                cmd += ["--api-host", "0.0.0.0"]
            procs.append(start(cmd + extra))
            procs.append(start(bridge))
            print("Loading the models (the first run downloads them; up to a minute)...")
            if not wait_for_server(180, procs):
                print("multicam.py didn't start its page server: look for an error above.")
                return
            open_pages(query)
            print("1. In the multicam window, drag a box over each piece of furniture a camera watches and")
            print("   type its name in this terminal, exactly as in the room map (Sofa, Desk, Coat hook...).")
            print("2. If the page shows the wrong room, use 'Draw the room' there and press 'Build the room'.")
            print("3. Put your keys down: the event log says 'appeared' and the 3D room reacts.")
            print("Ctrl+C here (or q in the multicam window) to stop.\n")
            procs[0].wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_all(procs)
        print("Stopped.")


if __name__ == "__main__":
    main()
