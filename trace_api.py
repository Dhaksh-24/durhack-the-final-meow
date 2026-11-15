"""trace_api.py -- the small HTTP server trace.html talks to.

multicam.py starts it in a background thread (see --api-port / --api-host), so
it can hand out the live frames the tracker already holds. Standard library
plus OpenCV only; no keys, and nothing leaves your network.

  GET /                          trace.html
  GET /api/cameras               [{"id", "name", "live", "frame_w", "frame_h"}]
  GET /api/items                 remembered items, refreshed by multicam about twice a second
  GET /api/events?limit=20       latest sightings from events.jsonl, newest first
  GET /api/thumbs/<id>.jpg       item thumbnail                 (memory/thumbs)
  GET /api/snapshots/<id>.jpg    frame it was last seen in      (memory/snapshots)
  GET /api/live/<camera>.mjpg    live video of one camera (MJPEG, about LIVE_FPS frames a second)
  GET /api/live/<camera>.jpg     the latest single frame
  GET /api/speech?text=..&voice=female|male
                                 Trace's voice, streamed: Microsoft's neural voice (edge-tts,
                                 online) plays as it's made, so it starts in under a second.
                                 Without internet: Kokoro (offline) WAV in one piece.
  POST /api/speech {"text", "voice"}
                                 The same, in one piece (the page uses it for Kokoro, one
                                 sentence at a time). X-Trace-Voice says which engine spoke.
                                 503 when neither works; the page then uses the browser's voice.
  GET /api/voice                 {"engine": "edge"|"kokoro"|"none"}

  POST /api/ask {"message", "history"}
                                 "where are my keys?" -> {"reply", "item_id", "state", "by",
                                 "prediction"}, from whereabouts.py (Gemini, or its rules when
                                 offline). An empty reply = not about a tracked item; the page
                                 then answers from its own rules.
  GET /api/whereabouts           every item's status, story and ranked places + the timeline
  GET /api/predict?item=keys     one item's ranked places and the evidence behind them
  GET /api/heatmap/<camera>.jpg  the camera's latest frame with the people heatmap and places

The live video is the raw camera picture; trace.html draws the item boxes on
top itself.

By default the server only listens on this computer (127.0.0.1). To open Trace
on a phone, start multicam.py with --api-host 0.0.0.0 and browse to
http://<this-computer's-ip>:8000/ -- but anyone on the same Wi-Fi can then see
the cameras, so only do that on a network you trust.
"""
import json
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

try:                    # Trace's voice: Microsoft neural (edge-tts, online) or Kokoro (offline)
    import tts
    VOICE = tts.available()
except Exception:
    tts, VOICE = None, False

ROOT = Path(__file__).resolve().parent
LIVE_FPS = 8            # frames a second per viewer; phones on Wi-Fi cope well with this
JPEG_QUALITY = 72
STALL_SECS = 8          # end a live stream after this long without a new frame
SPEECH_MAX = 600        # characters per /api/speech request; Trace's lines are far shorter


class TraceAPI:
    def __init__(self, cams, mem_dir, events_file):
        self.cams = {c.id: c for c in cams}
        self.order = [c.id for c in cams]
        self.mem_dir, self.events_file = Path(mem_dir), Path(events_file)
        # Replaced wholesale by multicam.py (one assignment), so a request never sees half a list.
        self.items = []
        self.brain = None       # whereabouts.Whereabouts, set by multicam.py
        self.httpd = None

    # ------------------------------------------------------------ data
    def cameras(self):
        out = []
        for cid in self.order:
            c = self.cams[cid]
            f, _ = c.latest()
            h, w = f.shape[:2] if f is not None else (None, None)
            out.append({"id": c.id, "name": c.name, "live": bool(c.ok), "frame_w": w, "frame_h": h})
        return out

    def events(self, limit):
        if not self.events_file.exists():
            return []
        with self.events_file.open() as f:
            lines = deque(f, maxlen=limit)
        out = []
        for line in reversed(lines):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    # ------------------------------------------------------------ server
    def start(self, host="127.0.0.1", port=8000):
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):        # the live streams would flood the console
                pass

            def _head(self, code, ctype, length=None, extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Expose-Headers", "X-Trace-Voice")
                self.send_header("Cache-Control", "no-store")
                if length is not None:
                    self.send_header("Content-Length", str(length))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()

            def _bytes(self, body, ctype, code=200):
                self._head(code, ctype, len(body))
                self.wfile.write(body)

            def _json(self, obj, code=200):
                self._bytes(json.dumps(obj).encode(), "application/json", code)

            def _file(self, path, ctype):
                if not path.is_file():
                    return self._json({"error": "not found"}, 404)
                self._bytes(path.read_bytes(), ctype)

            def do_OPTIONS(self):
                self._head(204, "text/plain", 0, {"Access-Control-Allow-Methods": "GET, POST, OPTIONS"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(min(n, 64 * 1024))
                p = urlparse(self.path).path
                if p == "/api/speech":
                    return self._speech(raw)
                if p == "/api/ask":
                    return self._ask(raw)
                self._json({"error": "not available yet"}, 501)

            def _ask(self, raw):
                if api.brain is None:
                    return self._json({"error": "no whereabouts engine"}, 501)
                try:
                    body = json.loads(raw or b"{}")
                    message = str(body.get("message", "")).strip()[:500]
                    history = body.get("history") if isinstance(body.get("history"), list) else []
                except (ValueError, AttributeError):
                    return self._json({"error": "bad json"}, 400)
                try:
                    return self._json(api.brain.ask(message, history))
                except Exception as exc:      # never take the page down; it has its own answers
                    print("ask: %s: %s" % (type(exc).__name__, exc))
                    return self._json({"reply": "", "item_id": None, "state": "idle"})

            def _speech(self, raw):
                if not VOICE:
                    return self._json({"error": "voice not installed (pip install -r requirements.txt)"}, 503)
                try:
                    body = json.loads(raw or b"{}")
                    text = str(body.get("text", "")).strip()[:SPEECH_MAX]
                except (ValueError, AttributeError):
                    return self._json({"error": "bad json"}, 400)
                if not text:
                    return self._json({"error": "no text"}, 400)
                # only the two named voices (tts.VOICES) can be asked for from the page
                gender = body.get("voice") if body.get("voice") in ("female", "male") else None
                try:
                    # Microsoft neural MP3 when online, else Kokoro WAV; pre-rendered lines from disk
                    data, mime, engine = tts.speech_file(text, gender)
                except Exception as exc:       # offline and no Kokoro, model not downloaded, ...
                    print("tts: could not say %r: %s: %s" % (text, type(exc).__name__, exc))
                    return self._json({"error": "voice failed"}, 503)
                self._head(200, mime, len(data), {"X-Trace-Voice": engine})
                self.wfile.write(data)

            def do_GET(self):
                url = urlparse(self.path)
                p = url.path
                try:
                    if p in ("/", "/trace.html"):
                        return self._file(ROOT / "trace.html", "text/html; charset=utf-8")
                    if p == "/api/cameras":
                        return self._json(api.cameras())
                    if p == "/api/voice":
                        return self._json({"engine": tts.current_engine() if VOICE else "none"})
                    if p == "/api/speech":
                        return self._speech_stream(parse_qs(url.query))
                    if p == "/api/items":
                        return self._json(api.items)
                    if p == "/api/events":
                        q = parse_qs(url.query)
                        limit = max(1, min(200, int((q.get("limit") or ["20"])[0])))
                        return self._json(api.events(limit))
                    if p == "/api/whereabouts" and api.brain:
                        return self._json(api.brain.snapshot())
                    if p == "/api/predict" and api.brain:
                        q = (parse_qs(url.query).get("item") or [""])[0]
                        oid = api.brain.find_item(q)
                        if oid is None:
                            return self._json({"error": "no tracked item matches %r" % q}, 404)
                        return self._json(api.brain.predict(oid))
                    m = re.fullmatch(r"/api/heatmap/([^/]+)\.jpg", p)
                    if m and api.brain:
                        cam = api.cams.get(m.group(1))
                        f, _ = cam.latest() if cam else (None, None)
                        if f is None:
                            return self._json({"error": "no frame from that camera"}, 404)
                        ok, jpg = cv2.imencode(".jpg", api.brain.overlay(cam.id, f.copy()),
                                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        return self._bytes(jpg.tobytes(), "image/jpeg")
                    m = re.fullmatch(r"/api/(thumbs|snapshots)/(\d+)\.jpg", p)
                    if m:
                        return self._file(api.mem_dir / m.group(1) / f"{m.group(2)}.jpg", "image/jpeg")
                    m = re.fullmatch(r"/api/live/([^/]+)\.(mjpg|jpg)", p)
                    if m:
                        cam = api.cams.get(m.group(1))
                        if cam is None:
                            return self._json({"error": "no such camera"}, 404)
                        return self._stream(cam) if m.group(2) == "mjpg" else self._still(cam)
                    return self._json({"error": "not found"}, 404)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def _speech_stream(self, q):
                if not VOICE:
                    return self._json({"error": "voice not installed (pip install -r requirements.txt)"}, 503)
                text = (q.get("text") or [""])[0].strip()[:SPEECH_MAX]
                if not text:
                    return self._json({"error": "no text"}, 400)
                v = (q.get("voice") or [""])[0]
                gender = v if v in ("female", "male") else None
                try:
                    mime, engine, chunks = tts.speech_stream(text, gender)
                except Exception as exc:
                    print("tts: could not say %r: %s: %s" % (text, type(exc).__name__, exc))
                    return self._json({"error": "voice failed"}, 503)
                # no Content-Length: the audio is still being made; the end of the connection ends it
                self.close_connection = True
                self._head(200, mime, extra={"X-Trace-Voice": engine, "Connection": "close"})
                try:
                    for chunk in chunks:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass                          # the page moved on (Trace was interrupted)
                except Exception as exc:          # the online voice dropped mid-line: end the clip there
                    print("tts: stream ended early: %s: %s" % (type(exc).__name__, exc))

            def _still(self, cam):
                f, _ = cam.latest()
                if f is None or not cam.ok:
                    return self._json({"error": "no signal"}, 503)
                ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                self._bytes(jpg.tobytes(), "image/jpeg")

            def _stream(self, cam):
                f, _ = cam.latest()
                if f is None or not cam.ok:
                    return self._json({"error": "no signal"}, 503)
                self.close_connection = True
                self._head(200, "multipart/x-mixed-replace; boundary=frame",
                           extra={"Connection": "close", "X-Accel-Buffering": "no"})
                last_t, fresh_at = 0.0, time.time()
                while api.httpd is not None:
                    t0 = time.time()
                    f, ft = cam.latest()
                    if f is None or ft == last_t:
                        if t0 - fresh_at > STALL_SECS:
                            break                       # camera stalled: let the page show "no signal"
                        time.sleep(0.03)
                        continue
                    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        b = jpg.tobytes()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(b)).encode() + b"\r\n\r\n" + b + b"\r\n")
                        self.wfile.flush()
                    last_t, fresh_at = ft, t0
                    time.sleep(max(0.0, 1.0 / LIVE_FPS - (time.time() - t0)))

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        shown = "localhost" if host in ("127.0.0.1", "localhost") else host
        print(f"Trace is at http://{shown}:{port}/" +
              ("  (cameras visible to anyone on this network)" if host == "0.0.0.0" else ""))
        if VOICE:
            if tts.ENGINE == "edge" and tts.edge_installed():
                print("Trace's voice: Microsoft %s / %s (online; the offline voice takes over without internet)"
                      % (tts.EDGE_VOICES["female"], tts.EDGE_VOICES["male"]))
            tts.warm_up(block=False)           # load Kokoro now (the fallback) so it's ready if needed
        else:
            print("Trace's voice: none installed, the page will use the browser's voice "
                  "(pip install -r requirements.txt)")
        return self

    def stop(self):
        httpd, self.httpd = self.httpd, None
        if httpd:
            httpd.shutdown()
            httpd.server_close()
