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
  POST /api/ask {"message", "history"}  (trace.html, the proper front end)
                                 -> {"reply", "item_id", "state": "searching"|"found"|"idle", ...}
                                 answered by guide.chat from the model the analytics page posted
                                 (below); 503 until it has, and trace.html then uses its own replies.
  POST /api/model {"room", "items": {id: bundle}}
                                 the analytics page (trace-sim, Real events mode) shares the
                                 model's current belief per item, about once a second
  GET  /api/model                what was last posted, and when
  POST /api/ask {"bundle", "utterance", "mode"}  (trace-sim's own Ask box)
                                 the guide (guide.py, SPEC 5): parses what was said into evidence
                                 and words the answer with the LLM (two calls at most), returns
                                 {"say", "places", "evidence", "chirp", "intent", "item", "source"}.
                                 No GEMINI_API_KEY: a rules parser, and "say" is null (the page
                                 then speaks its own template sentence).
  POST /api/chirp/<zone>         ask that zone's phone to beep; {"id", "phone": true if a phone
                                 page for that zone checked in during the last 5 s}
  GET  /api/chirp/<zone>         what a phone page polls: {"id"} of the latest chirp
  GET  /sim                      the 3D room and guide (trace-sim/index.html)
  GET  /phone?zone=<id>          the page a phone opens to beep for its zone
  POST /api/speech {"text", "voice": "female"|"male"}
                                 Trace's voice: a WAV of the line, made by tts.py (Kokoro,
                                 offline). 503 when kokoro isn't installed; the page then
                                 uses the browser's own voice instead.

The live video is the raw camera picture; trace.html draws the item boxes on
top itself.

Without multicam.py (no cameras, e.g. replaying events.jsonl):  python trace_api.py
then open http://localhost:8000/sim

By default the server only listens on this computer (127.0.0.1). To open Trace
on a phone, start multicam.py with --api-host 0.0.0.0 and browse to
http://<this-computer's-ip>:8000/ -- but anyone on the same Wi-Fi can then see
the cameras, so only do that on a network you trust.
"""
import io
import json
import re
import threading
import time
import wave
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

try:                    # Trace's offline voice; optional (kokoro needs Python 3.10 - 3.12)
    import tts
    VOICE = not tts.missing()
except Exception:
    tts, VOICE = None, False

ROOT = Path(__file__).resolve().parent
LIVE_FPS = 8            # frames a second per viewer; phones on Wi-Fi cope well with this
JPEG_QUALITY = 72
STALL_SECS = 8          # end a live stream after this long without a new frame
SPEECH_MAX = 600        # characters per /api/speech request; Trace's lines are far shorter
CHIRP_SEEN = 5.0        # a phone page that polled this recently counts as listening

PHONE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Trace phone</title>
<style>body{margin:0;font:18px system-ui,sans-serif;background:#0f1513;color:#e4ebe7;display:grid;place-items:center;min-height:100vh;text-align:center}
button{font:inherit;padding:16px 26px;border-radius:12px;border:0;background:#6cb6d6;color:#0b1418;font-weight:700}
.flash{animation:f .25s 6 alternate}@keyframes f{to{background:#f0a943}}p{color:#97a49d;max-width:22em;margin:12px auto}</style></head>
<body><main><h1 id="z">Trace phone</h1><button id="go">Tap to turn sound on</button><p id="s">This phone beeps when Trace sends someone to its place.</p></main>
<script>
const zone = new URLSearchParams(location.search).get("zone") || "";
document.getElementById("z").textContent = zone ? "Trace phone: " + zone : "Add ?zone=<id> to the address";
let ctx = null, last = null;
function beep() {
  if (!ctx) return;
  const t = ctx.currentTime;
  for (let k = 0; k < 3; k++) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = 1320; o.connect(g); g.connect(ctx.destination);
    g.gain.setValueAtTime(0.0001, t + k * 0.35); g.gain.exponentialRampToValueAtTime(0.6, t + k * 0.35 + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + k * 0.35 + 0.25); o.start(t + k * 0.35); o.stop(t + k * 0.35 + 0.3);
  }
  document.body.classList.remove("flash"); void document.body.offsetWidth; document.body.classList.add("flash");
}
document.getElementById("go").onclick = () => { ctx = ctx || new AudioContext(); ctx.resume(); beep();
  document.getElementById("go").textContent = "Sound on"; document.getElementById("s").textContent = "Listening for Trace. Keep this page open."; };
async function poll() {
  try { const r = await fetch("/api/chirp/" + encodeURIComponent(zone), { cache: "no-store" }); const j = await r.json();
    if (last !== null && j.id !== last) beep(); last = j.id; } catch (e) { /* laptop not reachable: try again */ }
  setTimeout(poll, 1000);
}
if (zone) poll();
</script></body></html>"""


def wav_bytes(audio, rate):
    """float32 mono samples -> a 16-bit WAV file in memory."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class TraceAPI:
    def __init__(self, cams, mem_dir, events_file):
        self.cams = {c.id: c for c in cams}
        self.order = [c.id for c in cams]
        self.mem_dir, self.events_file = Path(mem_dir), Path(events_file)
        # Replaced wholesale by multicam.py (one assignment), so a request never sees half a list.
        self.items = []
        self.httpd = None
        self.model = None       # {"room", "items": {id: bundle}, "received"} from the analytics page
        self.sessions = {}      # trace.html's searches in progress, per item (guide.chat)
        self.lock = threading.Lock()
        self.chirps = {}        # zone -> id of the latest chirp
        self.polls = {}         # zone -> when a phone page last asked

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
                if p == "/api/model":
                    try:
                        body = json.loads(raw or b"{}")
                        assert isinstance(body.get("items"), dict)
                    except (ValueError, AssertionError, AttributeError):
                        return self._json({"error": "bad json: needs items"}, 400)
                    body["received"] = time.time()
                    api.model = body
                    return self._json({"ok": True})
                m = re.fullmatch(r"/api/chirp/([a-z0-9-]+)", p)
                if m:
                    z = m.group(1)
                    api.chirps[z] = api.chirps.get(z, 0) + 1
                    return self._json({"id": api.chirps[z], "phone": time.time() - api.polls.get(z, 0) < CHIRP_SEEN})
                self._json({"error": "not available yet"}, 501)

            def _ask(self, raw):
                try:
                    body = json.loads(raw or b"{}")
                except ValueError:
                    return self._json({"error": "bad json"}, 400)
                if isinstance(body, dict) and "message" in body and "bundle" not in body:
                    return self._chat(body)
                try:
                    bundle, said = body["bundle"], str(body.get("utterance", "")).strip()[:500]
                    mode = body.get("mode") if body.get("mode") in ("default", "blind", "dementia") else "default"
                except (ValueError, KeyError, AttributeError, TypeError):
                    return self._json({"error": "bad json: needs bundle and utterance"}, 400)
                import guide                   # imported here so multicam.py starts without it
                try:
                    res = guide.ask(bundle, said, mode)
                except Exception as exc:
                    print("guide: %s: %s" % (type(exc).__name__, exc))
                    return self._json({"error": "guide failed"}, 500)
                for e in res.get("errors") or []:
                    print("guide:", e)
                self._json(res)

            def _chat(self, body):
                import guide
                message = str(body.get("message", "")).strip()[:500]
                history = body.get("history") if isinstance(body.get("history"), list) else []
                mode = body.get("mode") if body.get("mode") in ("default", "blind", "dementia") else "default"
                try:
                    with api.lock:                  # one turn at a time: the sessions are shared
                        res = guide.chat(message, history, api.model, api.sessions, mode)
                except guide.NoModel:
                    return self._json({"error": "no model yet: open /sim in Real events mode"}, 503)
                except Exception as exc:
                    print("guide chat: %s: %s" % (type(exc).__name__, exc))
                    return self._json({"error": "guide failed"}, 500)
                # trace.html's own item ids come from multicam's memory: the most recent of that kind
                same = [it for it in api.items if it.get("category") == res["item"]]
                res["item_id"] = max(same, key=lambda it: it.get("last_time") or 0)["id"] if same else None
                if res.get("chirp"):
                    api.chirps[res["chirp"]] = api.chirps.get(res["chirp"], 0) + 1
                if res.get("why"):
                    print("guide chat:", res["why"])
                self._json(res)

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
                    audio = tts.audio(text, gender)    # pre-rendered lines come straight from disk
                except Exception as exc:       # model not downloaded, no espeak, ...
                    print("tts: could not say %r: %s: %s" % (text, type(exc).__name__, exc))
                    return self._json({"error": "voice failed"}, 503)
                self._bytes(wav_bytes(audio, tts.RATE), "audio/wav")

            def do_GET(self):
                url = urlparse(self.path)
                p = url.path
                try:
                    if p in ("/", "/trace.html"):
                        return self._file(ROOT / "trace.html", "text/html; charset=utf-8")
                    if p in ("/sim", "/sim/", "/trace-sim/", "/trace-sim/index.html"):
                        return self._file(ROOT / "trace-sim" / "index.html", "text/html; charset=utf-8")
                    if p == "/trace-sim/sample-events.jsonl":
                        return self._file(ROOT / "trace-sim" / "sample-events.jsonl", "application/x-ndjson")
                    if p == "/phone":
                        return self._bytes(PHONE_PAGE.encode(), "text/html; charset=utf-8")
                    m = re.fullmatch(r"/api/chirp/([a-z0-9-]+)", p)
                    if m:
                        api.polls[m.group(1)] = time.time()
                        return self._json({"id": api.chirps.get(m.group(1), 0)})
                    if p == "/api/cameras":
                        return self._json(api.cameras())
                    if p == "/api/items":
                        return self._json(api.items)
                    if p == "/api/model":
                        return self._json(api.model or {})
                    if p == "/api/events":
                        q = parse_qs(url.query)
                        limit = max(1, min(200, int((q.get("limit") or ["20"])[0])))
                        return self._json(api.events(limit))
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
            tts.warm_up(block=False)           # load Kokoro now so the first reply isn't slow
        else:
            print("Trace's voice: kokoro not installed, the page will use the browser's voice "
                  "(see README, Voice)")
        return self

    def stop(self):
        httpd, self.httpd = self.httpd, None
        if httpd:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    # Standalone: the guide, the 3D page and phone chirps without cameras (multicam.py normally
    # starts this server itself, with its cameras).
    import argparse
    ap = argparse.ArgumentParser(description="Trace's HTTP server without cameras")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 lets phones on your Wi-Fi reach /phone")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    srv = TraceAPI([], ROOT / "memory", ROOT / "events.jsonl").start(a.host, a.port)
    print("3D room and guide: http://localhost:%d/sim   phone beeper: http://<this-laptop>:%d/phone?zone=<id>" % (a.port, a.port))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop()
