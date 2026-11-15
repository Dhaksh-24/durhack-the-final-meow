"""trace_api.py -- the small HTTP server trace.html talks to.

multicam.py starts it in a background thread (see --api-port / --api-host), so
it can hand out the live frames the tracker already holds. Standard library
plus OpenCV only; no keys, and nothing leaves your network.

  GET /                          trace.html
  GET /model                     same as /sim: the model page (3D room, probabilities)
  GET/POST /api/room             room.json: the room map the page built, loaded by every browser
  GET /api/cameras               [{"id", "name", "live", "frame_w", "frame_h"}]
  GET /api/items                 remembered items, refreshed by multicam about twice a second
                                 (without multicam: built from the model the /sim page posted)
  GET /api/events?limit=20       latest sightings, newest first (events.jsonl with multicam,
                                 trace-events.jsonl standalone: appeared / moved only)
  GET /api/thumbs/<id>.jpg       item thumbnail                 (memory/thumbs)
  GET /api/snapshots/<id>.jpg    frame it was last seen in      (memory/snapshots)
  GET /api/live/<camera>.mjpg    live video of one camera (MJPEG, about LIVE_FPS frames a second)
  GET /api/live/<camera>.jpg     the latest single frame
  POST /api/ask {"message", "history"}  (trace.html, the proper front end)
                                 -> {"reply", "item_id", "state": "searching"|"found"|"idle", ...}
                                 answered by guide.chat from the model the analytics page posted
                                 (below). Until it has, whereabouts.py answers from multicam's own
                                 person trail (Gemini, or its rules offline); 503 if neither can,
                                 and trace.html then uses its own replies.
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

  GET /api/whereabouts           every item's status, story and ranked places + the timeline
  GET /api/predict?item=keys     one item's ranked places and the evidence behind them
  GET /api/heatmap/<camera>.jpg  the camera's latest frame with the people heatmap and places

  GET /api/speech?text=..&voice=female|male
                                 Trace's voice, streamed: Microsoft's neural voice (edge-tts,
                                 online) plays as it's made, so it starts in under a second.
                                 Without internet: Kokoro (offline) WAV in one piece.
  POST /api/speech {"text", "voice"}
                                 The same, in one piece (the page uses it for Kokoro, one
                                 sentence at a time). X-Trace-Voice says which engine spoke.
                                 503 when neither works; the page then uses the browser's voice.
  GET /api/voice                 {"engine": "edge"|"kokoro"|"none"}

The live video is the raw camera picture; trace.html draws the item boxes on
top itself.

Without multicam.py (no cameras, e.g. replaying events.jsonl):  python trace_api.py
then open http://localhost:8000/sim

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


def ago_secs(text):
    """The page's agoText ("just now", "5 minutes ago", "2 h 10 min ago") back to seconds."""
    t = str(text or "")
    h = re.search(r"(\d+) h", t)
    m = re.search(r"(\d+) min", t)
    return (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0)


class TraceAPI:
    def __init__(self, cams, mem_dir, events_file):
        self.cams = {c.id: c for c in cams}
        self.order = [c.id for c in cams]
        self.mem_dir, self.events_file = Path(mem_dir), Path(events_file)
        # Replaced wholesale by multicam.py (one assignment), so a request never sees half a list.
        self.items = []
        self.brain = None       # whereabouts.Whereabouts, set by multicam.py
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

    def zone_names(self):
        """zone id -> name, from the places in the model the page posted ({} before it has)."""
        for b in ((self.model or {}).get("items") or {}).values():
            return {p["id"]: p["name"] for p in b.get("places") or [] if "id" in p and "name" in p}
        return {}

    def model_items(self):
        """Without multicam (run_trace.py --demo, a replay) nothing fills self.items, and trace.html
        showed "No items yet" while Trace answered from the model. This builds the same item shape
        from the model the /sim page posted: every item a phone has seen, in view or last seen."""
        m = self.model or {}
        received, names, out = m.get("received") or time.time(), self.zone_names(), []
        for key, b in (m.get("items") or {}).items():
            vis, ls = b.get("visible_now"), b.get("last_seen") or {}
            zone = vis or ls.get("zone")
            if not zone:
                continue                        # never seen: nothing to remember yet
            where, noun = names.get(zone, zone), b.get("item_name") or key
            plural = bool(b.get("item_plural"))
            if vis:
                sentence, ago = "The %s phone can see your %s right now." % (where.lower(), noun), 0
            else:
                ago = ago_secs(ls.get("ago"))
                sentence = "Your %s %s last seen on the %s%s." % (
                    noun, "were" if plural else "was", where.lower(), " " + ls["ago"] if ls.get("ago") else "")
                top = (b.get("search_plan") or [{}])[0]
                if top.get("zone") and top["zone"] != zone:
                    sentence += " Best place to look now: the %s (%s)." % (str(top.get("place", top["zone"])).lower(), top.get("word", "possibly"))
            out.append({"id": b.get("item") or key, "name": noun[:1].upper() + noun[1:], "category": key,
                        "camera": None, "camera_name": where + " phone", "zone": where,
                        "last_time": received - ago, "box": None, "frame_w": None, "frame_h": None,
                        "sentence": sentence, "confidence": None, "guess": False, "flag": None,
                        "in_view": bool(vis)})
        return out

    def events(self, limit):
        if not self.events_file.exists():
            return []
        with self.events_file.open() as f:
            lines = deque(f, maxlen=limit * 5)     # room for the person events skipped below
        out, names = [], self.zone_names()
        for line in reversed(lines):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if "event" in e:                       # trace-events.jsonl: keep sightings, readable
                if e["event"] not in ("appeared", "moved") or not e.get("item"):
                    continue
                cat = e.get("category") or str(e["item"]).rsplit("-", 1)[0]
                e = dict(e, item=cat[:1].upper() + cat[1:], category=cat,
                         zone=names.get(e.get("zone"), e.get("zone")), camera_name=e.get("camera_name") or e.get("camera"))
            out.append(e)
            if len(out) >= limit:
                break
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
                p = urlparse(self.path).path
                # the model bundle grows with the room (places, the best question's factors per zone):
                # cut short, it wouldn't parse and Trace would keep answering from a stale model
                raw = self.rfile.read(min(n, (2048 if p == "/api/model" else 64) * 1024))
                if p == "/api/speech":
                    return self._speech(raw)
                if p == "/api/ask":
                    return self._ask(raw)
                if p == "/api/room":                 # the page saves the room it built
                    try:
                        room = json.loads(raw.decode("utf-8"))
                        assert isinstance(room.get("zones"), list) and room["zones"]
                    except Exception:
                        return self._json({"error": "needs a zone map with a zones list"}, 400)
                    (ROOT / "room.json").write_text(json.dumps(room, indent=2))
                    return self._json({"saved": "room.json", "zones": len(room["zones"])})
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
                    # no /sim tab posting the model: whereabouts.py answers from multicam's own trail
                    if api.brain is not None:
                        return self._whereabouts(message, history)
                    return self._json({"error": "no model yet: open /sim in Real events mode"}, 503)
                except Exception as exc:
                    print("guide chat: %s: %s" % (type(exc).__name__, exc))
                    if api.brain is not None:
                        return self._whereabouts(message, history)
                    return self._json({"error": "guide failed"}, 500)
                # trace.html's own item ids come from multicam's memory: the most recent of that kind
                same = [it for it in api.items if it.get("category") == res["item"]]
                newest = max(same, key=lambda it: it.get("last_time") or 0) if same else None
                res["item_id"] = newest["id"] if newest else None
                # The camera tracker sees it but the model was never told (no zone event): say what
                # the camera saw rather than "no phone has seen your keys". Only before any check, so
                # "not there" still moves on through the model's plan.
                ses = api.sessions.get(res["item"]) or {}
                bundle = ((api.model or {}).get("items") or {}).get(res["item"])
                if newest and res["state"] != "found" and not ses.get("tried"):
                    line = guide.camera_reply(bundle, newest, time.time())
                    if line:
                        print("guide chat: the camera sees %s (zone %r) but the model doesn't: draw a zone box "
                              "around it in the camera window, named as in the room map" % (res["item"], newest.get("zone")))
                        if ses:
                            ses["last"] = line
                        res.update(reply=line, state="found", source="camera", why="model had no sighting; camera memory used")
                if res.get("chirp"):
                    api.chirps[res["chirp"]] = api.chirps.get(res["chirp"], 0) + 1
                if res.get("why"):
                    print("guide chat:", res["why"])
                self._json(res)

            def _whereabouts(self, message, history):
                try:
                    res = api.brain.ask(message, history)
                except Exception as exc:      # never take the page down; it has its own answers
                    print("ask: %s: %s" % (type(exc).__name__, exc))
                    return self._json({"error": "whereabouts failed"}, 500)
                if not res.get("reply"):      # not about a tracked item: the page answers itself
                    return self._json({"error": "not about a tracked item"}, 503)
                return self._json(res)

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
                    if p in ("/sim", "/sim/", "/model", "/model/", "/trace-sim/", "/trace-sim/index.html"):
                        return self._file(ROOT / "trace-sim" / "index.html", "text/html; charset=utf-8")
                    if p == "/trace-sim/sample-events.jsonl":
                        return self._file(ROOT / "trace-sim" / "sample-events.jsonl", "application/x-ndjson")
                    if p == "/phone":
                        return self._bytes(PHONE_PAGE.encode(), "text/html; charset=utf-8")
                    m = re.fullmatch(r"/api/chirp/([a-z0-9-]+)", p)
                    if m:
                        api.polls[m.group(1)] = time.time()
                        return self._json({"id": api.chirps.get(m.group(1), 0)})
                    if p == "/api/room":                 # the room the page last built, shared by every browser
                        f = ROOT / "room.json"
                        return self._file(f, "application/json") if f.exists() else self._json({"error": "no room yet"}, 404)
                    if p == "/api/target":               # where Trace is sending the person (the sonar)
                        import guide
                        return self._json(dict(guide.TARGET))
                    if p == "/api/cameras":
                        return self._json(api.cameras())
                    if p == "/api/voice":
                        return self._json({"engine": tts.current_engine() if VOICE else "none"})
                    if p == "/api/speech":
                        return self._speech_stream(parse_qs(url.query))
                    if p == "/api/items":
                        return self._json(api.items or api.model_items())
                    if p == "/api/model":
                        return self._json(api.model or {})
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


if __name__ == "__main__":
    # Standalone: the guide, the 3D page and phone chirps without cameras (multicam.py normally
    # starts this server itself, with its cameras).
    import argparse
    ap = argparse.ArgumentParser(description="Trace's HTTP server without cameras")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 lets phones on your Wi-Fi reach /phone")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    # trace-events.jsonl: what run_trace.py --demo and the bridge use (cleared at each start),
    # not multicam's old events.jsonl, which showed hours-old sightings
    srv = TraceAPI([], ROOT / "memory", ROOT / "trace-events.jsonl").start(a.host, a.port)
    print("3D room and guide: http://localhost:%d/sim   phone beeper: http://<this-laptop>:%d/phone?zone=<id>" % (a.port, a.port))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.stop()
