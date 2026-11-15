"""tts.py -- Trace's voice.

Two engines:

  edge    (default) Microsoft's neural voices through edge-tts: very natural, free,
          no API key, about half a second a sentence. Needs internet, and the
          text of each line goes to Microsoft's speech service (the camera video
          never does). Voices: EDGE_VOICES.
  kokoro  Kokoro-82M on this computer: offline, nothing leaves the machine, a
          bit more synthetic. Used whenever edge is unavailable (no internet,
          edge-tts not installed), or always with TRACE_VOICE=kokoro.

Kokoro: no cloud, no API key, no rate limit. The model (~330 MB) downloads once
from Hugging Face and is read from the local cache after that, so the voice
keeps working with the network unplugged.

    import tts
    tts.warm_up()                                  # load the model, ~1 short synth
    tts.speak("Your keys are on the table.")       # returns at once, plays in the background
    tts.speak("Found them.", block=True)           # wait until it has been said
    tts.set_voice("male")                          # or "female", or a voice id: "bf_emma"
    audio = tts.synth("Let me have a look.")       # float32 numpy array at 24 kHz
    tts.save("Let me have a look.", "look.wav")

From the command line:

    python tts.py "Found them."                    # say it
    python tts.py "Found them." --male
    python tts.py "Found them." --voice bf_emma --speed 0.85
    python tts.py "Found them." --out found.wav    # write a file instead of playing
    python tts.py --samples                        # samples/<voice>.wav, one per candidate voice
    python tts.py --prerender                      # audio/<voice>/<slug>.wav for LINES

    python tts.py --audition                       # samples/audition/*.wav: Kokoro voices, blends, speeds
    python tts.py --audition-edge                  # samples/audition-edge/*.mp3: Microsoft voices
    python tts.py "Found them." --engine kokoro    # force one engine

Voices: ids starting "a" are American English, "b" British English; the
first letter picks the Kokoro pipeline. Pre-rendered lines (--prerender) are
played straight from disk when speak() gets exactly that text.

Blends: "af_heart:0.6,af_bella:0.4" mixes two (or more) voices by weight,
which is the main way to get a warmer or softer sound out of Kokoro. Use a
blend anywhere a voice id goes: VOICES, --voice, set_voice().

Playback goes through one background thread and a queue, so speak() never
holds up the camera loop. A newer line replaces anything still waiting (we
only care about the latest answer), and the same sentence is not said twice
within REPEAT_WINDOW seconds. With no audio device the line is written to
out.wav instead, with one warning.

Kokoro needs Python 3.10 - 3.12 (its spaCy dependency has no 3.13+ build).
"""
import argparse
import importlib.util
import io
import os
import queue
import re
import sys
import threading
import time
import warnings
import wave

import numpy as np

# "edge" (natural, online) or "kokoro" (offline). TRACE_VOICE=kokoro keeps everything local.
ENGINE = os.environ.get("TRACE_VOICE", "edge").strip().lower()
# Microsoft's most conversational voices; see --audition-edge for others (Sonia, Ryan, Emma, ...)
EDGE_VOICES = {"female": "en-US-AvaMultilingualNeural", "male": "en-US-AndrewMultilingualNeural"}
EDGE_RATE = "+0%"            # e.g. "-5%" for a touch slower
EDGE_AUDITION = ["en-US-AvaMultilingualNeural", "en-US-EmmaMultilingualNeural", "en-US-JennyNeural",
                 "en-GB-SoniaNeural", "en-GB-LibbyNeural", "en-US-AndrewMultilingualNeural",
                 "en-US-BrianMultilingualNeural", "en-US-GuyNeural", "en-GB-RyanNeural", "en-GB-ThomasNeural"]
EDGE_RETRY_SECS = 60         # after a failure (no internet), use Kokoro this long before trying edge again

REPO = "hexgrad/Kokoro-82M"
RATE = 24000                 # Kokoro always outputs 24 kHz mono; edge is asked for 24 kHz too

# Blends picked from --audition (clips 03 and 14): af_heart is Kokoro's best voice,
# af_bella adds warmth; am_puck darkens am_michael (and alone it peaks near clipping).
VOICES = {"female": "af_heart:0.6,af_bella:0.4", "male": "am_michael:0.6,am_puck:0.4"}
CALM_SPEED = 0.95            # 0.9 came out near 143 words a minute; natural talk is ~150-160
SPEED = CALM_SPEED           # the CLI's --speed changes this
PAUSE = 0.15                 # seconds of silence between sentences
REPEAT_WINDOW = 5.0          # do not repeat the same sentence within this
CACHE_MAX = 200              # synthesised lines kept in memory

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(HERE, "samples")
AUDIO_DIR = os.path.join(HERE, "audio")
FALLBACK_WAV = os.path.join(HERE, "out.wav")

# --samples: the candidates, all saying the same line, so the team can pick.
SAMPLE_VOICES = ["af_heart", "af_bella", "af_nicole", "bf_emma",
                 "am_michael", "am_fenrir", "bm_george", "bm_fable"]
SAMPLE_LINE = ("Hi, I'm Trace. Take your time, there's no rush. "
               "We'll find it together.")

# --audition: single voices, blends and speeds, all saying a real Trace answer,
# so you can pick by ear. Put the winners in VOICES (and CALM_SPEED).
AUDITION_LINE = ("Found them. Your keys are on the table, just left of the laptop. "
                 "I spotted them on the Hall camera a minute ago.")
AUDITION = [
    # female
    ("af_heart", 0.90), ("af_heart", 1.00),
    ("af_heart:0.6,af_bella:0.4", 0.95), ("af_heart:0.5,af_bella:0.5", 0.95),
    ("af_bella", 0.95), ("af_heart:0.7,af_nicole:0.3", 0.95),
    ("af_heart:0.6,bf_emma:0.4", 0.95), ("bf_emma", 0.95),
    # male
    ("am_michael", 0.90), ("am_michael", 1.00),
    ("am_michael:0.5,am_fenrir:0.5", 0.95), ("am_fenrir", 0.95),
    ("am_puck", 0.95), ("am_michael:0.6,am_puck:0.4", 0.95),
    ("bm_george", 0.95), ("bm_fable", 0.95),
]

# --prerender: the fixed demo lines. speak() plays these from disk.
GREETING = "Hi, I'm Trace. What are you looking for?"
LINES = [
    GREETING,
    "Let me have a look.",
    "Found them.",
    "Found it.",
    "I can't see them yet. Try moving the camera slowly around the room.",
    "I can't see it yet. Try moving the camera slowly around the room.",
    # trace.html's conversation: the follow-up questions and short replies, so they're instant
    "Anything else?",
    "What else can I do?",
    "Anything else I can find?",
    "Okay.",
    "Okay. I'm here if you need me.",
    "I'm here if you need me.",              # the page speaks long lines sentence by sentence
]

# Kokoro splits on this, then we put PAUSE seconds between the pieces.
SENTENCE_END = r"(?<=[.!?])\s+"

# Hugging Face prints a long note about symlinks on every Windows download.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_voice = VOICES["female"]
_model = None                # one KModel, shared by every pipeline
_pipelines = {}              # lang code ("a" / "b") -> KPipeline
_cache = {}                  # (text, voice, speed) -> audio
_lock = threading.Lock()     # Kokoro is not thread-safe: one synth at a time

_queue = queue.Queue()
_worker = None
_last_text, _last_at = None, 0.0
_no_device = False
last_latency = None          # seconds from speak() to sound, for the last line


def missing():
    """Names of the packages the offline (Kokoro) voice needs but cannot find, [] if none."""
    return [name for name in ("kokoro", "soundfile")
            if importlib.util.find_spec(name) is None]


def edge_installed():
    return importlib.util.find_spec("edge_tts") is not None


def available():
    """True when Trace can speak at all (edge or Kokoro)."""
    return (ENGINE == "edge" and edge_installed()) or not missing()


# --------------------------------------------------------------------------
# Edge (Microsoft neural voices, online)
# --------------------------------------------------------------------------

_edge_cache = {}             # (text, voice, rate) -> mp3 bytes
_edge_down_until = 0.0


def _gender_of(voice):
    """Kokoro voice id -> "female"/"male" when it is one of VOICES, else None."""
    for gender, v in VOICES.items():
        if v == voice:
            return gender
    return None


def _edge_voice(voice):
    """The edge voice to use for this Kokoro voice, or None to use Kokoro."""
    if ENGINE != "edge" or time.time() < _edge_down_until or not edge_installed():
        return None
    gender = _gender_of(voice)
    return EDGE_VOICES.get(gender) if gender else None


def _edge_saved(text, voice):
    """MP3 bytes of a pre-rendered online-voice line (works offline too), or None."""
    gender = _gender_of(voice)
    if ENGINE != "edge" or not gender or text not in LINES:
        return None
    path = os.path.join(AUDIO_DIR, "edge", EDGE_VOICES[gender], slug(text) + ".mp3")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def _edge_failed(exc):
    global _edge_down_until
    _edge_down_until = time.time() + EDGE_RETRY_SECS
    print("tts: online voice unavailable (%s: %s); using the offline voice for %ds"
          % (type(exc).__name__, exc, EDGE_RETRY_SECS))


def edge_stream(text, edge_voice, rate=None):
    """Yield MP3 chunks (24 kHz mono) of text in a Microsoft neural voice as they arrive,
    so playback can start after ~0.7 s instead of waiting for the whole clip. Cached and
    pre-rendered lines come back in one piece. Raises before the first chunk on failure."""
    rate = EDGE_RATE if rate is None else rate
    key = (text, edge_voice, rate)
    if key in _edge_cache:
        yield _edge_cache[key]
        return
    path = os.path.join(AUDIO_DIR, "edge", edge_voice, slug(text) + ".mp3")
    if text in LINES and rate == EDGE_RATE and os.path.exists(path):
        with open(path, "rb") as f:          # pre-rendered: instant, and works offline
            yield f.read()
        return

    import asyncio
    import edge_tts
    loop = asyncio.new_event_loop()
    try:
        err = None
        # The multilingual voices now and then send nothing back for very short lines
        # ("Okay.", "Found them."); the same words without the full stop usually work.
        for attempt in dict.fromkeys([text, text.rstrip(".!?") or text]):
            stream = edge_tts.Communicate(attempt, edge_voice, rate=rate,
                                          connect_timeout=4, receive_timeout=15).stream()
            got = bytearray()
            try:
                while True:
                    try:
                        chunk = loop.run_until_complete(stream.__anext__())
                    except StopAsyncIteration:
                        break
                    if chunk["type"] == "audio" and chunk["data"]:
                        got.extend(chunk["data"])
                        yield chunk["data"]
            except edge_tts.exceptions.NoAudioReceived as exc:
                if got:
                    break
                err = exc
                continue
            if got:
                if len(_edge_cache) >= CACHE_MAX:
                    _edge_cache.pop(next(iter(_edge_cache)))
                _edge_cache[key] = bytes(got)
                return
        raise err or RuntimeError("no audio came back")
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()


def edge_mp3(text, edge_voice, rate=None):
    """The whole MP3 of text in a Microsoft neural voice. Raises on failure."""
    return b"".join(edge_stream(text, edge_voice, rate))


def _decode_mp3(data):
    import soundfile as sf
    audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != RATE:                          # edge is asked for 24 kHz, but just in case
        n = int(len(audio) * RATE / rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio).astype(np.float32)
    return audio


def wav_bytes(audio, rate=RATE):
    """float32 mono samples -> a 16-bit WAV file in memory."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def speech_file(text, gender=None):
    """(audio bytes, mime type, engine) for one line, for the web page.
    Edge MP3 when it can, else Kokoro WAV. Raises when neither works."""
    text = text.strip()
    voice = _resolve(gender)
    ev = _edge_voice(voice)
    if ev:
        try:
            return edge_mp3(text, ev), "audio/mpeg", "edge"
        except Exception as exc:
            _edge_failed(exc)
    saved = _edge_saved(text, voice)         # offline, but this line was pre-rendered: same voice
    if saved:
        return saved, "audio/mpeg", "edge"
    if missing():
        raise RuntimeError("no voice: the online voice failed and kokoro is not installed")
    return wav_bytes(_kokoro_audio(text, voice)), "audio/wav", "kokoro"


def speech_stream(text, gender=None):
    """(mime type, engine, chunks): like speech_file(), but the online voice's audio comes
    out as it's made. Falls back to speech_file() (Kokoro, in one piece) when it can't."""
    import itertools
    text = text.strip()
    ev = _edge_voice(_resolve(gender))
    if ev:
        chunks = edge_stream(text, ev)
        try:
            first = next(chunks)              # no internet shows up here, before anything is sent
            return "audio/mpeg", "edge", itertools.chain([first], chunks)
        except StopIteration:
            pass
        except Exception as exc:
            _edge_failed(exc)
    data, mime, engine = speech_file(text, gender)
    return mime, engine, iter([data])


def current_engine():
    """What Trace would speak with right now: "edge", "kokoro" or "none"."""
    if _edge_voice(VOICES["female"]):
        return "edge"
    return "none" if missing() else "kokoro"


# --------------------------------------------------------------------------
# Choosing a voice
# --------------------------------------------------------------------------

VOICE_ID = re.compile(r"^[ab][fm]_[a-z]+$")


def _parts(voice):
    """"af_heart" -> [("af_heart", 1.0)];
    "af_heart:0.6,af_bella:0.4" -> [("af_heart", 0.6), ("af_bella", 0.4)]."""
    out = []
    for bit in voice.split(","):
        name, _, weight = bit.strip().partition(":")
        out.append((name.strip(), float(weight) if weight else 1.0))
    return out


def _resolve(gender=None):
    """"female" / "male" / a voice id or blend / None (the current voice) -> voice."""
    if gender is None:
        return _voice
    voice = VOICES.get(gender, gender)
    try:
        ok = all(VOICE_ID.match(n) and w > 0 for n, w in _parts(voice))
    except ValueError:
        ok = False
    if not ok:
        raise ValueError("unknown voice %r: use %s, a Kokoro voice id such as "
                         "bf_emma, or a blend such as af_heart:0.6,af_bella:0.4"
                         % (gender, " / ".join(VOICES)))
    return voice


def _voice_dir(voice):
    """Folder name for a voice or blend ("af_heart:0.6,af_bella:0.4" has
    characters Windows won't allow)."""
    return re.sub(r"[^A-Za-z0-9_.]+", "-", voice).strip("-")


def set_voice(gender):
    """Switch the default voice: "female", "male", or a voice id."""
    global _voice
    _voice = _resolve(gender)
    return _voice


def current_voice():
    return _voice


# --------------------------------------------------------------------------
# Loading the model -- once, and from the local cache when we can
# --------------------------------------------------------------------------

def _fetch(filename):
    """Local path of one file from the Kokoro repo.

    Tries the cache first without touching the network, so a venue with no
    (or captive-portal) wifi does not stall on a version check. Downloads
    only when the file has never been fetched.
    """
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(REPO, filename, local_files_only=True)
    except Exception:
        print("tts: downloading %s from Hugging Face (once)" % filename)
        return hf_hub_download(REPO, filename)


def _pipeline(lang):
    """The KPipeline for "a" (American) or "b" (British), built on first use."""
    global _model
    if lang in _pipelines:
        return _pipelines[lang]
    # Kokoro's internals warn about things we cannot change.
    warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
    warnings.filterwarnings("ignore", message=".*weight_norm.*")
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
    import torch
    from kokoro import KModel, KPipeline
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = KModel(repo_id=REPO, config=_fetch("config.json"),
                        model=_fetch(KModel.MODEL_NAMES[REPO])).to(device).eval()
    _pipelines[lang] = KPipeline(lang_code=lang, repo_id=REPO, model=_model)
    return _pipelines[lang]


def warm_up(block=True):
    """Load the model and say one short line silently, so the first real
    reply is not slow. Returns the seconds it took.

    block=False does the same in a background thread and returns None at
    once; a failure there (say, the model was never downloaded and there is
    no internet) is printed as one line rather than raised.
    """
    if missing():                         # no Kokoro: nothing to load (edge needs no warm-up)
        print("voice: %s" % ("online (Microsoft %s)" % EDGE_VOICES["female"] if ENGINE == "edge" and edge_installed()
                             else "none -- pip install -r requirements.txt"))
        return None
    if not block:
        threading.Thread(target=_warm_up_quietly, daemon=True).start()
        return None
    t0 = time.perf_counter()
    synth("Hello.")
    took = time.perf_counter() - t0
    print("voice ready in %.1fs (%s)" % (took, _voice))
    return took


def _warm_up_quietly():
    try:
        warm_up()
    except Exception as exc:
        print("tts: voice failed to load: %s: %s" % (type(exc).__name__, exc))


# --------------------------------------------------------------------------
# Making audio
# --------------------------------------------------------------------------

def synth(text, gender=None, speed=None):
    """Text -> float32 numpy array at RATE, sentences spaced by PAUSE."""
    voice = _resolve(gender)
    speed = SPEED if speed is None else speed
    key = (text, voice, speed)
    if key in _cache:
        return _cache[key]

    gap = np.zeros(int(RATE * PAUSE), dtype=np.float32)
    parts = []
    with _lock:
        pipe = _pipeline(voice[0])
        pack = _voice_pack(voice)
        for result in pipe(text, voice=pack, speed=speed,
                           split_pattern=SENTENCE_END):
            if result.audio is None:
                continue
            if parts:
                parts.append(gap)
            parts.append(result.audio.cpu().numpy().astype(np.float32))
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    _remember(key, audio)
    return audio


_packs = {}                  # blend spec -> mixed voice tensor


def _voice_pack(voice):
    """What Kokoro gets as voice=: a local .pt path for one voice (so it never
    goes to the network), or the weighted mix of several for a blend."""
    parts = _parts(voice)
    if len(parts) == 1:
        return _fetch("voices/%s.pt" % parts[0][0])
    if voice not in _packs:
        import torch
        total = sum(w for _, w in parts)
        _packs[voice] = sum(torch.load(_fetch("voices/%s.pt" % name), weights_only=True) * (w / total)
                            for name, w in parts)
    return _packs[voice]


def _remember(key, audio):
    if len(_cache) >= CACHE_MAX:
        _cache.pop(next(iter(_cache)))       # oldest first
    _cache[key] = audio


def save(text, path, gender=None, speed=None):
    """Synthesise text and write it to a wav file. Returns the path."""
    import soundfile as sf
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    sf.write(path, synth(text, gender, speed), RATE)
    return path


def slug(text):
    """"I can't see them yet." -> "i-can-t-see-them-yet" (a file name)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def _prerendered(text, voice):
    """Path of the pre-rendered file for exactly this text, or None."""
    if text not in LINES or SPEED != CALM_SPEED:
        return None
    path = os.path.join(AUDIO_DIR, _voice_dir(voice), slug(text) + ".wav")
    return path if os.path.exists(path) else None


def audio(text, gender=None):
    """Float32 samples at RATE: the online voice when it's up, else Kokoro
    (pre-rendered lines straight from disk either way)."""
    return _audio_for(text.strip(), _resolve(gender))


def _audio_for(text, voice):
    ev = _edge_voice(voice)
    if ev:
        try:
            return _decode_mp3(edge_mp3(text, ev))
        except Exception as exc:          # offline, blocked, ...: the offline voice takes over
            _edge_failed(exc)
    saved = _edge_saved(text, voice)
    if saved:
        return _decode_mp3(saved)
    return _kokoro_audio(text, voice)


def _kokoro_audio(text, voice):
    """Pre-rendered file if there is one, else synthesise (both cached)."""
    key = (text, voice, SPEED)
    if key in _cache:
        return _cache[key]
    path = _prerendered(text, voice)
    if path is None:
        return synth(text, voice)
    import soundfile as sf
    audio, _ = sf.read(path, dtype="float32")
    _remember(key, audio)
    return audio


# --------------------------------------------------------------------------
# Playing it -- one background thread, so the camera loop never waits
# --------------------------------------------------------------------------

def speak(text, gender=None, block=False):
    """Say text aloud. Returns at once unless block=True.

    Returns False when the line was skipped because the same sentence was
    queued less than REPEAT_WINDOW seconds ago.
    """
    global _worker, _last_text, _last_at
    text = text.strip()
    if not text:
        return False
    voice = _resolve(gender)
    now = time.time()
    if text == _last_text and now - _last_at < REPEAT_WINDOW:
        return False
    _last_text, _last_at = text, now

    # Only the newest line matters: drop whatever is still waiting.
    while True:
        try:
            _, _, _, old_done = _queue.get_nowait()
        except queue.Empty:
            break
        old_done.set()

    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_play_loop, daemon=True)
        _worker.start()
    done = threading.Event()
    _queue.put((text, voice, time.perf_counter(), done))
    if block:
        done.wait()
    return True


def _play_loop():
    global last_latency
    while True:
        text, voice, asked_at, done = _queue.get()
        try:
            audio = _audio_for(text, voice)
            last_latency = time.perf_counter() - asked_at
            _play(audio)
        except Exception as exc:          # never let one bad line kill the voice
            print("tts: could not say %r: %s: %s" % (text, type(exc).__name__, exc))
        finally:
            done.set()


def _play(audio):
    """Play through the default output; with none, write FALLBACK_WAV."""
    global _no_device
    if not _no_device:
        try:
            import sounddevice as sd
            sd.play(audio, RATE)
            sd.wait()
            return
        except Exception as exc:          # no PortAudio, no device, device busy
            _no_device = True
            print("tts: WARNING no audio device (%s: %s) -- writing each line "
                  "to %s instead" % (type(exc).__name__, exc, FALLBACK_WAV))
    import soundfile as sf
    sf.write(FALLBACK_WAV, audio, RATE)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def write_samples():
    for voice in SAMPLE_VOICES:
        path = save(SAMPLE_LINE, os.path.join(SAMPLES_DIR, voice + ".wav"), voice)
        print("wrote %s" % os.path.relpath(path, HERE))


def prerender(voices):
    global SPEED
    SPEED = CALM_SPEED                    # speak() only uses files made at this speed
    for voice in voices:
        for line in LINES:
            path = os.path.join(AUDIO_DIR, _voice_dir(voice), slug(line) + ".wav")
            save(line, path, voice)
            print("wrote %s" % os.path.relpath(path, HERE))


def audition():
    """samples/audition/NN_<voice>_<speed>.wav for every AUDITION entry."""
    folder = os.path.join(SAMPLES_DIR, "audition")
    for i, (voice, speed) in enumerate(AUDITION, 1):
        path = os.path.join(folder, "%02d_%s_%.2f.wav" % (i, _voice_dir(voice), speed))
        save(AUDITION_LINE, path, voice, speed)
        print("wrote %s" % os.path.relpath(path, HERE))
    print("Listen, then put your favourites in VOICES (and CALM_SPEED) at the top of tts.py.")


def prerender_edge():
    """audio/edge/<voice>/<slug>.mp3 for LINES in both EDGE_VOICES: instant, and they
    still play when the internet is down."""
    for ev in EDGE_VOICES.values():
        for line in LINES:
            path = os.path.join(AUDIO_DIR, "edge", ev, slug(line) + ".mp3")
            if os.path.exists(path):
                continue
            try:
                data = edge_mp3(line, ev)
            except Exception:                 # one retry: the service drops the odd request
                time.sleep(1)
                data = edge_mp3(line, ev)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            print("wrote %s" % os.path.relpath(path, HERE))


def audition_edge():
    """samples/audition-edge/NN_<voice>.mp3: the same Trace answer in each EDGE_AUDITION voice."""
    folder = os.path.join(SAMPLES_DIR, "audition-edge")
    os.makedirs(folder, exist_ok=True)
    for i, ev in enumerate(EDGE_AUDITION, 1):
        path = os.path.join(folder, "%02d_%s.mp3" % (i, ev))
        try:
            data = edge_mp3(AUDITION_LINE, ev)
        except Exception as exc:              # a voice Microsoft has retired, a network blip, ...
            print("skipped %s (%s)" % (ev, type(exc).__name__))
            continue
        with open(path, "wb") as f:
            f.write(data)
        print("wrote %s" % os.path.relpath(path, HERE))
    print("Listen, then put your favourites in EDGE_VOICES at the top of tts.py.")


def main():
    global SPEED, ENGINE
    parser = argparse.ArgumentParser(description="Trace's offline voice (Kokoro-82M).")
    parser.add_argument("text", nargs="?", help="what to say")
    parser.add_argument("--male", action="store_true", help="use %s" % VOICES["male"])
    parser.add_argument("--voice", help="a Kokoro voice id, e.g. af_heart, bf_emma")
    parser.add_argument("--speed", type=float, default=CALM_SPEED)
    parser.add_argument("--out", help="write a wav file instead of playing")
    parser.add_argument("--samples", action="store_true",
                        help="write samples/<voice>.wav for each candidate voice")
    parser.add_argument("--prerender", action="store_true",
                        help="write audio/<voice>/<slug>.wav for the fixed lines")
    parser.add_argument("--audition", action="store_true",
                        help="write samples/audition/*.wav: Kokoro voices, blends and speeds to pick from")
    parser.add_argument("--audition-edge", action="store_true",
                        help="write samples/audition-edge/*.mp3: Microsoft neural voices to pick from")
    parser.add_argument("--engine", choices=["edge", "kokoro"], help="force one engine (default: %s)" % ENGINE)
    args = parser.parse_args()

    if args.engine:
        ENGINE = args.engine
    if args.audition_edge:
        if not edge_installed():
            sys.exit("needs edge-tts: pip install edge-tts")
        audition_edge()
        return
    if missing() and not (ENGINE == "edge" and edge_installed()):
        sys.exit("the voice needs %s: pip install \"kokoro>=0.9.4\" soundfile "
                 "sounddevice (Python 3.10-3.12)" % ", ".join(missing()))
    SPEED = args.speed
    voice = set_voice(args.voice or ("male" if args.male else "female"))

    if args.samples:
        write_samples()
        return
    if args.audition:
        audition()
        return
    if args.prerender:
        if ENGINE == "edge" and edge_installed():
            prerender_edge()
        if not missing():
            prerender([args.voice] if args.voice else list(VOICES.values()))
        return
    if not args.text:
        parser.error("give some text, or --samples / --prerender")

    t0 = time.perf_counter()
    who = _edge_voice(voice) or voice
    if args.out:
        import soundfile as sf
        samples = audio(args.text)
        sf.write(args.out, samples, RATE)
        print("wrote %s (%.1fs of audio, %s, in %.1fs)"
              % (args.out, len(samples) / RATE, who, time.perf_counter() - t0))
    else:
        speak(args.text, block=True)
        print("said it with %s in %.1fs" % (who, time.perf_counter() - t0))


if __name__ == "__main__":
    main()
