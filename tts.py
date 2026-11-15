"""tts.py -- Trace's voice: calm, offline speech with Kokoro-82M.

No cloud, no API key, no rate limit. The model (~330 MB) downloads once
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

Voices: ids starting "a" are American English, "b" British English; the
first letter picks the Kokoro pipeline. Pre-rendered lines (--prerender) are
played straight from disk when speak() gets exactly that text.

Playback goes through one background thread and a queue, so speak() never
holds up the camera loop. A newer line replaces anything still waiting (we
only care about the latest answer), and the same sentence is not said twice
within REPEAT_WINDOW seconds. With no audio device the line is written to
out.wav instead, with one warning.

Kokoro needs Python 3.10 - 3.12 (its spaCy dependency has no 3.13+ build).
"""
import argparse
import importlib.util
import os
import queue
import re
import sys
import threading
import time
import warnings

import numpy as np

REPO = "hexgrad/Kokoro-82M"
RATE = 24000                 # Kokoro always outputs 24 kHz mono

VOICES = {"female": "af_heart", "male": "am_michael"}
CALM_SPEED = 0.9             # a touch slower than normal speech
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

# --prerender: the fixed demo lines. speak() plays these from disk.
GREETING = "Hi, I'm Trace. What are you looking for?"
LINES = [
    GREETING,
    "Let me have a look.",
    "Found them.",
    "Found it.",
    "I can't see them yet. Try moving the camera slowly around the room.",
    "I can't see it yet. Try moving the camera slowly around the room.",
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
    """Names of the packages the voice needs but cannot find, [] if none."""
    return [name for name in ("kokoro", "soundfile")
            if importlib.util.find_spec(name) is None]


# --------------------------------------------------------------------------
# Choosing a voice
# --------------------------------------------------------------------------

def _resolve(gender=None):
    """"female" / "male" / a voice id / None (the current voice) -> voice id."""
    if gender is None:
        return _voice
    voice = VOICES.get(gender, gender)
    if voice[:1] not in ("a", "b") or "_" not in voice:
        raise ValueError("unknown voice %r: use %s, or a Kokoro voice id such "
                         "as bf_emma" % (gender, " / ".join(VOICES)))
    return voice


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
    reply is not slow. Returns the seconds it took (None when block=False,
    which does the same in a background thread)."""
    if not block:
        threading.Thread(target=warm_up, daemon=True).start()
        return None
    t0 = time.perf_counter()
    synth("Hello.")
    took = time.perf_counter() - t0
    print("voice ready in %.1fs (%s)" % (took, _voice))
    return took


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
        # A local path, so Kokoro does not go to the network for the voice.
        voice_file = _fetch("voices/%s.pt" % voice)
        for result in pipe(text, voice=voice_file, speed=speed,
                           split_pattern=SENTENCE_END):
            if result.audio is None:
                continue
            if parts:
                parts.append(gap)
            parts.append(result.audio.cpu().numpy().astype(np.float32))
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    _remember(key, audio)
    return audio


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
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def _prerendered(text, voice):
    """Path of the pre-rendered file for exactly this text, or None."""
    if text not in LINES or SPEED != CALM_SPEED:
        return None
    path = os.path.join(AUDIO_DIR, voice, slug(text) + ".wav")
    return path if os.path.exists(path) else None


def _audio_for(text, voice):
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
            path = os.path.join(AUDIO_DIR, voice, slug(line) + ".wav")
            save(line, path, voice)
            print("wrote %s" % os.path.relpath(path, HERE))


def main():
    global SPEED
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
    args = parser.parse_args()

    if missing():
        sys.exit("the voice needs %s: pip install \"kokoro>=0.9.4\" soundfile "
                 "sounddevice (Python 3.10-3.12)" % ", ".join(missing()))
    SPEED = args.speed
    voice = set_voice(args.voice or ("male" if args.male else "female"))

    if args.samples:
        write_samples()
        return
    if args.prerender:
        prerender([args.voice] if args.voice else list(VOICES.values()))
        return
    if not args.text:
        parser.error("give some text, or --samples / --prerender")

    t0 = time.perf_counter()
    if args.out:
        save(args.text, args.out)
        seconds = len(synth(args.text)) / RATE
        print("wrote %s (%.1fs of audio, %s, in %.1fs)"
              % (args.out, seconds, voice, time.perf_counter() - t0))
    else:
        speak(args.text, block=True)
        print("said it with %s in %.1fs" % (voice, time.perf_counter() - t0))


if __name__ == "__main__":
    main()
