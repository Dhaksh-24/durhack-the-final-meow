# Talking to Trace: the page, the voice and "Hey Trace"

Open **http://localhost:8000/** (`trace.html`, served by `trace_api.py`; `python run_trace.py` starts it) and press **Wake Trace**.

- **Live cameras:** press **Watch live** on a camera to see it live, with Trace's boxes
  drawn on top. Opening `trace.html` as a file without `multicam.py` shows demo data
  and a drawing of the room instead.
- **Voice:** Trace reads every answer aloud: Microsoft's neural voice (edge-tts) when
  online, Kokoro (offline) without internet, and the browser's voice if neither works. The **Voice** button opens its settings:
  read aloud on/off and **Female / Male** (remembered per browser). You can also say
  or type "use the male voice".
- **"Hey Trace":** turn on **Hey Trace** in the header. Then say "Hey Trace" (chime)
  and your question, or "Hey Trace, where are my keys?" in one breath. After that it's a
  conversation: Trace answers, asks "Anything else?" and listens for about 7 seconds (soft
  chime), so you don't repeat "Hey Trace". Silence or "that's all" ends it. Things to say:
  - "where are my keys?" (Trace may offer "Want to see the Hall camera live?": say yes or no)
  - "open the living room camera", "switch to the hall camera", "next camera", "go back"
  - "what can you see?", "what do you remember?", "show me", "say that again"
  - "use the male voice", "mute" / "talk to me", "go to sleep" (asks first), "help"

  The mic button in the chat starts the same conversation without the wake word.
  - Listening uses the browser's speech recognition (Chrome, Edge or Safari). In
    Chrome the audio goes to Google to be turned into text; the camera video doesn't.
  - Browsers only allow the microphone on a secure page, so voice commands work at
    `http://localhost:8000` on this computer, but not on a phone at `http://<laptop-ip>:8000`.

## The voice (`tts.py`)

`tts.py` has two engines:

- **edge (default):** Microsoft's neural voices through
  [edge-tts](https://github.com/rany2/edge-tts). Very natural, free, no API key, and
  streamed so it starts in under a second. It needs internet, and **the text of each
  line goes to Microsoft's speech service** (camera video never does).
- **Kokoro (offline):** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on the
  CPU. Nothing leaves the computer. It takes over automatically when edge fails (no
  internet), for 60 s before edge is tried again. Set `TRACE_VOICE=kokoro` to use only
  Kokoro, so no text leaves the machine.

`python tts.py --audition-edge` writes the Microsoft voices to compare, and
`EDGE_VOICES` at the top of `tts.py` picks the pair. The rest of this section is about
Kokoro. `legacy/live_scene.py` uses it too.

### Install

Kokoro needs **Python 3.10 to 3.12** because its spaCy dependency has no build for 3.13+.
On Windows, the `py` launcher can add 3.12 next to a newer Python:

```bash
py install 3.12
py -V:3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(or just `pip install "kokoro>=0.9.4" soundfile sounddevice` into an existing 3.10–3.12 env)

Kokoro uses **espeak-ng** for words that aren't in its dictionary, such as names:

- **Windows:** download the `.msi` installer from the
  [espeak-ng GitHub releases](https://github.com/espeak-ng/espeak-ng/releases) and run it.
  (pip also installs a bundled copy through `espeakng-loader`, so this is a backup.)
- **macOS:** `brew install espeak-ng`
- **Linux:** `sudo apt install espeak-ng`

### Do this once AT HOME, before the venue

The first run downloads the model (~330 MB), each voice (~0.5 MB each) and spaCy's
small English model. Do it on a good connection:

```bash
python tts.py --samples      # downloads everything, writes samples/<voice>.wav
python tts.py --prerender    # writes audio/<voice>/*.wav for the fixed demo lines
```

Everything after that comes from the local cache, with no network calls, not even
version checks. Listen to `samples/` to pick a voice. The candidates are `af_heart`,
`af_bella`, `af_nicole`, `bf_emma`, `am_michael`, `am_fenrir`, `bm_george` and
`bm_fable`. `samples/` and `audio/` are git-ignored.

### Use it

```bash
python tts.py "Found them. Your keys are on the table, just left of the laptop."
python tts.py "Found them." --male
python tts.py "Found them." --voice bf_emma --speed 0.85
python tts.py "Found them." --out found.wav
```

```python
import tts
tts.warm_up()                       # load the model once (~10 s), then replies are quick
tts.speak("Let me have a look.")    # non-blocking; plays on a background thread
```

In `legacy/live_scene.py` (the old one-camera script), Trace greets you at startup and says where each item is. It says
it once, and again only if the item moves somewhere else. Keys:

- `v`: voice on/off
- `m`: switch between the female and male voice

`--male` starts with the male voice and `--no-voice` runs silently. If kokoro isn't
installed, it prints how to install it and carries on without a voice.

### Finding a warmer voice

Kokoro can **blend** voices: `af_heart:0.6,af_bella:0.4` mixes two voices by weight,
and works anywhere a voice id goes (`VOICES`, `--voice`, `set_voice()`). To compare
options by ear:

```bash
python tts.py --audition     # samples/audition/NN_<voice>_<speed>.wav, same Trace line each
```

That writes 16 clips: single voices, blends and two speeds for both female and male.
Put your favourites in `VOICES` (and the speed in `CALM_SPEED`) at the top of `tts.py`.
The Female / Male switch in the app uses `VOICES["female"]` and `VOICES["male"]`. Then
re-run `python tts.py --prerender`.

### Switching voices

- At runtime: press `m` in `legacy/live_scene.py`, or call `tts.set_voice("male")`,
  `tts.set_voice("female")` or `tts.set_voice("bf_emma")`.
- Default pair: edit `VOICES = {"female": "af_heart:0.6,af_bella:0.4", "male": "am_michael:0.6,am_puck:0.4"}` in
  `tts.py`. Voice ids starting with `a` are American and `b` are British. The right
  pipeline is chosen from that first letter.
- After changing a voice, re-run `python tts.py --prerender`. For a voice outside
  `VOICES`, use `python tts.py --prerender --voice bf_emma`.

If there is no audio device, each line is written to `out.wav` with a warning, and
nothing crashes.
