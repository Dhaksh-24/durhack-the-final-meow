# durhack-the-final-meow

## Run Trace (the app)

```bash
python multicam.py                    # cameras from cameras.json, or webcam 0
```

Then open **http://localhost:8000/** and press **Wake Trace**. `multicam.py` serves
the page (`trace.html`) through `trace_api.py`, with:

- **Live cameras:** press **Watch live** on a camera to see it live, with Trace's boxes
  drawn on top. Opening `trace.html` as a file without `multicam.py` shows demo data
  and a drawing of the room instead.
- **Voice:** Trace reads every answer aloud, using Kokoro (`tts.py`) when it's
  installed and the browser's voice otherwise. The **Voice** button turns it off.
- **"Hey Trace":** turn on **Hey Trace** in the header. Then say "Hey Trace" (chime)
  and your question, or "Hey Trace, where are my keys?" in one breath. You can also say
  "say that again", "stop", "go back", or "show me the hall camera". The mic button in
  the chat listens once without the wake word.
  - Listening uses the browser's speech recognition (Chrome, Edge or Safari). In
    Chrome the audio goes to Google to be turned into text; the camera video doesn't.
  - Browsers only allow the microphone on a secure page, so voice commands work at
    `http://localhost:8000` on this computer, but not on a phone at `http://<laptop-ip>:8000`.

For phones on the same Wi-Fi, run `python multicam.py --api-host 0.0.0.0`, then open
`http://<laptop-ip>:8000/`. Anyone on that network can then watch the cameras.

## Voice (offline)

Trace speaks with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), running
locally on the CPU. There is no cloud TTS and no API key, so it has no rate limits and
keeps working with the network unplugged. `tts.py` is the voice and `live_scene.py`
uses it.

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

In `live_scene.py`, Trace greets you at startup and says where each item is. It says
it once, and again only if the item moves somewhere else. Keys:

- `v`: voice on/off
- `m`: switch between the female and male voice

`--male` starts with the male voice and `--no-voice` runs silently. If kokoro isn't
installed, it prints how to install it and carries on without a voice.

### Switching voices

- At runtime: press `m` in `live_scene.py`, or call `tts.set_voice("male")`,
  `tts.set_voice("female")` or `tts.set_voice("bf_emma")`.
- Default pair: edit `VOICES = {"female": "af_heart", "male": "am_michael"}` in
  `tts.py`. Voice ids starting with `a` are American and `b` are British. The right
  pipeline is chosen from that first letter.
- After changing a voice, re-run `python tts.py --prerender`. For a voice outside
  `VOICES`, use `python tts.py --prerender --voice bf_emma`.

If there is no audio device, each line is written to `out.wav` with a warning, and
nothing crashes.
