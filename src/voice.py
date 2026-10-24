"""Voice layer: Gnani STT (ears) and Gnani TTS (voice). No other speech service is used anywhere.

SDK usage follows the capstone brief (pip install gnani-vachana; key in GNANI_API_KEY).
"""
import re
import time
import wave
from datetime import datetime
from pathlib import Path

from . import config

RECOVERY_REPLY = "Sorry, I didn't catch that. Could you say it again?"

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]


def detect_lang(text, default=None):
    """Very small heuristic: Devanagari -> hi-IN, otherwise the default (en-IN handles Hinglish)."""
    if re.search(r"[\u0900-\u097F]", text or ""):
        return "hi-IN"
    return default or config.DEFAULT_LANG


def speechify(text):
    """Make a reply easy to read aloud: strip markdown, say dates with the year, group long numbers."""
    t = re.sub(r"[*_`#>]+", "", text or "")
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)

    def iso(m):
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"
        except ValueError:
            return m.group(0)

    t = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", iso, t)
    t = re.sub(r"₹\s?([\d,]+(?:\.\d+)?)", lambda m: f"{m.group(1).replace(',', '')} rupees", t)
    t = re.sub(r"\bRs\.?\s?([\d,]+)", lambda m: f"{m.group(1).replace(',', '')} rupees", t)

    def group_id(m):  # ORD-1234567 -> "ORD 123 456 7" so the TTS reads it in chunks
        prefix, digits = m.group(1), m.group(2)
        chunks = [digits[i:i + 3] for i in range(0, len(digits), 3)]
        return (prefix + " " if prefix else "") + " ".join(chunks)

    t = re.sub(r"\b([A-Za-z]{0,5})[-_]?(\d{6,})\b", group_id, t)
    return re.sub(r"\s+", " ", t).strip()


def _wav_seconds(path):
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def _split_wav(path, max_s=55):
    """STT REST accepts clips up to 60 s; split longer WAVs into <=55 s pieces."""
    secs = _wav_seconds(path)
    if secs is None or secs <= 58:
        return [Path(path)]
    pieces = []
    with wave.open(str(path), "rb") as w:
        params, rate = w.getparams(), w.getframerate()
        step = int(rate * max_s)
        i = 0
        while True:
            frames = w.readframes(step)
            if not frames:
                break
            out = Path(path).with_suffix(f".part{i}.wav")
            with wave.open(str(out), "wb") as o:
                o.setparams(params)
                o.writeframes(frames)
            pieces.append(out)
            i += 1
    return pieces


class Voice:
    def __init__(self, voice=None, model=None):
        from gnani.stt import GnaniSTTClient  # official SDK: pip install gnani-vachana
        from gnani.tts import GnaniTTSClient

        self.stt = GnaniSTTClient()  # reads GNANI_API_KEY
        self.tts = GnaniTTSClient()
        self.voice = voice or config.GNANI_VOICE
        self.model = model or config.GNANI_TTS_MODEL  # always name the model explicitly

    def transcribe(self, audio_path, lang=None, retries=2):
        lang = lang or config.DEFAULT_LANG
        parts = []
        for piece in _split_wav(audio_path):
            for attempt in range(retries + 1):
                try:
                    out = self.stt.transcribe(str(piece), language_code=lang)
                    parts.append((out.get("transcript") or "").strip())
                    break
                except Exception as exc:
                    if attempt == retries:
                        print(f"[stt] failed for {piece}: {exc}")
                    time.sleep(1.0 * (attempt + 1))
        return " ".join(p for p in parts if p).strip()

    def speak(self, text, out_path, lang=None, retries=2):
        lang = lang or config.DEFAULT_LANG
        spoken = speechify(text)
        for attempt in range(retries + 1):
            try:
                self.tts.synthesize(spoken, voice=self.voice, model=self.model, language=lang,
                                    output_file=str(out_path))
                return str(out_path)
            except Exception as exc:
                if attempt == retries:
                    print(f"[tts] failed: {exc}")
                    return None
                time.sleep(1.0 * (attempt + 1))

    def turn(self, audio_path, agent, out_path, lang=None):
        """audio in -> (heard, reply, audio_out). Recovers politely from empty/unclear audio."""
        heard = self.transcribe(audio_path, lang)
        reply = agent.respond(heard) if heard.strip() else RECOVERY_REPLY
        reply_lang = detect_lang(reply, lang)
        return heard, reply, self.speak(reply, out_path, reply_lang)
