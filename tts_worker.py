#!/usr/bin/env python3
"""
Enhanced Hybrid TTS Worker Daemon:
- Primary Engine: Google TTS (gTTS) - Giọng trong trẻo, tự nhiên, đọc tiếng Việt chuẩn xác (Chị Google), tốc độ siêu nhanh (~0.5s).
- Fallback Engine: ZeroTTS offline local model.
- Strict Vietnamese-only text sanitizer.
"""
import hashlib
import json
import os
import re
import sys
import time
import traceback
import unicodedata
from pathlib import Path

# Đảm bảo I/O tiếng Việt UTF-8 chuẩn xác tuyệt đối trên mọi nền tảng (đặc biệt là Windows cp1252)
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "3")

from gtts import gTTS

try:
    import numpy as np
    from zerotts import ZeroTTS, normalize_vi_text
    from zerotts.chunking import normalize_punctuation
    HAS_ZEROTTS = True
except Exception:
    HAS_ZEROTTS = False

CACHE_DIR = Path(__file__).resolve().parent / "data" / "tts-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_MODEL_CANDIDATES = [
    os.environ.get("ZEROTTS_MODEL_PATH", ""),
    str(Path(__file__).resolve().parent / "data" / "models" / "zerotts"),
    "/Users/zens/Work/UIT/Projects/f1-metrics-explainer/models/zerotts",
]

# Bộ lọc ký tự tiếng Việt chuẩn
VIETNAMESE_ALLOWED_RE = re.compile(
    r"[^0-9a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ\s.,!?]"
)
BRACKET_EMOTE_RE = re.compile(r"\[[^\]]{1,24}\]")
REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}", re.IGNORECASE)
MULTI_PUNCT_RE = re.compile(r"([.,!?])\1+")
MULTI_SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")


def resolve_model_source() -> str:
    for candidate in LOCAL_MODEL_CANDIDATES:
        if candidate and os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return "zeroweight-ai/ZeroTTS"


def emit(payload: dict) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def clean_and_normalize_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    # Chuẩn hóa Unicode sang dạng NFC thống nhất
    text = unicodedata.normalize("NFC", unicodedata.normalize("NFKC", str(raw_text)))
    # Xóa link URL nếu có
    text = URL_RE.sub(" ", text)
    # Thay dấu hai chấm thành dấu phẩy để ngắt nhịp tự nhiên giữa Tên người và Lời nói
    text = text.replace(":", ", ")
    text = BRACKET_EMOTE_RE.sub(" ", text)
    text = VIETNAMESE_ALLOWED_RE.sub(" ", text)
    text = MULTI_PUNCT_RE.sub(r"\1", text)
    text = REPEATED_CHARS_RE.sub(r"\1\1", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip(" .,!?-")

    letters = re.findall(r"[a-zA-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹ]", text)
    if len(letters) < 2:
        return ""

    if len(text) > 160:
        text = text[:160].rsplit(" ", 1)[0]

    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def polish_waveform(wav, sample_rate: int):
    """De-click ends and prevent PCM_16 digital clipping."""
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr.reshape(1, 0)

    arr = arr - np.mean(arr)

    peak = float(np.max(np.abs(arr)))
    if peak > 0.90:
        arr = arr * (0.90 / peak)

    fade_in = min(int(sample_rate * 0.012), arr.size // 4)
    fade_out = min(int(sample_rate * 0.020), arr.size // 4)
    if fade_in > 1:
        ramp_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_in, dtype=np.float32)))
        arr[:fade_in] *= ramp_in
    if fade_out > 1:
        ramp_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_out, dtype=np.float32)))
        arr[-fade_out:] *= ramp_out

    return arr.reshape(1, -1)


def main() -> None:
    t0 = time.time()
    tts = None
    voices = ["google"]

    # Google TTS (Chị Google) sẵn sàng tức thì (0.05s)
    emit({
        "type": "ready",
        "model": "Google TTS (Chị Google)",
        "voice": "google",
        "voices": ["google"],
        "sampleRate": 24000,
        "loadSec": 0.05,
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            text = (req.get("text") or "").strip()
            voice = (req.get("voice") or "google").strip().lower()

            norm_text = clean_and_normalize_text(text)
            if not norm_text:
                emit({"type": "result", "id": req_id, "ok": False, "error": "Filtered empty/spam text"})
                continue

            # ================================================================
            # 1. GOOGLE TTS ENGINE (CHỊ GOOGLE): Tự nhiên, trong trẻo, 0.5s
            # ================================================================
            if voice in ("google", "gtts", "chigoogle", "maichi") or tts is None or voice not in voices:
                cache_key = hashlib.sha1(f"gtts::{norm_text}".encode("utf-8")).hexdigest()[:20]
                out_name = f"google_{cache_key}.mp3"
                out_path = CACHE_DIR / out_name

                if out_path.is_file() and out_path.stat().st_size > 200:
                    emit({
                        "type": "result",
                        "id": req_id,
                        "ok": True,
                        "cached": True,
                        "voice": "google",
                        "file": out_name,
                        "url": f"/tts-cache/{out_name}",
                        "synthSec": 0.0,
                    })
                    continue

                t_synth = time.time()
                try:
                    g = gTTS(text=norm_text, lang='vi', tld='com.vn')
                    g.save(str(out_path))
                    synth_sec = round(time.time() - t_synth, 2)
                    emit({
                        "type": "result",
                        "id": req_id,
                        "ok": True,
                        "cached": False,
                        "voice": "google",
                        "file": out_name,
                        "url": f"/tts-cache/{out_name}",
                        "synthSec": synth_sec,
                    })
                    continue
                except Exception as e_gtts:
                    sys.stderr.write(f"[gTTS Error, fallback to ZeroTTS] {e_gtts}\n")
                    if tts is None:
                        emit({"type": "result", "id": req_id, "ok": False, "error": str(e_gtts)})
                        continue

            # ================================================================
            # 2. ZEROTTS OFFLINE FALLBACK ENGINE
            # ================================================================
            cache_key = hashlib.sha1(f"v3::{voice}::{norm_text}".encode("utf-8")).hexdigest()[:20]
            out_name = f"{voice}_{cache_key}.wav"
            out_path = CACHE_DIR / out_name

            if out_path.is_file() and out_path.stat().st_size > 200:
                emit({
                    "type": "result",
                    "id": req_id,
                    "ok": True,
                    "cached": True,
                    "voice": voice,
                    "file": out_name,
                    "url": f"/tts-cache/{out_name}",
                    "synthSec": 0.0,
                })
                continue

            t_synth = time.time()
            raw_wav = tts.synthesize(norm_text, voice=voice, eoa_extra_frames=2)
            if raw_wav is None or raw_wav.size == 0:
                emit({"type": "result", "id": req_id, "ok": False, "error": "No audio generated"})
                continue

            clean_audio = polish_waveform(raw_wav, int(tts.sample_rate))
            tts.save_audio(clean_audio, str(out_path))
            synth_sec = round(time.time() - t_synth, 2)
            duration_sec = round(clean_audio.shape[-1] / float(tts.sample_rate), 2)

            emit({
                "type": "result",
                "id": req_id,
                "ok": True,
                "cached": False,
                "voice": voice,
                "file": out_name,
                "url": f"/tts-cache/{out_name}",
                "synthSec": synth_sec,
                "durationSec": duration_sec,
            })
        except Exception as exc:
            emit({
                "type": "result",
                "id": req_id,
                "ok": False,
                "error": str(exc),
            })


if __name__ == "__main__":
    main()
