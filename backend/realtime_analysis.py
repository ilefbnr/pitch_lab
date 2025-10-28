from typing import Any, Dict, Optional, Tuple
import base64, io, time

# Optional deps (graceful fallbacks)
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:
    librosa = None  # type: ignore


# -----------------------
# Helpers
# -----------------------
def _strip_data_url_prefix(s: str) -> str:
    if not s:
        return s
    if ";base64," in s:
        return s.split(";base64,", 1)[1]
    if "base64," in s:
        return s.split("base64,", 1)[1]
    return s

def _decode_image(b64: str) -> Optional["np.ndarray"]:
    if np is None or cv2 is None or not b64:
        return None
    try:
        raw = base64.b64decode(_strip_data_url_prefix(b64))
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def _decode_audio(b64: str, sr_hint: Optional[int] = None) -> Tuple[Optional["np.ndarray"], Optional[int]]:
    if np is None or not b64:
        return None, None
    raw = base64.b64decode(_strip_data_url_prefix(b64))
    if sf is not None:
        try:
            data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if hasattr(data, "ndim") and data.ndim > 1:
                data = data.mean(axis=1)
            return data, int(rate)
        except Exception:
            pass
    try:
        if sr_hint is None:
            return None, None
        pcm = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        return pcm, int(sr_hint)
    except Exception:
        return None, None


# -----------------------
# Video analysis (OpenCV)
# -----------------------
_last_face_ts = 0.0
_FACE_MIN_INTERVAL_MS = 80

def _analyze_faces(img_bgr: "np.ndarray") -> Dict[str, Any]:
    global _last_face_ts
    if np is None or cv2 is None:
        return {"note": "numpy/opencv not available"}
    now = time.time() * 1000.0
    if now - _last_face_ts < _FACE_MIN_INTERVAL_MS:
        return {"skipped": True}
    _last_face_ts = now

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    det = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    faces = int(len(det))
    return {"faces": faces, "metrics": [], "engagement_visual": float(min(1.0, faces / 2.0))}

# -----------------------
# Audio analysis
# -----------------------
def _estimate_pitch_hz(y: "np.ndarray", sr: int) -> Optional[float]:
    try:
        if librosa is not None and y.size > 0:
            f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, win_length=1024)
            f0 = f0[np.isfinite(f0)]
            if f0.size:
                return float(np.median(f0))
    except Exception:
        pass
    try:
        y = y - y.mean()
        corr = np.correlate(y, y, mode="full")[len(y) - 1 :]
        corr[: int(sr / 20)] = 0  # ignore <50Hz
        peak = int(np.argmax(corr[: int(sr / 2)]))
        if peak > 0:
            return float(sr / peak)
    except Exception:
        pass
    return None

def _analyze_audio(y: "np.ndarray", sr: int) -> Dict[str, Any]:
    if np is None:
        return {"note": "numpy not available"}
    if y.size == 0 or sr <= 0:
        return {"note": "empty audio"}
    rms = float(np.sqrt(np.mean(y.astype("float64") ** 2)))
    zcr = float(((y[:-1] * y[1:]) < 0).mean()) if y.size > 1 else 0.0
    pitch = _estimate_pitch_hz(y, sr)
    speaking_prob = float(min(1.0, max(0.0, (rms - 0.01) * 20.0)))
    return {
        "rms": round(rms, 4),
        "zcr": round(zcr, 4),
        "pitch_hz": round(pitch, 1) if pitch else None,
        "speaking_prob": round(speaking_prob, 3),
    }

# -----------------------
# Public API
# -----------------------
def process(message: Dict[str, Any]) -> Dict[str, Any]:
    event = (message or {}).get("event")
    data = (message or {}).get("data", {})
    out: Dict[str, Any] = {"event": event}

    if event == "video_frame":
        img_b64 = data.get("image_b64") or data.get("image") or ""
        img = _decode_image(img_b64) if img_b64 else None
        out["video"] = _analyze_faces(img) if img is not None else {"error": "invalid image"}
    elif event == "audio_chunk":
        sr_hint = data.get("sr")
        audio_b64 = data.get("audio_b64") or data.get("audio") or ""
        y, sr = _decode_audio(audio_b64, sr_hint=sr_hint)
        out["audio"] = _analyze_audio(y, sr) if (y is not None and sr is not None) else {"error": "invalid audio"}
    else:
        out["note"] = "unknown event"

    vis = out.get("video", {})
    aud = out.get("audio", {})
    vis_eng = float(vis.get("engagement_visual", 0.0)) if isinstance(vis, dict) else 0.0
    speak = float(aud.get("speaking_prob", 0.0)) if isinstance(aud, dict) else 0.0
    out["engagement_score"] = round(min(1.0, 0.6 * vis_eng + 0.8 * speak), 3)
    return out

realtime_analyzer = type("RealtimeAnalyzer", (), {"process": staticmethod(process)})()