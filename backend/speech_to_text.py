import os

def transcribe_audio(path: str) -> str:
    """
    Minimal transcription. Tries openai/whisper if available, else returns placeholder.
    """
    try:
        import whisper  # optional
        model = whisper.load_model("small")
        result = model.transcribe(path)
        return result.get("text", "").strip()
    except Exception:
        return f"[transcription unavailable] ({os.path.basename(path)})"