def analyze(path: str) -> dict:
    return {
        "happy": 0.12,
        "sad": 0.08,
        "angry": 0.05,
        "neutral": 0.75,
        "notes": f"video emotion stub for {path}",
    }

video_emotion_analyzer = type("V", (), {"analyze": staticmethod(analyze)})()