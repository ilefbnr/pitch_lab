from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import json
import uuid
import shutil
import subprocess
import base64
import json
import uuid
import shutil
import subprocess
import asyncio
from database import get_db, engine
from models import Base, User, Pitch
from schemas import PitchResponse
from speech_to_text import transcribe_audio
from voice_analysis import ml_voice_analyzer
from realtime_analysis import realtime_analyzer
from video_emotion_analysis import video_emotion_analyzer

try:
    from pydub import AudioSegment  # type: ignore[import-not-found]
except Exception:
    AudioSegment = None  # type: ignore

# Create tables
Base.metadata.create_all(bind=engine)

# Uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

app = FastAPI(title="Pitch Recording API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8050",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ helpers ------------
def ensure_default_user():
    db = next(get_db())
    try:
        if not db.query(User).filter(User.username == "default").first():
            db.add(User(username="default", email="default@local", created_at=datetime.utcnow()))
            db.commit()
    finally:
        db.close()

ensure_default_user()

def make_json_safe(obj: Any):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)

def save_upload_file(upload: UploadFile, dest_dir: Path, filename: Optional[str] = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = filename or f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex}_{upload.filename or 'upload'}"
    dest = dest_dir / name
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest
def convert_to_wav(input_path: str, output_path: str) -> None:
    if AudioSegment is not None:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    else:
        # ffmpeg fallback
        cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def extract_audio_from_video(video_path: str) -> Tuple[bool, Optional[str]]:
    out_audio = str(Path(video_path).with_suffix(".wav"))
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", out_audio]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, out_audio
    except Exception:
        return False, None

# ------------ websocket manager ------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.metrics: Dict[str, Dict] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.setdefault(session_id, []).append(websocket)
        self.metrics.setdefault(session_id, {"messages": 0, "last": None})

    def disconnect(self, session_id: str, websocket: WebSocket):
        conns = self.active_connections.get(session_id, [])
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            self.active_connections.pop(session_id, None)
            self.metrics.pop(session_id, None)

    async def broadcast(self, session_id: str, message: dict):
        for ws in list(self.active_connections.get(session_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                try:
                    await ws.close()
                except Exception:
                    pass
                self.disconnect(session_id, ws)

manager = ConnectionManager()

# ------------ routes ------------
@app.get("/")
async def root():
    return {"service": "PitchLab backend", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "active_realtime_sessions": len(manager.active_connections),
    }

@app.post("/pitches", response_model=PitchResponse)
async def create_pitch(
    title: str = Form(...),
    description: str = Form(""),
    audio_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    audio_path = None
    video_path = None
    transcript = ""
    analysis_result: Dict[str, Any] = {}

    # save uploads
    try:
        if audio_file:
            saved = save_upload_file(audio_file, uploads_dir)
            audio_path = str(saved)
            if saved.suffix.lower() in [".webm", ".ogg", ".m4a"]:
                wav_path = str(saved.with_suffix(".wav"))
                try:
                    convert_to_wav(str(saved), wav_path)
                    audio_path = wav_path
                except Exception:
                    pass

        if video_file:
            saved_v = save_upload_file(video_file, uploads_dir)
            video_path = str(saved_v)
            ok, extracted = extract_audio_from_video(video_path)
            if ok and extracted:
                audio_path = extracted
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploads: {e}")

    # transcription
    try:
        if audio_path:
            transcript = transcribe_audio(audio_path)
    except Exception:
        transcript = "Transcription failed"

    # analysis
    try:
        if audio_path and ml_voice_analyzer:
            analysis_result["voice"] = ml_voice_analyzer.analyze(audio_path)
        if video_path and video_emotion_analyzer:
            analysis_result["video"] = video_emotion_analyzer.analyze(video_path)
    except Exception as e:
        analysis_result["error"] = f"{e}"

    # persist
    db_pitch = Pitch(
        title=title,
        description=description,
        audio_file_path=audio_path,
        video_file_path=video_path,
        transcript=transcript,
        analysis_result=json.dumps(make_json_safe(analysis_result)),
        created_at=datetime.utcnow(),
        user_id=1,
    )
    db.add(db_pitch)
    db.commit()
    db.refresh(db_pitch)

    return {
        "id": db_pitch.id,
        "title": db_pitch.title,
        "description": db_pitch.description,
        "transcript": db_pitch.transcript,
        "audio_file_path": db_pitch.audio_file_path,
        "video_file_path": db_pitch.video_file_path,
        "analysis_result": json.loads(db_pitch.analysis_result) if db_pitch.analysis_result else None,
        "created_at": db_pitch.created_at,
        "user_id": db_pitch.user_id,
    }

@app.get("/pitches", response_model=List[PitchResponse])
async def list_pitches(db: Session = Depends(get_db)):
    rows = db.query(Pitch).order_by(Pitch.created_at.desc()).all()
    out = []
    for p in rows:
        out.append({
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "transcript": p.transcript,
            "audio_file_path": p.audio_file_path,
            "video_file_path": p.video_file_path,
            "analysis_result": json.loads(p.analysis_result) if p.analysis_result else None,
            "created_at": p.created_at,
            "user_id": p.user_id,
        })
    return out

@app.get("/pitches/{pitch_id}", response_model=PitchResponse)
async def get_pitch(pitch_id: int, db: Session = Depends(get_db)):
    p = db.query(Pitch).filter(Pitch.id == pitch_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Pitch not found")
    return {
        "id": p.id,
        "title": p.title,
        "description": p.description,
        "transcript": p.transcript,
        "audio_file_path": p.audio_file_path,
        "video_file_path": p.video_file_path,
        "analysis_result": json.loads(p.analysis_result) if p.analysis_result else None,
        "created_at": p.created_at,
        "user_id": p.user_id,
    }

@app.get("/audio/{pitch_id}")
async def get_pitch_audio(pitch_id: int, db: Session = Depends(get_db)):
    p = db.query(Pitch).filter(Pitch.id == pitch_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Pitch not found")
    if not p.audio_file_path:
        raise HTTPException(status_code=404, detail="No audio for this pitch")
    path = Path(p.audio_file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on server")
    # best-effort content-type
    media_type = "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)

@app.post("/pitches/{pitch_id}/investor-response")
async def investor_response(pitch_id: int, db: Session = Depends(get_db)):
    p = db.query(Pitch).filter(Pitch.id == pitch_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Pitch not found")
    transcript = (p.transcript or "").lower()
    score = 50 + 15 * ("market" in transcript or "customers" in transcript) + 15 * ("revenue" in transcript or "monetiz" in transcript) + 10 * ("team" in transcript)
    sentiment = "positive" if score >= 75 else ("negative" if score < 50 else "neutral")
    return {
        "pitch_id": p.id,
        "investor_interest_score": score,
        "summary": (p.transcript[:300] + "...") if p.transcript else "",
        "recommendation": "Proceed to follow-up" if score >= 65 else "Refine pitch and metrics",
        "sentiment": sentiment,
    }

@app.post("/realtime/start-session")
async def start_realtime_session():
    session_id = uuid.uuid4().hex
    manager.metrics[session_id] = {"messages": 0, "last": None}
    return {"session_id": session_id}

@app.get("/realtime/session/{session_id}/metrics")
async def get_session_metrics(session_id: str):
    m = manager.metrics.get(session_id)
    if not m:
        raise HTTPException(status_code=404, detail="Session not found")
    return m

@app.websocket("/ws/realtime/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                obj = json.loads(data)
            except Exception:
                obj = {"raw": data}
            metrics = manager.metrics.setdefault(session_id, {"messages": 0, "last": None})
            metrics["messages"] += 1
            metrics["last"] = datetime.utcnow().isoformat()
            analysis = None
            if realtime_analyzer:
                try:
                    analysis = realtime_analyzer.process(obj)
                except Exception:
                    analysis = {"note": "realtime analyzer failure"}
            payload = {"event": "message", "data": obj, "analysis": analysis, "timestamp": datetime.utcnow().isoformat()}
            await manager.broadcast(session_id, payload)
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
        manager.disconnect(session_id, websocket)

@app.post("/video/emotion-analysis")
async def video_emotion_analysis_endpoint(video_file: UploadFile = File(...)):
    saved = save_upload_file(video_file, uploads_dir)
    if video_emotion_analyzer:
        result = video_emotion_analyzer.analyze(str(saved))
        return {"result": result}
    return {"result": "video emotion analyzer not available"}

@app.post("/audio/enhanced-emotion-analysis")
async def audio_enhanced_analysis(audio_file: UploadFile = File(...)):
    saved = save_upload_file(audio_file, uploads_dir)
    if ml_voice_analyzer:
        return {"result": ml_voice_analyzer.analyze(str(saved))}
    return {"result": "no analyzer available"}

@app.post("/video/realtime-emotion")
async def video_realtime_emotion(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    image_file: Optional[UploadFile] = File(default=None),
):
    image_b64: Optional[str] = None
    if image_file:
        raw = await image_file.read()
        image_b64 = "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    elif payload and isinstance(payload, dict):
        image_b64 = payload.get("image_b64") or payload.get("image")
    if not image_b64:
        raise HTTPException(status_code=400, detail="Provide image_file or image_b64")
    return realtime_analyzer.process({"event": "video_frame", "data": {"image_b64": image_b64}})

@app.post("/audio/realtime-emotion")
async def audio_realtime_emotion(
    payload: Optional[Dict[str, Any]] = Body(default=None),
    audio_file: Optional[UploadFile] = File(default=None),
):
    audio_b64: Optional[str] = None
    sr: Optional[int] = None
    if audio_file:
        raw = await audio_file.read()
        audio_b64 = "data:audio/wav;base64," + base64.b64encode(raw).decode("ascii")
    elif payload and isinstance(payload, dict):
        audio_b64 = payload.get("audio_b64") or payload.get("audio")
        sr = payload.get("sr")
    if not audio_b64:
        raise HTTPException(status_code=400, detail="Provide audio_file or audio_b64")
    return realtime_analyzer.process({"event": "audio_chunk", "data": {"audio_b64": audio_b64, "sr": sr}})

@app.websocket("/realtime-pitch")
async def websocket_alias(websocket: WebSocket):
    session_id = "default"
    await manager.connect(session_id, websocket)
    # Annonce d’état initiale
    await websocket.send_json({"event": "ready", "session_id": session_id, "ts": datetime.utcnow().isoformat()})

    # Tâche keepalive (envoie un heartbeat toutes les 25s)
    async def _keepalive():
        try:
            while True:
                await asyncio.sleep(25)
                await websocket.send_json({"event": "heartbeat", "ts": datetime.utcnow().isoformat()})
        except Exception:
            pass

    ka = asyncio.create_task(_keepalive())
    try:
        while True:
            # Accepte texte ou binaire
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("text") if "text" in msg else None
            if data is None and "bytes" in msg and msg["bytes"] is not None:
                # tente de décoder binaire en texte
                try:
                    data = msg["bytes"].decode("utf-8", errors="ignore")
                except Exception:
                    data = None

            try:
                obj = json.loads(data) if data else {"raw": data}
            except Exception:
                obj = {"raw": data}

            metrics = manager.metrics.setdefault(session_id, {"messages": 0, "last": None})
            metrics["messages"] += 1
            metrics["last"] = datetime.utcnow().isoformat()

            analysis = None
            if realtime_analyzer:
                try:
                    analysis = realtime_analyzer.process(obj)
                except Exception:
                    analysis = {"note": "realtime analyzer failure"}

            payload = {
                "event": "message",
                "data": obj,
                "analysis": analysis,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await manager.broadcast(session_id, payload)
    except WebSocketDisconnect:
        pass
    finally:
        ka.cancel()
        manager.disconnect(session_id, websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)