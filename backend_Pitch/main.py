from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import os
import uuid
import aiofiles
import json
import asyncio
from pathlib import Path
import subprocess
import tempfile
import librosa
import numpy as np
from typing import Optional, Tuple
import ffmpeg
import base64
from pydub import AudioSegment
import cv2
from io import BytesIO
from PIL import Image

from database import get_db, engine
from models import Base, User, Pitch
from schemas import UserCreate, UserLogin, UserResponse, PitchCreate, PitchResponse
# Auth imports removed - no authentication required
from file_storage import storage
from speech_to_text import transcribe_audio
from voice_analysis import ml_voice_analyzer
from investor_ai import investor_ai
from realtime_analysis import realtime_analyzer

# Phase 4: Import advanced features
try:
    from video_emotion_analysis import video_emotion_analyzer
    VIDEO_ANALYSIS_AVAILABLE = True
    print("✅ Video emotion analysis loaded successfully")
except Exception as e:
    VIDEO_ANALYSIS_AVAILABLE = False
    print(f"⚠️ Video emotion analysis not available: {e}")
    # Create a mock analyzer
    class MockVideoEmotionAnalyzer:
        async def analyze_video_file(self, *args, **kwargs):
            return {'error': 'Video analysis not available', 'emotions': {'neutral': 1.0}}
        async def analyze_frame_base64(self, *args, **kwargs):
            return {'error': 'Video analysis not available', 'emotions': {'neutral': 1.0}}
    video_emotion_analyzer = MockVideoEmotionAnalyzer()

try:
    from enhanced_voice_emotion import enhanced_voice_analyzer
    ENHANCED_VOICE_AVAILABLE = True
    print("✅ Enhanced voice emotion analysis loaded successfully")
except Exception as e:
    ENHANCED_VOICE_AVAILABLE = False
    print(f"⚠️ Enhanced voice emotion analysis not available: {e}")
    # Create a mock analyzer
    class MockEnhancedVoiceAnalyzer:
        async def analyze_voice_emotions(self, *args, **kwargs):
            return {'error': 'Enhanced voice analysis not available', 'emotions': {'neutral': 1.0}}
    enhanced_voice_analyzer = MockEnhancedVoiceAnalyzer()

try:
    from comprehensive_reporting import comprehensive_report_generator
    COMPREHENSIVE_REPORTING_AVAILABLE = True
    print("✅ Comprehensive reporting loaded successfully")
except Exception as e:
    COMPREHENSIVE_REPORTING_AVAILABLE = False
    print(f"⚠️ Comprehensive reporting not available: {e}")
    # Create a mock report generator
    class MockComprehensiveReportGenerator:
        async def generate_comprehensive_report(self, *args, **kwargs):
            return {'error': 'Comprehensive reporting not available', 'report': 'Basic report only'}
    comprehensive_report_generator = MockComprehensiveReportGenerator()

# Create tables
Base.metadata.create_all(bind=engine)

# Ensure default user exists for non-authenticated access
def ensure_default_user():
    """Ensure a default user exists for non-authenticated operations"""
    db = next(get_db())
    try:
        # Check if default user exists
        default_user = db.query(User).filter(User.id == 1).first()
        if not default_user:
            # Create default user
            default_user = User(
                email="default@pitch-lab.com",
                username="default_user",
                hashed_password="not_used"  # Password not needed since auth is removed
            )
            db.add(default_user)
            db.commit()
            print("✅ Default user created for non-authenticated access")
        else:
            print("✅ Default user already exists")
    except Exception as e:
        print(f"❌ Error ensuring default user: {e}")
        db.rollback()
    finally:
        db.close()

# Ensure default user exists on startup
ensure_default_user()

app = FastAPI(title="Pitch Recording API", version="1.0.0")

# CORS middleware - Updated for new backend port and common frontend ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server  
        "http://localhost:8080",  # Legacy port
        "http://localhost:8050",  # Current backend port (for testing)
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://localhost:8001",  # Current backend port (for testing)
        "http://127.0.0.1:8001",

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security removed - no authentication required

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        # Don't call accept() here - it should be called in the main endpoint
        self.active_connections[session_id] = websocket
        self.user_sessions[user_id] = session_id
        print(f"WebSocket connected: session {session_id}, user {user_id}")
    
    def disconnect(self, session_id: str, user_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        print(f"WebSocket disconnected: session {session_id}")
    
    async def send_personal_message(self, message: dict, session_id: str):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                # Check if WebSocket is still open
                if hasattr(websocket, 'client_state') and websocket.client_state.name != 'CONNECTED':
                    print(f"⚠️ WebSocket {session_id} not connected (state: {websocket.client_state.name}), removing from active connections")
                    self.disconnect(session_id, "unknown")
                    return
                
                # Ensure all data is JSON serializable
                safe_message = self._make_json_safe(message)
                message_json = json.dumps(safe_message)
                
                # Send message with better error handling
                await websocket.send_text(message_json)
                print(f"✅ Message sent to {session_id}: {message.get('event', 'unknown_event')}")
                
            except (TypeError, ValueError) as e:
                print(f"❌ JSON encoding error for {session_id}: {e}")
                print(f"   Message data: {message}")
                # Try to send a simpler error message
                try:
                    await websocket.send_text(json.dumps({'error': 'Failed to encode message'}))
                except:
                    pass
            except ConnectionResetError as e:
                print(f"❌ Connection reset for {session_id}: {e}")
                self.disconnect(session_id, "unknown")
            except Exception as e:
                print(f"❌ Error sending message to {session_id}: {type(e).__name__}: {str(e)}")
                print(f"   WebSocket state: {getattr(websocket, 'client_state', 'unknown')}")
                # Disconnect on WebSocket errors to prevent spam
                if "Cannot call" in str(e) or "close message" in str(e):
                    print(f"🔌 Disconnecting broken WebSocket {session_id}")
                    self.disconnect(session_id, "unknown")
                pass
        else:
            print(f"⚠️ Attempted to send message to non-existent session: {session_id}")
    
    def _make_json_safe(self, obj):
        """Convert objects to JSON-safe format including numpy types"""
        import numpy as np
        from datetime import datetime, date
        import decimal
        
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item') and callable(obj.item):  # numpy scalar
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return {k: self._make_json_safe(v) for k, v in obj.__dict__.items()}
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            try:
                return [self._make_json_safe(item) for item in obj]
            except (TypeError, AttributeError):
                return str(obj)
        else:
            return str(obj)  # Convert everything else to string
    
    async def emit_to_session(self, event: str, data: dict, session_id: str = None):
        """Emit event to specific session or all sessions"""
        message = {
            'event': event,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        if session_id:
            await self.send_personal_message(message, session_id)
        else:
            # Broadcast to all connections
            for session_id, websocket in self.active_connections.items():
                await self.send_personal_message(message, session_id)

manager = ConnectionManager()

def make_json_safe(obj):
    """Global helper to convert objects to JSON-safe format including numpy types"""
    import numpy as np
    from datetime import datetime, date
    import decimal
    
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item') and callable(obj.item):  # numpy scalar
        return obj.item()
    elif hasattr(obj, '__dict__'):
        return {k: make_json_safe(v) for k, v in obj.__dict__.items()}
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            return [make_json_safe(item) for item in obj]
        except (TypeError, AttributeError):
            return str(obj)
    else:
        return str(obj)  # Convert everything else to string

# Create uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Pitch Recording API", "port": 8001, "auth": "disabled"}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "backend_port": 8001,
        "authentication": "disabled",
        "websocket_endpoint": "/ws/realtime/{session_id}",
        "active_connections": len(manager.active_connections) if 'manager' in globals() else 0
    }

# Registration endpoint removed - no authentication required

# Login endpoint removed - no authentication required

# get_current_user function removed - no authentication required

@app.post("/pitches", response_model=PitchResponse)
async def create_pitch(
    title: str = Form(...),
    description: str = Form(""),
    audio_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),  # Phase 4: Optional video input
    db: Session = Depends(get_db)
):
    # Handle audio file if provided
    file_path = None
    if audio_file:
        audio_content = await audio_file.read()
        
        # Validate audio content
        if not audio_content or len(audio_content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file received. Please record audio before uploading.")
        
        if len(audio_content) < 1024:  # Less than 1KB
            raise HTTPException(status_code=400, detail="Audio file too small. Please record for at least a few seconds.")
        
        # Save audio file using storage system
        try:
            file_path = await storage.save_audio_file(audio_content, audio_file.filename or "recording.wav")
            
            # Double-check the saved file
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise HTTPException(status_code=500, detail="Failed to save audio file properly.")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save audio file: {str(e)}")
    
    # Phase 4: Save video file if provided and detect audio
    video_file_path = None
    extracted_audio_path = None
    
    if video_file:
        try:
            video_content = await video_file.read()
            video_file_path = await storage.save_video_file(video_content, video_file.filename or "recording.mp4")
            
            # Detect audio in video file
            has_audio_in_video, extracted_audio_path = detect_audio_in_video(video_file_path)
            
            if has_audio_in_video and extracted_audio_path:
                print(f"✅ Audio detected in video: {extracted_audio_path}")
                # Use the extracted audio as the primary audio source
                file_path = extracted_audio_path
            else:
                print("ℹ️ No meaningful audio detected in video - proceeding with video-only analysis")
                
        except Exception as e:
            print(f"Video save failed: {e}")  # Log but don't fail the request
    
    # Transcribe audio if available (from separate audio file or extracted from video)
    transcript = ""
    if file_path:
        try:
            if extracted_audio_path and file_path == extracted_audio_path:
                # Transcribe audio extracted from video
                transcript = await transcribe_audio_from_video(video_file_path)
            else:
                # Transcribe separate audio file
                transcript = await transcribe_audio(file_path)
        except Exception as e:
            transcript = f"Transcription failed: {str(e)}"
    else:
        transcript = "No audio provided - video-only pitch"
    
    # Phase 4: Enhanced multimodal analysis
    analysis_results = {}
    
    # Determine content type based on actual audio presence
    has_audio = file_path is not None and transcript and not transcript.startswith("Transcription failed")
    has_video = video_file_path is not None
    
    if has_video and has_audio:
        content_type = "multimodal"
    elif has_video and not has_audio:
        content_type = "video_only"
    elif has_audio and not has_video:
        content_type = "audio_only"
    else:
        content_type = "unknown"
    
    print(f"📊 Content type determined: {content_type} (has_audio: {has_audio}, has_video: {has_video})")
    print(f"⏱️ Starting analysis for {content_type} content...")
    
    # Perform ML voice analysis (existing) - only if audio is available
    analysis_result = None
    if has_audio and transcript and not transcript.startswith("Transcription failed"):
        print(f"🎵 Starting audio analysis...")
        try:
            # Use pydub for better audio format support
            from pydub import AudioSegment
            import tempfile
            
            # Load audio with pydub (supports more formats)
            audio = AudioSegment.from_file(file_path)
            
            # Calculate audio duration - file_path is already the correct audio file
            duration = None
            try:
                import librosa
                y, sr = librosa.load(file_path)
                duration = librosa.get_duration(y=y, sr=sr)
            except Exception as e:
                print(f"Librosa duration calculation failed: {e}")
                # Fallback: use pydub to get duration
                duration = len(audio) / 1000.0  # Duration in seconds
            
            # Perform ML analysis
            analysis = await ml_voice_analyzer.analyze_transcript(
                transcript=transcript,
                audio_duration=duration,
                audio_file_path=file_path
            )
            
            # Convert analysis to JSON string
            analysis_result = json.dumps(analysis, default=str)
            analysis_results['voice_ml_analysis'] = analysis
            
        except Exception as e:
            print(f"ML Analysis failed: {e}")  # Log for debugging
            analysis_result = json.dumps({
                'error': f'ML Analysis failed: {str(e)}',
                'confidence_score': 0,
                'overall_grade': 'F'
            })
    
    # Phase 4: Enhanced voice emotion analysis - only if audio is available
    if has_audio:
        try:
            enhanced_voice_analysis = await enhanced_voice_analyzer.analyze_voice_emotions(file_path)
            analysis_results['enhanced_voice_emotion'] = enhanced_voice_analysis
        except Exception as e:
            print(f"Enhanced voice analysis failed: {e}")
    
    # Phase 4: Video emotion analysis if video provided
    if has_video:
        try:
            print(f"📹 Starting video emotion analysis...")
            video_analysis = await video_emotion_analyzer.analyze_video_file(video_file_path)
            analysis_results['video_emotion_analysis'] = video_analysis
            print(f"✅ Video emotion analysis completed")
        except Exception as e:
            print(f"❌ Video analysis failed: {e}")
    
    # Create comprehensive analysis based on content type
    if content_type == "video_only":
        try:
            # Generate comprehensive video-only analysis
            video_only_analysis = {
                'content_type': 'video_only',
                'analysis_summary': {
                    'title': 'Video-Only Pitch Analysis',
                    'description': 'Analysis of visual presentation without audio content',
                    'focus_areas': [
                        'Facial expressions and emotional engagement',
                        'Body language and presentation confidence',
                        'Visual storytelling effectiveness',
                        'Professional appearance and demeanor'
                    ]
                },
                'video_analysis': analysis_results.get('video_emotion_analysis', {}),
                'recommendations': {
                    'visual_presentation': [
                        'Maintain consistent eye contact with the camera',
                        'Use expressive facial features to convey enthusiasm',
                        'Practice confident body language and posture',
                        'Consider adding audio narration for better engagement'
                    ],
                    'technical_improvements': [
                        'Ensure good lighting for clear facial visibility',
                        'Use a stable camera setup to avoid distractions',
                        'Practice smooth, deliberate movements',
                        'Consider using visual aids or slides'
                    ]
                },
                'scoring': {
                    'emotional_engagement': analysis_results.get('video_emotion_analysis', {}).get('business_appropriateness', 0.5),
                    'visual_confidence': analysis_results.get('video_emotion_analysis', {}).get('emotion_stability', 0.5),
                    'overall_grade': 'B' if analysis_results.get('video_emotion_analysis', {}).get('business_appropriateness', 0.5) > 0.6 else 'C'
                }
            }
            
            # Convert to JSON string
            analysis_result = json.dumps(video_only_analysis, default=str)
            analysis_results['comprehensive_video_analysis'] = video_only_analysis
            
        except Exception as e:
            print(f"Video-only analysis failed: {e}")
            analysis_result = json.dumps({
                'error': f'Video-only analysis failed: {str(e)}',
                'content_type': 'video_only',
                'confidence_score': 0,
                'overall_grade': 'F'
            })
    
    elif content_type == "multimodal":
        try:
            # Combine audio and video analysis for multimodal content
            multimodal_analysis = {
                'content_type': 'multimodal',
                'analysis_summary': {
                    'title': 'Multimodal Pitch Analysis',
                    'description': 'Comprehensive analysis of both audio and visual presentation',
                    'focus_areas': [
                        'Verbal communication and clarity',
                        'Facial expressions and emotional engagement',
                        'Overall presentation confidence',
                        'Content structure and delivery'
                    ]
                },
                'audio_analysis': analysis_results.get('voice_ml_analysis', {}),
                'video_analysis': analysis_results.get('video_emotion_analysis', {}),
                'enhanced_voice_analysis': analysis_results.get('enhanced_voice_emotion', {}),
                'recommendations': {
                    'communication': [
                        'Balance verbal and visual elements effectively',
                        'Ensure audio clarity matches visual confidence',
                        'Practice synchronized delivery of content'
                    ],
                    'presentation': [
                        'Maintain consistent energy throughout',
                        'Use gestures to emphasize key points',
                        'Practice smooth transitions between topics'
                    ]
                },
                'scoring': {
                    'audio_confidence': analysis_results.get('voice_ml_analysis', {}).get('confidence_score', 0.5),
                    'visual_confidence': analysis_results.get('video_emotion_analysis', {}).get('emotion_stability', 0.5),
                    'overall_grade': 'A' if (analysis_results.get('voice_ml_analysis', {}).get('confidence_score', 0.5) + 
                                           analysis_results.get('video_emotion_analysis', {}).get('business_appropriateness', 0.5)) / 2 > 0.8 else 'B'
                }
            }
            
            # Convert to JSON string
            analysis_result = json.dumps(multimodal_analysis, default=str)
            analysis_results['comprehensive_multimodal_analysis'] = multimodal_analysis
            
        except Exception as e:
            print(f"Multimodal analysis failed: {e}")
            analysis_result = json.dumps({
                'error': f'Multimodal analysis failed: {str(e)}',
                'content_type': 'multimodal',
                'confidence_score': 0,
                'overall_grade': 'F'
            })
    
    else:  # audio_only or unknown
        # Use existing audio analysis logic
        pass
    
    # Create pitch record
    print(f"💾 Saving pitch to database...")
    # Use default user ID since authentication is removed
    default_user_id = 1  # You can change this or create a test user
    db_pitch = Pitch(
        title=title,
        description=description,
        audio_file_path=file_path if file_path else None,  # Only set if audio file exists
        video_file_path=video_file_path,  # Phase 4: Save video file path
        transcript=transcript,
        analysis_result=analysis_result,
        user_id=default_user_id
    )
    db.add(db_pitch)
    db.commit()
    db.refresh(db_pitch)
    print(f"✅ Pitch analysis completed successfully!")
    
    return PitchResponse(
        id=db_pitch.id,
        title=db_pitch.title,
        description=db_pitch.description,
        transcript=db_pitch.transcript,
        audio_file_path=db_pitch.audio_file_path if db_pitch.audio_file_path else None,
        video_file_path=db_pitch.video_file_path,  # Phase 4: Include video file path
        analysis_result=db_pitch.analysis_result,
        created_at=db_pitch.created_at,
        user_id=db_pitch.user_id
    )

@app.get("/pitches", response_model=list[PitchResponse])
async def get_pitches(
    db: Session = Depends(get_db)
):
    # Return all pitches since authentication is removed
    pitches = db.query(Pitch).all()
    return [
        PitchResponse(
            id=pitch.id,
            title=pitch.title,
            description=pitch.description,
            transcript=pitch.transcript,
            audio_file_path=pitch.audio_file_path if pitch.audio_file_path else None,
            video_file_path=pitch.video_file_path,  # Phase 4: Include video file path
            analysis_result=pitch.analysis_result,
            created_at=pitch.created_at,
            user_id=pitch.user_id
        )
        for pitch in pitches
    ]

@app.get("/pitches/{pitch_id}", response_model=PitchResponse)
async def get_pitch(
    pitch_id: int,
    db: Session = Depends(get_db)
):
    # Return pitch by ID without user filtering since authentication is removed
    pitch = db.query(Pitch).filter(Pitch.id == pitch_id).first()
    
    if not pitch:
        raise HTTPException(
            status_code=404,
            detail="Pitch not found"
        )
    
    return PitchResponse(
        id=pitch.id,
        title=pitch.title,
        description=pitch.description,
        transcript=pitch.transcript,
        audio_file_path=pitch.audio_file_path if pitch.audio_file_path else None,
        video_file_path=pitch.video_file_path,  # Phase 4: Include video file path
        analysis_result=pitch.analysis_result,
        created_at=pitch.created_at,
        user_id=pitch.user_id
    )

@app.get("/audio/{pitch_id}")
async def get_audio(
    pitch_id: int,
    db: Session = Depends(get_db)
):
    # Return audio without user filtering since authentication is removed
    pitch = db.query(Pitch).filter(Pitch.id == pitch_id).first()
    
    if not pitch:
        raise HTTPException(
            status_code=404,
            detail="Pitch not found"
        )
    
    # Check if audio file path exists and is not None
    if not pitch.audio_file_path:
        raise HTTPException(
            status_code=404,
            detail="No audio file available for this pitch"
        )
    
    if not Path(pitch.audio_file_path).exists():
        raise HTTPException(
            status_code=404,
            detail="Audio file not found"
        )
    
    return FileResponse(
        pitch.audio_file_path,
        media_type="audio/wav",
        filename=f"pitch_{pitch_id}.wav"
    )

@app.post("/pitches/{pitch_id}/investor-response")
async def generate_investor_response(
    pitch_id: int,
    db: Session = Depends(get_db)
):
    """Generate AI investor response for a specific pitch"""
    # Get the pitch without user filtering since authentication is removed
    pitch = db.query(Pitch).filter(Pitch.id == pitch_id).first()
    
    if not pitch:
        raise HTTPException(
            status_code=404,
            detail="Pitch not found"
        )
    
    # Generate investor response
    try:
        response = await investor_ai.generate_investor_response(
            transcript=pitch.transcript,
            analysis_result=pitch.analysis_result
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate investor response: {str(e)}"
        )

@app.get("/pitches/{pitch_id}/analysis")
async def get_pitch_analysis(
    pitch_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed analysis results for a specific pitch"""
    # Get pitch analysis without user filtering since authentication is removed
    pitch = db.query(Pitch).filter(Pitch.id == pitch_id).first()
    
    if not pitch:
        raise HTTPException(
            status_code=404,
            detail="Pitch not found"
        )
    
    if not pitch.analysis_result:
        raise HTTPException(
            status_code=404,
            detail="Analysis not available for this pitch"
        )
    
    try:
        import json
        analysis = json.loads(pitch.analysis_result)
        return analysis
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid analysis data"
        )

@app.websocket("/ws/realtime/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time pitch analysis"""
    default_user_id = "1"
    
    try:
        # Accept connection first
        await websocket.accept()
        print(f"🔌 WebSocket connection accepted for session: {session_id}")
        
        # Register connection (no authentication required)
        await manager.connect(websocket, session_id, default_user_id)
        print(f"📝 Connection registered in manager for session: {session_id}")
        
        # Start real-time analysis session
        try:
            await realtime_analyzer.start_session(session_id, int(default_user_id))
            print(f"Real-time session started for {session_id}")
        except Exception as e:
            print(f"Failed to start real-time session: {e}")
            await websocket.send_text(json.dumps({'error': f'Failed to start session: {str(e)}'}))
            await websocket.close()
            return
        
        # Create emit callback for real-time updates
        async def emit_callback(event: str, data: dict):
            try:
                await manager.emit_to_session(event, data, session_id)
            except Exception as e:
                print(f"Error emitting event {event}: {e}")
        
        # Send connection confirmation
        await manager.emit_to_session('connected', {
            'session_id': session_id,
            'user_id': default_user_id,
            'status': 'ready_for_audio'
        }, session_id)
        
        print(f"WebSocket fully connected and ready for session {session_id}")
        
        try:
            while True:
                # Receive message from client
                try:
                    message = await websocket.receive()
                except WebSocketDisconnect:
                    print(f"Client disconnected from session {session_id}")
                    break
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
                
                if message['type'] == 'websocket.receive':
                    if 'bytes' in message:
                        # Process audio data
                        audio_data = message['bytes']
                        
                        # Process audio chunk with real-time analysis
                        result = await realtime_analyzer.process_audio_chunk(
                            session_id=session_id,
                            audio_data=audio_data,
                            emit_callback=emit_callback
                        )
                        
                    elif 'text' in message:
                        # Handle text commands
                        try:
                            command = json.loads(message['text'])
                            command_type = command.get('type')
                            
                            if command_type == 'request_investor_response':
                                # Generate live investor response
                                response = await realtime_analyzer.generate_live_investor_response(
                                    session_id=session_id,
                                    emit_callback=emit_callback
                                )
                                
                            elif command_type == 'end_session':
                                # End the session
                                summary = await realtime_analyzer.end_session(session_id)
                                await manager.emit_to_session('session_ended', summary, session_id)
                                break
                                
                            elif command_type == 'get_metrics':
                                # Send current metrics
                                await manager.emit_to_session('current_metrics', {
                                    'metrics': realtime_analyzer.live_metrics
                                }, session_id)
                                
                        except json.JSONDecodeError:
                            await manager.emit_to_session('error', {
                                'message': 'Invalid command format'
                            }, session_id)
                
        except WebSocketDisconnect:
            # Handle client disconnect
            print(f"Client disconnected from session {session_id}")
            
        except Exception as e:
            print(f"WebSocket error in session {session_id}: {e}")
            await manager.emit_to_session('error', {
                'message': f'Server error: {str(e)}'
            }, session_id)
            
    except Exception as e:
        print(f"WebSocket connection error: {e}")
        
    finally:
        # Clean up - this is critical to stop processing
        print(f"🧹 Cleaning up WebSocket session: {session_id}")
        manager.disconnect(session_id, default_user_id if 'default_user_id' in locals() else 'unknown')
        try:
            # End the session to stop all processing immediately
            await realtime_analyzer.end_session(session_id)
            print(f"✅ Real-time session {session_id} ended and cleaned up")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup error for session {session_id}: {cleanup_error}")

@app.post("/realtime/start-session")
async def start_realtime_session():
    """Start a new real-time analysis session"""
    session_id = str(uuid.uuid4())
    default_user_id = 1  # Use default user ID since authentication is removed
    
    try:
        result = await realtime_analyzer.start_session(session_id, default_user_id)
        return {
            'session_id': session_id,
            'websocket_url': f'/ws/realtime/{session_id}',
            'status': 'ready',
            'user_id': default_user_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start session: {str(e)}"
        )

@app.get("/realtime/session/{session_id}/metrics")
async def get_session_metrics(
    session_id: str
):
    """Get current metrics for a real-time session"""
    try:
        if session_id in realtime_analyzer.session_data:
            session = realtime_analyzer.session_data[session_id]
            # Remove user access check since authentication is removed
            
            return {
                'session_id': session_id,
                'live_metrics': realtime_analyzer.live_metrics,
                'session_info': {
                    'start_time': session['start_time'].isoformat(),
                    'total_chunks': session['total_chunks'],
                    'analysis_count': len(session['analysis_results'])
                }
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

# ========================
# Phase 4: Advanced Features Endpoints
# ========================

@app.post("/video/emotion-analysis")
async def analyze_video_emotions(
    video_file: UploadFile = File(...)
):
    """Analyze emotions from uploaded video file"""
    try:
        # Save video file temporarily
        video_content = await video_file.read()
        video_path = await storage.save_video_file(video_content, video_file.filename or "video.mp4")
        
        # Analyze video emotions
        analysis = await video_emotion_analyzer.analyze_video_file(video_path)
        
        result = {
            'video_analysis': analysis,
            'filename': video_file.filename,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return make_json_safe(result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video emotion analysis failed: {str(e)}"
        )

@app.post("/video/realtime-emotion")
async def analyze_realtime_video_frame(
    frame_data: dict
):
    """Analyze emotions from a real-time video frame"""
    try:
        frame_base64 = frame_data.get('frame')
        if not frame_base64:
            raise HTTPException(status_code=400, detail="Frame data required")
        
        # Analyze frame emotions
        analysis = await video_emotion_analyzer.analyze_frame_base64(frame_base64)
        
        result = {
            'frame_analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        return make_json_safe(result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Real-time video analysis failed: {str(e)}"
        )

@app.post("/video/comprehensive-analysis")
async def analyze_video_comprehensive(
    video_file: UploadFile = File(...),
    include_emotion_analysis: bool = True,
    include_presentation_metrics: bool = True
):
    """Comprehensive video analysis for presentation skills"""
    try:
        if not VIDEO_ANALYSIS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Video analysis not available")
        
        # Save video file
        video_content = await video_file.read()
        video_path = await storage.save_video_file(video_content, video_file.filename or "analysis_video.mp4")
        
        # Perform comprehensive analysis
        analysis_results = {}
        
        # Emotion analysis
        if include_emotion_analysis:
            try:
                emotion_analysis = await video_emotion_analyzer.analyze_video_file(video_path)
                analysis_results['emotion_analysis'] = emotion_analysis
            except Exception as e:
                print(f"Emotion analysis failed: {e}")
                analysis_results['emotion_analysis'] = {'error': str(e)}
        
        # Presentation metrics (if available)
        if include_presentation_metrics:
            try:
                # Add basic presentation metrics
                presentation_metrics = {
                    'video_duration': 0,  # Would need to extract from video
                    'analysis_quality': 'high' if analysis_results.get('emotion_analysis', {}).get('error') is None else 'medium',
                    'recommendations_count': len(analysis_results.get('emotion_analysis', {}).get('recommendations', [])),
                    'confidence_indicators': {
                        'emotional_stability': analysis_results.get('emotion_analysis', {}).get('emotion_stability', 0.5),
                        'business_appropriateness': analysis_results.get('emotion_analysis', {}).get('business_appropriateness', 0.5),
                        'engagement_level': analysis_results.get('emotion_analysis', {}).get('average_confidence', 0.5)
                    }
                }
                analysis_results['presentation_metrics'] = presentation_metrics
            except Exception as e:
                print(f"Presentation metrics failed: {e}")
                analysis_results['presentation_metrics'] = {'error': str(e)}
        
        # Generate comprehensive summary
        comprehensive_summary = {
            'analysis_type': 'comprehensive_video_analysis',
            'content_type': 'video_only',
            'analysis_summary': {
                'title': 'Comprehensive Video Presentation Analysis',
                'description': 'Detailed analysis of visual presentation skills and emotional engagement',
                'focus_areas': [
                    'Facial expressions and emotional engagement',
                    'Presentation confidence and professionalism',
                    'Visual storytelling effectiveness',
                    'Technical presentation quality'
                ]
            },
            'detailed_results': analysis_results,
            'overall_assessment': {
                'emotional_engagement_score': analysis_results.get('emotion_analysis', {}).get('business_appropriateness', 0.5),
                'presentation_confidence': analysis_results.get('emotion_analysis', {}).get('emotion_stability', 0.5),
                'overall_grade': 'A' if analysis_results.get('emotion_analysis', {}).get('business_appropriateness', 0.5) > 0.8 else 
                               'B' if analysis_results.get('emotion_analysis', {}).get('business_appropriateness', 0.5) > 0.6 else 
                               'C' if analysis_results.get('emotion_analysis', {}).get('business_appropriateness', 0.5) > 0.4 else 'D'
            },
            'improvement_areas': [
                'Enhance facial expressiveness for better engagement',
                'Practice consistent emotional delivery',
                'Improve camera presence and eye contact',
                'Consider adding audio narration for better impact'
            ]
        }
        
        return comprehensive_summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive video analysis failed: {str(e)}")

@app.post("/audio/enhanced-emotion-analysis")
async def analyze_enhanced_voice_emotions(
    audio_file: UploadFile = File(...)
):
    """Perform enhanced voice emotion analysis"""
    try:
        # Save audio file temporarily
        audio_content = await audio_file.read()
        audio_path = await storage.save_audio_file(audio_content, audio_file.filename or "audio.wav")
        
        # Perform enhanced voice emotion analysis
        analysis = await enhanced_voice_analyzer.analyze_voice_emotions(audio_path)
        
        result = {
            'enhanced_voice_analysis': analysis,
            'filename': audio_file.filename,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return make_json_safe(result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced voice emotion analysis failed: {str(e)}"
        )

@app.post("/pitches/{pitch_id}/comprehensive-report")
async def generate_comprehensive_report(
    pitch_id: int,
    report_type: str = "detailed_analysis",
    db: Session = Depends(get_db)
):
    """Generate comprehensive post-pitch report with visualizations"""
    try:
        # Get the pitch without user filtering since authentication is removed
        pitch = db.query(Pitch).filter(Pitch.id == pitch_id).first()
        
        if not pitch:
            raise HTTPException(status_code=404, detail="Pitch not found")
        
        # Prepare pitch data
        pitch_data = {
            'id': pitch.id,
            'title': pitch.title,
            'description': pitch.description,
            'transcript': pitch.transcript,
            'created_at': pitch.created_at.isoformat()
        }
        
        # Gather all analysis results
        analysis_results = {}
        
        # Include existing ML analysis
        if pitch.analysis_result:
            try:
                import json
                analysis_results['voice_ml_analysis'] = json.loads(pitch.analysis_result)
            except json.JSONDecodeError:
                pass
        
        # Perform enhanced voice analysis if audio file exists
        if pitch.audio_file_path and Path(pitch.audio_file_path).exists():
            try:
                enhanced_analysis = await enhanced_voice_analyzer.analyze_voice_emotions(pitch.audio_file_path)
                analysis_results['enhanced_voice_emotion'] = enhanced_analysis
            except Exception as e:
                print(f"Enhanced voice analysis failed: {e}")
        
        # Perform video analysis if video file exists (check for video file with same base name)
        audio_path = Path(pitch.audio_file_path)
        video_extensions = ['.mp4', '.avi', '.mov', '.webm']
        for ext in video_extensions:
            video_path = audio_path.with_suffix(ext)
            if video_path.exists():
                try:
                    video_analysis = await video_emotion_analyzer.analyze_video_file(str(video_path))
                    analysis_results['video_emotion_analysis'] = video_analysis
                    break
                except Exception as e:
                    print(f"Video analysis failed: {e}")
        
        # Generate comprehensive report
        report = await comprehensive_report_generator.generate_comprehensive_report(
            pitch_data=pitch_data,
            analysis_results=analysis_results,
            report_type=report_type
        )
        
        # Make sure all data is JSON-safe before returning
        safe_report = make_json_safe(report)
        
        return safe_report
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )

@app.get("/reports/templates")
async def get_report_templates():
    """Get available report templates"""
    return {
        'templates': [
            {
                'type': 'executive_summary',
                'name': 'Executive Summary',
                'description': 'High-level overview with key insights and scores'
            },
            {
                'type': 'detailed_analysis',
                'name': 'Detailed Analysis',
                'description': 'Comprehensive breakdown of all analysis components'
            },
            {
                'type': 'improvement_plan',
                'name': 'Improvement Plan',
                'description': 'Focused recommendations and practice schedule'
            },
            {
                'type': 'benchmarking',
                'name': 'Benchmarking Report',
                'description': 'Comparison with industry standards and top performers'
            }
        ]
    }

@app.post("/multimodal/analyze")
async def multimodal_analysis(
    audio_file: UploadFile = File(...),
    video_file: Optional[UploadFile] = File(None),
    include_enhanced_voice: bool = True,
    include_video_emotion: bool = True
):
    """Perform comprehensive multimodal analysis"""
    try:
        analysis_results = {}
        
        # Save and analyze audio
        audio_content = await audio_file.read()
        audio_path = await storage.save_audio_file(audio_content, audio_file.filename or "audio.wav")
        
        # Transcribe audio
        try:
            transcript = await transcribe_audio(audio_path)
            analysis_results['transcript'] = transcript
        except Exception as e:
            transcript = f"Transcription failed: {str(e)}"
            analysis_results['transcript'] = transcript
        
        # ML voice analysis
        if transcript and not transcript.startswith("Transcription failed"):
            try:
                import librosa
                y, sr = librosa.load(audio_path)
                duration = librosa.get_duration(y=y, sr=sr)
                
                voice_ml_analysis = await ml_voice_analyzer.analyze_transcript(
                    transcript=transcript,
                    audio_duration=duration,
                    audio_file_path=audio_path
                )
                analysis_results['voice_ml_analysis'] = voice_ml_analysis
            except Exception as e:
                print(f"ML voice analysis failed: {e}")
        
        # Enhanced voice emotion analysis
        if include_enhanced_voice:
            try:
                enhanced_voice_analysis = await enhanced_voice_analyzer.analyze_voice_emotions(audio_path)
                analysis_results['enhanced_voice_emotion'] = enhanced_voice_analysis
            except Exception as e:
                print(f"Enhanced voice analysis failed: {e}")
        
        # Video analysis if provided
        if video_file and include_video_emotion:
            try:
                video_content = await video_file.read()
                video_path = await storage.save_video_file(video_content, video_file.filename or "video.mp4")
                
                video_analysis = await video_emotion_analyzer.analyze_video_file(video_path)
                analysis_results['video_emotion_analysis'] = video_analysis
            except Exception as e:
                print(f"Video analysis failed: {e}")
        
        result = {
            'multimodal_analysis': analysis_results,
            'analysis_timestamp': datetime.now().isoformat(),
            'components_analyzed': list(analysis_results.keys())
        }
        
        return make_json_safe(result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multimodal analysis failed: {str(e)}"
        )
    
import ffmpeg
import os

def convert_webm_to_wav(input_path, output_path):
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception as e:
        print(f'ffmpeg conversion error: {e}')
        return False

def detect_audio_in_video(video_path: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if video contains audio and extract audio if present.
    Returns (has_audio, audio_file_path)
    """
    try:
        # Use ffprobe to check if video has audio stream
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', 
            '-show_streams', video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFprobe failed with return code {result.returncode}")
            print(f"FFprobe stderr: {result.stderr}")
            print(f"FFprobe stdout: {result.stdout}")
            return False, None
        
        # Parse the JSON output
        streams_info = json.loads(result.stdout)
        audio_streams = [stream for stream in streams_info.get('streams', []) if stream.get('codec_type') == 'audio']
        has_audio = len(audio_streams) > 0
        
        print(f"📊 Video analysis: Found {len(audio_streams)} audio stream(s) in {os.path.basename(video_path)}")
        if audio_streams:
            for i, stream in enumerate(audio_streams):
                codec = stream.get('codec_name', 'unknown')
                print(f"   Audio stream {i}: codec={codec}")
        
        if not has_audio:
            print(f"❌ No audio streams found in video")
            return False, None
        
        # Extract audio from video
        audio_path = video_path.replace('.mp4', '_audio.wav').replace('.webm', '_audio.wav')
        print(f"🎵 Extracting audio to: {os.path.basename(audio_path)}")
        
        extract_cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '1', audio_path, '-y'
        ]
        
        extract_result = subprocess.run(extract_cmd, capture_output=True)
        if extract_result.returncode != 0:
            print(f"❌ Audio extraction failed with return code {extract_result.returncode}")
            print(f"FFmpeg stderr: {extract_result.stderr.decode('utf-8') if extract_result.stderr else 'No error message'}")
            return False, None
        else:
            print(f"✅ Audio extraction completed successfully")
        
        # Check if extracted audio has meaningful content (not just silence)
        try:
            audio = AudioSegment.from_wav(audio_path)
            audio_dBFS = audio.dBFS
            audio_duration = len(audio) / 1000.0  # Duration in seconds
            
            print(f"🔊 Audio analysis: dBFS={audio_dBFS:.2f}, duration={audio_duration:.2f}s")
            
            # More lenient threshold for meaningful audio
            # Normal speech is typically between -20 to -60 dBFS
            # Background noise is usually below -70 dBFS
            if audio_dBFS > -70 and audio_duration > 0.5:  # Much more lenient threshold
                print(f"✅ Audio detected: {audio_dBFS:.2f} dBFS > -70 dBFS threshold")
                return True, audio_path
            else:
                print(f"❌ Audio rejected: {audio_dBFS:.2f} dBFS <= -70 dBFS threshold or duration too short")
                # Clean up the extracted file if it's just silence
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return False, None
        except Exception as e:
            print(f"Audio analysis failed: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return False, None
            
    except Exception as e:
        print(f"Audio detection failed: {e}")
        return False, None

async def transcribe_audio_from_video(video_path: str) -> str:
    """
    Extract audio from video and transcribe it
    """
    try:
        # Extract audio from video
        audio_path = video_path.replace('.mp4', '_temp_audio.wav').replace('.webm', '_temp_audio.wav')
        extract_cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', audio_path, '-y'
        ]
        
        result = subprocess.run(extract_cmd, capture_output=True)
        if result.returncode != 0:
            return f"Audio extraction failed: {result.stderr}"
        
        # Transcribe the extracted audio
        transcript = await transcribe_audio(audio_path)
        
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return transcript
        
    except Exception as e:
        return f"Video transcription failed: {str(e)}"
    

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Pitch Recording API...")
    print("=" * 50)
    print(f"🌐 Server: http://localhost:8001")
    print(f"📋 API Docs: http://localhost:8001/docs")
    print(f"🔌 WebSocket: ws://localhost:8001/ws/realtime/{{session_id}}")
    print(f"🔓 Authentication: DISABLED")
    print(f"📊 Health Check: http://localhost:8001/health")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        timeout_keep_alive=120,  # 2 minutes keep-alive
        timeout_graceful_shutdown=30,  # 30 seconds graceful shutdown
        access_log=True
    )
