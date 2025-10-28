import asyncio
import whisper
from pathlib import Path
import os
from dotenv import load_dotenv
import torch

load_dotenv()

# Initialize Whisper model
# Available models: tiny, base, small, medium, large
# tiny: fastest, least accurate (~39 MB)
# base: good balance (~74 MB) 
# small: better accuracy (~244 MB)
# medium: high accuracy (~769 MB)
# large: best accuracy (~1550 MB)

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
print(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")

try:
    # Load the model once at startup
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    print(f"✅ Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully")
    
    # Check if CUDA is available for faster processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
except Exception as e:
    print(f"❌ Error loading Whisper model: {e}")
    whisper_model = None

async def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper (local model)
    """
    if not whisper_model:
        return await mock_transcribe_audio(audio_file_path)
    
    try:
        # Check if file exists
        if not Path(audio_file_path).exists():
            return "Error: Audio file not found"
        
        print(f"Transcribing audio file: {audio_file_path}")
        
        # Run Whisper transcription in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            whisper_model.transcribe, 
            audio_file_path
        )
        
        # Extract the transcribed text
        transcript = result["text"].strip()
        
        if transcript:
            print(f"✅ Transcription completed: {len(transcript)} characters")
            return transcript
        else:
            return "No speech detected in the audio recording."
            
    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

async def transcribe_audio_with_options(audio_file_path: str, language: str = None) -> dict:
    """
    Advanced transcription with additional options
    Returns detailed result including confidence and timing
    """
    if not whisper_model:
        return {
            "text": await mock_transcribe_audio(audio_file_path),
            "language": "unknown",
            "confidence": 0.0
        }
    
    try:
        if not Path(audio_file_path).exists():
            return {
                "text": "Error: Audio file not found",
                "language": "unknown", 
                "confidence": 0.0
            }
        
        print(f"Advanced transcribing: {audio_file_path}")
        
        # Transcribe with language detection and additional options
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: whisper_model.transcribe(
                audio_file_path,
                language=language,
                word_timestamps=True,
                fp16=False  # Use fp32 for better compatibility
            )
        )
        
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "confidence": calculate_average_confidence(result.get("segments", [])),
            "segments": result.get("segments", [])
        }
        
    except Exception as e:
        error_msg = f"Advanced transcription error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "text": error_msg,
            "language": "unknown",
            "confidence": 0.0
        }

def calculate_average_confidence(segments):
    """Calculate average confidence from segments"""
    if not segments:
        return 0.0
    
    total_confidence = 0.0
    total_words = 0
    
    for segment in segments:
        if "words" in segment:
            for word in segment["words"]:
                if "probability" in word:
                    total_confidence += word["probability"]
                    total_words += 1
    
    return total_confidence / total_words if total_words > 0 else 0.0

# Fallback mock for development/testing
async def mock_transcribe_audio(audio_file_path: str) -> str:
    """
    Mock transcription for development purposes
    """
    await asyncio.sleep(1)  # Simulate processing time
    return "This is a mock transcription. Install Whisper with 'pip install openai-whisper' to enable local speech-to-text transcription."
