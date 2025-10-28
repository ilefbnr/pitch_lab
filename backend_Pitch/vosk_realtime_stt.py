import json
import logging
import numpy as np
import os
from typing import Optional, Dict, Any
import vosk
import wave
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class VoskRealtimeSTT:
    """
    Real-time speech-to-text using Vosk
    Optimized for streaming audio chunks
    """
    
    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 16000):
        """Initialize Vosk real-time STT"""
        self.sample_rate = sample_rate
        self.model = None
        self.recognizer = None
        
        # Set default model path
        if model_path is None:
            model_path = "models/vosk-model-small-en-us-0.15"
        
        self.model_path = Path(model_path)
        
        # Initialize Vosk
        self._initialize_vosk()
    
    def _initialize_vosk(self):
        """Initialize Vosk model and recognizer"""
        try:
            # Check if model exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Vosk model not found at {self.model_path}")
            
            # Load Vosk model
            self.model = vosk.Model(str(self.model_path))
            
            # Create recognizer
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            
            # Enable partial results for real-time feedback
            self.recognizer.SetWords(True)
            
            logger.info(f"✅ Vosk initialized with model: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            raise
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Process audio chunk and return transcription results
        
        Args:
            audio_data: numpy array of audio samples (float32, mono)
            
        Returns:
            Dict with transcription results
        """
        try:
            if self.recognizer is None:
                print("❌ Vosk recognizer not initialized")
                return {
                    'error': 'Vosk not initialized',
                    'transcript': '',
                    'partial': '',
                    'final': False
                }
            
            # Convert float32 to int16
            if audio_data.dtype == np.float32:
                # Convert from [-1, 1] to [-32768, 32767]
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data.astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_int16.tobytes()
            
            # Process with Vosk
            accept_result = self.recognizer.AcceptWaveform(audio_bytes)
            
            if accept_result:
                # Final result
                result = json.loads(self.recognizer.Result())
                transcript = result.get('text', '').strip()
                if transcript:  # Only print if we have actual text
                    print(f"✅ FINAL: '{transcript}'")
                return {
                    'transcript': transcript,
                    'partial': '',
                    'final': True,
                    'confidence': result.get('confidence', 0.0),
                    'words': result.get('words', [])
                }
            else:
                # Partial result
                partial_result = json.loads(self.recognizer.PartialResult())
                partial_text = partial_result.get('partial', '').strip()
                if partial_text:  # Only print if we have actual partial text
                    print(f"📝 PARTIAL: '{partial_text}'")
                return {
                    'transcript': '',
                    'partial': partial_text,
                    'final': False,
                    'confidence': 0.0,
                    'words': []
                }
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                'error': f'Processing failed: {str(e)}',
                'transcript': '',
                'partial': '',
                'final': False
            }
    
    def finalize_recognition(self) -> Dict[str, Any]:
        """
        Get final recognition result
        Used when ending a session
        """
        try:
            if self.recognizer is None:
                return {'transcript': '', 'final': True}
            
            # Get final result
            final_result = json.loads(self.recognizer.FinalResult())
            return {
                'transcript': final_result.get('text', ''),
                'final': True,
                'confidence': final_result.get('confidence', 0.0),
                'words': final_result.get('words', [])
            }
            
        except Exception as e:
            logger.error(f"Error finalizing recognition: {e}")
            return {
                'transcript': '',
                'final': True,
                'error': str(e)
            }
    
    def reset(self):
        """Reset recognizer for new session"""
        try:
            if self.model:
                self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
                self.recognizer.SetWords(True)
                logger.debug("Vosk recognizer reset")
        except Exception as e:
            logger.error(f"Error resetting Vosk: {e}")
    
    def transcribe_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe a complete audio file
        Useful for testing and fallback scenarios
        """
        try:
            # Read audio file
            with wave.open(audio_path, 'rb') as wf:
                # Check if audio format is compatible
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != self.sample_rate:
                    logger.warning(f"Audio format mismatch: channels={wf.getnchannels()}, width={wf.getsampwidth()}, rate={wf.getframerate()}")
                
                # Reset recognizer
                self.reset()
                
                # Process in chunks
                chunk_size = 4000  # bytes
                transcripts = []
                
                while True:
                    data = wf.readframes(chunk_size)
                    if len(data) == 0:
                        break
                    
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        if result.get('text'):
                            transcripts.append(result['text'])
                
                # Get final result
                final_result = json.loads(self.recognizer.FinalResult())
                if final_result.get('text'):
                    transcripts.append(final_result['text'])
                
                # Combine all transcripts
                full_transcript = ' '.join(transcripts).strip()
                
                return {
                    'transcript': full_transcript,
                    'final': True,
                    'method': 'file_transcription'
                }
                
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}")
            return {
                'transcript': '',
                'final': True,
                'error': str(e)
            }
    
    def check_audio_quality(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Check audio quality for speech detection
        """
        try:
            # Calculate basic audio metrics
            if len(audio_data) == 0:
                return {
                    'has_speech': False,
                    'energy': 0.0,
                    'max_amplitude': 0.0,
                    'rms_energy': 0.0
                }
            
            # RMS energy
            rms_energy = float(np.sqrt(np.mean(audio_data ** 2)))
            
            # Max amplitude
            max_amplitude = float(np.max(np.abs(audio_data)))
            
            # Simple energy threshold for speech detection
            energy_threshold = 0.0001  # Extremely low threshold for Vosk (increased sensitivity)
            has_speech = rms_energy > energy_threshold
            
            return {
                'has_speech': has_speech,
                'energy': rms_energy,
                'max_amplitude': max_amplitude,
                'rms_energy': rms_energy,
                'duration': len(audio_data) / self.sample_rate
            }
            
        except Exception as e:
            logger.error(f"Error checking audio quality: {e}")
            return {
                'has_speech': False,
                'energy': 0.0,
                'max_amplitude': 0.0,
                'rms_energy': 0.0,
                'error': str(e)
            }

# Global instance
vosk_stt = VoskRealtimeSTT()
