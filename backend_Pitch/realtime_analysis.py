import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import librosa
import soundfile as sf
from io import BytesIO
import tempfile
import os
from typing import Tuple

from voice_analysis import ml_voice_analyzer
from investor_ai import investor_ai
from vosk_realtime_stt import vosk_stt

logger = logging.getLogger(__name__)

class RealTimeAnalyzer:
    """
    Real-time voice analysis for streaming audio data
    Processes audio chunks and provides live feedback
    """
    
    def __init__(self):
        """Initialize real-time analyzer"""
        self.chunk_size = 1024  # Audio chunk size
        self.sample_rate = 16000  # Standard sample rate
        self.buffer = []  # Audio buffer
        self.transcript_buffer = ""  # Transcript accumulation
        self.analysis_history = []  # Store analysis results
        self.session_data = {}  # Session-specific data
        self.is_processing = False
        self.last_processing_time = 0  # Track when we last processed
        
        # Real-time metrics
        self.live_metrics = {
            'volume_level': 0.0,
            'speaking_pace': 0.0,
            'confidence_trend': [],
            'emotion_trend': [],
            'pitch_variation': 0.0,
            'speaking_time': 0.0,
            'pause_count': 0,
            'last_update': None
        }
        
        # Chunk processing settings - optimized for instant real-time response
        self.min_chunk_duration = 0.1  # Very small minimum (Vosk handles this well)
        self.max_buffer_size = 5.0  # Smaller buffer for faster response
        self.transcription_threshold = 0.3  # Process very quickly (300ms)
        self.silence_threshold = 0.0001  # Very sensitive threshold
        self.processing_interval = 0.1  # Process every 100ms for instant response
        
    async def start_session(self, session_id: str, user_id: int) -> Dict[str, Any]:
        """Start a new real-time analysis session"""
        self.session_data[session_id] = {
            'user_id': user_id,
            'start_time': datetime.now(),
            'total_chunks': 0,
            'total_audio_duration': 0.0,
            'accumulated_transcript': "",
            'analysis_results': []
        }
        
        # Reset metrics
        self.live_metrics = {
            'volume_level': 0.0,
            'speaking_pace': 0.0,
            'confidence_trend': [],
            'emotion_trend': [],
            'pitch_variation': 0.0,
            'speaking_time': 0.0,
            'pause_count': 0,
            'last_update': datetime.now().isoformat()
        }
        
        # Reset Vosk recognizer for new session
        try:
            vosk_stt.reset()
            logger.info(f"✅ Vosk recognizer reset for session: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to reset Vosk: {e}")
        
        logger.info(f"Started real-time session: {session_id}")
        return {
            'session_id': session_id,
            'status': 'started',
            'metrics': self.live_metrics,
            'stt_engine': 'vosk'
        }
    
    async def process_audio_chunk(
        self, 
        session_id: str, 
        audio_data: bytes,
        emit_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process incoming audio chunk and return real-time analysis"""
        if session_id not in self.session_data:
            print(f"⚠️ Session {session_id} not found, ignoring audio chunk")
            return {
                'session_id': session_id,
                'error': f"Session {session_id} not found",
                'chunk_processed': False
            }
        
        try:
            # Convert audio bytes to numpy array
            audio_array = self._bytes_to_audio_array(audio_data)
            
            # Add to buffer
            self.buffer.extend(audio_array)
            
            # Update session data
            session = self.session_data[session_id]
            session['total_chunks'] += 1
            
            # Calculate real-time metrics
            metrics = await self._calculate_live_metrics(audio_array)
            
            # Process buffer if sufficient data accumulated and contains speech
            analysis_result = None
            buffer_duration = len(self.buffer) / self.sample_rate
            current_time = datetime.now().timestamp()
            
            # Process immediately for real-time response - check every chunk
            should_process = False
            
            if buffer_duration >= self.transcription_threshold:
                # Check if enough time has passed since last processing
                if current_time - self.last_processing_time >= self.processing_interval:
                    should_process = True
            
            # Also process if buffer gets moderately full (faster cleanup)
            if buffer_duration > 2.0:
                should_process = True
            
            if should_process:
                # Check if buffer contains meaningful audio (not just silence)
                recent_audio = np.array(self.buffer[-int(self.sample_rate * 1):])  # Last 1 second
                audio_energy = np.mean(recent_audio ** 2) if len(recent_audio) > 0 else 0
                
                if audio_energy > self.silence_threshold:
                    self.last_processing_time = current_time
                    analysis_result = await self._process_buffer_chunk(session_id)
                    
                    # Emit real-time updates if callback provided
                    if emit_callback and analysis_result:
                        await emit_callback('analysis_update', {
                            'session_id': session_id,
                            'metrics': metrics,
                            'analysis': analysis_result,
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Always emit live metrics
            if emit_callback:
                await emit_callback('live_metrics', {
                    'session_id': session_id,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                })
            
            return {
                'session_id': session_id,
                'chunk_processed': True,
                'live_metrics': metrics,
                'analysis': analysis_result
            }
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                'session_id': session_id,
                'error': f"Processing failed: {str(e)}",
                'chunk_processed': False
            }
    
    async def _calculate_live_metrics(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Calculate real-time audio metrics"""
        try:
            # Volume level (RMS)
            volume_level = float(np.sqrt(np.mean(audio_array ** 2)))
            
            # Pitch variation using zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_array)[0]
            pitch_variation = float(np.std(zcr))
            
            # Speech detection (simple energy-based)
            is_speaking = volume_level > 0.01  # Threshold for speech detection
            
            # Update metrics
            self.live_metrics.update({
                'volume_level': round(volume_level * 100, 2),  # Convert to percentage
                'pitch_variation': round(pitch_variation, 3),
                'is_speaking': bool(is_speaking),  # Ensure boolean conversion
                'last_update': datetime.now().isoformat()
            })
            
            # Speaking time tracking
            if is_speaking:
                chunk_duration = len(audio_array) / self.sample_rate
                self.live_metrics['speaking_time'] += chunk_duration
            
            return self.live_metrics
            
        except Exception as e:
            logger.error(f"Error calculating live metrics: {e}")
            return self.live_metrics
    
    async def _process_buffer_chunk(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Process accumulated buffer for analysis using Vosk real-time STT"""
        try:
            if self.is_processing:
                logger.debug("Already processing, skipping this chunk")
                return None
            
            self.is_processing = True
            
            # Convert buffer to audio array with validation
            buffer_array = np.array(self.buffer, dtype=np.float32)
            
            if len(buffer_array) == 0:
                logger.warning("Empty buffer for processing")
                return None
            
            # For real-time Vosk, process very short chunks (100ms minimum)
            duration = len(buffer_array) / self.sample_rate
            if duration < 0.1:
                print(f"⚡ Buffer too short for transcription: {duration:.2f}s (need at least 0.1s)")
                return None
            
            # Check audio quality before processing - but be more lenient for real-time
            quality_check = vosk_stt.check_audio_quality(buffer_array)
            
            # For real-time, try processing even with low energy - let Vosk decide
            if quality_check.get('energy', 0) > 0.000001:  # Very low threshold for real-time
                print(f"🎤 Processing audio: energy={quality_check.get('energy', 0):.6f}, duration={quality_check.get('duration', 0):.2f}s")
            else:
                print(f"🔇 Audio too quiet: {quality_check.get('energy', 0):.6f}")
                return None
            
            try:
                session = self.session_data[session_id]
                
                # Use Vosk for real-time transcription
                logger.info(f"🎤 Starting Vosk transcription for chunk ({duration:.2f}s)...")
                vosk_result = vosk_stt.process_audio_chunk(buffer_array)
                
                # Extract transcript from Vosk result
                chunk_transcript = ""
                is_final = vosk_result.get('final', False)
                
                if is_final and vosk_result.get('transcript'):
                    chunk_transcript = vosk_result['transcript'].strip()
                    print(f"✅ Final transcript: '{chunk_transcript}'")
                elif vosk_result.get('partial'):
                    # For partial results, show immediately for real-time feedback
                    chunk_transcript = vosk_result['partial'].strip()
                    print(f"📝 Partial transcript: '{chunk_transcript}'")
                else:
                    print(f"❌ No transcript in Vosk result: {vosk_result}")
                
                print(f"🎤 Vosk result summary: transcript='{chunk_transcript}', final={is_final}, length={len(chunk_transcript)}")
                
                # Create analysis result
                analysis_result = {
                    'chunk_id': session['total_chunks'],
                    'transcript': chunk_transcript,
                    'timestamp': datetime.now().isoformat(),
                    'duration': duration,
                    'audio_energy': float(quality_check.get('energy', 0)),
                    'has_speech': quality_check.get('has_speech', False),
                    'is_final': is_final,
                    'vosk_confidence': vosk_result.get('confidence', 0.0)
                }
                
                # Only accumulate final transcripts to avoid duplicates
                if is_final and chunk_transcript:
                    # Accumulate transcript
                    session['accumulated_transcript'] += " " + chunk_transcript
                    
                    # Perform ML analysis on transcript chunk for final results
                    try:
                        # Create temporary file for ML analysis
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                            temp_path = temp_file.name
                        
                        sf.write(temp_path, buffer_array, self.sample_rate)
                        
                        analysis = await ml_voice_analyzer.analyze_transcript(
                            transcript=chunk_transcript,
                            audio_duration=duration,
                            audio_file_path=temp_path
                        )
                        
                        analysis_result['analysis'] = analysis
                        
                        # Update trends if analysis successful
                        if 'error' not in analysis:
                            self._update_analysis_trends(analysis)
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
                    except Exception as e:
                        logger.warning(f"ML analysis failed: {e}")
                        analysis_result['analysis'] = {
                            'error': f'Analysis failed: {str(e)}',
                            'confidence_score': 0
                        }
                
                elif chunk_transcript:  # Partial result - show immediately
                    # For partial results, provide immediate feedback for real-time response
                    analysis_result['analysis'] = {
                        'message': f'LIVE: "{chunk_transcript}"',
                        'transcript': chunk_transcript,
                        'audio_quality': 'live',
                        'confidence_score': vosk_result.get('confidence', 0) * 100,
                        'is_partial': True,
                        'real_time': True,
                        'status': 'partial'
                    }
                    print(f"🚀 PARTIAL RESULT TO FRONTEND: '{chunk_transcript}'")
                else:
                    # No transcript but audio detected
                    analysis_result['analysis'] = {
                        'message': 'Listening...',
                        'audio_quality': 'detected',
                        'confidence_score': 0
                    }
                
                # Store analysis result
                session['analysis_results'].append(analysis_result)
                
                # Clear buffer after processing
                self._manage_buffer()
                logger.debug(f"✅ Processed buffer with Vosk: {duration:.2f}s, transcript: '{chunk_transcript}' (final: {is_final})")
                
                return analysis_result
                
            except Exception as e:
                logger.error(f"Error processing buffer with Vosk: {e}")
                return None
                
            finally:
                self.is_processing = False
            
        except Exception as e:
            logger.error(f"Critical error in buffer processing: {e}")
            self.is_processing = False
            return None
    
    async def _transcribe_audio_chunk(self, audio_path: str) -> Optional[str]:
        """Transcribe audio chunk with improved error handling"""
        try:
            # Check file exists and has content
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                return None
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size < 1000:  # Less than 1KB likely empty or corrupted
                logger.warning(f"Audio file too small ({file_size} bytes): {audio_path}")
                return None
            
            # Check audio duration
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                duration = len(y) / sr
                if duration < 2.0:  # Need at least 2 seconds for Whisper to work properly
                    logger.debug(f"Audio too short for transcription ({duration:.2f}s)")
                    return None
                    
                # Check if audio has meaningful content (not just silence)
                energy = np.mean(y ** 2)
                logger.info(f"🔊 Audio energy check: {energy:.8f}")
                if energy < self.silence_threshold:  # Use the configured threshold
                    logger.debug(f"Audio too quiet for transcription (energy: {energy:.8f})")
                    return None
                    
            except Exception as e:
                logger.warning(f"Failed to analyze audio file: {e}")
                # Continue with transcription attempt anyway
            
            # Try optimized transcription first
            try:
                from optimized_speech_to_text import optimized_stt
                result = await optimized_stt.transcribe_with_quality_check(audio_path)
                
                if result.get('transcript'):
                    logger.debug(f"Transcription successful: {result['transcript'][:50]}...")
                    return result['transcript']
                elif result.get('skipped_reason'):
                    logger.debug(f"Transcription skipped: {result['skipped_reason']}")
                    return None
                elif result.get('error'):
                    logger.warning(f"Optimized transcription failed: {result['error']}")
                else:
                    logger.warning(f"Optimized transcription failed: Unknown error")
                    
            except ImportError:
                logger.debug("Optimized speech-to-text not available, using fallback")
            except Exception as e:
                logger.warning(f"Optimized transcription error: {e}")
            
            # Fallback to basic transcription
            try:
                from speech_to_text import transcribe_audio
                transcript = await transcribe_audio(audio_path)
                if transcript and not transcript.startswith("Transcription failed"):
                    logger.debug(f"Fallback transcription successful")
                    return transcript
                else:
                    logger.debug(f"Fallback transcription result: {transcript}")
                    return None
                    
            except Exception as fallback_error:
                logger.warning(f"Fallback transcription failed: {fallback_error}")
            
            # If all transcription methods fail, check if it's just silence
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                energy = np.mean(y ** 2)
                if energy < 0.001:
                    logger.debug("Audio appears to be silence - no transcription needed")
                    return None
                else:
                    logger.info("Audio contains sound but transcription failed - may be non-speech")
                    return None
            except Exception as e:
                logger.debug(f"Could not analyze audio file: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Critical error in transcription: {e}")
            return None
    
    def _update_analysis_trends(self, analysis: Dict[str, Any]):
        """Update trend data from analysis results"""
        try:
            # Confidence trend
            confidence_score = analysis.get('confidence_score', 0)
            self.live_metrics['confidence_trend'].append(confidence_score)
            
            # Keep only recent trend data (last 10 points)
            if len(self.live_metrics['confidence_trend']) > 10:
                self.live_metrics['confidence_trend'] = self.live_metrics['confidence_trend'][-10:]
            
            # Emotion trend
            emotion_analysis = analysis.get('emotion_analysis', {})
            if 'dominant_emotion' in emotion_analysis:
                emotion_entry = {
                    'emotion': emotion_analysis['dominant_emotion'],
                    'timestamp': datetime.now().isoformat()
                }
                self.live_metrics['emotion_trend'].append(emotion_entry)
                
                # Keep only recent emotions
                if len(self.live_metrics['emotion_trend']) > 10:
                    self.live_metrics['emotion_trend'] = self.live_metrics['emotion_trend'][-10:]
            
            # Speaking pace from analysis
            speaking_pace = analysis.get('speaking_pace', {})
            if 'words_per_minute' in speaking_pace:
                self.live_metrics['speaking_pace'] = speaking_pace['words_per_minute']
            
        except Exception as e:
            logger.error(f"Error updating trends: {e}")
    
    def _manage_buffer(self):
        """Manage buffer size to prevent memory issues - keep sliding window for real-time"""
        # Keep only the most recent audio for continuous processing
        max_buffer_length = int(2.0 * self.sample_rate)  # Keep last 2 seconds
        
        if len(self.buffer) > max_buffer_length:
            # Keep the most recent audio
            keep_length = int(1.0 * self.sample_rate)  # Keep last 1 second
            self.buffer = self.buffer[-keep_length:]
    
    def _bytes_to_audio_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array with better WebSocket support"""
        try:
            if len(audio_bytes) < 4:
                logger.warning(f"Audio chunk too small: {len(audio_bytes)} bytes")
                return np.array([])

            header = audio_bytes[:4]

            # Helper to resample and normalize to float32 mono at self.sample_rate
            def _to_target_rate_and_normalize(samples: np.ndarray, sr: int) -> np.ndarray:
                if samples.ndim > 1:
                    samples = np.mean(samples, axis=1)
                if sr != self.sample_rate and len(samples) > 0:
                    try:
                        samples = librosa.resample(samples.astype(np.float32), orig_sr=sr, target_sr=self.sample_rate)
                    except Exception as e:
                        logger.warning(f"Resample failed ({sr}->${self.sample_rate}): {e}")
                # Ensure float32 and clamp
                samples = samples.astype(np.float32)
                if len(samples) > 0:
                    max_abs = np.max(np.abs(samples))
                    if max_abs > 1.0:
                        samples = samples / max_abs
                return samples

            # Try container decoding first (WebM/Opus, OGG, etc.) via pydub/ffmpeg
            try:
                from pydub import AudioSegment
                decoded = None
                bio = BytesIO(audio_bytes)

                # Attempt format guesses
                for fmt in ["webm", "ogg", "opus", "wav"]:
                    try:
                        bio.seek(0)
                        decoded = AudioSegment.from_file(bio, format=fmt)
                        logger.info(f"🎧 Decoded WebSocket audio via pydub: format={fmt}")
                        break
                    except Exception:
                        continue

                if decoded is not None:
                    sample_width = decoded.sample_width  # bytes per sample
                    channels = decoded.channels
                    frame_rate = decoded.frame_rate
                    array_type_max = float(2 ** (8 * sample_width - 1))
                    raw_samples = np.array(decoded.get_array_of_samples())
                    if channels > 1:
                        raw_samples = raw_samples.reshape((-1, channels))
                        raw_samples = np.mean(raw_samples, axis=1)  # to mono
                    # Normalize to [-1, 1]
                    samples = raw_samples.astype(np.float32) / array_type_max
                    return _to_target_rate_and_normalize(samples, frame_rate)
                else:
                    logger.info("⚠️ pydub could not decode audio (no matching container). Ensure ffmpeg is installed and in PATH.")
            except Exception as e:
                logger.info(f"⚠️ pydub/ffmpeg decode failed or not available: {e}")

            # Fallbacks: try raw PCM16, then float32 (assume browser AudioContext at 48000 Hz)
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio_array) == 0:
                    return np.array([])
                return _to_target_rate_and_normalize(audio_array, 48000)
            except Exception as e:
                logger.debug(f"PCM16 conversion failed: {e}")
                try:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    if len(audio_array) == 0:
                        return np.array([])
                    return _to_target_rate_and_normalize(audio_array, 48000)
                except Exception as e2:
                    logger.error(f"Failed to convert audio bytes: {e2}")
                    return np.array([])
                    
        except Exception as e:
            logger.error(f"Error converting audio bytes: {e}")
            return np.array([])
    
    async def generate_live_investor_response(
        self, 
        session_id: str,
        emit_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Generate real-time investor response based on current session"""
        try:
            if session_id not in self.session_data:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.session_data[session_id]
            
            # Get accumulated data
            full_transcript = session['accumulated_transcript']
            recent_analysis = session['analysis_results'][-1] if session['analysis_results'] else None
            
            if not full_transcript.strip():
                return {
                    'type': 'investor_response',
                    'message': "I'm listening... please continue with your pitch.",
                    'investor_type': 'encouraging'
                }
            
            # Generate contextual investor response
            if recent_analysis:
                analysis_data = json.dumps(recent_analysis['analysis'], default=str)
            else:
                analysis_data = None
            
            response = await investor_ai.generate_investor_response(
                transcript=full_transcript,
                analysis_result=analysis_data
            )
            
            # Add real-time context
            response['session_context'] = {
                'total_speaking_time': self.live_metrics['speaking_time'],
                'confidence_trend': self.live_metrics['confidence_trend'][-3:],  # Last 3 points
                'current_pace': self.live_metrics['speaking_pace']
            }
            
            # Emit real-time investor response
            if emit_callback:
                await emit_callback('investor_response', {
                    'session_id': session_id,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating live investor response: {e}")
            return {
                'type': 'investor_response',
                'error': f"Response generation failed: {str(e)}",
                'message': "I'm having trouble processing your pitch right now. Please continue."
            }
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End real-time session and return summary"""
        try:
            if session_id not in self.session_data:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.session_data[session_id]
            end_time = datetime.now()
            
            # Calculate session summary
            total_duration = (end_time - session['start_time']).total_seconds()
            
            summary = {
                'session_id': session_id,
                'total_duration': total_duration,
                'speaking_time': self.live_metrics['speaking_time'],
                'speaking_ratio': self.live_metrics['speaking_time'] / total_duration if total_duration > 0 else 0,
                'total_chunks': session['total_chunks'],
                'final_transcript': session['accumulated_transcript'],
                'analysis_count': len(session['analysis_results']),
                'confidence_trend': self.live_metrics['confidence_trend'],
                'emotion_summary': self._summarize_emotions(),
                'final_metrics': self.live_metrics.copy()
            }
            
            # Finalize Vosk recognition to get any remaining transcript
            try:
                final_vosk_result = vosk_stt.finalize_recognition()
                if final_vosk_result.get('transcript'):
                    final_transcript = final_vosk_result['transcript'].strip()
                    if final_transcript and final_transcript not in summary['final_transcript']:
                        summary['final_transcript'] += " " + final_transcript
                        logger.info(f"📝 Added final Vosk transcript: '{final_transcript}'")
            except Exception as e:
                logger.warning(f"Failed to finalize Vosk recognition: {e}")
            
            # Clean up session data
            del self.session_data[session_id]
            
            # Reset for next session
            self.buffer = []
            self.transcript_buffer = ""
            self.is_processing = False
            self.last_processing_time = 0
            
            print(f"🔚 Session {session_id} ended and cleaned up")
            logger.info(f"Ended real-time session: {session_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return {
                'session_id': session_id,
                'error': f"Session end failed: {str(e)}"
            }
    
    def _summarize_emotions(self) -> Dict[str, Any]:
        """Summarize emotion trends from the session"""
        try:
            emotions = [entry['emotion'] for entry in self.live_metrics['emotion_trend']]
            if not emotions:
                return {'dominant_emotion': 'neutral', 'emotion_changes': 0}
            
            # Count emotion frequencies
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Find dominant emotion
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            
            # Count emotion changes
            emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotion_distribution': emotion_counts,
                'emotion_changes': emotion_changes,
                'emotional_stability': 'stable' if emotion_changes < 3 else 'variable'
            }
            
        except Exception as e:
            logger.error(f"Error summarizing emotions: {e}")
            return {'dominant_emotion': 'neutral', 'emotion_changes': 0}

# Global real-time analyzer instance
realtime_analyzer = RealTimeAnalyzer()
