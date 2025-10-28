import librosa
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from pathlib import Path
import json

# Try to import advanced emotion analysis libraries
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedVoiceEmotionAnalyzer:
    """
    Enhanced voice emotion analysis with tone, pitch, and prosodic features
    Complementary to the existing ML voice analyzer
    """
    
    def __init__(self):
        """Initialize enhanced voice emotion analyzer"""
        self.models_loaded = False
        self.emotion_mapping = {
            'happy': {'pitch_range': (150, 300), 'energy_range': (0.6, 1.0), 'tempo_range': (120, 180)},
            'confident': {'pitch_range': (100, 200), 'energy_range': (0.5, 0.9), 'tempo_range': (100, 160)},
            'nervous': {'pitch_range': (200, 400), 'energy_range': (0.3, 0.7), 'tempo_range': (150, 220)},
            'sad': {'pitch_range': (80, 150), 'energy_range': (0.1, 0.4), 'tempo_range': (60, 100)},
            'angry': {'pitch_range': (150, 350), 'energy_range': (0.7, 1.0), 'tempo_range': (130, 200)},
            'calm': {'pitch_range': (90, 180), 'energy_range': (0.3, 0.6), 'tempo_range': (80, 120)},
            'excited': {'pitch_range': (180, 350), 'energy_range': (0.8, 1.0), 'tempo_range': (140, 200)}
        }
        
        # Business context weights for different emotions
        self.business_emotion_scores = {
            'confident': 1.0,
            'calm': 0.9,
            'happy': 0.8,
            'excited': 0.6,
            'nervous': 0.3,
            'sad': 0.2,
            'angry': 0.1
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize voice emotion analysis models"""
        try:
            # Set up librosa for audio processing
            logger.info("Enhanced voice emotion analyzer initialized")
            self.models_loaded = True
            
        except Exception as e:
            logger.error(f"Error initializing enhanced voice models: {e}")
            self.models_loaded = False
    
    async def analyze_voice_emotions(self, audio_file_path: str) -> Dict[str, Any]:
        """Comprehensive voice emotion analysis"""
        try:
            if not self.models_loaded:
                return {'error': 'Enhanced voice emotion models not available'}
            
            # Handle different audio formats with pydub
            try:
                from pydub import AudioSegment
                import tempfile
                import os
                
                # Load audio with pydub (supports more formats)
                audio = AudioSegment.from_file(audio_file_path)
                
                # Convert to a format librosa can handle if needed
                if not audio_file_path.endswith('.wav'):
                    # Create temporary wav file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                        audio.export(temp_wav.name, format='wav')
                        wav_path = temp_wav.name
                else:
                    wav_path = audio_file_path
                    
                # Load audio file
                y, sr = librosa.load(wav_path, sr=None)
                
                # Clean up temporary file if created
                if wav_path != audio_file_path:
                    os.unlink(wav_path)
                    
            except ImportError:
                # Fallback to librosa only
                y, sr = librosa.load(audio_file_path, sr=None)
            
            # Extract comprehensive features
            features = await self._extract_voice_features(y, sr)
            
            # Analyze emotions from features
            emotion_analysis = self._analyze_emotions_from_features(features)
            
            # Prosodic analysis
            prosodic_analysis = self._analyze_prosodic_features(y, sr)
            
            # Voice quality analysis
            voice_quality = self._analyze_voice_quality(y, sr)
            
            # Business appropriateness
            business_analysis = self._analyze_business_appropriateness(emotion_analysis, prosodic_analysis)
            
            return {
                'voice_emotion_analysis': emotion_analysis,
                'prosodic_features': prosodic_analysis,
                'voice_quality': voice_quality,
                'business_analysis': business_analysis,
                'audio_duration': len(y) / sr,
                'sample_rate': sr
            }
            
        except Exception as e:
            logger.error(f"Voice emotion analysis failed: {e}")
            return {'error': f'Voice emotion analysis failed: {str(e)}'}
    
    async def _extract_voice_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive voice features"""
        try:
            features = {}
            
            # Fundamental frequency (pitch) analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            
            # Remove NaN values and calculate pitch statistics
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 0:
                features['pitch'] = {
                    'mean': float(np.mean(f0_clean)),
                    'std': float(np.std(f0_clean)),
                    'min': float(np.min(f0_clean)),
                    'max': float(np.max(f0_clean)),
                    'range': float(np.max(f0_clean) - np.min(f0_clean)),
                    'variation_coefficient': float(np.std(f0_clean) / np.mean(f0_clean)) if np.mean(f0_clean) > 0 else 0
                }
            else:
                features['pitch'] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'range': 0, 'variation_coefficient': 0
                }
            
            # Energy and intensity features
            rms = librosa.feature.rms(y=y)[0]
            features['energy'] = {
                'mean': float(np.mean(rms)),
                'std': float(np.std(rms)),
                'max': float(np.max(rms)),
                'dynamic_range': float(np.max(rms) - np.min(rms))
            }
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            features['spectral'] = {
                'centroid_mean': float(np.mean(spectral_centroids)),
                'centroid_std': float(np.std(spectral_centroids)),
                'rolloff_mean': float(np.mean(spectral_rolloff)),
                'zcr_mean': float(np.mean(zero_crossing_rate)),
                'zcr_std': float(np.std(zero_crossing_rate))
            }
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['rhythm'] = {
                'tempo': float(tempo),
                'beat_count': len(beats),
                'rhythm_regularity': self._calculate_rhythm_regularity(beats, sr)
            }
            
            # MFCC features (voice timbre)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc'] = {
                f'mfcc_{i+1}_mean': float(np.mean(mfccs[i])) for i in range(min(13, mfccs.shape[0]))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {'error': f'Feature extraction failed: {str(e)}'}
    
    def _calculate_rhythm_regularity(self, beats: np.ndarray, sr: int) -> float:
        """Calculate rhythm regularity from beat timestamps"""
        try:
            if len(beats) < 3:
                return 0.0
            
            # Calculate inter-beat intervals
            beat_times = beats / sr
            intervals = np.diff(beat_times)
            
            # Regularity is inverse of interval variation
            if len(intervals) > 1 and np.mean(intervals) > 0:
                cv = np.std(intervals) / np.mean(intervals)
                return max(0.0, 1.0 - cv)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Rhythm regularity calculation failed: {e}")
            return 0.5
    
    def _analyze_emotions_from_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotions based on extracted voice features"""
        try:
            if 'error' in features:
                return features
            
            emotion_scores = {}
            
            # Extract key feature values
            pitch_mean = features.get('pitch', {}).get('mean', 0)
            pitch_variation = features.get('pitch', {}).get('variation_coefficient', 0)
            energy_mean = features.get('energy', {}).get('mean', 0)
            tempo = features.get('rhythm', {}).get('tempo', 0)
            
            # Score each emotion based on feature ranges
            for emotion, ranges in self.emotion_mapping.items():
                score = 0.0
                
                # Pitch scoring
                pitch_range = ranges['pitch_range']
                if pitch_range[0] <= pitch_mean <= pitch_range[1]:
                    score += 0.4
                else:
                    # Penalize deviation from ideal range
                    deviation = min(abs(pitch_mean - pitch_range[0]), abs(pitch_mean - pitch_range[1]))
                    score += max(0, 0.4 - (deviation / 100) * 0.1)
                
                # Energy scoring
                energy_range = ranges['energy_range']
                if energy_range[0] <= energy_mean <= energy_range[1]:
                    score += 0.3
                
                # Tempo scoring
                tempo_range = ranges['tempo_range']
                if tempo_range[0] <= tempo <= tempo_range[1]:
                    score += 0.3
                
                emotion_scores[emotion] = max(0.0, min(1.0, score))
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v / total_score for k, v in emotion_scores.items()}
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else 'neutral'
            
            return {
                'emotion_scores': emotion_scores,
                'dominant_emotion': dominant_emotion,
                'confidence': emotion_scores.get(dominant_emotion, 0.0),
                'pitch_characteristics': self._classify_pitch_characteristics(features.get('pitch', {})),
                'energy_characteristics': self._classify_energy_characteristics(features.get('energy', {})),
                'vocal_stability': 1.0 - min(1.0, pitch_variation) if pitch_variation else 0.5
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis from features failed: {e}")
            return {'error': f'Emotion analysis failed: {str(e)}'}
    
    def _classify_pitch_characteristics(self, pitch_features: Dict[str, float]) -> str:
        """Classify pitch characteristics for business context"""
        try:
            mean_pitch = pitch_features.get('mean', 0)
            pitch_variation = pitch_features.get('variation_coefficient', 0)
            
            if mean_pitch < 100:
                pitch_level = "low"
            elif mean_pitch > 200:
                pitch_level = "high"
            else:
                pitch_level = "moderate"
            
            if pitch_variation < 0.1:
                variation_level = "monotone"
            elif pitch_variation > 0.3:
                variation_level = "highly_variable"
            else:
                variation_level = "appropriately_varied"
            
            return f"{pitch_level}_pitch_with_{variation_level}_variation"
            
        except Exception as e:
            return "unknown_pitch_pattern"
    
    def _classify_energy_characteristics(self, energy_features: Dict[str, float]) -> str:
        """Classify energy characteristics for business context"""
        try:
            mean_energy = energy_features.get('mean', 0)
            dynamic_range = energy_features.get('dynamic_range', 0)
            
            if mean_energy < 0.3:
                energy_level = "low"
            elif mean_energy > 0.7:
                energy_level = "high"
            else:
                energy_level = "moderate"
            
            if dynamic_range < 0.2:
                dynamics = "flat"
            elif dynamic_range > 0.6:
                dynamics = "dynamic"
            else:
                dynamics = "balanced"
            
            return f"{energy_level}_energy_with_{dynamics}_dynamics"
            
        except Exception as e:
            return "unknown_energy_pattern"
    
    def _analyze_prosodic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze prosodic features (rhythm, stress, intonation)"""
        try:
            # Speaking rate analysis
            # Estimate syllables using energy peaks
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            
            # Find peaks in energy (potential syllable boundaries)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rms, height=np.mean(rms) * 0.5, distance=int(sr * 0.1 / 512))
            
            duration = len(y) / sr
            estimated_syllables = len(peaks)
            syllables_per_second = estimated_syllables / duration if duration > 0 else 0
            
            # Pause analysis
            # Detect silent regions
            silence_threshold = np.mean(rms) * 0.1
            silent_frames = rms < silence_threshold
            
            # Find pause durations
            pause_starts = []
            pause_ends = []
            in_pause = False
            
            for i, is_silent in enumerate(silent_frames):
                if is_silent and not in_pause:
                    pause_starts.append(i)
                    in_pause = True
                elif not is_silent and in_pause:
                    pause_ends.append(i)
                    in_pause = False
            
            # Handle case where audio ends in pause
            if in_pause and pause_starts:
                pause_ends.append(len(silent_frames) - 1)
            
            # Calculate pause statistics
            pause_durations = []
            for start, end in zip(pause_starts, pause_ends):
                duration_seconds = (end - start) * 512 / sr
                if duration_seconds > 0.1:  # Only count pauses longer than 100ms
                    pause_durations.append(duration_seconds)
            
            return {
                'speaking_rate': {
                    'syllables_per_second': round(syllables_per_second, 2),
                    'estimated_syllables': estimated_syllables,
                    'assessment': self._assess_speaking_rate(syllables_per_second)
                },
                'pause_analysis': {
                    'total_pauses': len(pause_durations),
                    'average_pause_duration': round(np.mean(pause_durations), 2) if pause_durations else 0,
                    'total_pause_time': round(sum(pause_durations), 2),
                    'pause_frequency': round(len(pause_durations) / duration, 2) if duration > 0 else 0,
                    'assessment': self._assess_pause_patterns(pause_durations, duration)
                },
                'fluency_score': self._calculate_fluency_score(syllables_per_second, pause_durations, duration)
            }
            
        except Exception as e:
            logger.error(f"Prosodic analysis failed: {e}")
            return {'error': f'Prosodic analysis failed: {str(e)}'}
    
    def _assess_speaking_rate(self, syllables_per_second: float) -> str:
        """Assess speaking rate for business presentations"""
        if syllables_per_second < 2.5:
            return "Too slow - may lose audience attention"
        elif syllables_per_second > 5.5:
            return "Too fast - may be difficult to follow"
        elif 3.0 <= syllables_per_second <= 4.5:
            return "Optimal pace for business presentations"
        else:
            return "Acceptable pace with room for improvement"
    
    def _assess_pause_patterns(self, pause_durations: List[float], total_duration: float) -> str:
        """Assess pause patterns for effective communication"""
        if not pause_durations:
            return "No strategic pauses detected - consider adding pauses for emphasis"
        
        avg_pause = np.mean(pause_durations)
        pause_frequency = len(pause_durations) / total_duration if total_duration > 0 else 0
        
        if pause_frequency < 0.1:
            return "Too few pauses - add strategic pauses for clarity"
        elif pause_frequency > 0.5:
            return "Too many pauses - work on fluency"
        elif 0.5 <= avg_pause <= 1.5:
            return "Good use of strategic pauses"
        else:
            return "Pause timing could be improved"
    
    def _calculate_fluency_score(self, syllables_per_second: float, pause_durations: List[float], duration: float) -> float:
        """Calculate overall fluency score"""
        try:
            score = 0.5  # Base score
            
            # Rate component (30% weight)
            if 3.0 <= syllables_per_second <= 4.5:
                score += 0.3
            elif 2.5 <= syllables_per_second <= 5.5:
                score += 0.2
            else:
                score += 0.1
            
            # Pause component (20% weight)
            if pause_durations:
                pause_frequency = len(pause_durations) / duration if duration > 0 else 0
                if 0.1 <= pause_frequency <= 0.3:
                    score += 0.2
                elif 0.05 <= pause_frequency <= 0.5:
                    score += 0.1
            
            return round(min(1.0, score), 3)
            
        except Exception as e:
            logger.error(f"Fluency score calculation failed: {e}")
            return 0.5
    
    def _analyze_voice_quality(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze voice quality characteristics"""
        try:
            # Jitter (pitch perturbation)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0_clean = f0[~np.isnan(f0)]
            
            if len(f0_clean) > 1:
                jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0
            else:
                jitter = 0
            
            # Shimmer (amplitude perturbation)
            rms = librosa.feature.rms(y=y)[0]
            if len(rms) > 1:
                shimmer = np.std(np.diff(rms)) / np.mean(rms) if np.mean(rms) > 0 else 0
            else:
                shimmer = 0
            
            # Harmonic-to-noise ratio estimation
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Simple HNR estimation based on spectral characteristics
            harmonicity = np.mean(spectral_centroids) / np.mean(spectral_rolloff) if np.mean(spectral_rolloff) > 0 else 0
            
            return {
                'jitter': round(jitter, 4),
                'shimmer': round(shimmer, 4),
                'harmonicity': round(harmonicity, 4),
                'voice_quality_assessment': self._assess_voice_quality(jitter, shimmer, harmonicity),
                'professional_rating': self._rate_professional_voice_quality(jitter, shimmer)
            }
            
        except Exception as e:
            logger.error(f"Voice quality analysis failed: {e}")
            return {'error': f'Voice quality analysis failed: {str(e)}'}
    
    def _assess_voice_quality(self, jitter: float, shimmer: float, harmonicity: float) -> str:
        """Assess overall voice quality"""
        issues = []
        
        if jitter > 0.02:
            issues.append("pitch instability")
        if shimmer > 0.1:
            issues.append("amplitude instability")
        if harmonicity < 0.5:
            issues.append("reduced voice clarity")
        
        if not issues:
            return "Excellent voice quality with stable pitch and amplitude"
        elif len(issues) == 1:
            return f"Good voice quality with minor {issues[0]}"
        else:
            return f"Voice quality needs improvement: {', '.join(issues)}"
    
    def _rate_professional_voice_quality(self, jitter: float, shimmer: float) -> str:
        """Rate voice quality for professional presentations"""
        if jitter < 0.01 and shimmer < 0.05:
            return "Professional quality - excellent for investor presentations"
        elif jitter < 0.02 and shimmer < 0.1:
            return "Good quality - suitable for business presentations"
        else:
            return "Consider voice training for improved professional presentation"
    
    def _analyze_business_appropriateness(self, emotion_analysis: Dict, prosodic_analysis: Dict) -> Dict[str, Any]:
        """Analyze business appropriateness of voice characteristics"""
        try:
            business_score = 0.5  # Base score
            recommendations = []
            
            # Emotion appropriateness
            if 'error' not in emotion_analysis:
                dominant_emotion = emotion_analysis.get('dominant_emotion', 'neutral')
                emotion_score = self.business_emotion_scores.get(dominant_emotion, 0.5)
                business_score += (emotion_score - 0.5) * 0.4  # 40% weight
                
                if emotion_score < 0.5:
                    recommendations.append(f"Work on projecting more confidence - current tone suggests {dominant_emotion}")
                elif emotion_score > 0.8:
                    recommendations.append("Excellent emotional tone for business presentations")
            
            # Speaking rate appropriateness
            if 'error' not in prosodic_analysis:
                rate_data = prosodic_analysis.get('speaking_rate', {})
                syllables_per_second = rate_data.get('syllables_per_second', 0)
                
                if 3.0 <= syllables_per_second <= 4.5:
                    business_score += 0.2
                    recommendations.append("Optimal speaking pace for investor presentations")
                elif syllables_per_second < 2.5:
                    business_score -= 0.1
                    recommendations.append("Increase speaking pace to maintain audience engagement")
                elif syllables_per_second > 5.5:
                    business_score -= 0.1
                    recommendations.append("Slow down speaking pace for better comprehension")
                
                # Fluency score
                fluency = prosodic_analysis.get('fluency_score', 0.5)
                business_score += (fluency - 0.5) * 0.3  # 30% weight
            
            # Normalize score
            business_score = max(0.0, min(1.0, business_score))
            
            # Overall assessment
            if business_score >= 0.8:
                overall_assessment = "Excellent - professional presentation quality"
            elif business_score >= 0.6:
                overall_assessment = "Good - suitable for business presentations"
            elif business_score >= 0.4:
                overall_assessment = "Acceptable - some improvements needed"
            else:
                overall_assessment = "Needs significant improvement for professional settings"
            
            return {
                'business_score': round(business_score, 3),
                'overall_assessment': overall_assessment,
                'recommendations': recommendations,
                'strengths': self._identify_voice_strengths(emotion_analysis, prosodic_analysis),
                'improvement_areas': self._identify_improvement_areas(emotion_analysis, prosodic_analysis)
            }
            
        except Exception as e:
            logger.error(f"Business analysis failed: {e}")
            return {'error': f'Business analysis failed: {str(e)}'}
    
    def _identify_voice_strengths(self, emotion_analysis: Dict, prosodic_analysis: Dict) -> List[str]:
        """Identify voice strengths for positive reinforcement"""
        strengths = []
        
        try:
            if 'error' not in emotion_analysis:
                confidence = emotion_analysis.get('confidence', 0)
                if confidence > 0.7:
                    strengths.append("Strong emotional expression")
                
                vocal_stability = emotion_analysis.get('vocal_stability', 0)
                if vocal_stability > 0.8:
                    strengths.append("Excellent vocal stability")
            
            if 'error' not in prosodic_analysis:
                fluency_score = prosodic_analysis.get('fluency_score', 0)
                if fluency_score > 0.7:
                    strengths.append("Good speech fluency")
                
                pause_data = prosodic_analysis.get('pause_analysis', {})
                if "Good use" in pause_data.get('assessment', ''):
                    strengths.append("Effective use of pauses")
            
            if not strengths:
                strengths.append("Baseline voice characteristics suitable for improvement")
                
        except Exception as e:
            logger.error(f"Strength identification failed: {e}")
        
        return strengths
    
    def _identify_improvement_areas(self, emotion_analysis: Dict, prosodic_analysis: Dict) -> List[str]:
        """Identify specific areas for voice improvement"""
        improvements = []
        
        try:
            if 'error' not in emotion_analysis:
                dominant_emotion = emotion_analysis.get('dominant_emotion', 'neutral')
                if dominant_emotion in ['nervous', 'sad']:
                    improvements.append("Voice emotion projection")
                
                vocal_stability = emotion_analysis.get('vocal_stability', 0)
                if vocal_stability < 0.6:
                    improvements.append("Vocal stability and consistency")
            
            if 'error' not in prosodic_analysis:
                rate_data = prosodic_analysis.get('speaking_rate', {})
                if "Too" in rate_data.get('assessment', ''):
                    improvements.append("Speaking rate adjustment")
                
                fluency_score = prosodic_analysis.get('fluency_score', 0)
                if fluency_score < 0.6:
                    improvements.append("Overall speech fluency")
                
        except Exception as e:
            logger.error(f"Improvement area identification failed: {e}")
        
        return improvements if improvements else ["Continue practicing for enhanced presentation skills"]

# Create global enhanced voice emotion analyzer instance
enhanced_voice_analyzer = EnhancedVoiceEmotionAnalyzer()
