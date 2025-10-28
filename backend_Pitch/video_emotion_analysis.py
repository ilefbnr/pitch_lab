import cv2
import numpy as np
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import base64
import io
from PIL import Image
import json
from datetime import datetime

# Try to import DeepFace and fallback gracefully
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully for video emotion analysis")
except Exception as e:
    DEEPFACE_AVAILABLE = False
    print(f"⚠️ DeepFace not available - using fallback emotion analysis: {e}")
    # Create a mock DeepFace for graceful degradation
    class MockDeepFace:
        @staticmethod
        def analyze(*args, **kwargs):
            return [{
                'emotion': {'neutral': 0.7, 'happy': 0.2, 'sad': 0.1},
                'dominant_emotion': 'neutral'
            }]
    DeepFace = MockDeepFace

# Try to import MediaPipe for face detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded successfully for face detection")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ MediaPipe not available - using basic face detection: {e}")
    # Create a mock MediaPipe for graceful degradation
    class MockMediaPipe:
        pass
    mp = MockMediaPipe()

logger = logging.getLogger(__name__)

class VideoEmotionAnalyzer:
    """
    Advanced video emotion recognition using facial analysis
    Supports real-time and batch processing
    """
    
    def __init__(self):
        """Initialize video emotion analyzer"""
        self.models_loaded = False
        self.face_detection = None
        self.face_mesh = None
        self.emotion_history = []
        self.frame_cache = []
        
        # Emotion mapping for business context
        self.business_emotion_weights = {
            'happy': 1.0,      # Positive for pitches
            'confident': 1.0,   # Ideal for presentations
            'neutral': 0.7,     # Acceptable baseline
            'surprise': 0.8,    # Shows engagement
            'fear': -0.5,       # Nervous appearance
            'sad': -0.8,        # Negative impression
            'angry': -1.0,      # Very negative
            'disgust': -0.9     # Unprofessional
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize face detection and analysis models"""
        try:
            if MEDIAPIPE_AVAILABLE:
                mp_face_detection = mp.solutions.face_detection
                mp_face_mesh = mp.solutions.face_mesh
                
                self.face_detection = mp_face_detection.FaceDetection(
                    model_selection=1,  # Full range model
                    min_detection_confidence=0.7
                )
                
                self.face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                
                logger.info("MediaPipe face detection initialized")
            
            if DEEPFACE_AVAILABLE:
                # Pre-load DeepFace models
                try:
                    # Initialize models by running a dummy analysis
                    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False)
                    logger.info("DeepFace emotion models initialized")
                except Exception as e:
                    logger.warning(f"DeepFace initialization warning: {e}")
            
            self.models_loaded = True
            
        except Exception as e:
            logger.error(f"Error initializing video emotion models: {e}")
            self.models_loaded = False
    
    async def analyze_video_file(self, video_path: str) -> Dict[str, Any]:
        """Analyze emotions from a complete video file"""
        try:
            if not self.models_loaded:
                return {'error': 'Video emotion models not available'}
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': f'Could not open video file: {video_path}'}
            
            emotions_over_time = []
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Analyze every 30th frame (roughly 1 frame per second for 30fps video)
            frame_skip = max(1, int(fps))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    timestamp = frame_count / fps
                    emotion_result = await self._analyze_frame_emotions(frame)
                    
                    if emotion_result and 'error' not in emotion_result:
                        emotion_result['timestamp'] = timestamp
                        emotions_over_time.append(emotion_result)
                
                frame_count += 1
                
                # Progress tracking for long videos
                if frame_count % (frame_skip * 10) == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Video analysis progress: {progress:.1f}%")
            
            cap.release()
            
            # Analyze results
            analysis = self._analyze_emotion_trends(emotions_over_time)
            analysis['total_frames_analyzed'] = len(emotions_over_time)
            analysis['video_duration'] = total_frames / fps if fps > 0 else 0
            
            return analysis
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return {'error': f'Video analysis failed: {str(e)}'}
    
    async def analyze_frame_base64(self, frame_data: str) -> Dict[str, Any]:
        """Analyze emotions from a base64 encoded frame"""
        try:
            if not self.models_loaded:
                return {'error': 'Video emotion models not available'}
            
            # Decode base64 image
            image_data = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
            image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            return await self._analyze_frame_emotions(frame)
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return {'error': f'Frame analysis failed: {str(e)}'}
    
    async def _analyze_frame_emotions(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze emotions in a single frame"""
        try:
            # Detect faces first
            faces = self._detect_faces(frame)
            
            if not faces:
                return {
                    'emotions': {},
                    'dominant_emotion': 'no_face',
                    'confidence': 0.0,
                    'face_detected': False,
                    'business_score': 0.0
                }
            
            # Use the largest face
            face_region = faces[0]
            
            # Extract emotion using DeepFace if available
            if DEEPFACE_AVAILABLE:
                try:
                    result = DeepFace.analyze(
                        face_region, 
                        actions=['emotion'], 
                        enforce_detection=False
                    )
                    
                    # Handle both list and dict returns from DeepFace
                    if isinstance(result, list):
                        result = result[0]
                    
                    emotions = result.get('emotion', {})
                    dominant_emotion = result.get('dominant_emotion', 'neutral')
                    
                    # Convert numpy values to Python native types for JSON serialization
                    emotions_serializable = {}
                    for emotion, value in emotions.items():
                        if hasattr(value, 'item'):  # numpy scalar
                            emotions_serializable[emotion] = float(value.item())
                        else:
                            emotions_serializable[emotion] = float(value)
                    
                    # Calculate business appropriateness score
                    business_score = self._calculate_business_emotion_score(emotions_serializable)
                    
                    return {
                        'emotions': emotions_serializable,
                        'dominant_emotion': dominant_emotion.lower(),
                        'confidence': float(max(emotions_serializable.values())) if emotions_serializable else 0.0,
                        'face_detected': True,
                        'business_score': float(business_score),
                        'face_region_size': [int(face_region.shape[0]), int(face_region.shape[1])]
                    }
                    
                except Exception as e:
                    logger.warning(f"DeepFace emotion analysis failed: {e}")
            
            # Fallback to basic analysis
            return self._basic_emotion_analysis(face_region)
            
        except Exception as e:
            logger.error(f"Frame emotion analysis failed: {e}")
            return {'error': f'Frame analysis failed: {str(e)}'}
    
    def _detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect faces in frame and return cropped face regions"""
        faces = []
        
        try:
            if MEDIAPIPE_AVAILABLE and self.face_detection:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(rgb_frame)
                
                if results.detections:
                    h, w, _ = frame.shape
                    
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Convert relative coordinates to absolute
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Ensure coordinates are within frame bounds
                        x = max(0, x)
                        y = max(0, y)
                        width = min(width, w - x)
                        height = min(height, h - y)
                        
                        # Extract face region
                        face_region = frame[y:y+height, x:x+width]
                        if face_region.size > 0:
                            faces.append(face_region)
            
            else:
                # Fallback to OpenCV Haar Cascades
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_coords = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in face_coords:
                    face_region = frame[y:y+h, x:x+w]
                    faces.append(face_region)
        
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
        
        return faces
    
    def _basic_emotion_analysis(self, face_region: np.ndarray) -> Dict[str, Any]:
        """Basic emotion analysis when DeepFace is not available"""
        try:
            # Simple heuristic based on face region properties
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate basic features
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Basic emotion estimation (simplified)
            emotions = {
                'neutral': 0.6,
                'happy': 0.2 if brightness > 120 else 0.1,
                'sad': 0.1 if brightness < 100 else 0.05,
                'angry': 0.05,
                'surprise': 0.1 if contrast > 30 else 0.05,
                'fear': 0.05,
                'disgust': 0.05
            }
            
            dominant_emotion = max(emotions, key=emotions.get)
            business_score = self._calculate_business_emotion_score(emotions)
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': float(emotions[dominant_emotion]),
                'face_detected': True,
                'business_score': float(business_score),
                'method': 'basic_heuristic'
            }
            
        except Exception as e:
            logger.error(f"Basic emotion analysis failed: {e}")
            return {'error': f'Basic analysis failed: {str(e)}'}
    
    def _calculate_business_emotion_score(self, emotions: Dict[str, float]) -> float:
        """Calculate business appropriateness score from emotions"""
        try:
            score = 0.0
            total_weight = 0.0
            
            for emotion, value in emotions.items():
                emotion_lower = emotion.lower()
                weight = self.business_emotion_weights.get(emotion_lower, 0.0)
                score += value * weight
                total_weight += abs(weight)
            
            # Normalize to 0-1 scale
            if total_weight > 0:
                normalized_score = (score + total_weight) / (2 * total_weight)
                return max(0.0, min(1.0, normalized_score))
            
            return 0.5  # Neutral score
            
        except Exception as e:
            logger.error(f"Business score calculation failed: {e}")
            return 0.5
    
    def _analyze_emotion_trends(self, emotions_over_time: List[Dict]) -> Dict[str, Any]:
        """Analyze emotion trends throughout the video"""
        try:
            if not emotions_over_time:
                return {
                    'error': 'No emotion data to analyze',
                    'dominant_emotion': 'unknown',
                    'emotion_stability': 0.0,
                    'business_appropriateness': 0.0
                }
            
            # Calculate dominant emotions
            emotion_counts = {}
            business_scores = []
            confidence_scores = []
            
            for frame_result in emotions_over_time:
                if 'error' not in frame_result:
                    emotion = frame_result.get('dominant_emotion', 'neutral')
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    business_scores.append(frame_result.get('business_score', 0.5))
                    confidence_scores.append(frame_result.get('confidence', 0.0))
            
            # Overall statistics
            total_frames = len(emotions_over_time)
            dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
            
            # Calculate stability (less variance = more stable)
            emotion_changes = 0
            for i in range(1, len(emotions_over_time)):
                if emotions_over_time[i].get('dominant_emotion') != emotions_over_time[i-1].get('dominant_emotion'):
                    emotion_changes += 1
            
            stability = 1.0 - (emotion_changes / max(1, total_frames - 1))
            
            # Business metrics
            avg_business_score = float(sum(business_scores) / len(business_scores)) if business_scores else 0.5
            avg_confidence = float(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0
            
            # Emotion distribution
            emotion_percentages = {
                emotion: (count / total_frames) * 100 
                for emotion, count in emotion_counts.items()
            }
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotion_distribution': emotion_percentages,
                'emotion_stability': round(stability, 3),
                'business_appropriateness': round(avg_business_score, 3),
                'average_confidence': round(avg_confidence, 3),
                'total_emotion_changes': emotion_changes,
                'analysis_quality': 'high' if avg_confidence > 0.7 else 'medium' if avg_confidence > 0.4 else 'low',
                'recommendations': self._generate_video_recommendations(
                    dominant_emotion, stability, avg_business_score
                )
            }
            
        except Exception as e:
            logger.error(f"Emotion trend analysis failed: {e}")
            return {'error': f'Trend analysis failed: {str(e)}'}
    
    def _generate_video_recommendations(self, dominant_emotion: str, stability: float, business_score: float) -> List[str]:
        """Generate recommendations based on video emotion analysis"""
        recommendations = []
        
        try:
            # Enhanced emotion-based recommendations
            if dominant_emotion in ['sad', 'fear']:
                recommendations.append("Work on projecting more confidence and enthusiasm in your facial expressions")
                recommendations.append("Practice power poses before recording to boost confidence")
                recommendations.append("Focus on positive self-talk to improve emotional state")
            elif dominant_emotion == 'angry':
                recommendations.append("Practice relaxation techniques to appear more approachable and friendly")
                recommendations.append("Use breathing exercises to maintain calm composure")
                recommendations.append("Focus on the positive aspects of your pitch")
            elif dominant_emotion == 'neutral':
                recommendations.append("Add more facial expressiveness to engage your audience better")
                recommendations.append("Practice varying your expressions to show enthusiasm")
                recommendations.append("Use hand gestures to complement your facial expressions")
            elif dominant_emotion in ['happy', 'surprise']:
                recommendations.append("Great emotional engagement! Your expressions convey enthusiasm effectively")
                recommendations.append("Maintain this positive energy throughout your presentation")
                recommendations.append("Consider adding more vocal variety to complement your expressions")
            
            # Enhanced stability recommendations
            if stability < 0.5:
                recommendations.append("Work on maintaining consistent emotional expression throughout your pitch")
                recommendations.append("Practice your pitch multiple times to build emotional consistency")
                recommendations.append("Focus on one emotion (confidence) and maintain it")
            elif stability > 0.8:
                recommendations.append("Good emotional consistency, but consider varying your expressions for emphasis")
                recommendations.append("Add strategic emotional peaks to highlight key points")
                recommendations.append("Use facial expressions to emphasize important moments")
            
            # Enhanced business appropriateness recommendations
            if business_score < 0.4:
                recommendations.append("Focus on professional, confident facial expressions appropriate for business settings")
                recommendations.append("Practice in front of a mirror to see your expressions")
                recommendations.append("Consider taking a public speaking course")
            elif business_score > 0.8:
                recommendations.append("Excellent professional demeanor - your expressions are well-suited for investor presentations")
                recommendations.append("Your visual confidence will help build investor trust")
                recommendations.append("Consider adding more storytelling elements to your pitch")
            
            # Enhanced engagement recommendations
            recommendations.append("Maintain eye contact with the camera to simulate direct investor engagement")
            recommendations.append("Use the 'triangle technique' - look at camera, then slightly left, then right")
            recommendations.append("Practice your pitch with a timer to ensure optimal pacing")
            recommendations.append("Consider using visual aids to complement your facial expressions")
            
            # Technical recommendations
            recommendations.append("Ensure good lighting to make your expressions clearly visible")
            recommendations.append("Use a stable camera setup to avoid distracting movements")
            recommendations.append("Practice in the same environment you'll record in")
            
            if not recommendations:
                recommendations.append("Continue practicing to maintain strong visual presentation skills")
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Unable to generate specific recommendations - continue practicing your presentation skills")
        
        return recommendations
    
    async def start_realtime_session(self, session_id: str) -> Dict[str, Any]:
        """Start a real-time video emotion analysis session"""
        try:
            self.emotion_history = []
            self.frame_cache = []
            
            return {
                'session_id': session_id,
                'status': 'ready',
                'models_available': {
                    'deepface': DEEPFACE_AVAILABLE,
                    'mediapipe': MEDIAPIPE_AVAILABLE,
                    'opencv': True
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start realtime video session: {e}")
            return {'error': f'Failed to start session: {str(e)}'}
    
    async def process_realtime_frame(self, session_id: str, frame_data: str) -> Dict[str, Any]:
        """Process a real-time video frame"""
        try:
            result = await self.analyze_frame_base64(frame_data)
            
            if 'error' not in result:
                # Add timestamp
                result['timestamp'] = datetime.now().isoformat()
                
                # Store in history
                self.emotion_history.append(result)
                
                # Keep only recent history (last 60 seconds assuming 1 fps)
                if len(self.emotion_history) > 60:
                    self.emotion_history = self.emotion_history[-60:]
                
                # Add trend information
                result['session_trends'] = self._calculate_session_trends()
            
            return result
            
        except Exception as e:
            logger.error(f"Realtime frame processing failed: {e}")
            return {'error': f'Frame processing failed: {str(e)}'}
    
    def _calculate_session_trends(self) -> Dict[str, Any]:
        """Calculate real-time emotion trends for current session"""
        try:
            if len(self.emotion_history) < 2:
                return {'insufficient_data': True}
            
            recent_emotions = [entry.get('dominant_emotion', 'neutral') for entry in self.emotion_history[-10:]]
            recent_business_scores = [entry.get('business_score', 0.5) for entry in self.emotion_history[-10:]]
            
            # Calculate trends
            current_emotion = recent_emotions[-1] if recent_emotions else 'neutral'
            avg_business_score = float(sum(recent_business_scores) / len(recent_business_scores)) if recent_business_scores else 0.5
            
            # Emotion consistency in recent frames
            unique_emotions = len(set(recent_emotions))
            consistency = 1.0 - (unique_emotions / len(recent_emotions)) if recent_emotions else 0.0
            
            return {
                'current_emotion': current_emotion,
                'recent_business_score': round(avg_business_score, 3),
                'emotion_consistency': round(consistency, 3),
                'frames_analyzed': len(self.emotion_history),
                'trend_direction': 'improving' if len(recent_business_scores) >= 2 and recent_business_scores[-1] > recent_business_scores[0] else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Session trends calculation failed: {e}")
            return {'error': f'Trends calculation failed: {str(e)}'}

# Create global video emotion analyzer instance
video_emotion_analyzer = VideoEmotionAnalyzer()
