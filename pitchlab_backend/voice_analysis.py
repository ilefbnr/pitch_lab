import re
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import librosa
import soundfile as sf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy

logger = logging.getLogger(__name__)

class MLVoiceAnalyzer:
    """
    Machine Learning-based voice analysis for pitch evaluation
    Uses pre-trained models for comprehensive speech analysis
    """
    
    def __init__(self):
        """Initialize ML models and analyzers"""
        self.models_loaded = False
        
        # Sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Reference templates for business pitches
        self.pitch_templates = [
            "Our innovative solution addresses a significant market opportunity with proven results and scalable growth potential.",
            "We have developed a revolutionary product that disrupts the market and provides competitive advantages.",
            "Our business model demonstrates strong revenue potential with clear value proposition for customers.",
            "The market demand is substantial and our team has the expertise to execute this vision successfully."
        ]
        
        # Initialize TF-IDF vectorizer for similarity analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Load models after initializing all attributes
        self._load_models()
        
    def _load_models(self):
        """Load all required ML models"""
        try:
            # Load emotion classification model
            try:
                self.emotion_classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=-1  # Use CPU
                )
                logger.info("Emotion classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load emotion classifier: {e}")
                self.emotion_classifier = None
            
            # Load confidence/uncertainty detection model
            try:
                self.confidence_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",  # We'll repurpose this for confidence detection
                    device=-1
                )
                logger.info("Confidence classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load confidence classifier: {e}")
                self.confidence_classifier = None
            
            # Load spaCy model for linguistic analysis
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Installing...")
                try:
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy model downloaded and loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download spaCy model: {e}")
                    self.nlp = None
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {e}")
                self.nlp = None
            
            # Fit TF-IDF on pitch templates
            try:
                self.tfidf_vectorizer.fit(self.pitch_templates)
                logger.info("TF-IDF vectorizer fitted successfully")
            except Exception as e:
                logger.error(f"Failed to fit TF-IDF vectorizer: {e}")
            
            self.models_loaded = True
            logger.info("ML models initialization completed")
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            self.models_loaded = False
    
    async def analyze_transcript(self, transcript: str, audio_duration: float = None, audio_file_path: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive ML-based analysis of the pitch transcript and audio
        """
        try:
            if not self.models_loaded:
                self._load_models()
            
            if not transcript or len(transcript.strip()) < 10:
                return {
                    'error': "Transcript too short for meaningful analysis",
                    'confidence_score': 0,
                    'overall_grade': 'F'
                }
            
            # Clean and prepare text
            clean_text = self._clean_text(transcript)
            
            # Perform ML-based analyses
            analysis = {
                'basic_stats': await self._analyze_basic_stats(clean_text),
                'emotion_analysis': await self._analyze_emotions(clean_text),
                'confidence_analysis': await self._analyze_confidence(clean_text),
                'sentiment_analysis': await self._analyze_sentiment(clean_text),
                'linguistic_analysis': await self._analyze_linguistics(clean_text),
                'pitch_similarity': await self._analyze_pitch_similarity(clean_text),
                'readability_analysis': await self._analyze_readability(clean_text),
                'audio_analysis': await self._analyze_audio(audio_file_path) if audio_file_path else None,
                'speaking_pace': self._calculate_speaking_pace(clean_text, audio_duration),
                'confidence_score': 0,
                'overall_grade': 'B',
                'recommendations': []
            }
            
            # Calculate overall confidence score and grade
            analysis['confidence_score'] = await self._calculate_ml_confidence_score(analysis)
            analysis['overall_grade'] = self._calculate_grade(analysis['confidence_score'])
            analysis['recommendations'] = await self._generate_ml_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"ML Analysis failed: {e}")
            return {
                'error': f"Analysis failed: {str(e)}",
                'basic_stats': {'word_count': 0, 'sentence_count': 0},
                'confidence_score': 0,
                'overall_grade': 'F'
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        return text
    
    async def _analyze_basic_stats(self, text: str) -> Dict[str, Any]:
        """Analyze basic text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': round(len(words) / max(len(sentences), 1), 1),
            'unique_words': len(set(words)),
            'vocabulary_richness': round(len(set(words)) / max(len(words), 1), 2),
            'character_count': len(text)
        }
    
    async def _analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Analyze emotional content using transformer model"""
        try:
            if not self.emotion_classifier:
                return {
                    'error': 'Emotion classifier not available',
                    'emotion_scores': {'neutral': 1.0},
                    'dominant_emotion': 'neutral',
                    'emotional_stability': 'Unable to analyze - model not loaded',
                    'pitch_appropriateness': 'Unable to analyze - model not loaded'
                }
            
            # Split text into chunks for better analysis
            chunks = self._split_text_chunks(text, max_length=512)
            all_emotions = []
            
            for chunk in chunks:
                if len(chunk.strip()) > 10:
                    emotions = self.emotion_classifier(chunk)
                    all_emotions.extend(emotions)
            
            if not all_emotions:
                return {'error': 'No emotions detected'}
            
            # Aggregate emotions
            emotion_scores = {}
            for emotion_result in all_emotions:
                emotion = emotion_result['label']
                score = emotion_result['score']
                if emotion in emotion_scores:
                    emotion_scores[emotion] += score
                else:
                    emotion_scores[emotion] = score
            
            # Normalize scores
            total_score = sum(emotion_scores.values())
            for emotion in emotion_scores:
                emotion_scores[emotion] = round(emotion_scores[emotion] / total_score, 3)
            
            # Get dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            
            return {
                'emotion_scores': emotion_scores,
                'dominant_emotion': dominant_emotion,
                'emotional_stability': self._assess_emotional_stability(emotion_scores),
                'pitch_appropriateness': self._assess_pitch_emotions(emotion_scores)
            }
            
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {'error': f'Emotion analysis failed: {str(e)}'}
    
    async def _analyze_confidence(self, text: str) -> Dict[str, Any]:
        """Analyze confidence level using linguistic patterns and ML"""
        try:
            if not self.nlp:
                # Fallback to basic analysis without spaCy
                confidence_score = 50.0  # Default neutral score
                return {
                    'confidence_score': confidence_score,
                    'confidence_indicators': {'error': 'spaCy model not available'},
                    'assessment': 'Unable to analyze - model not loaded'
                }
            
            # Use spaCy for linguistic analysis
            doc = self.nlp(text)
            
            # Analyze linguistic confidence indicators
            confidence_indicators = {
                'modal_verbs': 0,  # will, can, must vs might, could, may
                'certainty_adverbs': 0,  # definitely, certainly vs possibly, maybe
                'first_person_assertions': 0,  # "I know" vs "I think"
                'question_marks': 0,
                'hedge_words': 0  # sort of, kind of, etc.
            }
            
            # Strong confidence indicators
            strong_modals = ['will', 'can', 'must', 'shall']
            weak_modals = ['might', 'could', 'may', 'would']
            certainty_words = ['definitely', 'certainly', 'absolutely', 'clearly']
            hedge_words = ['maybe', 'perhaps', 'possibly', 'probably', 'sort of', 'kind of']
            
            for token in doc:
                if token.lemma_ in strong_modals:
                    confidence_indicators['modal_verbs'] += 1
                elif token.lemma_ in weak_modals:
                    confidence_indicators['modal_verbs'] -= 1
                elif token.lemma_ in certainty_words:
                    confidence_indicators['certainty_adverbs'] += 1
                elif token.lemma_ in hedge_words:
                    confidence_indicators['hedge_words'] += 1
            
            # Count questions (can indicate uncertainty)
            confidence_indicators['question_marks'] = text.count('?')
            
            # Calculate confidence score
            confidence_score = (
                confidence_indicators['modal_verbs'] * 2 +
                confidence_indicators['certainty_adverbs'] * 3 -
                confidence_indicators['hedge_words'] * 2 -
                confidence_indicators['question_marks'] * 1
            )
            
            # Normalize to 0-100 scale
            normalized_score = max(0, min(100, 50 + confidence_score * 5))
            
            return {
                'confidence_score': round(normalized_score, 1),
                'confidence_indicators': confidence_indicators,
                'assessment': self._assess_confidence_level(normalized_score)
            }
            
        except Exception as e:
            logger.error(f"Confidence analysis failed: {e}")
            return {'error': f'Confidence analysis failed: {str(e)}'}
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER sentiment analyzer"""
        try:
            sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Determine overall sentiment
            if sentiment_scores['compound'] >= 0.05:
                overall_sentiment = 'Positive'
            elif sentiment_scores['compound'] <= -0.05:
                overall_sentiment = 'Negative'
            else:
                overall_sentiment = 'Neutral'
            
            return {
                'sentiment_scores': sentiment_scores,
                'overall_sentiment': overall_sentiment,
                'positivity_ratio': round(sentiment_scores['pos'], 3),
                'pitch_sentiment_assessment': self._assess_pitch_sentiment(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'error': f'Sentiment analysis failed: {str(e)}'}
    
    async def _analyze_linguistics(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features using spaCy"""
        try:
            if not self.nlp:
                # Fallback analysis without spaCy
                words = text.split()
                sentences = text.split('.')
                return {
                    'pos_distribution': {'error': 'spaCy model not available'},
                    'named_entities': [],
                    'avg_sentence_length': round(len(words) / max(len(sentences), 1), 1),
                    'linguistic_complexity': 'Unable to analyze - model not loaded'
                }
            
            doc = self.nlp(text)
            
            # Extract linguistic features
            pos_counts = {}
            named_entities = []
            
            for token in doc:
                pos = token.pos_
                if pos in pos_counts:
                    pos_counts[pos] += 1
                else:
                    pos_counts[pos] = 1
            
            for ent in doc.ents:
                named_entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
            
            # Calculate linguistic complexity
            avg_sentence_length = np.mean([len(sent.text.split()) for sent in doc.sents])
            
            return {
                'pos_distribution': pos_counts,
                'named_entities': named_entities[:10],  # Limit to top 10
                'avg_sentence_length': round(avg_sentence_length, 1),
                'linguistic_complexity': self._assess_linguistic_complexity(pos_counts, avg_sentence_length)
            }
            
        except Exception as e:
            logger.error(f"Linguistic analysis failed: {e}")
            return {'error': f'Linguistic analysis failed: {str(e)}'}
    
    async def _analyze_pitch_similarity(self, text: str) -> Dict[str, Any]:
        """Analyze similarity to successful pitch templates"""
        try:
            # Transform text using TF-IDF
            text_vector = self.tfidf_vectorizer.transform([text])
            template_vectors = self.tfidf_vectorizer.transform(self.pitch_templates)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(text_vector, template_vectors)[0]
            
            max_similarity = float(np.max(similarities))
            avg_similarity = float(np.mean(similarities))
            
            return {
                'max_similarity': round(max_similarity, 3),
                'avg_similarity': round(avg_similarity, 3),
                'similarity_assessment': self._assess_pitch_similarity(max_similarity)
            }
            
        except Exception as e:
            logger.error(f"Pitch similarity analysis failed: {e}")
            return {'error': f'Pitch similarity analysis failed: {str(e)}'}
    
    async def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability using textstat"""
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            flesch_kincaid = textstat.flesch_kincaid_grade(text)
            gunning_fog = textstat.gunning_fog(text)
            
            return {
                'flesch_reading_ease': round(flesch_score, 1),
                'flesch_kincaid_grade': round(flesch_kincaid, 1),
                'gunning_fog_index': round(gunning_fog, 1),
                'readability_assessment': self._assess_readability(flesch_score)
            }
            
        except Exception as e:
            logger.error(f"Readability analysis failed: {e}")
            return {'error': f'Readability analysis failed: {str(e)}'}
    
    async def _analyze_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Analyze audio features using librosa"""
        try:
            if not audio_file_path or not Path(audio_file_path).exists():
                return {'error': 'Audio file not found'}
            
            # Load audio file
            y, sr = librosa.load(audio_file_path)
            
            # Extract audio features
            features = {
                'duration': float(librosa.get_duration(y=y, sr=sr)),
                'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
                'spectral_centroids': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'mfcc_features': np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)[:13].tolist()
            }
            
            # Analyze pitch variation
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_variation'] = {
                    'mean_pitch': float(np.mean(pitch_values)),
                    'pitch_std': float(np.std(pitch_values)),
                    'pitch_range': float(np.max(pitch_values) - np.min(pitch_values))
                }
            
            features['audio_quality_assessment'] = self._assess_audio_quality(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {'error': f'Audio analysis failed: {str(e)}'}
    
    def _calculate_speaking_pace(self, text: str, duration: float = None) -> Dict[str, Any]:
        """Calculate speaking pace metrics"""
        if duration is None or duration <= 0:
            return {
                'words_per_minute': None,
                'assessment': 'Unable to calculate - no duration provided'
            }
        
        word_count = len(text.split())
        words_per_minute = round((word_count / duration) * 60, 1)
        
        return {
            'words_per_minute': words_per_minute,
            'assessment': self._assess_speaking_pace(words_per_minute)
        }
    
    def _split_text_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks for transformer models"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _assess_emotional_stability(self, emotion_scores: Dict[str, float]) -> str:
        """Assess emotional stability of the pitch"""
        # Check for extreme emotions
        max_score = max(emotion_scores.values())
        if max_score > 0.8:
            return "High emotional intensity - may overwhelm audience"
        elif max_score > 0.6:
            return "Good emotional engagement"
        else:
            return "Low emotional engagement - consider adding passion"
    
    def _assess_pitch_emotions(self, emotion_scores: Dict[str, float]) -> str:
        """Assess if emotions are appropriate for a business pitch"""
        positive_emotions = ['joy', 'optimism', 'love']
        confidence_emotions = ['joy', 'optimism']
        
        positive_score = sum(emotion_scores.get(emotion, 0) for emotion in positive_emotions)
        
        if positive_score > 0.5:
            return "Excellent - positive emotions appropriate for pitch"
        elif positive_score > 0.3:
            return "Good emotional tone for business presentation"
        else:
            return "Consider adding more enthusiasm and positivity"
    
    def _assess_confidence_level(self, score: float) -> str:
        """Assess confidence level based on linguistic analysis"""
        if score >= 80:
            return "Very confident delivery"
        elif score >= 60:
            return "Confident delivery"
        elif score >= 40:
            return "Moderately confident"
        else:
            return "Lacks confidence - practice assertive language"
    
    def _assess_pitch_sentiment(self, sentiment_scores: Dict[str, float]) -> str:
        """Assess if sentiment is appropriate for pitch"""
        compound = sentiment_scores['compound']
        if compound >= 0.5:
            return "Excellent - very positive tone"
        elif compound >= 0.1:
            return "Good positive sentiment"
        elif compound >= -0.1:
            return "Neutral tone - consider adding more enthusiasm"
        else:
            return "Too negative for a business pitch"
    
    def _assess_linguistic_complexity(self, pos_counts: Dict[str, int], avg_length: float) -> str:
        """Assess linguistic complexity"""
        total_words = sum(pos_counts.values())
        noun_ratio = pos_counts.get('NOUN', 0) / max(total_words, 1)
        adj_ratio = pos_counts.get('ADJ', 0) / max(total_words, 1)
        
        if avg_length > 25:
            return "Sentences too complex - simplify for clarity"
        elif avg_length > 15 and noun_ratio > 0.3:
            return "Good complexity with rich vocabulary"
        elif avg_length < 8:
            return "Too simple - add more detail"
        else:
            return "Appropriate complexity level"
    
    def _assess_pitch_similarity(self, similarity: float) -> str:
        """Assess similarity to successful pitch patterns"""
        if similarity >= 0.7:
            return "Excellent - follows proven pitch patterns"
        elif similarity >= 0.5:
            return "Good structure similar to successful pitches"
        elif similarity >= 0.3:
            return "Decent structure but could be improved"
        else:
            return "Poor structure - study successful pitch examples"
    
    def _assess_readability(self, flesch_score: float) -> str:
        """Assess text readability"""
        if 60 <= flesch_score <= 80:
            return "Perfect readability for business audience"
        elif 50 <= flesch_score <= 90:
            return "Good readability"
        elif flesch_score < 50:
            return "Too complex - simplify language"
        else:
            return "Too simple - add more sophistication"
    
    def _assess_audio_quality(self, features: Dict[str, Any]) -> str:
        """Assess audio quality and delivery"""
        assessments = []
        
        # Check tempo
        tempo = features.get('tempo', 0)
        if 80 <= tempo <= 120:
            assessments.append("Good speaking rhythm")
        elif tempo < 80:
            assessments.append("Speaking too slowly")
        else:
            assessments.append("Speaking too quickly")
        
        # Check pitch variation
        pitch_var = features.get('pitch_variation', {})
        if pitch_var:
            pitch_std = pitch_var.get('pitch_std', 0)
            if pitch_std > 50:
                assessments.append("Good vocal variety")
            else:
                assessments.append("Monotone delivery - add vocal variety")
        
        return "; ".join(assessments) if assessments else "Audio analysis incomplete"
    
    def _assess_speaking_pace(self, wpm: float) -> str:
        """Assess speaking pace"""
        if 140 <= wpm <= 180:
            return "Perfect speaking pace for presentations"
        elif 120 <= wpm <= 200:
            return "Good speaking pace"
        elif wpm < 120:
            return "Too slow - increase pace"
        else:
            return "Too fast - slow down for clarity"
    
    async def _calculate_ml_confidence_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate confidence score based on ML analysis results"""
        score = 50  # Base score
        
        # Emotion analysis contribution
        emotion_analysis = analysis.get('emotion_analysis', {})
        if 'error' not in emotion_analysis:
            emotion_scores = emotion_analysis.get('emotion_scores', {})
            positive_emotions = emotion_scores.get('joy', 0) + emotion_scores.get('optimism', 0)
            score += positive_emotions * 20
        
        # Confidence analysis contribution
        confidence_analysis = analysis.get('confidence_analysis', {})
        if 'error' not in confidence_analysis:
            confidence_score = confidence_analysis.get('confidence_score', 50)
            score += (confidence_score - 50) * 0.3
        
        # Sentiment analysis contribution
        sentiment_analysis = analysis.get('sentiment_analysis', {})
        if 'error' not in sentiment_analysis:
            compound_score = sentiment_analysis.get('sentiment_scores', {}).get('compound', 0)
            score += compound_score * 25
        
        # Pitch similarity contribution
        similarity_analysis = analysis.get('pitch_similarity', {})
        if 'error' not in similarity_analysis:
            max_similarity = similarity_analysis.get('max_similarity', 0)
            score += max_similarity * 30
        
        # Readability contribution
        readability_analysis = analysis.get('readability_analysis', {})
        if 'error' not in readability_analysis:
            flesch_score = readability_analysis.get('flesch_reading_ease', 50)
            if 60 <= flesch_score <= 80:
                score += 10
            elif 50 <= flesch_score <= 90:
                score += 5
        
        return max(0, min(100, int(score)))
    
    def _calculate_grade(self, score: int) -> str:
        """Calculate letter grade based on score"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    async def _generate_ml_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate AI-powered recommendations based on ML analysis"""
        recommendations = []
        
        # Emotion-based recommendations
        emotion_analysis = analysis.get('emotion_analysis', {})
        if 'error' not in emotion_analysis:
            dominant_emotion = emotion_analysis.get('dominant_emotion', '')
            if dominant_emotion in ['fear', 'sadness', 'anger']:
                recommendations.append("Focus on positive outcomes and opportunities rather than problems")
            elif dominant_emotion == 'surprise':
                recommendations.append("Great enthusiasm! Ensure your message is clear and structured")
        
        # Confidence-based recommendations
        confidence_analysis = analysis.get('confidence_analysis', {})
        if 'error' not in confidence_analysis:
            confidence_score = confidence_analysis.get('confidence_score', 50)
            if confidence_score < 60:
                recommendations.append("Use more assertive language and eliminate hedge words")
            elif confidence_score > 90:
                recommendations.append("Excellent confidence! Consider adding humility to connect with audience")
        
        # Sentiment recommendations
        sentiment_analysis = analysis.get('sentiment_analysis', {})
        if 'error' not in sentiment_analysis:
            compound = sentiment_analysis.get('sentiment_scores', {}).get('compound', 0)
            if compound < 0.1:
                recommendations.append("Add more positive language and enthusiasm to your pitch")
        
        # Similarity recommendations
        similarity_analysis = analysis.get('pitch_similarity', {})
        if 'error' not in similarity_analysis:
            max_similarity = similarity_analysis.get('max_similarity', 0)
            if max_similarity < 0.3:
                recommendations.append("Study successful pitch structures and incorporate proven patterns")
        
        # Linguistic recommendations
        linguistic_analysis = analysis.get('linguistic_analysis', {})
        if 'error' not in linguistic_analysis:
            avg_length = linguistic_analysis.get('avg_sentence_length', 15)
            if avg_length > 25:
                recommendations.append("Shorten sentences for better clarity and impact")
            elif avg_length < 8:
                recommendations.append("Add more detail and depth to your explanations")
        
        # Audio recommendations
        audio_analysis = analysis.get('audio_analysis', {})
        if audio_analysis and 'error' not in audio_analysis:
            tempo = audio_analysis.get('tempo', 100)
            if tempo < 80:
                recommendations.append("Increase your speaking pace to maintain engagement")
            elif tempo > 140:
                recommendations.append("Slow down to ensure clarity and comprehension")
        
        if not recommendations:
            recommendations.append("Excellent pitch! Your delivery shows strong communication skills across all areas")
        
        return recommendations

# Create global ML analyzer instance
ml_voice_analyzer = MLVoiceAnalyzer()
