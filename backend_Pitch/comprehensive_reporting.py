import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

class ComprehensiveReportGenerator:
    """
    Generate detailed post-pitch reports with visualizations and actionable insights
    Combines multimodal analysis results into comprehensive reports
    """
    
    def __init__(self):
        """Initialize report generator"""
        self.report_templates = {
            'executive_summary': self._generate_executive_summary,
            'detailed_analysis': self._generate_detailed_analysis,
            'improvement_plan': self._generate_improvement_plan,
            'benchmarking': self._generate_benchmarking_report
        }
        
        # Configure matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Report scoring weights
        self.analysis_weights = {
            'voice_ml_analysis': 0.25,
            'enhanced_voice_emotion': 0.20,
            'video_emotion_analysis': 0.20,
            'speech_content': 0.20,
            'delivery_metrics': 0.15
        }
    
    async def generate_comprehensive_report(
        self, 
        pitch_data: Dict[str, Any], 
        analysis_results: Dict[str, Any],
        report_type: str = 'detailed_analysis'
    ) -> Dict[str, Any]:
        """Generate a comprehensive pitch analysis report"""
        try:
            # Prepare data for analysis
            processed_data = self._process_analysis_data(analysis_results)
            
            # Generate visualizations
            visualizations = await self._generate_all_visualizations(processed_data)
            
            # Generate text analysis
            text_analysis = self._generate_text_analysis(processed_data)
            
            # Generate recommendations
            recommendations = self._generate_comprehensive_recommendations(processed_data)
            
            # Calculate overall scores
            scores = self._calculate_comprehensive_scores(processed_data)
            
            # Generate report based on type
            if report_type in self.report_templates:
                report_content = await self.report_templates[report_type](
                    pitch_data, processed_data, visualizations, text_analysis, recommendations, scores
                )
            else:
                report_content = await self._generate_detailed_analysis(
                    pitch_data, processed_data, visualizations, text_analysis, recommendations, scores
                )
            
            return {
                'report_id': f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'pitch_id': pitch_data.get('id'),
                'generated_at': datetime.now().isoformat(),
                'report_type': report_type,
                'content': report_content,
                'visualizations': visualizations,
                'overall_scores': scores,
                'recommendations': recommendations,
                'metadata': {
                    'analysis_components': list(analysis_results.keys()),
                    'visualization_count': len(visualizations),
                    'recommendation_count': len(recommendations)
                }
            }
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {'error': f'Report generation failed: {str(e)}'}
    
    def _process_analysis_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and normalize analysis data for reporting"""
        processed = {
            'voice_ml': {},
            'enhanced_voice': {},
            'video_emotion': {},
            'content_analysis': {},
            'delivery_metrics': {},
            'timestamps': [],
            'metrics_timeline': []
        }
        
        try:
            # Process ML voice analysis
            if 'voice_ml_analysis' in analysis_results:
                voice_ml = analysis_results['voice_ml_analysis']
                processed['voice_ml'] = {
                    'confidence_score': voice_ml.get('confidence_analysis', {}).get('confidence_score', 0),
                    'emotions': voice_ml.get('emotion_analysis', {}).get('emotion_scores', {}),
                    'sentiment': voice_ml.get('sentiment_analysis', {}).get('overall_sentiment', 'neutral'),
                    'speaking_pace': voice_ml.get('speaking_pace', {}).get('words_per_minute', 0),
                    'overall_grade': voice_ml.get('overall_grade', 'C')
                }
            
            # Process enhanced voice emotion analysis
            if 'enhanced_voice_emotion' in analysis_results:
                enhanced_voice = analysis_results['enhanced_voice_emotion']
                processed['enhanced_voice'] = {
                    'emotion_scores': enhanced_voice.get('voice_emotion_analysis', {}).get('emotion_scores', {}),
                    'prosodic_features': enhanced_voice.get('prosodic_features', {}),
                    'voice_quality': enhanced_voice.get('voice_quality', {}),
                    'business_score': enhanced_voice.get('business_analysis', {}).get('business_score', 0.5)
                }
            
            # Process video emotion analysis
            if 'video_emotion_analysis' in analysis_results:
                video_emotion = analysis_results['video_emotion_analysis']
                processed['video_emotion'] = {
                    'dominant_emotion': video_emotion.get('dominant_emotion', 'neutral'),
                    'emotion_distribution': video_emotion.get('emotion_distribution', {}),
                    'stability': video_emotion.get('emotion_stability', 0.5),
                    'business_appropriateness': video_emotion.get('business_appropriateness', 0.5)
                }
            
            # Process content analysis
            if 'content_analysis' in analysis_results:
                content = analysis_results['content_analysis']
                processed['content_analysis'] = {
                    'word_count': content.get('basic_stats', {}).get('word_count', 0),
                    'readability': content.get('readability_analysis', {}).get('flesch_reading_ease', 0),
                    'similarity_score': content.get('pitch_similarity', {}).get('max_similarity', 0),
                    'linguistic_complexity': content.get('linguistic_analysis', {})
                }
            
            # Process real-time metrics if available
            if 'realtime_metrics' in analysis_results:
                realtime = analysis_results['realtime_metrics']
                processed['delivery_metrics'] = {
                    'volume_consistency': realtime.get('volume_consistency', 0.5),
                    'pace_variation': realtime.get('pace_variation', 0.5),
                    'speaking_time_ratio': realtime.get('speaking_time_ratio', 0.5)
                }
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
        
        return processed
    
    async def _generate_all_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate all visualizations and return as base64 encoded images"""
        visualizations = {}
        
        try:
            # Emotion analysis radar chart
            visualizations['emotion_radar'] = self._create_emotion_radar_chart(data)
            
            # Performance metrics dashboard
            visualizations['performance_dashboard'] = self._create_performance_dashboard(data)
            
            # Voice quality analysis
            visualizations['voice_quality'] = self._create_voice_quality_chart(data)
            
            # Improvement areas chart
            visualizations['improvement_areas'] = self._create_improvement_areas_chart(data)
            
            # Timeline visualization if data available
            if data.get('metrics_timeline'):
                visualizations['timeline'] = self._create_timeline_visualization(data)
            
            # Benchmarking chart
            visualizations['benchmarking'] = self._create_benchmarking_chart(data)
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
        
        return visualizations
    
    def _create_emotion_radar_chart(self, data: Dict[str, Any]) -> str:
        """Create radar chart for emotion analysis"""
        try:
            # Combine voice and video emotions
            voice_emotions = data.get('voice_ml', {}).get('emotions', {})
            enhanced_emotions = data.get('enhanced_voice', {}).get('emotion_scores', {})
            video_emotions = data.get('video_emotion', {}).get('emotion_distribution', {})
            
            # Normalize and combine emotions
            all_emotions = set()
            all_emotions.update(voice_emotions.keys())
            all_emotions.update(enhanced_emotions.keys())
            all_emotions.update(video_emotions.keys())
            
            emotions = list(all_emotions)[:8]  # Limit to 8 emotions for readability
            
            voice_scores = [voice_emotions.get(emotion, 0) for emotion in emotions]
            enhanced_scores = [enhanced_emotions.get(emotion, 0) for emotion in emotions]
            video_scores = [video_emotions.get(emotion, 0) / 100 if emotion in video_emotions else 0 for emotion in emotions]
            
            # Create radar chart using plotly
            fig = go.Figure()
            
            if voice_scores:
                fig.add_trace(go.Scatterpolar(
                    r=voice_scores,
                    theta=emotions,
                    fill='toself',
                    name='Voice ML Analysis',
                    line_color='blue'
                ))
            
            if enhanced_scores:
                fig.add_trace(go.Scatterpolar(
                    r=enhanced_scores,
                    theta=emotions,
                    fill='toself',
                    name='Enhanced Voice Analysis',
                    line_color='green'
                ))
            
            if video_scores and any(score > 0 for score in video_scores):
                fig.add_trace(go.Scatterpolar(
                    r=video_scores,
                    theta=emotions,
                    fill='toself',
                    name='Video Emotion Analysis',
                    line_color='red'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Multimodal Emotion Analysis",
                width=600,
                height=500
            )
            
            # Convert to base64
            img_buffer = io.BytesIO()
            fig.write_image(img_buffer, format='png')
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Emotion radar chart generation failed: {e}")
            return ""
    
    def _create_performance_dashboard(self, data: Dict[str, Any]) -> str:
        """Create comprehensive performance dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Voice Confidence', 'Speaking Pace', 'Content Quality', 'Overall Business Score'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Voice confidence gauge
            confidence_score = data.get('voice_ml', {}).get('confidence_score', 0)
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=confidence_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Voice Confidence"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}
            ), row=1, col=1)
            
            # Speaking pace gauge
            speaking_pace = data.get('voice_ml', {}).get('speaking_pace', 0)
            pace_score = min(100, max(0, (speaking_pace - 120) / 60 * 100)) if speaking_pace > 0 else 0
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=pace_score,
                title={'text': "Speaking Pace"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 80], 'color': "gray"}]}
            ), row=1, col=2)
            
            # Content quality gauge
            content_score = data.get('content_analysis', {}).get('similarity_score', 0) * 100
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=content_score,
                title={'text': "Content Quality"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}]}
            ), row=2, col=1)
            
            # Overall business score
            business_score = data.get('enhanced_voice', {}).get('business_score', 0.5) * 100
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=business_score,
                title={'text': "Business Readiness"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "gray"}]}
            ), row=2, col=2)
            
            fig.update_layout(
                height=600,
                width=800,
                title="Performance Dashboard"
            )
            
            # Convert to base64
            img_buffer = io.BytesIO()
            fig.write_image(img_buffer, format='png')
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Performance dashboard generation failed: {e}")
            return ""
    
    def _create_voice_quality_chart(self, data: Dict[str, Any]) -> str:
        """Create voice quality analysis chart"""
        try:
            # Extract voice quality metrics
            voice_quality = data.get('enhanced_voice', {}).get('voice_quality', {})
            prosodic = data.get('enhanced_voice', {}).get('prosodic_features', {})
            
            metrics = {
                'Jitter (Pitch Stability)': 1 - min(1, voice_quality.get('jitter', 0) * 50),
                'Shimmer (Amplitude Stability)': 1 - min(1, voice_quality.get('shimmer', 0) * 10),
                'Harmonicity': min(1, voice_quality.get('harmonicity', 0.5)),
                'Speaking Rate': min(1, prosodic.get('speaking_rate', {}).get('syllables_per_second', 3) / 5),
                'Fluency': prosodic.get('fluency_score', 0.5)
            }
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                )
            ])
            
            fig.update_layout(
                title="Voice Quality Analysis",
                xaxis_title="Voice Quality Metrics",
                yaxis_title="Score (0-1)",
                yaxis=dict(range=[0, 1]),
                width=700,
                height=400
            )
            
            # Convert to base64
            img_buffer = io.BytesIO()
            fig.write_image(img_buffer, format='png')
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Voice quality chart generation failed: {e}")
            return ""
    
    def _create_improvement_areas_chart(self, data: Dict[str, Any]) -> str:
        """Create improvement areas priority chart"""
        try:
            # Calculate improvement priorities based on scores
            areas = {
                'Voice Confidence': 100 - data.get('voice_ml', {}).get('confidence_score', 50),
                'Emotional Expression': (1 - data.get('enhanced_voice', {}).get('business_score', 0.5)) * 100,
                'Content Structure': (1 - data.get('content_analysis', {}).get('similarity_score', 0.5)) * 100,
                'Speaking Pace': abs(data.get('voice_ml', {}).get('speaking_pace', 150) - 150) / 150 * 100,
                'Visual Presentation': (1 - data.get('video_emotion', {}).get('business_appropriateness', 0.5)) * 100
            }
            
            # Sort by priority (higher score = more improvement needed)
            sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
            
            fig = go.Figure(data=[
                go.Bar(
                    y=[area[0] for area in sorted_areas],
                    x=[area[1] for area in sorted_areas],
                    orientation='h',
                    marker_color=px.colors.sequential.Reds_r
                )
            ])
            
            fig.update_layout(
                title="Improvement Priority Areas",
                xaxis_title="Improvement Needed (0-100)",
                yaxis_title="Skills Areas",
                width=700,
                height=400
            )
            
            # Convert to base64
            img_buffer = io.BytesIO()
            fig.write_image(img_buffer, format='png')
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Improvement areas chart generation failed: {e}")
            return ""
    
    def _create_timeline_visualization(self, data: Dict[str, Any]) -> str:
        """Create timeline visualization of metrics during pitch"""
        try:
            # This would use real-time data if available
            # For now, create a sample timeline
            timeline_data = data.get('metrics_timeline', [])
            
            if not timeline_data:
                # Create sample data
                time_points = list(range(0, 60, 5))  # Every 5 seconds for 1 minute
                confidence_trend = [50 + 20 * np.sin(t / 10) + np.random.normal(0, 5) for t in time_points]
                emotion_trend = [0.6 + 0.3 * np.cos(t / 15) + np.random.normal(0, 0.1) for t in time_points]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=confidence_trend,
                    mode='lines+markers',
                    name='Confidence Score',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=[e * 100 for e in emotion_trend],
                    mode='lines+markers',
                    name='Emotional Engagement',
                    line=dict(color='red'),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="Performance Timeline During Pitch",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Confidence Score",
                    yaxis2=dict(
                        title="Emotional Engagement (%)",
                        overlaying='y',
                        side='right'
                    ),
                    width=800,
                    height=400
                )
            
            # Convert to base64
            img_buffer = io.BytesIO()
            fig.write_image(img_buffer, format='png')
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Timeline visualization generation failed: {e}")
            return ""
    
    def _create_benchmarking_chart(self, data: Dict[str, Any]) -> str:
        """Create benchmarking comparison chart"""
        try:
            # Create comparison with industry benchmarks
            categories = ['Voice Confidence', 'Content Quality', 'Emotional Expression', 'Delivery', 'Overall']
            
            user_scores = [
                data.get('voice_ml', {}).get('confidence_score', 50),
                data.get('content_analysis', {}).get('similarity_score', 0.5) * 100,
                data.get('enhanced_voice', {}).get('business_score', 0.5) * 100,
                data.get('video_emotion', {}).get('business_appropriateness', 0.5) * 100,
                self._calculate_overall_score(data)
            ]
            
            # Industry benchmarks (example values)
            industry_avg = [65, 70, 60, 55, 62]
            top_performers = [85, 90, 85, 80, 85]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=user_scores,
                theta=categories,
                fill='toself',
                name='Your Performance',
                line_color='blue'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=industry_avg,
                theta=categories,
                fill='toself',
                name='Industry Average',
                line_color='orange'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=top_performers,
                theta=categories,
                fill='toself',
                name='Top Performers',
                line_color='green'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Performance Benchmarking",
                width=600,
                height=500
            )
            
            # Convert to base64
            img_buffer = io.BytesIO()
            fig.write_image(img_buffer, format='png')
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Benchmarking chart generation failed: {e}")
            return ""
    
    def _calculate_overall_score(self, data: Dict[str, Any]) -> float:
        """Calculate weighted overall performance score"""
        try:
            # Get actual scores with more realistic defaults
            voice_confidence = data.get('voice_ml', {}).get('confidence_score', 0)
            business_score = data.get('enhanced_voice', {}).get('business_score', 0.3)
            video_appropriateness = data.get('video_emotion', {}).get('business_appropriateness', 0.3)
            content_score = data.get('content_analysis', {}).get('similarity_score', 0.3)
            delivery_score = data.get('delivery_metrics', {}).get('volume_consistency', 0.3)
            
            # Normalize scores to 0-1 range
            scores = {
                'voice_ml_analysis': voice_confidence / 100,
                'enhanced_voice_emotion': business_score,
                'video_emotion_analysis': video_appropriateness,
                'speech_content': content_score,
                'delivery_metrics': delivery_score
            }
            
            # Calculate weighted score
            weighted_score = sum(
                scores[component] * self.analysis_weights.get(component, 0.2)
                for component in scores
            )
            
            # Apply penalties for poor performance
            final_score = weighted_score * 100
            
            # Additional penalties for critical issues
            if voice_confidence < 20:  # Very low confidence
                final_score *= 0.7
            if business_score < 0.3:  # Poor emotional appropriateness
                final_score *= 0.8
            if video_appropriateness < 0.3:  # Poor visual presentation
                final_score *= 0.8
            
            return min(100, max(0, final_score))
            
        except Exception as e:
            logger.error(f"Overall score calculation failed: {e}")
            return 30.0  # Lower default for failed analysis
    
    def _generate_text_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive text analysis"""
        try:
            analysis = {
                'executive_summary': self._generate_executive_text(data),
                'strengths': self._identify_key_strengths(data),
                'weaknesses': self._identify_key_weaknesses(data),
                'recommendations': self._generate_specific_recommendations(data),
                'next_steps': self._generate_next_steps(data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Text analysis generation failed: {e}")
            return {'error': f'Text analysis failed: {str(e)}'}
    
    def _generate_executive_text(self, data: Dict[str, Any]) -> str:
        """Generate executive summary text"""
        try:
            overall_score = self._calculate_overall_score(data)
            dominant_emotion = data.get('video_emotion', {}).get('dominant_emotion', 'neutral')
            confidence_score = data.get('voice_ml', {}).get('confidence_score', 50)
            
            if overall_score >= 90:
                performance_level = "excellent"
            elif overall_score >= 80:
                performance_level = "very good"
            elif overall_score >= 70:
                performance_level = "good"
            elif overall_score >= 60:
                performance_level = "satisfactory"
            elif overall_score >= 50:
                performance_level = "needs improvement"
            else:
                performance_level = "requires significant improvement"
            
            summary = f"""
            Your pitch demonstration shows {performance_level} performance with an overall score of {overall_score:.1f}/100. 
            
            Key highlights:
            - Voice confidence level: {confidence_score}/100
            - Dominant emotional tone: {dominant_emotion}
            - Business readiness score: {data.get('enhanced_voice', {}).get('business_score', 0.5) * 100:.1f}/100
            
            This analysis combines voice emotion recognition, video facial analysis, content structure evaluation, 
            and delivery metrics to provide comprehensive feedback on your presentation skills.
            """
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Executive text generation failed: {e}")
            return "Unable to generate executive summary due to analysis error."
    
    def _identify_key_strengths(self, data: Dict[str, Any]) -> List[str]:
        """Identify key strengths from analysis"""
        strengths = []
        
        try:
            # Voice confidence strength
            confidence_score = data.get('voice_ml', {}).get('confidence_score', 0)
            if confidence_score > 80:
                strengths.append(f"Excellent voice confidence ({confidence_score}/100)")
            elif confidence_score > 65:
                strengths.append(f"Good voice confidence ({confidence_score}/100)")
            
            # Emotional expression strength
            business_score = data.get('enhanced_voice', {}).get('business_score', 0)
            if business_score > 0.8:
                strengths.append(f"Excellent professional emotional expression ({business_score * 100:.1f}/100)")
            elif business_score > 0.6:
                strengths.append(f"Good professional emotional expression ({business_score * 100:.1f}/100)")
            
            # Content quality strength
            content_score = data.get('content_analysis', {}).get('similarity_score', 0)
            if content_score > 0.7:
                strengths.append(f"Excellent content structure ({content_score * 100:.1f}/100)")
            elif content_score > 0.5:
                strengths.append(f"Good content structure ({content_score * 100:.1f}/100)")
            
            # Visual presentation strength
            video_score = data.get('video_emotion', {}).get('business_appropriateness', 0)
            if video_score > 0.8:
                strengths.append(f"Excellent visual presentation ({video_score * 100:.1f}/100)")
            elif video_score > 0.6:
                strengths.append(f"Good visual presentation ({video_score * 100:.1f}/100)")
            
            # Voice quality strength
            voice_quality = data.get('enhanced_voice', {}).get('voice_quality', {})
            jitter = voice_quality.get('jitter', 1)
            if jitter < 0.02:
                strengths.append("Excellent voice stability and clarity")
            
            if not strengths:
                strengths.append("Baseline performance suitable for focused improvement")
                
        except Exception as e:
            logger.error(f"Strength identification failed: {e}")
        
        return strengths
    
    def _identify_key_weaknesses(self, data: Dict[str, Any]) -> List[str]:
        """Identify key weaknesses from analysis"""
        weaknesses = []
        
        try:
            # Voice confidence weakness
            confidence_score = data.get('voice_ml', {}).get('confidence_score', 0)
            if confidence_score < 30:
                weaknesses.append(f"Very low voice confidence - significant improvement needed ({confidence_score}/100)")
            elif confidence_score < 50:
                weaknesses.append(f"Voice confidence needs improvement ({confidence_score}/100)")
            
            # Speaking pace issues
            speaking_pace = data.get('voice_ml', {}).get('speaking_pace', 150)
            if speaking_pace < 100:
                weaknesses.append("Speaking pace very slow - may lose audience attention")
            elif speaking_pace < 120:
                weaknesses.append("Speaking pace too slow - consider increasing to 140-160 WPM")
            elif speaking_pace > 180:
                weaknesses.append("Speaking pace too fast - may be difficult to follow")
            
            # Emotional expression issues
            business_score = data.get('enhanced_voice', {}).get('business_score', 0.3)
            if business_score < 0.3:
                weaknesses.append(f"Emotional expression not suitable for business settings ({business_score * 100:.1f}/100)")
            elif business_score < 0.5:
                weaknesses.append(f"Emotional expression needs improvement ({business_score * 100:.1f}/100)")
            
            # Video presentation issues
            video_score = data.get('video_emotion', {}).get('business_appropriateness', 0.3)
            if video_score < 0.3:
                weaknesses.append(f"Visual presentation not suitable for business settings ({video_score * 100:.1f}/100)")
            elif video_score < 0.5:
                weaknesses.append(f"Visual presentation needs improvement ({video_score * 100:.1f}/100)")
            
            # Content structure issues
            content_score = data.get('content_analysis', {}).get('similarity_score', 0.3)
            if content_score < 0.3:
                weaknesses.append("Content structure significantly differs from successful pitch patterns")
            elif content_score < 0.5:
                weaknesses.append("Content structure could be more aligned with successful pitch patterns")
            
            # Check for critical issues
            dominant_emotion = data.get('video_emotion', {}).get('dominant_emotion', 'neutral')
            if dominant_emotion in ['fear', 'sad', 'angry']:
                weaknesses.append(f"Dominant emotion '{dominant_emotion}' may negatively impact audience perception")
            
            # Check for very low scores across multiple areas
            low_scores = 0
            if confidence_score < 30: low_scores += 1
            if business_score < 0.3: low_scores += 1
            if video_score < 0.3: low_scores += 1
            if content_score < 0.3: low_scores += 1
            
            if low_scores >= 3:
                weaknesses.append("Multiple areas need significant improvement - consider comprehensive training program")
            
        except Exception as e:
            logger.error(f"Weakness identification failed: {e}")
        
        return weaknesses if weaknesses else ["Continue practicing to enhance overall presentation skills"]
    
    def _generate_specific_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        try:
            # Voice-based recommendations
            confidence_score = data.get('voice_ml', {}).get('confidence_score', 50)
            if confidence_score < 60:
                recommendations.append("Practice using more assertive language and eliminate hedge words like 'maybe', 'perhaps'")
            
            # Emotion-based recommendations
            dominant_emotion = data.get('video_emotion', {}).get('dominant_emotion', 'neutral')
            if dominant_emotion in ['sad', 'fear', 'nervous']:
                recommendations.append("Work on projecting confidence through both voice tone and facial expressions")
            
            # Speaking pace recommendations
            speaking_pace = data.get('voice_ml', {}).get('speaking_pace', 150)
            if speaking_pace < 120:
                recommendations.append("Increase speaking pace to 140-160 words per minute for better engagement")
            elif speaking_pace > 180:
                recommendations.append("Slow down to 140-160 words per minute to improve comprehension")
            
            # Voice quality recommendations
            voice_quality = data.get('enhanced_voice', {}).get('voice_quality', {})
            jitter = voice_quality.get('jitter', 0)
            if jitter > 0.02:
                recommendations.append("Practice breathing exercises and vocal warm-ups to improve voice stability")
            
            # Content recommendations
            content_score = data.get('content_analysis', {}).get('similarity_score', 0.5)
            if content_score < 0.5:
                recommendations.append("Study successful pitch structures and incorporate proven patterns")
            
            # Pause and fluency recommendations
            prosodic = data.get('enhanced_voice', {}).get('prosodic_features', {})
            fluency_score = prosodic.get('fluency_score', 0.5)
            if fluency_score < 0.6:
                recommendations.append("Practice adding strategic pauses for emphasis and clarity")
            
            if not recommendations:
                recommendations.append("Continue regular practice to maintain and enhance your presentation skills")
                
        except Exception as e:
            logger.error(f"Specific recommendations generation failed: {e}")
        
        return recommendations
    
    def _generate_next_steps(self, data: Dict[str, Any]) -> List[str]:
        """Generate specific next steps for improvement"""
        next_steps = []
        
        try:
            overall_score = self._calculate_overall_score(data)
            
            if overall_score < 50:
                next_steps.extend([
                    "1. Focus on building basic confidence through daily vocal exercises",
                    "2. Record yourself daily and practice with the feedback system",
                    "3. Work with a presentation coach or take a public speaking course",
                    "4. Practice pitch structure using proven templates"
                ])
            elif overall_score < 70:
                next_steps.extend([
                    "1. Refine specific areas identified in the weakness analysis",
                    "2. Practice with real investors or experienced entrepreneurs",
                    "3. Record multiple versions and compare improvements",
                    "4. Focus on storytelling and emotional engagement techniques"
                ])
            else:
                next_steps.extend([
                    "1. Fine-tune delivery for different investor types",
                    "2. Prepare for Q&A scenarios and difficult questions",
                    "3. Practice pitch variations for different time constraints",
                    "4. Seek feedback from industry experts and successful entrepreneurs"
                ])
            
            # Add specific technical improvements
            if data.get('voice_ml', {}).get('confidence_score', 50) < 60:
                next_steps.append("5. Complete confidence-building exercises and assertiveness training")
            
            if data.get('enhanced_voice', {}).get('business_score', 0.5) < 0.6:
                next_steps.append("6. Work on professional voice coaching for business presentations")
                
        except Exception as e:
            logger.error(f"Next steps generation failed: {e}")
        
        return next_steps if next_steps else ["Continue using the platform for regular practice and improvement"]
    
    def _generate_comprehensive_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive recommendations across all analysis dimensions"""
        try:
            return {
                'immediate_actions': self._generate_immediate_actions(data),
                'short_term_goals': self._generate_short_term_goals(data),
                'long_term_development': self._generate_long_term_development(data),
                'practice_exercises': self._generate_practice_exercises(data),
                'resources': self._generate_recommended_resources(data)
            }
        except Exception as e:
            logger.error(f"Comprehensive recommendations generation failed: {e}")
            return {'error': f'Recommendations generation failed: {str(e)}'}
    
    def _generate_immediate_actions(self, data: Dict[str, Any]) -> List[str]:
        """Generate immediate actionable items"""
        actions = []
        
        confidence_score = data.get('voice_ml', {}).get('confidence_score', 50)
        if confidence_score < 50:
            actions.append("Record yourself reading business articles aloud to build vocal confidence")
        
        speaking_pace = data.get('voice_ml', {}).get('speaking_pace', 150)
        if speaking_pace < 120 or speaking_pace > 180:
            actions.append("Practice speaking with a metronome app to regulate pace")
        
        return actions if actions else ["Continue practicing with the current analysis system"]
    
    def _generate_short_term_goals(self, data: Dict[str, Any]) -> List[str]:
        """Generate 30-day improvement goals"""
        goals = []
        
        overall_score = self._calculate_overall_score(data)
        target_score = min(overall_score + 15, 90)
        
        goals.append(f"Achieve overall presentation score of {target_score}/100 within 30 days")
        
        confidence_score = data.get('voice_ml', {}).get('confidence_score', 50)
        if confidence_score < 70:
            goals.append("Increase voice confidence score to above 70/100")
        
        return goals
    
    def _generate_long_term_development(self, data: Dict[str, Any]) -> List[str]:
        """Generate long-term development recommendations"""
        development = [
            "Master pitch delivery for different investor personalities",
            "Develop expertise in handling challenging Q&A sessions",
            "Build a repertoire of compelling business stories and case studies"
        ]
        
        return development
    
    def _generate_practice_exercises(self, data: Dict[str, Any]) -> List[str]:
        """Generate specific practice exercises"""
        exercises = [
            "Daily 5-minute vocal warm-up routine",
            "Record and analyze one practice pitch per week",
            "Practice pitch elevator versions (30s, 1min, 5min)"
        ]
        
        confidence_score = data.get('voice_ml', {}).get('confidence_score', 50)
        if confidence_score < 60:
            exercises.append("Power posing and confidence-building exercises before recording")
        
        return exercises
    
    def _generate_recommended_resources(self, data: Dict[str, Any]) -> List[str]:
        """Generate recommended learning resources"""
        resources = [
            "Book: 'Pitch Anything' by Oren Klaff",
            "Online Course: Coursera's 'Introduction to Public Speaking'",
            "App: Orai for daily speech practice",
            "YouTube: TED Talks on effective presentations"
        ]
        
        return resources
    
    def _calculate_comprehensive_scores(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive scoring metrics"""
        try:
            scores = {
                'overall_score': self._calculate_overall_score(data),
                'voice_confidence': data.get('voice_ml', {}).get('confidence_score', 50),
                'emotional_appropriateness': data.get('enhanced_voice', {}).get('business_score', 0.5) * 100,
                'visual_presentation': data.get('video_emotion', {}).get('business_appropriateness', 0.5) * 100,
                'content_quality': data.get('content_analysis', {}).get('similarity_score', 0.5) * 100,
                'delivery_effectiveness': data.get('delivery_metrics', {}).get('volume_consistency', 0.5) * 100
            }
            
            # Calculate grade with stricter thresholds
            overall = scores['overall_score']
            if overall >= 95:
                grade = 'A+'
            elif overall >= 90:
                grade = 'A'
            elif overall >= 85:
                grade = 'A-'
            elif overall >= 80:
                grade = 'B+'
            elif overall >= 75:
                grade = 'B'
            elif overall >= 70:
                grade = 'B-'
            elif overall >= 65:
                grade = 'C+'
            elif overall >= 60:
                grade = 'C'
            elif overall >= 55:
                grade = 'C-'
            elif overall >= 50:
                grade = 'D+'
            elif overall >= 45:
                grade = 'D'
            else:
                grade = 'F'
            
            scores['letter_grade'] = grade
            
            return scores
            
        except Exception as e:
            logger.error(f"Comprehensive scoring failed: {e}")
            return {'overall_score': 50.0, 'letter_grade': 'C'}
    
    async def _generate_executive_summary(self, pitch_data, processed_data, visualizations, text_analysis, recommendations, scores):
        """Generate executive summary report"""
        return {
            'report_type': 'Executive Summary',
            'overall_score': scores.get('overall_score', 50),
            'letter_grade': scores.get('letter_grade', 'C'),
            'key_insights': text_analysis.get('executive_summary', ''),
            'top_strengths': text_analysis.get('strengths', [])[:3],
            'priority_improvements': text_analysis.get('weaknesses', [])[:3],
            'next_actions': recommendations.get('immediate_actions', [])
        }
    
    async def _generate_detailed_analysis(self, pitch_data, processed_data, visualizations, text_analysis, recommendations, scores):
        """Generate detailed analysis report"""
        return {
            'report_type': 'Detailed Analysis',
            'scores': scores,
            'analysis_breakdown': {
                'voice_analysis': processed_data.get('voice_ml', {}),
                'enhanced_voice': processed_data.get('enhanced_voice', {}),
                'video_emotion': processed_data.get('video_emotion', {}),
                'content_analysis': processed_data.get('content_analysis', {}),
                'delivery_metrics': processed_data.get('delivery_metrics', {})
            },
            'text_analysis': text_analysis,
            'recommendations': recommendations,
            'improvement_timeline': '30-90 days for significant improvement'
        }
    
    async def _generate_improvement_plan(self, pitch_data, processed_data, visualizations, text_analysis, recommendations, scores):
        """Generate improvement-focused report"""
        return {
            'report_type': 'Improvement Plan',
            'current_level': scores.get('letter_grade', 'C'),
            'target_level': 'A' if scores.get('overall_score', 50) < 85 else 'A+',
            'improvement_plan': recommendations,
            'practice_schedule': self._generate_practice_schedule(scores),
            'milestone_tracking': self._generate_milestones(scores)
        }
    
    async def _generate_benchmarking_report(self, pitch_data, processed_data, visualizations, text_analysis, recommendations, scores):
        """Generate benchmarking comparison report"""
        return {
            'report_type': 'Benchmarking Analysis',
            'your_scores': scores,
            'industry_comparison': self._generate_industry_comparison(scores),
            'percentile_ranking': self._calculate_percentile_ranking(scores),
            'competitive_analysis': 'Based on analysis of successful pitch patterns'
        }
    
    def _generate_practice_schedule(self, scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate personalized practice schedule"""
        overall_score = scores.get('overall_score', 50)
        
        if overall_score < 50:
            return {
                'daily': ['10-minute vocal warm-up', 'Confidence-building exercises'],
                'weekly': ['Record full pitch practice', 'Review analysis feedback'],
                'monthly': ['Seek mentor feedback', 'Update pitch content']
            }
        elif overall_score < 70:
            return {
                'daily': ['5-minute vocal practice', 'Presentation skills reading'],
                'weekly': ['Record pitch variations', 'Practice Q&A scenarios'],
                'monthly': ['Present to practice audience', 'Analyze successful pitches']
            }
        else:
            return {
                'daily': ['Voice maintenance exercises'],
                'weekly': ['Advanced presentation techniques practice'],
                'monthly': ['Real investor practice sessions', 'Industry presentation study']
            }
    
    def _generate_milestones(self, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate improvement milestones"""
        current_score = scores.get('overall_score', 50)
        
        milestones = []
        targets = [60, 70, 80, 85, 90]
        
        for target in targets:
            if target > current_score:
                weeks_needed = max(2, int((target - current_score) / 5))
                milestones.append({
                    'target_score': target,
                    'timeline': f"{weeks_needed} weeks",
                    'focus_areas': self._get_focus_areas_for_score(target)
                })
                
        return milestones[:3]  # Return next 3 milestones
    
    def _get_focus_areas_for_score(self, target_score: int) -> List[str]:
        """Get focus areas for achieving target score"""
        if target_score <= 60:
            return ['Basic confidence', 'Voice stability']
        elif target_score <= 70:
            return ['Content structure', 'Emotional expression']
        elif target_score <= 80:
            return ['Delivery refinement', 'Professional presence']
        else:
            return ['Advanced techniques', 'Investor-specific adaptation']
    
    def _generate_industry_comparison(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Generate industry comparison analysis"""
        overall_score = scores.get('overall_score', 50)
        
        if overall_score >= 80:
            comparison = "Top 20% of presenters"
        elif overall_score >= 65:
            comparison = "Above average presenter"
        elif overall_score >= 50:
            comparison = "Average presenter"
        else:
            comparison = "Below average - significant improvement potential"
        
        return {
            'overall_ranking': comparison,
            'voice_confidence': 'Based on voice analysis benchmarks',
            'content_quality': 'Compared to successful pitch patterns'
        }
    
    def _calculate_percentile_ranking(self, scores: Dict[str, float]) -> Dict[str, int]:
        """Calculate percentile rankings"""
        overall_score = scores.get('overall_score', 50)
        
        # Simplified percentile calculation
        percentile = min(99, max(1, int(overall_score * 1.2)))
        
        return {
            'overall_percentile': percentile,
            'voice_confidence_percentile': min(99, max(1, int(scores.get('voice_confidence', 50) * 1.1))),
            'content_quality_percentile': min(99, max(1, int(scores.get('content_quality', 50) * 1.3)))
        }

# Create global comprehensive report generator instance
comprehensive_report_generator = ComprehensiveReportGenerator()
