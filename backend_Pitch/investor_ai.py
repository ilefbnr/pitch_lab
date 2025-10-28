import asyncio
import logging
import json
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class InvestorAI:
    """
    AI-powered virtual investor system that generates realistic questions,
    comments, and feedback based on pitch transcripts and analysis results.
    """
    
    def __init__(self):
        # Different investor personas with unique characteristics
        self.investor_personas = {
            "venture_capitalist": {
                "name": "Sarah Chen",
                "title": "Senior Partner at TechVentures",
                "style": "analytical",
                "focus": ["scalability", "market_size", "team", "traction"],
                "personality": "direct and data-focused"
            },
            "angel_investor": {
                "name": "Michael Rodriguez",
                "title": "Angel Investor & Former Founder",
                "style": "supportive",
                "focus": ["product_market_fit", "founder_passion", "execution"],
                "personality": "encouraging but probing"
            },
            "growth_investor": {
                "name": "Jennifer Kim",
                "title": "Growth Equity Partner",
                "style": "strategic",
                "focus": ["revenue_model", "growth_metrics", "competitive_advantage"],
                "personality": "strategic and forward-thinking"
            },
            "sector_specialist": {
                "name": "David Thompson",
                "title": "Healthcare Tech Specialist",
                "style": "expert",
                "focus": ["regulatory", "market_dynamics", "differentiation"],
                "personality": "deeply knowledgeable and thorough"
            }
        }
        
        # Question templates based on common investor concerns
        self.question_templates = {
            "traction": [
                "What traction have you gained so far?",
                "Can you share your current user metrics?",
                "How many customers are paying for your solution?",
                "What's your month-over-month growth rate?"
            ],
            "market": [
                "How large is your addressable market?",
                "Who are your main competitors?",
                "What's your go-to-market strategy?",
                "How do you plan to capture market share?"
            ],
            "business_model": [
                "How do you make money?",
                "What's your unit economics?",
                "When do you expect to be profitable?",
                "What are your key revenue drivers?"
            ],
            "team": [
                "What's your background in this space?",
                "How did you identify this problem?",
                "What makes your team uniquely qualified?",
                "Do you have domain expertise?"
            ],
            "funding": [
                "How much are you raising?",
                "What will you use the funding for?",
                "What's your burn rate?",
                "When will you need your next round?"
            ],
            "technology": [
                "What's your technology differentiator?",
                "How defensible is your solution?",
                "Do you have any intellectual property?",
                "What are the technical risks?"
            ]
        }
        
        # Feedback templates based on analysis results
        self.feedback_templates = {
            "confidence": {
                "high": [
                    "I appreciate your confident delivery.",
                    "Your certainty is compelling.",
                    "You sound very sure of your vision."
                ],
                "low": [
                    "I'd like to see more conviction in your pitch.",
                    "You seem uncertain about some key points.",
                    "Try to project more confidence in your solution."
                ]
            },
            "clarity": {
                "high": [
                    "Your message is very clear.",
                    "I understand the problem and solution well.",
                    "Good job articulating the value proposition."
                ],
                "low": [
                    "Can you clarify what exactly you do?",
                    "I'm not entirely clear on your solution.",
                    "The value proposition could be clearer."
                ]
            },
            "passion": {
                "high": [
                    "I can feel your passion for this problem.",
                    "Your enthusiasm is infectious.",
                    "It's clear you care deeply about this."
                ],
                "low": [
                    "I'd like to understand what drives you here.",
                    "Tell me more about why this matters to you.",
                    "What's your personal connection to this problem?"
                ]
            }
        }
    
    async def generate_investor_response(
        self, 
        transcript: str, 
        analysis_result: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive investor response based on pitch transcript and analysis
        """
        try:
            # Parse analysis if provided
            analysis = {}
            if analysis_result:
                try:
                    analysis = json.loads(analysis_result)
                except:
                    pass
            
            # Select a random investor persona
            persona_key = random.choice(list(self.investor_personas.keys()))
            persona = self.investor_personas[persona_key]
            
            # Generate response components
            initial_reaction = await self._generate_initial_reaction(transcript, analysis, persona)
            questions = await self._generate_questions(transcript, analysis, persona)
            feedback = await self._generate_feedback(analysis, persona)
            follow_up = await self._generate_follow_up(transcript, persona)
            
            return {
                "investor": persona,
                "timestamp": datetime.now().isoformat(),
                "initial_reaction": initial_reaction,
                "questions": questions,
                "feedback": feedback,
                "follow_up": follow_up,
                "overall_interest": self._calculate_interest_level(analysis),
                "key_concerns": self._identify_concerns(transcript, analysis),
                "suggested_improvements": self._suggest_improvements(analysis)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate investor response: {e}")
            return {
                "error": f"Failed to generate investor response: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _generate_initial_reaction(
        self, 
        transcript: str, 
        analysis: Dict[str, Any], 
        persona: Dict[str, str]
    ) -> str:
        """Generate initial reaction based on investor persona"""
        
        confidence_score = analysis.get('confidence_score', 50)
        overall_grade = analysis.get('overall_grade', 'C')
        
        if persona['style'] == 'analytical':
            if confidence_score >= 80:
                return f"Interesting pitch. I'd like to dive deeper into the numbers and market dynamics."
            else:
                return f"I see potential here, but I need more data to evaluate the opportunity."
        
        elif persona['style'] == 'supportive':
            if confidence_score >= 70:
                return f"I love the energy and passion. This sounds like a compelling opportunity."
            else:
                return f"There's something here. Let's work through some of the details together."
        
        elif persona['style'] == 'strategic':
            return f"This could fit our portfolio thesis. Help me understand the strategic positioning."
        
        else:  # expert
            return f"From a sector perspective, this addresses a real need. Let's explore the specifics."
    
    async def _generate_questions(
        self, 
        transcript: str, 
        analysis: Dict[str, Any], 
        persona: Dict[str, str]
    ) -> List[str]:
        """Generate relevant questions based on investor focus areas"""
        
        questions = []
        focus_areas = persona['focus']
        
        # Select 2-3 questions based on investor focus
        for area in focus_areas[:3]:
            if area in self.question_templates:
                question = random.choice(self.question_templates[area])
                questions.append(question)
        
        # Add analysis-specific questions
        sentiment_analysis = analysis.get('sentiment_analysis', {})
        if sentiment_analysis and not sentiment_analysis.get('error'):
            sentiment = sentiment_analysis.get('overall_sentiment', 'Neutral')
            if sentiment == 'Negative':
                questions.append("I noticed some negative language. What challenges are you facing?")
        
        emotion_analysis = analysis.get('emotion_analysis', {})
        if emotion_analysis and not emotion_analysis.get('error'):
            dominant_emotion = emotion_analysis.get('dominant_emotion', '')
            if dominant_emotion == 'fear':
                questions.append("You seem concerned about something. What are your biggest risks?")
        
        return questions[:4]  # Limit to 4 questions
    
    async def _generate_feedback(
        self, 
        analysis: Dict[str, Any], 
        persona: Dict[str, str]
    ) -> List[str]:
        """Generate constructive feedback based on analysis"""
        
        feedback = []
        
        # Confidence feedback
        confidence_score = analysis.get('confidence_score', 50)
        if confidence_score >= 80:
            feedback.extend(random.sample(self.feedback_templates['confidence']['high'], 1))
        elif confidence_score < 60:
            feedback.extend(random.sample(self.feedback_templates['confidence']['low'], 1))
        
        # Clarity feedback based on readability
        readability = analysis.get('readability_analysis', {})
        if readability and not readability.get('error'):
            flesch_score = readability.get('flesch_reading_ease', 50)
            if 60 <= flesch_score <= 80:
                feedback.extend(random.sample(self.feedback_templates['clarity']['high'], 1))
            elif flesch_score < 40:
                feedback.extend(random.sample(self.feedback_templates['clarity']['low'], 1))
        
        # Emotion-based feedback
        emotion_analysis = analysis.get('emotion_analysis', {})
        if emotion_analysis and not emotion_analysis.get('error'):
            emotion_scores = emotion_analysis.get('emotion_scores', {})
            joy_score = emotion_scores.get('joy', 0)
            if joy_score > 0.3:
                feedback.extend(random.sample(self.feedback_templates['passion']['high'], 1))
            elif joy_score < 0.1:
                feedback.extend(random.sample(self.feedback_templates['passion']['low'], 1))
        
        return feedback
    
    async def _generate_follow_up(
        self, 
        transcript: str, 
        persona: Dict[str, str]
    ) -> str:
        """Generate follow-up action based on investor type"""
        
        if persona['style'] == 'analytical':
            return "Send me your financial projections and market research data."
        elif persona['style'] == 'supportive':
            return "I'd like to schedule a follow-up call to discuss next steps."
        elif persona['style'] == 'strategic':
            return "Let's set up a meeting with our portfolio companies in this space."
        else:  # expert
            return "I'll introduce you to our domain experts for technical due diligence."
    
    def _calculate_interest_level(self, analysis: Dict[str, Any]) -> str:
        """Calculate investor interest level based on analysis"""
        
        confidence_score = analysis.get('confidence_score', 50)
        overall_grade = analysis.get('overall_grade', 'C')
        
        if confidence_score >= 85 and overall_grade in ['A', 'B']:
            return "High Interest"
        elif confidence_score >= 70 and overall_grade in ['B', 'C']:
            return "Moderate Interest"
        elif confidence_score >= 50:
            return "Cautious Interest"
        else:
            return "Low Interest"
    
    def _identify_concerns(
        self, 
        transcript: str, 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify key concerns based on analysis"""
        
        concerns = []
        
        try:
            # Check confidence issues
            confidence_analysis = analysis.get('confidence_analysis', {})
            if confidence_analysis and not confidence_analysis.get('error'):
                confidence_score = confidence_analysis.get('confidence_score', 50)
                if confidence_score < 60:
                    concerns.append("Lack of confidence in delivery")
            
            # Check for too many hedge words
            if confidence_analysis:
                indicators = confidence_analysis.get('confidence_indicators', {})
                hedge_words = indicators.get('hedge_words', 0)
                if hedge_words > 3:
                    concerns.append("Too much uncertain language")
            
            # Check sentiment issues
            sentiment_analysis = analysis.get('sentiment_analysis', {})
            if sentiment_analysis and not sentiment_analysis.get('error'):
                compound = sentiment_analysis.get('sentiment_scores', {}).get('compound', 0)
                if compound < 0:
                    concerns.append("Negative tone throughout pitch")
            
            # Check readability
            readability = analysis.get('readability_analysis', {})
            if readability and not readability.get('error'):
                flesch_score = readability.get('flesch_reading_ease', 50)
                if flesch_score < 30:
                    concerns.append("Message too complex to understand")
            
            # Check video analysis if available
            video_analysis = analysis.get('video_analysis', {})
            if video_analysis:
                business_appropriateness = video_analysis.get('business_appropriateness', 0.5)
                if business_appropriateness < 0.6:
                    concerns.append("Presentation style may not be appropriate for business context")
                
                emotion_stability = video_analysis.get('emotion_stability', 0.5)
                if emotion_stability < 0.5:
                    concerns.append("Inconsistent emotional delivery throughout presentation")
            
        except Exception as e:
            logger.error(f"Error identifying concerns: {e}")
            concerns = ["Unable to analyze specific concerns due to data format"]
        
        # Ensure concerns is a list and limit to top 3
        if isinstance(concerns, list):
            return concerns[:3]
        else:
            return ["Analysis data format issue"]
    
    def _suggest_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest specific improvements based on analysis"""
        
        improvements = []
        
        try:
            # Get ML recommendations if available
            ml_recommendations = analysis.get('recommendations', [])
            if isinstance(ml_recommendations, list) and ml_recommendations:
                # Translate technical recommendations to investor language
                for rec in ml_recommendations[:3]:
                    if isinstance(rec, str):
                        if "filler words" in rec.lower():
                            improvements.append("Practice pausing instead of using filler words")
                        elif "confidence" in rec.lower():
                            improvements.append("Use more assertive language to build credibility")
                        elif "powerful" in rec.lower():
                            improvements.append("Include more compelling business terminology")
                        elif "pace" in rec.lower():
                            improvements.append("Adjust speaking pace for better audience engagement")
            
            # Add investor-specific suggestions
            confidence_score = analysis.get('confidence_score', 50)
            if confidence_score < 70:
                improvements.append("Practice your pitch to build confidence and reduce hesitation")
            
            emotion_analysis = analysis.get('emotion_analysis', {})
            if emotion_analysis and not emotion_analysis.get('error'):
                emotion_scores = emotion_analysis.get('emotion_scores', {})
                if emotion_scores.get('joy', 0) < 0.2:
                    improvements.append("Show more enthusiasm for your solution and market")
            
            # Add video-specific improvements
            video_analysis = analysis.get('video_analysis', {})
            if video_analysis:
                business_appropriateness = video_analysis.get('business_appropriateness', 0.5)
                if business_appropriateness < 0.7:
                    improvements.append("Practice maintaining professional demeanor throughout presentation")
                
                emotion_stability = video_analysis.get('emotion_stability', 0.5)
                if emotion_stability < 0.6:
                    improvements.append("Work on consistent emotional expression and engagement")
            
        except Exception as e:
            logger.error(f"Error suggesting improvements: {e}")
            improvements = ["Focus on clear communication and confident delivery"]
        
        # Ensure improvements is a list and limit to 4
        if isinstance(improvements, list):
            return improvements[:4]
        else:
            return ["Practice your presentation skills"]

# Create global investor AI instance
investor_ai = InvestorAI()
