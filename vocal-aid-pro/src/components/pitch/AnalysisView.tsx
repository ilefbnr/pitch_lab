import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './AnalysisView.css';

interface AnalysisResult {
  // Audio-based analysis structure
  basic_stats?: {
    word_count: number;
    sentence_count: number;
    avg_words_per_sentence: number;
    vocabulary_richness: number;
  };
  emotion_analysis?: {
    emotion_scores: Record<string, number>;
    dominant_emotion: string;
    emotional_stability: string;
    pitch_appropriateness: string;
    error?: string;
  };
  confidence_analysis?: {
    confidence_score: number;
    assessment: string;
    error?: string;
  };
  sentiment_analysis?: {
    overall_sentiment: string;
    positivity_ratio: number;
    pitch_sentiment_assessment: string;
    error?: string;
  };
  pitch_similarity?: {
    max_similarity: number;
    avg_similarity: number;
    similarity_assessment: string;
    error?: string;
  };
  readability_analysis?: {
    flesch_reading_ease: number;
    flesch_kincaid_grade: number;
    readability_assessment: string;
    error?: string;
  };
  speaking_pace?: {
    words_per_minute: number;
    assessment: string;
  };
  confidence_score?: number;
  overall_grade?: string;
  recommendations?: string[];
  
  // Video-only analysis structure
  content_type?: string;
  analysis_summary?: {
    title: string;
    description: string;
    focus_areas: string[];
  };
  video_analysis?: {
    dominant_emotion?: string;
    emotion_stability?: number;
    business_appropriateness?: number;
    average_confidence?: number;
    recommendations?: string[];
  };
  scoring?: {
    emotional_engagement?: number;
    visual_confidence?: number;
    overall_grade?: string;
    audio_confidence?: number;
  };
  
  // Multimodal analysis structure
  audio_analysis?: {
    confidence_score?: number;
    overall_grade?: string;
    speaking_pace?: {
      words_per_minute: number;
      assessment: string;
    };
    basic_stats?: {
      word_count: number;
      sentence_count: number;
      avg_words_per_sentence: number;
      vocabulary_richness: number;
    };
  };
  enhanced_voice_analysis?: {
    business_score?: number;
    emotion_scores?: Record<string, number>;
  };
  
  error?: string;
}

interface InvestorResponse {
  investor?: {
    name: string;
    title: string;
    style: string;
    personality: string;
  };
  initial_reaction?: string;
  questions?: string[];
  feedback?: string[];
  follow_up?: string;
  overall_interest?: string;
  key_concerns?: string[];
  suggested_improvements?: string[];
  timestamp?: string;
  error?: string;
}

interface AnalysisViewProps {
  pitchId: number;
  onClose: () => void;
}

const AnalysisView: React.FC<AnalysisViewProps> = ({ pitchId, onClose }) => {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [investorResponse, setInvestorResponse] = useState<InvestorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingInvestor, setLoadingInvestor] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'analysis' | 'investor'>('analysis');

  const fetchAnalysis = useCallback(async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`http://localhost:8050/pitches/${pitchId}/analysis`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setAnalysis(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load analysis');
    } finally {
      setLoading(false);
    }
  }, [pitchId]);

  useEffect(() => {
    fetchAnalysis();
  }, [fetchAnalysis]);

  const generateInvestorResponse = async () => {
    setLoadingInvestor(true);
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(
        `http://localhost:8050/pitches/${pitchId}/investor-response`,
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );
      console.log('Investor response data:', response.data);
      setInvestorResponse(response.data);
      setActiveTab('investor');
    } catch (err: any) {
      console.error('Investor response error:', err);
      setError(err.response?.data?.detail || 'Failed to generate investor response');
    } finally {
      setLoadingInvestor(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return '#4CAF50';
    if (score >= 60) return '#FF9800';
    return '#F44336';
  };

  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'A': return '#4CAF50';
      case 'B': return '#8BC34A';
      case 'C': return '#FF9800';
      case 'D': return '#FF5722';
      default: return '#F44336';
    }
  };

  if (loading) {
    return (
      <div className="analysis-overlay">
        <div className="analysis-modal">
          <div className="loading">Loading analysis...</div>
        </div>
      </div>
    );
  }

  if (error && !analysis) {
    return (
      <div className="analysis-overlay">
        <div className="analysis-modal">
          <div className="analysis-header">
            <h2>Analysis Error</h2>
            <button onClick={onClose} className="close-btn">&times;</button>
          </div>
          <div className="error">{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="analysis-overlay">
      <div className="analysis-modal">
        <div className="analysis-header">
          <h2>Pitch Analysis</h2>
          <button onClick={onClose} className="close-btn">&times;</button>
        </div>

        <div className="analysis-tabs">
          <button
            className={`tab ${activeTab === 'analysis' ? 'active' : ''}`}
            onClick={() => setActiveTab('analysis')}
          >
            AI Analysis
          </button>
          <button
            className={`tab ${activeTab === 'investor' ? 'active' : ''}`}
            onClick={() => setActiveTab('investor')}
          >
            Investor Response
          </button>
        </div>

        {activeTab === 'analysis' && analysis && (
          <div className="analysis-content">
            {analysis.error ? (
              <div className="error">{analysis.error}</div>
            ) : (
              <>
                {/* Content Type Header */}
                {analysis.content_type && (
                  <div className="content-type-header">
                    <h3>{analysis.analysis_summary?.title || 'Analysis Results'}</h3>
                    <p>{analysis.analysis_summary?.description || ''}</p>
                  </div>
                )}

                {/* Overall Score */}
                <div className="score-section">
                  <div className="overall-score">
                    <h3>Overall Performance</h3>
                    <div className="score-display">
                      <div 
                        className="score-circle"
                        style={{ 
                          borderColor: getScoreColor(
                            analysis.content_type === 'multimodal'
                              ? (analysis.audio_analysis?.confidence_score || analysis.scoring?.audio_confidence || analysis.scoring?.visual_confidence || 0)
                              : (analysis.confidence_score || analysis.scoring?.emotional_engagement || 0)
                          ) 
                        }}
                      >
                        <span style={{ 
                          color: getScoreColor(
                            analysis.content_type === 'multimodal'
                              ? (analysis.audio_analysis?.confidence_score || analysis.scoring?.audio_confidence || analysis.scoring?.visual_confidence || 0)
                              : (analysis.confidence_score || analysis.scoring?.emotional_engagement || 0)
                          ) 
                        }}>
                          {analysis.content_type === 'multimodal'
                            ? (analysis.audio_analysis?.confidence_score || analysis.scoring?.audio_confidence || analysis.scoring?.visual_confidence || 'N/A')
                            : (analysis.confidence_score || analysis.scoring?.emotional_engagement || 'N/A')
                          }
                        </span>
                      </div>
                      <div 
                        className="grade-badge"
                        style={{ 
                          backgroundColor: getGradeColor(
                            analysis.content_type === 'multimodal'
                              ? (analysis.scoring?.overall_grade || analysis.audio_analysis?.overall_grade || 'N/A')
                              : (analysis.overall_grade || analysis.scoring?.overall_grade || 'N/A')
                          ) 
                        }}
                      >
                        {analysis.content_type === 'multimodal'
                          ? (analysis.scoring?.overall_grade || analysis.audio_analysis?.overall_grade || 'N/A')
                          : (analysis.overall_grade || analysis.scoring?.overall_grade || 'N/A')
                        }
                      </div>
                    </div>
                  </div>
                </div>

                {/* Key Metrics */}
                <div className="metrics-grid">
                  <div className="metric-card">
                    <h4>Confidence Level</h4>
                    <div className="metric-value">
                      {analysis.content_type === 'multimodal' 
                        ? (analysis.audio_analysis?.confidence_score || analysis.scoring?.audio_confidence || analysis.scoring?.visual_confidence || 'N/A')
                        : (analysis.confidence_analysis?.confidence_score || analysis.scoring?.visual_confidence || 'N/A')
                      }
                    </div>
                    <div className="metric-label">
                      {analysis.content_type === 'multimodal' 
                        ? 'Audio confidence score'
                        : (analysis.confidence_analysis?.assessment || 'Visual confidence score')
                      }
                    </div>
                  </div>

                  <div className="metric-card">
                    <h4>Emotional Tone</h4>
                    <div className="metric-value">
                      {analysis.content_type === 'multimodal'
                        ? (analysis.video_analysis?.dominant_emotion || analysis.enhanced_voice_analysis?.emotion_scores?.confidence || 'N/A')
                        : (analysis.emotion_analysis?.dominant_emotion || analysis.video_analysis?.dominant_emotion || 'N/A')
                      }
                    </div>
                    <div className="metric-label">
                      {analysis.content_type === 'multimodal'
                        ? 'Emotional engagement'
                        : (analysis.emotion_analysis?.pitch_appropriateness || 'Emotional engagement')
                      }
                    </div>
                  </div>

                  <div className="metric-card">
                    <h4>Clarity</h4>
                    <div className="metric-value">
                      {analysis.content_type === 'multimodal'
                        ? (analysis.audio_analysis?.basic_stats?.vocabulary_richness ? (analysis.audio_analysis.basic_stats.vocabulary_richness * 100).toFixed(1) + '%' : 'N/A')
                        : (analysis.readability_analysis?.flesch_reading_ease?.toFixed(1) || 'N/A')
                      }
                    </div>
                    <div className="metric-label">
                      {analysis.content_type === 'multimodal'
                        ? 'Vocabulary richness'
                        : (analysis.readability_analysis?.readability_assessment || '')
                      }
                    </div>
                  </div>

                  <div className="metric-card">
                    <h4>Speaking Pace</h4>
                    <div className="metric-value">
                      {analysis.content_type === 'multimodal'
                        ? (analysis.audio_analysis?.speaking_pace?.words_per_minute || 'N/A') + ' WPM'
                        : (analysis.speaking_pace?.words_per_minute || 'N/A') + ' WPM'
                      }
                    </div>
                    <div className="metric-label">
                      {analysis.content_type === 'multimodal'
                        ? (analysis.audio_analysis?.speaking_pace?.assessment || '')
                        : (analysis.speaking_pace?.assessment || '')
                      }
                    </div>
                  </div>
                </div>

                {/* Detailed Analysis */}
                <div className="detailed-analysis">
                  <h3>Detailed Insights</h3>

                  {/* Video Analysis Section */}
                  {analysis.video_analysis && (
                    <div className="analysis-section">
                      <h4>Video Analysis</h4>
                      <p><strong>Dominant Emotion:</strong> {analysis.video_analysis.dominant_emotion || 'N/A'}</p>
                      <p><strong>Emotional Stability:</strong> {analysis.video_analysis.emotion_stability ? (analysis.video_analysis.emotion_stability * 100).toFixed(1) + '%' : 'N/A'}</p>
                      <p><strong>Business Appropriateness:</strong> {analysis.video_analysis.business_appropriateness ? (analysis.video_analysis.business_appropriateness * 100).toFixed(1) + '%' : 'N/A'}</p>
                      <p><strong>Average Confidence:</strong> {analysis.video_analysis.average_confidence ? (analysis.video_analysis.average_confidence * 100).toFixed(1) + '%' : 'N/A'}</p>
                    </div>
                  )}

                  {analysis.sentiment_analysis && !analysis.sentiment_analysis.error && (
                    <div className="analysis-section">
                      <h4>Sentiment Analysis</h4>
                      <p><strong>Overall Sentiment:</strong> {analysis.sentiment_analysis.overall_sentiment}</p>
                      <p><strong>Assessment:</strong> {analysis.sentiment_analysis.pitch_sentiment_assessment}</p>
                    </div>
                  )}

                  {analysis.pitch_similarity && !analysis.pitch_similarity.error && (
                    <div className="analysis-section">
                      <h4>Pitch Structure</h4>
                      <p><strong>Similarity to Successful Pitches:</strong> {(analysis.pitch_similarity.max_similarity * 100).toFixed(1)}%</p>
                      <p><strong>Assessment:</strong> {analysis.pitch_similarity.similarity_assessment}</p>
                    </div>
                  )}

                  {/* Basic Statistics - show for audio-based or multimodal analysis */}
                  {(analysis.basic_stats || (analysis.content_type === 'multimodal' && analysis.audio_analysis?.basic_stats)) && (
                    <div className="analysis-section">
                      <h4>Basic Statistics</h4>
                      <div className="stats-grid">
                        <div><strong>Words:</strong> {analysis.content_type === 'multimodal' ? (analysis.audio_analysis?.basic_stats?.word_count || 'N/A') : (analysis.basic_stats?.word_count || 'N/A')}</div>
                        <div><strong>Sentences:</strong> {analysis.content_type === 'multimodal' ? (analysis.audio_analysis?.basic_stats?.sentence_count || 'N/A') : (analysis.basic_stats?.sentence_count || 'N/A')}</div>
                        <div><strong>Avg Words/Sentence:</strong> {analysis.content_type === 'multimodal' ? (analysis.audio_analysis?.basic_stats?.avg_words_per_sentence || 'N/A') : (analysis.basic_stats?.avg_words_per_sentence || 'N/A')}</div>
                        <div><strong>Vocabulary Richness:</strong> {analysis.content_type === 'multimodal' ? (analysis.audio_analysis?.basic_stats?.vocabulary_richness ? (analysis.audio_analysis.basic_stats.vocabulary_richness * 100).toFixed(1) + '%' : 'N/A') : (analysis.basic_stats?.vocabulary_richness ? (analysis.basic_stats.vocabulary_richness * 100).toFixed(1) + '%' : 'N/A')}</div>
                      </div>
                    </div>
                  )}

                  {/* Focus Areas for Video Analysis */}
                  {analysis.analysis_summary?.focus_areas && (
                    <div className="analysis-section">
                      <h4>Focus Areas</h4>
                      <ul>
                        {analysis.analysis_summary.focus_areas.map((area, index) => (
                          <li key={index}>{area}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Recommendations */}
                {analysis.recommendations && analysis.recommendations.length > 0 && (
                  <div className="recommendations">
                    <h3>AI Recommendations</h3>
                    <ul>
                      {analysis.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Video Analysis Recommendations */}
                {analysis.video_analysis?.recommendations && analysis.video_analysis.recommendations.length > 0 && (
                  <div className="recommendations">
                    <h3>Video Analysis Recommendations</h3>
                    <ul>
                      {analysis.video_analysis.recommendations.map((rec, index) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {activeTab === 'investor' && (
          <div className="investor-content">
            {!investorResponse ? (
              <div className="generate-investor">
                <h3>Virtual Investor Feedback</h3>
                <p>Get realistic feedback from our AI investor panel based on your pitch analysis.</p>
                <button
                  onClick={generateInvestorResponse}
                  disabled={loadingInvestor}
                  className="generate-btn"
                >
                  {loadingInvestor ? 'Generating...' : 'Generate Investor Response'}
                </button>
              </div>
            ) : (
              <div className="investor-response">
                {investorResponse.error ? (
                  <div className="error">
                    <h3>Error Generating Investor Response</h3>
                    <p>{investorResponse.error}</p>
                    <button
                      onClick={generateInvestorResponse}
                      disabled={loadingInvestor}
                      className="generate-btn"
                    >
                      {loadingInvestor ? 'Retrying...' : 'Try Again'}
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="investor-info">
                      <h3>{investorResponse.investor?.name || 'Virtual Investor'}</h3>
                      <p className="investor-title">{investorResponse.investor?.title || 'AI Investor'}</p>
                      <p className="investor-style">Style: {investorResponse.investor?.style || 'N/A'} • {investorResponse.investor?.personality || 'N/A'}</p>
                    </div>

                    <div className="interest-level">
                      <h4>Interest Level: <span className={`interest ${(investorResponse.overall_interest || 'neutral').toLowerCase().replace(' ', '-')}`}>
                        {investorResponse.overall_interest || 'Neutral'}
                      </span></h4>
                    </div>

                    <div className="response-section">
                      <h4>Initial Reaction</h4>
                      <p className="reaction">{investorResponse.initial_reaction || 'No reaction available'}</p>
                    </div>

                    <div className="response-section">
                      <h4>Questions</h4>
                      <ul className="questions-list">
                        {investorResponse.questions && investorResponse.questions.length > 0 ? (
                          investorResponse.questions.map((question, index) => (
                            <li key={index}>{question}</li>
                          ))
                        ) : (
                          <li>No specific questions at this time.</li>
                        )}
                      </ul>
                    </div>

                    {investorResponse.feedback && investorResponse.feedback.length > 0 && (
                      <div className="response-section">
                        <h4>Feedback</h4>
                        <ul className="feedback-list">
                          {investorResponse.feedback.map((feedback, index) => (
                            <li key={index}>{feedback}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {investorResponse.key_concerns && investorResponse.key_concerns.length > 0 && (
                      <div className="response-section concerns">
                        <h4>Key Concerns</h4>
                        <ul>
                          {investorResponse.key_concerns.map((concern, index) => (
                            <li key={index}>{concern}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {investorResponse.suggested_improvements && investorResponse.suggested_improvements.length > 0 && (
                      <div className="response-section improvements">
                        <h4>Suggested Improvements</h4>
                        <ul>
                          {investorResponse.suggested_improvements.map((improvement, index) => (
                            <li key={index}>{improvement}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="response-section">
                      <h4>Next Steps</h4>
                      <p className="follow-up">{investorResponse.follow_up || 'No specific next steps provided.'}</p>
                    </div>

                    <button
                      onClick={generateInvestorResponse}
                      disabled={loadingInvestor}
                      className="regenerate-btn"
                    >
                      {loadingInvestor ? 'Generating...' : 'Generate New Response'}
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisView;
