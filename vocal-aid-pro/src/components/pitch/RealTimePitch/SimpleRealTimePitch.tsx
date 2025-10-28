import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './SimpleRealTimePitch.css';

interface Pitch {
  id: number;
  title: string;
  description: string;
  transcript: string;
  created_at: string;
  user_id: number;
  analysis_result?: string;
}

const SimpleRealTimePitch: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [pitchTitle, setPitchTitle] = useState('');
  const [recordingDuration, setRecordingDuration] = useState(0);

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Start recording
  const startRecording = async () => {
    try {
      setError(null);
      setCurrentTranscript('');
      setAnalysisResult(null);
      setRecordingDuration(0);
      
      // Get media stream
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      });
      
      streamRef.current = stream;

      // Setup MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      const audioChunks: Blob[] = [];

      // Collect audio data
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      // Handle recording stop
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await uploadAndAnalyze(audioBlob);
      };

      // Start recording
      mediaRecorder.start();
      setIsRecording(true);

      // Start duration counter
      intervalRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('Error starting recording:', error);
      setError('Failed to start recording. Please check microphone permissions.');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    setIsRecording(false);
  };

  // Upload and analyze audio - REMOVED AUTHENTICATION
  const uploadAndAnalyze = async (audioBlob: Blob) => {
    try {
      setError(null);
      
      if (!pitchTitle.trim()) {
        setPitchTitle(`Pitch ${new Date().toLocaleString()}`);
      }

      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'realtime-recording.wav');
      formData.append('title', pitchTitle || `Pitch ${new Date().toLocaleString()}`);
      formData.append('description', 'Real-time analysis recording');

      // Removed Authorization header - no token needed
      const response = await axios.post<Pitch>('http://localhost:8050/pitches', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      const pitch = response.data;
      setCurrentTranscript(pitch.transcript);
      
      // Parse analysis result if available
      if (pitch.analysis_result) {
        try {
          const analysis = JSON.parse(pitch.analysis_result);
          setAnalysisResult(analysis);
        } catch (e) {
          console.warn('Could not parse analysis result:', e);
        }
      }

    } catch (error) {
      console.error('Error uploading pitch:', error);
      setError('Failed to upload and analyze recording');
    }
  };

  // Format duration
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get confidence color
  const getConfidenceColor = (score: number) => {
    if (score >= 80) return '#28a745'; // green
    if (score >= 60) return '#ffc107'; // yellow
    return '#dc3545'; // red
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  return (
    <div className="realtime-pitch">
      <h2>Simple Real-Time Analysis</h2>
      <p>Record your pitch and get instant analysis using our traditional API!</p>
      
      {/* Error Display */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Title Input */}
      <div className="title-input">
        <label htmlFor="pitchTitle">Pitch Title:</label>
        <input
          type="text"
          id="pitchTitle"
          value={pitchTitle}
          onChange={(e) => setPitchTitle(e.target.value)}
          placeholder="Enter pitch title"
          disabled={isRecording}
        />
      </div>

      {/* Recording Controls */}
      <div className="recording-controls">
        {!isRecording ? (
          <button 
            className="start-button" 
            onClick={startRecording}
          >
            🎤 Start Recording
          </button>
        ) : (
          <div className="recording-actions">
            <div className="recording-status">
              <span className="recording-indicator">🔴</span>
              <span>Recording: {formatDuration(recordingDuration)}</span>
            </div>
            <button className="stop-button" onClick={stopRecording}>
              ⏹️ Stop & Analyze
            </button>
          </div>
        )}
      </div>

      {/* Live Recording Status */}
      {isRecording && (
        <div className="live-status">
          <h3>🎙️ Recording in Progress</h3>
          <p>Speak clearly into your microphone. Click "Stop & Analyze" when finished.</p>
          <div className="recording-visualization">
            <div className="pulse-animation"></div>
          </div>
        </div>
      )}

      {/* Transcript Results */}
      {currentTranscript && (
        <div className="transcript-results">
          <h3>📝 Transcript</h3>
          <div className="transcript-text">
            {currentTranscript}
          </div>
        </div>
      )}

      {/* Analysis Results */}
      {analysisResult && (
        <div className="analysis-results">
          <h3>📊 Analysis Results</h3>
          
          {analysisResult.error ? (
            <div className="analysis-error">
              <p>Analysis Error: {analysisResult.error}</p>
            </div>
          ) : (
            <div className="analysis-metrics">
              <div className="metric-card">
                <div className="metric-icon">🎯</div>
                <div className="metric-value" style={{ 
                  color: getConfidenceColor(analysisResult.confidence_score || 0) 
                }}>
                  {analysisResult.confidence_score || 0}%
                </div>
                <div className="metric-label">Confidence Score</div>
              </div>
              
              <div className="metric-card">
                <div className="metric-icon">📈</div>
                <div className="metric-value">{analysisResult.overall_grade || 'N/A'}</div>
                <div className="metric-label">Overall Grade</div>
              </div>
              
              {analysisResult.speaking_pace && (
                <div className="metric-card">
                  <div className="metric-icon">⚡</div>
                  <div className="metric-value">{analysisResult.speaking_pace} WPM</div>
                  <div className="metric-label">Speaking Pace</div>
                </div>
              )}
              
              {analysisResult.emotion_analysis && (
                <div className="metric-card">
                  <div className="metric-icon">😊</div>
                  <div className="metric-value">{analysisResult.emotion_analysis.dominant_emotion || 'neutral'}</div>
                  <div className="metric-label">Dominant Emotion</div>
                </div>
              )}
            </div>
          )}
          
          {analysisResult.feedback && Array.isArray(analysisResult.feedback) && (
            <div className="feedback-section">
              <h4>💡 Feedback</h4>
              <ul>
                {analysisResult.feedback.map((feedback: string, index: number) => (
                  <li key={index}>{feedback}</li>
                ))}
              </ul>
            </div>
          )}
          
          {analysisResult.suggestions && Array.isArray(analysisResult.suggestions) && (
            <div className="suggestions-section">
              <h4>🚀 Suggestions</h4>
              <ul>
                {analysisResult.suggestions.map((suggestion: string, index: number) => (
                  <li key={index}>{suggestion}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SimpleRealTimePitch;