import React from 'react';
import './LiveMetricsChart.css';

interface LiveMetricsChartProps {
  confidenceTrend: number[];
  emotionTrend: Array<{emotion: string; timestamp: string}>;
  volumeLevel: number;
  speakingPace: number;
  pitchVariation: number;
}

const LiveMetricsChart: React.FC<LiveMetricsChartProps> = ({
  confidenceTrend,
  emotionTrend,
  volumeLevel,
  speakingPace,
  pitchVariation
}) => {
  
  // Create confidence trend visualization
  const renderConfidenceTrend = () => {
    const maxPoints = 20;
    const points = confidenceTrend.slice(-maxPoints);
    const width = 300;
    const height = 100;
    
    if (points.length < 2) {
      return (
        <div className="chart-placeholder">
          <span>Collecting data...</span>
        </div>
      );
    }
    
    const pathData = points.map((value, index) => {
      const x = (index / (points.length - 1)) * width;
      const y = height - (value / 100) * height;
      return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
    
    return (
      <svg width={width} height={height} className="confidence-chart">
        <defs>
          <linearGradient id="confidenceGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#667eea" stopOpacity="0.8"/>
            <stop offset="100%" stopColor="#667eea" stopOpacity="0.2"/>
          </linearGradient>
        </defs>
        
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map(line => (
          <line
            key={line}
            x1="0"
            y1={height - (line / 100) * height}
            x2={width}
            y2={height - (line / 100) * height}
            stroke="#e1e8ed"
            strokeWidth="1"
            strokeDasharray="2,2"
          />
        ))}
        
        {/* Confidence line */}
        <path
          d={pathData}
          fill="none"
          stroke="#667eea"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        
        {/* Fill area under curve */}
        <path
          d={`${pathData} L ${width} ${height} L 0 ${height} Z`}
          fill="url(#confidenceGradient)"
        />
        
        {/* Data points */}
        {points.map((value, index) => {
          const x = (index / (points.length - 1)) * width;
          const y = height - (value / 100) * height;
          return (
            <circle
              key={index}
              cx={x}
              cy={y}
              r="4"
              fill="#667eea"
              className="data-point"
            />
          );
        })}
      </svg>
    );
  };
  
  // Volume level indicator
  const renderVolumeIndicator = () => {
    const normalizedVolume = Math.min(volumeLevel, 100);
    const barHeight = (normalizedVolume / 100) * 80;
    
    return (
      <div className="volume-indicator">
        <div className="volume-bar-container">
          <div 
            className="volume-bar"
            style={{ 
              height: `${barHeight}%`,
              backgroundColor: getVolumeColor(normalizedVolume)
            }}
          />
          <div className="volume-scale">
            {[100, 75, 50, 25, 0].map(mark => (
              <div key={mark} className="scale-mark">
                <span>{mark}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="volume-value">{normalizedVolume.toFixed(1)}%</div>
      </div>
    );
  };
  
  // Speaking pace gauge
  const renderSpeakingPaceGauge = () => {
    const minWPM = 100;
    const maxWPM = 200;
    const optimalMin = 140;
    const optimalMax = 180;
    
    const normalizedPace = Math.max(minWPM, Math.min(maxWPM, speakingPace));
    const angle = ((normalizedPace - minWPM) / (maxWPM - minWPM)) * 180 - 90;
    
    const isOptimal = speakingPace >= optimalMin && speakingPace <= optimalMax;
    
    return (
      <div className="pace-gauge">
        <svg width="120" height="80" viewBox="0 0 120 80">
          {/* Background arc */}
          <path
            d="M 20 60 A 40 40 0 0 1 100 60"
            fill="none"
            stroke="#e1e8ed"
            strokeWidth="8"
            strokeLinecap="round"
          />
          
          {/* Optimal range arc */}
          <path
            d="M 35 70 A 40 40 0 0 1 85 70"
            fill="none"
            stroke="#4ecdc4"
            strokeWidth="6"
            strokeOpacity="0.3"
            strokeLinecap="round"
          />
          
          {/* Current pace indicator */}
          <line
            x1="60"
            y1="60"
            x2={60 + 35 * Math.cos(angle * Math.PI / 180)}
            y2={60 + 35 * Math.sin(angle * Math.PI / 180)}
            stroke={isOptimal ? "#4ecdc4" : "#ff6b6b"}
            strokeWidth="3"
            strokeLinecap="round"
          />
          
          {/* Center dot */}
          <circle cx="60" cy="60" r="4" fill="#2c3e50" />
        </svg>
        
        <div className="pace-value">
          {speakingPace > 0 ? `${speakingPace.toFixed(0)} WPM` : 'Calculating...'}
        </div>
        <div className={`pace-status ${isOptimal ? 'optimal' : 'needs-adjustment'}`}>
          {isOptimal ? 'Optimal' : speakingPace < optimalMin ? 'Too Slow' : 'Too Fast'}
        </div>
      </div>
    );
  };
  
  // Emotion timeline
  const renderEmotionTimeline = () => {
    const recentEmotions = emotionTrend.slice(-10);
    
    if (recentEmotions.length === 0) {
      return (
        <div className="emotion-placeholder">
          <span>No emotion data yet</span>
        </div>
      );
    }
    
    const emotionEmojis: { [key: string]: string } = {
      joy: '😊',
      confidence: '😎',
      optimism: '🙂',
      surprise: '😲',
      fear: '😰',
      sadness: '😢',
      anger: '😠',
      neutral: '😐'
    };
    
    const emotionColors: { [key: string]: string } = {
      joy: '#4ecdc4',
      confidence: '#667eea',
      optimism: '#45b7d1',
      surprise: '#f39c12',
      fear: '#e74c3c',
      sadness: '#9b59b6',
      anger: '#e74c3c',
      neutral: '#95a5a6'
    };
    
    return (
      <div className="emotion-timeline">
        {recentEmotions.map((emotion, index) => (
          <div 
            key={index} 
            className="emotion-point"
            style={{ backgroundColor: emotionColors[emotion.emotion] || '#95a5a6' }}
            title={`${emotion.emotion} at ${new Date(emotion.timestamp).toLocaleTimeString()}`}
          >
            <span className="emotion-emoji">
              {emotionEmojis[emotion.emotion] || '😐'}
            </span>
          </div>
        ))}
      </div>
    );
  };
  
  // Pitch variation indicator
  const renderPitchVariation = () => {
    const variationLevel = Math.min(pitchVariation * 1000, 100); // Scale for display
    const bars = Array.from({ length: 10 }, (_, i) => {
      const height = Math.random() * variationLevel + 10; // Simulate pitch variation
      return height;
    });
    
    return (
      <div className="pitch-variation">
        <div className="pitch-bars">
          {bars.map((height, index) => (
            <div
              key={index}
              className="pitch-bar"
              style={{
                height: `${height}%`,
                backgroundColor: `hsl(${200 + height}, 70%, 60%)`
              }}
            />
          ))}
        </div>
        <div className="pitch-value">
          Variation: {(pitchVariation * 1000).toFixed(1)}
        </div>
      </div>
    );
  };
  
  const getVolumeColor = (volume: number) => {
    if (volume < 20) return '#e74c3c';
    if (volume < 40) return '#f39c12';
    if (volume < 70) return '#4ecdc4';
    return '#27ae60';
  };
  
  return (
    <div className="live-metrics-chart">
      <div className="chart-grid">
        {/* Confidence Trend */}
        <div className="chart-section">
          <h4>📈 Confidence Trend</h4>
          <div className="chart-container">
            {renderConfidenceTrend()}
          </div>
        </div>
        
        {/* Volume Level */}
        <div className="chart-section">
          <h4>🔊 Volume Level</h4>
          <div className="chart-container">
            {renderVolumeIndicator()}
          </div>
        </div>
        
        {/* Speaking Pace */}
        <div className="chart-section">
          <h4>⚡ Speaking Pace</h4>
          <div className="chart-container">
            {renderSpeakingPaceGauge()}
          </div>
        </div>
        
        {/* Emotion Timeline */}
        <div className="chart-section">
          <h4>😊 Emotion Timeline</h4>
          <div className="chart-container">
            {renderEmotionTimeline()}
          </div>
        </div>
        
        {/* Pitch Variation */}
        <div className="chart-section">
          <h4>🎵 Pitch Variation</h4>
          <div className="chart-container">
            {renderPitchVariation()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveMetricsChart;
