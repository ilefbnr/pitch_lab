import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { 
  Brain, 
  Volume2, 
  Timer, 
  AlertTriangle, 
  TrendingUp,
  Download,
  Share2,
  Eye
} from "lucide-react";

const FeedbackDisplay = () => {
  const analysisData = {
    overallScore: 84,
    clarity: 88,
    confidence: 79,
    pace: 165, // words per minute
    fillerWords: 12,
    emotionalEngagement: 85,
    duration: "8:32"
  };

  const fillerWordBreakdown = [
    { word: "um", count: 5 },
    { word: "uh", count: 3 },
    { word: "like", count: 2 },
    { word: "you know", count: 2 }
  ];

  const suggestions = [
    {
      type: "improvement",
      icon: TrendingUp,
      title: "Reduce Filler Words",
      description: "Try pausing instead of using filler words. Practice with silent intervals.",
      priority: "high"
    },
    {
      type: "strength",
      icon: Volume2,
      title: "Great Voice Projection",
      description: "Your voice volume and projection were excellent throughout the speech.",
      priority: "positive"
    },
    {
      type: "improvement",
      icon: Timer,
      title: "Speaking Pace",
      description: "Consider slowing down slightly. Aim for 140-160 words per minute.",
      priority: "medium"
    }
  ];

  return (
    <section className="py-20 bg-background">
      <div className="container mx-auto px-6">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6 mb-8">
            <div>
              <h2 className="text-3xl font-bold mb-2">Speech Analysis Results</h2>
              <p className="text-muted-foreground">
                Detailed feedback for "Product Launch Presentation" • {analysisData.duration}
              </p>
            </div>
            
            <div className="flex gap-3">
              <Button variant="outline">
                <Download className="w-4 h-4 mr-2" />
                Export Report
              </Button>
              <Button variant="outline">
                <Share2 className="w-4 h-4 mr-2" />
                Share
              </Button>
            </div>
          </div>

          {/* Overall Score */}
          <Card className="glass-card mb-8">
            <CardContent className="p-8">
              <div className="text-center">
                <div className="relative inline-flex items-center justify-center w-32 h-32 mb-4">
                  <svg className="w-32 h-32 transform -rotate-90">
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="transparent"
                      className="text-muted"
                    />
                    <circle
                      cx="64"
                      cy="64"
                      r="56"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="transparent"
                      strokeDasharray={`${2 * Math.PI * 56}`}
                      strokeDashoffset={`${2 * Math.PI * 56 * (1 - analysisData.overallScore / 100)}`}
                      className="text-primary transition-all duration-1000"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-3xl font-bold text-primary">{analysisData.overallScore}</span>
                  </div>
                </div>
                <h3 className="text-xl font-semibold mb-2">Overall Performance</h3>
                <Badge variant="default" className="text-sm">
                  Excellent
                </Badge>
              </div>
            </CardContent>
          </Card>

          <div className="grid lg:grid-cols-2 gap-8 mb-8">
            {/* Key Metrics */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-primary" />
                  Key Metrics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Speech Clarity</span>
                    <span className="text-sm text-muted-foreground">{analysisData.clarity}%</span>
                  </div>
                  <Progress value={analysisData.clarity} className="h-2" />
                </div>
                
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Vocal Confidence</span>
                    <span className="text-sm text-muted-foreground">{analysisData.confidence}%</span>
                  </div>
                  <Progress value={analysisData.confidence} className="h-2" />
                </div>
                
                <div>
                  <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Emotional Engagement</span>
                    <span className="text-sm text-muted-foreground">{analysisData.emotionalEngagement}%</span>
                  </div>
                  <Progress value={analysisData.emotionalEngagement} className="h-2" />
                </div>
                
                <div className="pt-4 border-t border-border/50">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Speaking Pace</span>
                    <Badge variant="secondary">{analysisData.pace} WPM</Badge>
                  </div>
                </div>
                
                <div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Filler Words</span>
                    <Badge variant="outline">{analysisData.fillerWords} total</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Filler Words Breakdown */}
            <Card className="glass-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-warning" />
                  Filler Words Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {fillerWordBreakdown.map((item, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-background/50 rounded-lg">
                      <span className="font-medium">"{item.word}"</span>
                      <Badge variant="outline">{item.count} times</Badge>
                    </div>
                  ))}
                </div>
                
                <div className="mt-6 p-4 bg-warning-light rounded-lg">
                  <p className="text-sm text-warning-foreground">
                    <strong>Tip:</strong> Try pausing for 1-2 seconds instead of using filler words. 
                    This will make your speech sound more confident and professional.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Suggestions */}
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="w-5 h-5 text-accent" />
                Personalized Recommendations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4">
                {suggestions.map((suggestion, index) => (
                  <div key={index} className="flex items-start gap-4 p-4 bg-background/50 rounded-lg hover:bg-background/80 transition-colors">
                    <div className={`p-2 rounded-lg ${
                      suggestion.priority === "positive" ? "bg-success/10" :
                      suggestion.priority === "high" ? "bg-warning/10" : "bg-primary/10"
                    }`}>
                      <suggestion.icon className={`w-5 h-5 ${
                        suggestion.priority === "positive" ? "text-success" :
                        suggestion.priority === "high" ? "text-warning" : "text-primary"
                      }`} />
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium mb-1">{suggestion.title}</h4>
                      <p className="text-sm text-muted-foreground">{suggestion.description}</p>
                    </div>
                    <Badge variant={
                      suggestion.priority === "positive" ? "default" :
                      suggestion.priority === "high" ? "destructive" : "secondary"
                    }>
                      {suggestion.priority === "positive" ? "Strength" : "Improve"}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default FeedbackDisplay;