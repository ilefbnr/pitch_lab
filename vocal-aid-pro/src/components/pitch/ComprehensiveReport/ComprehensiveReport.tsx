import React, { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BarChart3, Brain, Mic, Smile, Clock, TrendingUp, ChevronLeft, Sparkles } from "lucide-react";
import axios from "axios";

interface ComprehensiveReportProps {
  pitchId: number;
  onClose: () => void;
}

interface AnalysisData {
  confidence_score?: number;
  clarity_score?: number;
  emotion_analysis?: { dominant_emotion?: string };
  speaking_pace?: number;
  feedback?: string[];
  suggestions?: string[];
  overall_grade?: string;
}

const ComprehensiveReport: React.FC<ComprehensiveReportProps> = ({ pitchId, onClose }) => {
  const [report, setReport] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const response = await axios.get(`http://localhost:8001/pitches/${pitchId}/report`);
        setReport(response.data);
      } catch (err) {
        console.error("Error loading report:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchReport();
  }, [pitchId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh] text-muted-foreground">
        <Sparkles className="animate-spin mr-2" /> Generating your report...
      </div>
    );
  }

  if (!report) {
    return <div className="text-center text-muted-foreground py-10">❌ No report available for this pitch.</div>;
  }

  return (
    <section className="py-10 px-4">
      <div className="max-w-5xl mx-auto space-y-8">
        <div className="flex justify-start">
          <Button variant="outline" className="flex items-center gap-2" onClick={onClose}>
            <ChevronLeft className="w-4 h-4" />
            Back to Dashboard
          </Button>
        </div>

        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Comprehensive Analysis Report
          </h2>
          <p className="text-muted-foreground text-sm">Deep insights into your vocal delivery, pacing, and emotional tone</p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card className="glass-card p-6 text-center">
            <CardHeader>
              <Mic className="w-8 h-8 text-primary mx-auto mb-2" />
              <CardTitle>Confidence</CardTitle>
            </CardHeader>
            <CardContent>
              <Progress value={report.confidence_score || 0} className="mb-3" />
              <p className="text-2xl font-semibold">{report.confidence_score ? `${report.confidence_score}%` : "N/A"}</p>
            </CardContent>
          </Card>

          <Card className="glass-card p-6 text-center">
            <CardHeader>
              <Brain className="w-8 h-8 text-accent mx-auto mb-2" />
              <CardTitle>Clarity</CardTitle>
            </CardHeader>
            <CardContent>
              <Progress value={report.clarity_score || 0} className="mb-3" />
              <p className="text-2xl font-semibold">{report.clarity_score ? `${report.clarity_score}%` : "N/A"}</p>
            </CardContent>
          </Card>

          <Card className="glass-card p-6 text-center">
            <CardHeader>
              <Smile className="w-8 h-8 text-success mx-auto mb-2" />
              <CardTitle>Dominant Emotion</CardTitle>
            </CardHeader>
            <CardContent>
              <Badge variant="outline" className="px-3 py-1 text-base">
                {report.emotion_analysis?.dominant_emotion || "Neutral"}
              </Badge>
            </CardContent>
          </Card>
        </div>

        <div className="grid sm:grid-cols-2 gap-6">
          <Card className="glass-card p-6">
            <CardHeader className="flex items-center gap-2">
              <Clock className="text-primary w-5 h-5" />
              <CardTitle>Speaking Pace</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{report.speaking_pace ? `${report.speaking_pace} WPM` : "Not measured"}</p>
              <p className="text-muted-foreground text-sm">Ideal range: 130–160 words per minute</p>
            </CardContent>
          </Card>

          <Card className="glass-card p-6">
            <CardHeader className="flex items-center gap-2">
              <TrendingUp className="text-accent w-5 h-5" />
              <CardTitle>Overall Grade</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-4xl font-bold text-accent">{report.overall_grade || "N/A"}</p>
              <p className="text-muted-foreground text-sm mt-1">Calculated based on confidence, clarity, and delivery</p>
            </CardContent>
          </Card>
        </div>

        {report.feedback && report.feedback.length > 0 && (
          <Card className="glass-card p-6">
            <CardHeader className="flex items-center gap-2">
              <BarChart3 className="text-primary w-5 h-5" />
              <CardTitle>Feedback</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {report.feedback.map((fb, i) => (
                <div key={i} className="p-3 bg-background/50 border border-border/20 rounded-lg text-sm text-muted-foreground">
                  {fb}
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {report.suggestions && report.suggestions.length > 0 && (
          <Card className="glass-card p-6">
            <CardHeader className="flex items-center gap-2">
              <TrendingUp className="text-success w-5 h-5" />
              <CardTitle>Improvement Suggestions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {report.suggestions.map((sg, i) => (
                <div key={i} className="p-3 bg-background/50 border border-border/20 rounded-lg text-sm text-muted-foreground">
                  {sg}
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    </section>
  );
};

export default ComprehensiveReport;
