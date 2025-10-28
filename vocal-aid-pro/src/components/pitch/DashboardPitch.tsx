import React, { useState } from "react";
import {
  Mic,
  Video,
  Rocket,
  Library,
  BarChart3,
  Award,
  TrendingUp,
  Clock,
  Target,
} from "lucide-react";
import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

import AudioRecorder from "./AudioRecorder";
import PitchList from "./PitchList";
import RealTimePitch from "./RealTimePitch/RealTimePitch";
import VideoRecorder from "./VideoRecorder/VideoRecorder";
import ComprehensiveReport from "./ComprehensiveReport/ComprehensiveReport";

import "./DashboardPitch.css";

interface Pitch {
  id: number;
  title: string;
  description: string;
  transcript: string;
  created_at: string;
  user_id: number;
}

type TabType = "record" | "realtime" | "video" | "pitches";

const DashboardPitch: React.FC = () => {
  const [pitches, setPitches] = useState<Pitch[]>([]);
  const [activeTab, setActiveTab] = useState<TabType>("record");
  const [selectedPitchId, setSelectedPitchId] = useState<number | null>(null);
  const [showComprehensiveReport, setShowComprehensiveReport] = useState(false);
  const [isVideoRecording, setIsVideoRecording] = useState(false);

  const handleNewPitch = (newPitch: Pitch) => {
    setPitches((prev) => [newPitch, ...prev]);
  };

  const handleCloseReport = () => {
    setShowComprehensiveReport(false);
    setSelectedPitchId(null);
  };

  const handleVideoRecorded = (videoBlob: Blob) => {
    console.log("Video recorded:", videoBlob);
  };

  const handleStartVideoRecording = () => setIsVideoRecording(true);
  const handleStopVideoRecording = () => setIsVideoRecording(false);

  const renderTabContent = () => {
    switch (activeTab) {
      case "record":
        return <AudioRecorder onNewPitch={handleNewPitch} />;
      case "realtime":
        return <RealTimePitch />;
      case "video":
        return (
          <VideoRecorder
            onVideoRecorded={handleVideoRecorded}
            isRecording={isVideoRecording}
            onStartRecording={handleStartVideoRecording}
            onStopRecording={handleStopVideoRecording}
            enableFaceDetection={true}
          />
        );
      case "pitches":
        return <PitchList pitches={pitches} setPitches={setPitches} />;
      default:
        return null;
    }
  };

  // demo metrics (optional placeholders)
  const metrics = [
    { label: "Clarity Score", value: 87, change: "+5", trend: "up" },
    { label: "Confidence Level", value: 82, change: "+3", trend: "up" },
    { label: "Filler Words", value: 12, change: "-8", trend: "down" },
    { label: "Speaking Pace", value: 155, change: "+2", trend: "up" },
  ];

  return (
    <section className="py-20 bg-muted/30 min-h-screen">
      <div className="container mx-auto px-6 max-w-7xl">
        {/* Header */}
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6 mb-12">
          <div>
            <h2 className="text-3xl font-bold mb-2 gradient-text">🎙️ Pitch Lab Dashboard</h2>
            <p className="text-muted-foreground text-lg">
              Record, analyze, and track your speaking performance
            </p>
          </div>
          <div className="flex gap-3">
            <Button className="shadow-soft" onClick={() => setActiveTab("record")}>
              <Mic className="w-4 h-4 mr-2" />
              Record Speech
            </Button>
            <Button variant="outline" onClick={() => setActiveTab("video")}>
              <Video className="w-4 h-4 mr-2" />
              Video Analysis
            </Button>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {metrics.map((metric, index) => (
            <Card key={index} className="glass-card hover:shadow-medium transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm text-muted-foreground">{metric.label}</p>
                  <Badge variant={metric.trend === "up" ? "default" : "secondary"} className="text-xs">
                    {metric.change}
                  </Badge>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-bold">{metric.value}</span>
                  {(metric.label === "Clarity Score" || metric.label === "Confidence Level") && (
                    <span className="text-sm text-muted-foreground">%</span>
                  )}
                  {metric.label === "Speaking Pace" && (
                    <span className="text-sm text-muted-foreground">WPM</span>
                  )}
                  {metric.label === "Filler Words" && (
                    <span className="text-sm text-muted-foreground">/min</span>
                  )}
                </div>
                {(metric.label === "Clarity Score" || metric.label === "Confidence Level") && (
                  <Progress value={metric.value} className="mt-3" />
                )}
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Tabs */}
        <div className="flex flex-wrap justify-center gap-3 mb-10">
          <Button
            variant={activeTab === "record" ? "default" : "outline"}
            onClick={() => setActiveTab("record")}
            className="flex items-center gap-2"
          >
            <Mic className="w-4 h-4" /> Record
          </Button>
          <Button
            variant={activeTab === "realtime" ? "default" : "outline"}
            onClick={() => setActiveTab("realtime")}
            className="flex items-center gap-2"
          >
            <Rocket className="w-4 h-4" /> Real-Time
          </Button>
          <Button
            variant={activeTab === "video" ? "default" : "outline"}
            onClick={() => setActiveTab("video")}
            className="flex items-center gap-2"
          >
            <Video className="w-4 h-4" /> Video
          </Button>
          <Button
            variant={activeTab === "pitches" ? "default" : "outline"}
            onClick={() => setActiveTab("pitches")}
            className="flex items-center gap-2"
          >
            <Library className="w-4 h-4" /> My Pitches
          </Button>
        </div>

        {/* Content */}
        <Card className="glass-card shadow-medium">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {activeTab === "record" && (<><Mic className="w-5 h-5 text-primary" /> Record a New Pitch</>)}
              {activeTab === "realtime" && (<><Rocket className="w-5 h-5 text-accent" /> Real-Time Analysis</>)}
              {activeTab === "video" && (<><Video className="w-5 h-5 text-success" /> Video Pitch Analysis</>)}
              {activeTab === "pitches" && (<><BarChart3 className="w-5 h-5 text-primary" /> My Pitches</>)}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">{renderTabContent()}</CardContent>
        </Card>

        {/* Optional “quick actions” style row */}
        <div className="grid lg:grid-cols-3 gap-8 mt-12">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-primary" />
                Recent Tips
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="p-3 bg-background/50 rounded-lg text-sm text-muted-foreground">Pause between ideas for clarity.</div>
              <div className="p-3 bg-background/50 rounded-lg text-sm text-muted-foreground">Project your voice on key points.</div>
              <div className="p-3 bg-background/50 rounded-lg text-sm text-muted-foreground">Reduce filler words—short pauses are fine.</div>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5 text-accent" />
                Quick Actions
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button variant="outline" className="w-full justify-start" onClick={() => setActiveTab("realtime")}>
                <TrendingUp className="w-4 h-4 mr-2" />
                Start Real-Time Session
              </Button>
              <Button variant="outline" className="w-full justify-start" onClick={() => setActiveTab("pitches")}>
                <Award className="w-4 h-4 mr-2" />
                View Reports
              </Button>
              <Button variant="outline" className="w-full justify-start" onClick={() => setActiveTab("record")}>
                <Mic className="w-4 h-4 mr-2" />
                Record Audio
              </Button>
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Award className="w-5 h-5 text-success" />
                Weekly Goal
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Practice Sessions</span>
                  <span className="text-muted-foreground">3/5</span>
                </div>
                <Progress value={60} />
                <p className="text-xs text-muted-foreground">Complete 2 more sessions to reach your weekly goal!</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {showComprehensiveReport && selectedPitchId && (
          <ComprehensiveReport pitchId={selectedPitchId} onClose={handleCloseReport} />
        )}
      </div>
    </section>
  );
};

export default DashboardPitch;
