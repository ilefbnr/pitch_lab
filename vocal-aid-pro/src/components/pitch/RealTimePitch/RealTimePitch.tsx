import React, { useState, useEffect, useRef } from "react";
import { Mic, BarChart3, Wifi, Loader2, AlertCircle, CheckCircle2 } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

const RealTimePitch: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [metrics, setMetrics] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const connectSocket = () => {
    setLoading(true);
    setError(null);
    const socket = new WebSocket("ws://localhost:8001/realtime-pitch");
    socketRef.current = socket;

    socket.onopen = () => {
      setIsConnected(true);
      setLoading(false);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.transcript) setTranscript((prev) => prev + " " + data.transcript);
        if (data.metrics) setMetrics(data.metrics);
      } catch (e) {
        console.error("WS parse error", e);
      }
    };

    socket.onclose = () => {
      setIsConnected(false);
      setIsRecording(false);
      setLoading(false);
    };

    socket.onerror = (evt) => {
      console.error("WebSocket error:", evt);
      setError("WebSocket connection failed");
      setIsConnected(false);
      setLoading(false);
    };
  };

  const startStreaming = async () => {
    if (!isConnected) {
      connectSocket();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0 && socketRef.current?.readyState === WebSocket.OPEN) {
          socketRef.current.send(e.data);
        }
      };

      mediaRecorder.start(250);
      setIsRecording(true);
    } catch (err) {
      console.error(err);
      setError("Failed to start recording. Check microphone permissions.");
      setIsRecording(false);
    }
  };

  const stopStreaming = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.close();
    }
    setIsRecording(false);
    setIsConnected(false);
  };

  useEffect(() => {
    return () => stopStreaming();
  }, []);

  return (
    <Card className="glass-card shadow-medium p-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-2xl font-semibold">
          <BarChart3 className="w-6 h-6 text-accent" />
          Real-Time Speech Analysis
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="flex flex-wrap justify-center gap-3">
          {!isConnected ? (
            <Button onClick={connectSocket} disabled={loading}>
              {loading ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Connecting...</>) : (<><Wifi className="mr-2 h-4 w-4" /> Connect</>)}
            </Button>
          ) : !isRecording ? (
            <Button onClick={startStreaming} disabled={loading}>
              <Mic className="mr-2 h-4 w-4" /> Start Speaking
            </Button>
          ) : (
            <Button variant="destructive" onClick={stopStreaming}>
              <Mic className="mr-2 h-4 w-4" /> Stop
            </Button>
          )}
        </div>

        <div className="flex justify-center gap-4 text-center">
          <Badge variant={isConnected ? "default" : "secondary"}>{isConnected ? "Connected" : "Disconnected"}</Badge>
          <Badge variant={isRecording ? "default" : "secondary"}>{isRecording ? "Recording" : "Idle"}</Badge>
        </div>

        <div className="p-4 bg-muted/30 rounded-lg border border-border/20 h-48 overflow-y-auto">
          <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-wrap">
            {transcript || "Your real-time transcript will appear here..."}
          </p>
        </div>

        {Object.keys(metrics).length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(metrics).map(([key, value]) => (
              <Card key={key} className="bg-background/60 border border-border/20 p-4 text-center shadow-soft hover:shadow-medium transition-all">
                <h4 className="font-semibold capitalize">{key.replace(/_/g, " ")}</h4>
                <Progress value={Number(value) * 100} className="h-2 my-2" />
                <p className="text-sm text-muted-foreground">{(Number(value) * 100).toFixed(0)}%</p>
              </Card>
            ))}
          </div>
        )}

        {!isRecording && !isConnected && (
          <div className="flex justify-center text-muted-foreground text-sm">
            Connect and start speaking to analyze your pitch live.
          </div>
        )}
        {isConnected && !isRecording && (
          <div className="flex items-center justify-center gap-2 text-green-600 text-sm">
            <CheckCircle2 className="w-4 h-4" /> Connected — ready to record
          </div>
        )}
        {!isConnected && error && (
          <div className="flex items-center justify-center gap-2 text-destructive text-sm">
            <AlertCircle className="w-4 h-4" /> {error}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RealTimePitch;
