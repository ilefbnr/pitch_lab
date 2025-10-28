import React, { useState, useRef, useCallback, useEffect } from "react";
import { Video, Mic, Smile, StopCircle, AlertCircle, CheckCircle2 } from "lucide-react";
import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface VideoRecorderProps {
  onVideoRecorded: (videoBlob: Blob) => void;
  isRecording: boolean;
  onStartRecording: () => void;
  onStopRecording: () => void;
  enableFaceDetection?: boolean;
  onEmotionUpdate?: (emotion: any) => void;
}

const VideoRecorder: React.FC<VideoRecorderProps> = ({
  onVideoRecorded,
  isRecording,
  onStartRecording,
  onStopRecording,
  enableFaceDetection = false,
  onEmotionUpdate,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [error, setError] = useState("");
  const [hasPermission, setHasPermission] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState("😐");
  const [detectionStatus, setDetectionStatus] = useState("Idle");

  const initializeCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setHasPermission(true);
      setIsInitialized(true);
      setError("");
    } catch (err) {
      setError("Unable to access camera. Please allow permissions.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setHasPermission(false);
    setIsInitialized(false);
  }, []);

  const startRecording = useCallback(() => {
    if (!streamRef.current) {
      setError("Camera not initialized");
      return;
    }
    try {
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8,opus")
        ? "video/webm;codecs=vp8,opus"
        : "video/webm";
      const mediaRecorder = new MediaRecorder(streamRef.current, { mimeType });
      const chunks: Blob[] = [];
      mediaRecorder.ondataavailable = (e) => e.data.size > 0 && chunks.push(e.data);
      mediaRecorder.onstop = () => onVideoRecorded(new Blob(chunks, { type: "video/webm" }));
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(1000);
      onStartRecording();
      if (enableFaceDetection) setDetectionStatus("Analyzing...");
    } catch (err) {
      setError("Failed to start recording");
    }
  }, [onStartRecording, onVideoRecorded, enableFaceDetection]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    onStopRecording();
    stopCamera();
  }, [onStopRecording, stopCamera]);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isRecording && enableFaceDetection && videoRef.current && canvasRef.current) {
      const detectEmotion = async () => {
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext("2d");
        if (!ctx || !videoRef.current) return;
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0);
        const frameData = canvas.toDataURL("image/jpeg", 0.6);
        try {
          const res = await fetch("http://localhost:8001/video/realtime-emotion", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: frameData }),
          });
          if (res.ok) {
            const result = await res.json();
            const emotion = result.frame_analysis?.dominant_emotion || "neutral";
            const emojiMap: Record<string, string> = {
              happy: "😊",
              confident: "😎",
              neutral: "😐",
              surprise: "😲",
              fear: "😰",
              sad: "😢",
              angry: "😠",
            };
            setCurrentEmotion(emojiMap[emotion] || "😐");
            setDetectionStatus("Detecting...");
            onEmotionUpdate?.(result.frame_analysis);
          }
        } catch {
          setDetectionStatus("Error detecting emotion");
        }
        if (isRecording) timer = setTimeout(detectEmotion, 2000);
      };
      timer = setTimeout(detectEmotion, 1500);
    }
    return () => clearTimeout(timer);
  }, [isRecording, enableFaceDetection, onEmotionUpdate]);

  useEffect(() => {
    initializeCamera();
    return () => stopCamera();
  }, [initializeCamera, stopCamera]);

  return (
    <Card className="glass-card shadow-medium p-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-2xl font-semibold">
          <Video className="w-6 h-6 text-success" /> Video Recorder
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="relative rounded-xl overflow-hidden border border-border/20 bg-background/50 shadow-inner">
          <video ref={videoRef} autoPlay muted playsInline className={`w-full rounded-lg ${isRecording ? "ring-2 ring-primary" : ""}`} />
          <canvas ref={canvasRef} style={{ display: "none" }} />
          {isRecording && (
            <div className="absolute top-3 left-3 flex items-center gap-2 bg-primary/10 px-3 py-1 rounded-full backdrop-blur-md">
              <span className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
              <span className="text-sm font-medium text-primary">Recording...</span>
            </div>
          )}
          {enableFaceDetection && isRecording && (
            <div className="absolute bottom-3 right-3 bg-background/80 px-3 py-2 rounded-lg flex items-center gap-2 shadow-soft">
              <Smile className="w-4 h-4 text-primary" />
              <span className="text-lg">{currentEmotion}</span>
              <Badge variant="outline">{detectionStatus}</Badge>
            </div>
          )}
        </div>

        <div className="flex flex-wrap gap-3 justify-center">
          {!isInitialized ? (
            <Button onClick={initializeCamera} className="px-6">
              <Mic className="mr-2 h-5 w-5" /> Initialize Camera
            </Button>
          ) : !isRecording ? (
            <Button onClick={startRecording} disabled={!hasPermission} className="px-6">
              <Video className="mr-2 h-5 w-5" /> Start Recording
            </Button>
          ) : (
            <Button variant="destructive" onClick={stopRecording} className="px-6">
              <StopCircle className="mr-2 h-5 w-5" /> Stop Recording
            </Button>
          )}
        </div>

        <div className="grid grid-cols-3 gap-4 text-center">
          <div><Badge variant={hasPermission ? "default" : "secondary"}>Camera {hasPermission ? "Ready" : "Off"}</Badge></div>
          <div><Badge variant={isRecording ? "default" : "secondary"}>{isRecording ? "Recording" : "Idle"}</Badge></div>
          {enableFaceDetection && (<div><Badge variant="outline">{detectionStatus}</Badge></div>)}
        </div>

        {error && (<div className="flex items-center gap-2 text-destructive"><AlertCircle className="w-4 h-4" /> {error}</div>)}
        {!error && hasPermission && isInitialized && !isRecording && (
          <div className="flex items-center gap-2 text-green-600"><CheckCircle2 className="w-4 h-4" /> Ready to record</div>
        )}
      </CardContent>
    </Card>
  );
};

export default VideoRecorder;
