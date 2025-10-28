import React, { useState, useRef } from "react";
import axios from "axios";
import {
  Mic,
  StopCircle,
  Save,
  Loader2,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

interface Pitch {
  id: number;
  title: string;
  description: string;
  transcript: string;
  created_at: string;
  user_id: number;
}

interface AudioRecorderProps {
  onNewPitch: (pitch: Pitch) => void;
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onNewPitch }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [success, setSuccess] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const getSupportedMimeType = () => {
    const mimeTypes = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/wav",
      "audio/ogg;codecs=opus",
    ];
    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) return mimeType;
    }
    return "";
  };

  const startRecording = async () => {
    try {
      setError("");
      setSuccess(false);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
          channelCount: 1,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      const mimeType = getSupportedMimeType();
      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType || "audio/wav" });
        setAudioBlob(blob);
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }
        if (durationIntervalRef.current) clearInterval(durationIntervalRef.current);
        setRecordingDuration(0);
      };

      mediaRecorder.start(500);
      setIsRecording(true);
      const startTime = Date.now();
      durationIntervalRef.current = setInterval(() => {
        setRecordingDuration(Math.floor((Date.now() - startTime) / 1000));
      }, 1000);
    } catch (err) {
      console.error(err);
      setError("Failed to access microphone. Please allow permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (recordingDuration < 3) {
        setError("Please record for at least 3 seconds.");
        return;
      }
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const uploadPitch = async () => {
    if (!audioBlob || !title.trim()) {
      setError("Please provide a title and record audio first.");
      return;
    }

    setLoading(true);
    setError("");
    setSuccess(false);

    try {
      const formData = new FormData();
      const filename = audioBlob.type.includes("webm") ? "recording.webm" : "recording.wav";
      formData.append("audio_file", audioBlob, filename);
      formData.append("title", title);
      formData.append("description", description);

      const response = await axios.post("http://localhost:8001/pitches", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      onNewPitch(response.data);
      setTitle("");
      setDescription("");
      setAudioBlob(null);
      setSuccess(true);
    } catch (err) {
      console.error(err);
      setError("Failed to upload pitch. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <Card className="glass-card shadow-medium p-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-2xl font-semibold">
          <Mic className="w-6 h-6 text-primary" />
          Audio Pitch Recorder
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="grid gap-4">
          <Input placeholder="Pitch Title" value={title} onChange={(e) => setTitle(e.target.value)} />
          <Textarea placeholder="Description (optional)" value={description} onChange={(e) => setDescription(e.target.value)} />
        </div>

        <div className="flex flex-col items-center gap-4 py-4">
          {!isRecording ? (
            <Button size="lg" onClick={startRecording} className="w-48">
              <Mic className="mr-2 h-5 w-5" /> Start Recording
            </Button>
          ) : (
            <Button variant="destructive" size="lg" onClick={stopRecording} className="w-48">
              <StopCircle className="mr-2 h-5 w-5" /> Stop Recording
            </Button>
          )}

          <div className="w-full max-w-sm text-center">
            <Badge variant="outline" className="mb-2">
              {isRecording ? "Recording..." : "Idle"}
            </Badge>
            <Progress value={(recordingDuration / 60) * 100} className="h-2" />
            <p className="text-sm mt-2 text-muted-foreground">Duration: {formatDuration(recordingDuration)}</p>
          </div>
        </div>

        {audioBlob && (
          <div className="bg-muted/30 p-4 rounded-lg space-y-3 border border-border/20">
            <h4 className="font-semibold">Recording Preview</h4>
            <audio controls src={URL.createObjectURL(audioBlob)} className="w-full rounded-lg" />
            <Button onClick={uploadPitch} disabled={loading} className="w-full">
              {loading ? (<><Loader2 className="mr-2 w-4 h-4 animate-spin" /> Uploading...</>) : (<><Save className="mr-2 w-4 h-4" /> Save & Analyze</>)}
            </Button>
          </div>
        )}

        {error && (<div className="flex items-center gap-2 text-destructive"><AlertCircle className="w-4 h-4" /> {error}</div>)}
        {success && (<div className="flex items-center gap-2 text-green-600"><CheckCircle2 className="w-4 h-4" /> Pitch uploaded successfully!</div>)}
      </CardContent>
    </Card>
  );
};

export default AudioRecorder;
