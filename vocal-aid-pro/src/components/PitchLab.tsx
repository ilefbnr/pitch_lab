import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Mic, Video, Play, Square, Upload, BarChart3, Clock, Star, TrendingUp, Eye, MessageSquare, Download, Trash2, Edit, Zap } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { createClient } from '@supabase/supabase-js';
import { Input } from '@/components/ui/input'; // if you have shadcn input; else use <input>

const API_BASE = (process.env.VITE_API_URL || process.env.REACT_APP_API_URL || "http://localhost:8001").replace(/\/+$/, "");
const SUPABASE_URL = (process.env.VITE_SUPABASE_URL || process.env.REACT_APP_SUPABASE_URL || "");
const SUPABASE_KEY = (process.env.VITE_SUPABASE_ANON_KEY || process.env.REACT_APP_SUPABASE_ANON_KEY || "");
const SUPABASE_BUCKET = (process.env.VITE_SUPABASE_BUCKET || process.env.REACT_APP_SUPABASE_BUCKET || "pitches");

// Optional supabase client (only used if env vars provided)
const supabase = SUPABASE_URL && SUPABASE_KEY ? createClient(SUPABASE_URL, SUPABASE_KEY) : null;

interface Pitch {
  id: string;
  title: string;
  description?: string;
  type: 'audio' | 'video' | 'realtime';
  duration?: string;
  score?: number;
  sentiment?: 'positive' | 'neutral' | 'negative';
  createdAt?: string;
  transcript?: string;
  // backend may return file paths/urls:
  audio_file_path?: string | null;
  video_file_path?: string | null;
  analysis_result?: any;
}

const PitchLab: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingType, setRecordingType] = useState<'audio' | 'video'>('audio');
  const [recordingTime, setRecordingTime] = useState(0);
  const [pitches, setPitches] = useState<Pitch[]>([]);
  const [uploading, setUploading] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const { toast } = useToast();

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startTsRef = useRef<number>(0);
  const timerRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoPreviewRef = useRef<HTMLVideoElement | null>(null);

  // Realtime
  const [isRealtimeOn, setIsRealtimeOn] = useState(false);
  const [realtimeInfo, setRealtimeInfo] = useState<any>(null);
  const realtimeVideoRef = useRef<HTMLVideoElement | null>(null);
  const realtimeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const realtimeStreamRef = useRef<MediaStream | null>(null);
  const frameTimerRef = useRef<number | null>(null);
  const audioRecorderRef = useRef<MediaRecorder | null>(null);

  const [pitchTitle, setPitchTitle] = useState('My Pitch');
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const [pendingType, setPendingType] = useState<'audio'|'video'|null>(null);

  useEffect(() => {
    fetchPitchesList();
    return cleanupStream;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function cleanupStream() {
    if (timerRef.current) window.clearInterval(timerRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch {}
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  }

  const fetchPitchesList = async () => {
    try {
      const res = await fetch(`${API_BASE}/pitches`);
      if (res.ok) {
        const data = await res.json();
        setPitches(data);
      }
    } catch (err) {
      // ignore
    }
  };

  const startRecording = async (type: 'audio' | 'video') => {
    cleanupStream();
    setRecordingType(type);
    setAnalysis(null);

    try {
      const constraints: MediaStreamConstraints = type === 'audio'
        ? { audio: true, video: false }
        : { audio: true, video: { width: 1280, height: 720 } };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      // preview for video
      if (type === 'video' && videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream;
        videoPreviewRef.current.muted = true;
        videoPreviewRef.current.play().catch(() => {});
      }

      // choose mime
      let mimeType = '';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) mimeType = 'audio/webm;codecs=opus';
      else if (MediaRecorder.isTypeSupported('audio/webm')) mimeType = 'audio/webm';
      else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) mimeType = 'video/webm;codecs=vp9';
      else if (MediaRecorder.isTypeSupported('video/webm')) mimeType = 'video/webm';

      const options = mimeType ? { mimeType } : undefined;
      const mr = new MediaRecorder(stream, options as any);
      mediaRecorderRef.current = mr;
      chunksRef.current = [];

      mr.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };

      mr.onstop = () => {
        // stream tracks remain active until cleaned; we stop them in cleanupStream or after upload
      };

      mr.start();
      startTsRef.current = Date.now();
      setRecordingTime(0);
      setIsRecording(true);

      timerRef.current = window.setInterval(() => {
        setRecordingTime(Math.floor((Date.now() - startTsRef.current) / 1000));
      }, 250);

      toast({
        title: `${type === 'audio' ? 'Audio' : 'Video'} Recording Started`,
        description: 'Speak clearly. Recording must be at least 3 seconds.',
      });
    } catch (err: any) {
      toast({ title: 'Microphone/Camera Error', description: String(err), variant: 'destructive' });
    }
  };

  const stopRecording = async () => {
    if (!mediaRecorderRef.current) return;
    mediaRecorderRef.current.stop();
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    const recordedSeconds = Math.floor((Date.now() - startTsRef.current) / 1000);
    setRecordingTime(recordedSeconds);
    setIsRecording(false);

    // stop preview tracks now (but keep data)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoPreviewRef.current) {
      videoPreviewRef.current.pause();
      videoPreviewRef.current.srcObject = null;
    }

    if (recordedSeconds < 3) {
      toast({ title: 'Recording too short', description: 'Please record at least 3 seconds.', variant: 'destructive' });
      chunksRef.current = [];
      return;
    }

    const blob = new Blob(chunksRef.current, { type: chunksRef.current[0]?.type || (recordingType === 'audio' ? 'audio/webm' : 'video/webm') });
    const filename = `${recordingType}_pitch_${new Date().toISOString().replace(/[:.]/g, '-')}.${recordingType === 'audio' ? 'webm' : 'webm'}`;
    const file = new File([blob], filename, { type: blob.type });

    // Instead of uploading immediately, store and let user name + Save
    setPendingFile(file);
    setPendingType(recordingType);
    chunksRef.current = [];

    toast({ title: 'Ready to save', description: 'Add a title and click Save to upload for analysis.' });
  };

  const savePending = async () => {
    if (!pendingFile || !pendingType) return;
    await uploadToBackend(pendingFile, pendingType);
    setPendingFile(null);
    setPendingType(null);
  };

  const uploadToBackend = async (file: File, type: 'audio'|'video') => {
    try {
      setUploading(true);
      const fd = new FormData();
      fd.append('title', pitchTitle || `${type === 'audio' ? 'Audio' : 'Video'} pitch`);
      fd.append('description', '');
      if (type === 'audio') fd.append('audio_file', file, file.name);
      else fd.append('video_file', file, file.name);

      const res = await fetch(`${API_BASE}/pitches`, {
        method: 'POST',
        body: fd,
      });

      if (!res.ok) {
        const txt = await res.text().catch(()=>res.statusText);
        throw new Error(txt || 'Upload failed');
      }

      const data = await res.json();
      toast({ title: 'Upload successful', description: 'Analysis started / returned' });
      // backend returns new pitch data -> add to list and show analysis
      if (data) {
        setPitches(prev => [data, ...prev]);
        setAnalysis(data.analysis_result ?? data);
      }
    } catch (err: any) {
      toast({ title: 'Upload error', description: err.message || String(err), variant: 'destructive' });
    } finally {
      setUploading(false);
    }
  };

  const uploadToSupabase = async (file: File) => {
    if (!supabase) {
      toast({ title: 'Supabase not configured', description: 'Set SUPABASE env vars to use direct upload', variant: 'default' });
      return;
    }
    try {
      setUploading(true);
      const path = `${new Date().toISOString().slice(0,10)}/${file.name}`;
      const { data: up, error: upErr } = await supabase.storage.from(SUPABASE_BUCKET).upload(path, file, { upsert: false });
      if (upErr) throw upErr;
      if (!up) throw new Error('Upload failed: no data returned');

      const { data: pub } = supabase.storage.from(SUPABASE_BUCKET).getPublicUrl(up.path);
      const publicUrl = pub.publicUrl;

      toast({ title: 'Uploaded to Supabase', description: publicUrl });
      // Optionally notify backend with publicUrl/up.path
      // await fetch(`${API_BASE}/pitches`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ title:'From Supabase', storage_path: up.path, public_url: publicUrl }) });
    } catch (err: any) {
      toast({ title: 'Supabase upload error', description: err.message || String(err), variant: 'destructive' });
    } finally {
      setUploading(false);
    }
  };

  const playPitch = async (pitch: Pitch) => {
    try {
      if (pitch.audio_file_path) {
        // backend endpoint to stream audio by pitch id
        const res = await fetch(`${API_BASE}/audio/${pitch.id}`);
        if (!res.ok) throw new Error('Audio not available');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.play();
      } else if (pitch.video_file_path) {
        // open video in new tab
        window.open(pitch.video_file_path, '_blank');
      } else {
        toast({ title: 'No media', description: 'This pitch has no media file' });
      }
    } catch (err: any) {
      toast({ title: 'Playback error', description: err.message || String(err), variant: 'destructive' });
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-success';
      case 'negative': return 'text-destructive';
      default: return 'text-warning';
    }
  };

  const getSentimentBadge = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'bg-success text-white';
      case 'negative': return 'bg-destructive text-white';
      default: return 'bg-warning text-white';
    }
  };

  // -------- Realtime helpers ----------
  const startRealtime = async () => {
    try {
      // get A/V
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 }, audio: true });
      realtimeStreamRef.current = stream;

      if (realtimeVideoRef.current) {
        realtimeVideoRef.current.srcObject = stream;
        await realtimeVideoRef.current.play().catch(()=>{});
      }

      // frame sender (every 600ms)
      const sendFrame = async () => {
        if (!realtimeCanvasRef.current || !realtimeVideoRef.current) return;
        const v = realtimeVideoRef.current;
        const c = realtimeCanvasRef.current;
        c.width = v.videoWidth || 640;
        c.height = v.videoHeight || 360;
        const ctx = c.getContext('2d');
        if (!ctx) return;
        ctx.drawImage(v, 0, 0, c.width, c.height);
        const b64 = c.toDataURL('image/jpeg', 0.6);
        try {
          const res = await fetch(`${API_BASE}/video/realtime-emotion`, {
            method: 'POST',
            headers: { 'Content-Type':'application/json' },
            body: JSON.stringify({ image_b64: b64 }),
          });
          if (res.ok) {
            const data = await res.json();
            setRealtimeInfo((prev:any)=> ({ ...(prev||{}), video: data.video, engagement: data.engagement_score }));
          }
        } catch {}
      };
      frameTimerRef.current = window.setInterval(sendFrame, 600) as unknown as number;

      // audio recorder -> multipart uploads every 1s
      let mime = '';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) mime = 'audio/webm;codecs=opus';
      else if (MediaRecorder.isTypeSupported('audio/webm')) mime = 'audio/webm';
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } as any : undefined);
      audioRecorderRef.current = rec;
      rec.ondataavailable = async (e: BlobEvent) => {
        if (!e.data || e.data.size === 0) return;
        const fd = new FormData();
        const f = new File([e.data], `rt_${Date.now()}.webm`, { type: e.data.type || 'audio/webm' });
        fd.append('audio_file', f, f.name);
        try {
          const res = await fetch(`${API_BASE}/audio/realtime-emotion`, { method: 'POST', body: fd });
          if (res.ok) {
            const data = await res.json();
            setRealtimeInfo((prev:any)=> ({ ...(prev||{}), audio: data.audio, engagement: data.engagement_score }));
          }
        } catch {}
      };
      rec.start(1000); // 1s chunks

      setIsRealtimeOn(true);
      toast({ title: 'Realtime started', description: 'Camera and mic are streaming for analysis.' });
    } catch (e:any) {
      toast({ title: 'Realtime error', description: e?.message || String(e), variant: 'destructive' });
      stopRealtime();
    }
  };

  const stopRealtime = () => {
    try {
      if (audioRecorderRef.current && audioRecorderRef.current.state !== 'inactive') audioRecorderRef.current.stop();
    } catch {}
    audioRecorderRef.current = null;

    if (frameTimerRef.current) {
      window.clearInterval(frameTimerRef.current);
      frameTimerRef.current = null;
    }
    if (realtimeStreamRef.current) {
      realtimeStreamRef.current.getTracks().forEach(t => t.stop());
      realtimeStreamRef.current = null;
    }
    if (realtimeVideoRef.current) {
      try { realtimeVideoRef.current.pause(); } catch {}
      realtimeVideoRef.current.srcObject = null;
    }
    setIsRealtimeOn(false);
  };

  useEffect(() => () => stopRealtime(), []);

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-2">Pitch Lab</h1>
          <p className="text-muted-foreground">
            Record, analyze, and perfect your business pitches with AI-powered insights
          </p>
        </div>

        <Tabs defaultValue="record" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="record">📼 Record</TabsTrigger>
            <TabsTrigger value="realtime">🚀 Real-time</TabsTrigger>
            <TabsTrigger value="video">🎥 Video</TabsTrigger>
            <TabsTrigger value="pitches">📚 My Pitches</TabsTrigger>
          </TabsList>

          {/* Audio Recording Tab */}
          <TabsContent value="record">
            <div className="grid lg:grid-cols-1 gap-6">
              <div className="lg:col-span-2">
                <Card className="shadow-elegant">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Mic className="w-5 h-5 text-porter-blue" />
                      Audio Pitch Recording
                    </CardTitle>
                    <CardDescription>
                      Record your pitch and then save it for backend analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex items-center gap-2">
                      <Input placeholder="Pitch title" value={pitchTitle} onChange={e=>setPitchTitle(e.target.value)} />
                      <Button onClick={savePending} disabled={!pendingFile} variant="porter">Save</Button>
                    </div>

                    {!isRecording ? (
                      <div className="text-center py-8">
                        <Button onClick={() => startRecording('audio')} variant="porter" size="lg" className="px-8">
                          <Mic className="w-5 h-5 mr-2" />
                          Start Recording (Audio)
                        </Button>
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <div className="text-3xl font-mono text-porter-blue mb-6">
                          {formatTime(recordingTime)}
                        </div>
                        <Button onClick={stopRecording} variant="destructive" size="lg" className="px-8">
                          <Square className="w-5 h-5 mr-2" />
                          Stop Recording
                        </Button>
                      </div>
                    )}

                    {uploading && <Progress value={50} />}

                    {analysis && (
                      <div className="mt-4">
                        <h4 className="text-lg font-semibold">Analysis</h4>
                        <pre className="text-sm bg-muted p-3 rounded mt-2 overflow-auto">{JSON.stringify(analysis, null, 2)}</pre>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* Real-time Analysis Tab */}
          <TabsContent value="realtime">
            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-porter-blue" />
                  Real-time Pitch Analysis
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  {!isRealtimeOn ? (
                    <Button variant="porter" onClick={startRealtime}>
                      <Play className="w-5 h-5 mr-2" />
                      Start Real-time
                    </Button>
                  ) : (
                    <Button variant="destructive" onClick={stopRealtime}>
                      <Square className="w-5 h-5 mr-2" />
                      Stop
                    </Button>
                  )}
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <video ref={realtimeVideoRef} className="w-full rounded bg-black" />
                    <canvas ref={realtimeCanvasRef} className="hidden" />
                  </div>
                  <div className="text-sm">
                    <h4 className="font-semibold mb-2">Live Metrics</h4>
                    <pre className="bg-muted p-3 rounded overflow-auto">
                      {JSON.stringify(realtimeInfo, null, 2)}
                    </pre>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Video Recording Tab */}
          <TabsContent value="video">
            <Card className="shadow-elegant">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Video className="w-5 h-5 text-porter-blue" />
                  Video Pitch with Emotion Analysis
                </CardTitle>
                <CardDescription>
                  Record with video for enhanced emotion analysis and body language feedback
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                  <div className="w-full h-full relative">
                    <video ref={videoPreviewRef} className="w-full h-full object-cover rounded-lg" />
                    <div className="absolute left-4 bottom-4">
                      <Button onClick={() => isRecording ? stopRecording() : startRecording('video')} variant="porter">
                        {isRecording ? <Square className="w-4 h-4 mr-2" /> : <Video className="w-4 h-4 mr-2" />}
                        {isRecording ? 'Stop' : 'Record'}
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* My Pitches Tab */}
          <TabsContent value="pitches">
            <div className="space-y-6">
              <div className="flex justify-between items-center">
                <h2 className="text-xl font-semibold">Your Recorded Pitches</h2>
                <Badge variant="secondary">{pitches.length} pitches</Badge>
              </div>

              <div className="grid gap-4">
                {pitches.map((pitch) => (
                  <Card key={pitch.id} className="hover:shadow-lg transition-shadow">
                    <CardContent className="p-6">
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <h3 className="text-lg font-semibold">{pitch.title}</h3>
                            <Badge variant="outline" className="text-xs">
                              {pitch.type}
                            </Badge>
                            {pitch.sentiment && <Badge className={getSentimentBadge(pitch.sentiment)}>{pitch.sentiment}</Badge>}
                          </div>
                          <p className="text-muted-foreground text-sm mb-3">
                            {pitch.description}
                          </p>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <div className="flex items-center gap-1">
                              <Clock className="w-4 h-4" />
                              {pitch.duration}
                            </div>
                            <div className="flex items-center gap-1">
                              <TrendingUp className={`w-4 h-4 ${getSentimentColor(pitch.sentiment ?? '')}`} />
                              Score: {pitch.score ?? '-'}%
                            </div>
                            <span>{pitch.createdAt}</span>
                          </div>
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <Button variant="outline" size="sm" onClick={() => playPitch(pitch)}>
                            <Play className="w-4 h-4" />
                          </Button>
                          <Button variant="outline" size="sm">
                            <Download className="w-4 h-4" />
                          </Button>
                          <Button variant="outline" size="sm">
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default PitchLab;