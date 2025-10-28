import React from "react";
import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Play, BarChart3, Trash2, Calendar, Award } from "lucide-react";

interface Pitch {
  id: number;
  title: string;
  description: string;
  transcript: string;
  created_at: string;
  analysis_score?: number;
}

interface PitchListProps {
  pitches: Pitch[];
  setPitches: React.Dispatch<React.SetStateAction<Pitch[]>>;
}

const PitchList: React.FC<PitchListProps> = ({ pitches, setPitches }) => {
  const handleDeletePitch = async (id: number) => {
    try {
      await fetch(`http://localhost:8001/pitches/${id}`, { method: "DELETE" });
      setPitches((prev) => prev.filter((p) => p.id !== id));
    } catch (error) {
      console.error("Error deleting pitch:", error);
    }
  };

  return (
    <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      {pitches.length === 0 ? (
        <Card className="glass-card p-8 text-center shadow-soft col-span-full">
          <p className="text-muted-foreground text-lg">No pitches found. Record one to get started 🎙️</p>
        </Card>
      ) : (
        pitches.map((pitch) => (
          <Card key={pitch.id} className="glass-card shadow-medium p-4 hover:shadow-xl transition-all duration-300 border border-border/30">
            <CardHeader className="space-y-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg font-semibold">{pitch.title || "Untitled Pitch"}</CardTitle>
                <Badge variant="outline" className="text-xs">
                  <Calendar className="w-3 h-3 mr-1" />
                  {new Date(pitch.created_at).toLocaleDateString()}
                </Badge>
              </div>
              {pitch.description && <p className="text-sm text-muted-foreground line-clamp-2">{pitch.description}</p>}
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="bg-background/60 p-3 rounded-lg border border-border/20 h-20 overflow-y-auto">
                <p className="text-xs text-muted-foreground leading-relaxed">{pitch.transcript || "No transcript available."}</p>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  <span className="text-sm">Score</span>
                </div>
                <span className="text-sm font-semibold">
                  {pitch.analysis_score ? `${Math.round(pitch.analysis_score * 100)}%` : "N/A"}
                </span>
              </div>
              <Progress value={pitch.analysis_score ? pitch.analysis_score * 100 : 0} className="h-2" />

              <div className="flex justify-between mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex items-center gap-1"
                  onClick={() => window.open(`http://localhost:8001/pitches/${pitch.id}/report`, "_blank")}
                >
                  <Award className="w-4 h-4" />
                  Report
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  className="flex items-center gap-1"
                  onClick={() => window.open(`http://localhost:8001/audio/${pitch.id}`, "_blank")}
                >
                  <Play className="w-4 h-4" />
                  Play
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  className="flex items-center gap-1"
                  onClick={() => handleDeletePitch(pitch.id)}
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </Button>
              </div>
            </CardContent>
          </Card>
        ))
      )}
    </div>
  );
};

export default PitchList;
