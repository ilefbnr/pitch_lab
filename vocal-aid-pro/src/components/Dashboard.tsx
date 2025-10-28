import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { 
  Mic, 
  Upload, 
  TrendingUp, 
  Clock, 
  Target, 
  Award,
  BarChart3,
  Calendar,
  Play
} from "lucide-react";

const Dashboard = () => {
  const recentSessions = [
    {
      id: 1,
      title: "Product Launch Presentation",
      date: "Jan 15, 2025",
      duration: "8:32",
      score: 85,
      status: "completed"
    },
    {
      id: 2,
      title: "Team Meeting Pitch",
      date: "Jan 12, 2025",
      duration: "5:47",
      score: 78,
      status: "completed"
    },
    {
      id: 3,
      title: "Conference Talk",
      date: "Jan 10, 2025",
      duration: "12:15",
      score: 92,
      status: "completed"
    }
  ];

  const metrics = [
    { label: "Clarity Score", value: 87, change: "+5", trend: "up" },
    { label: "Confidence Level", value: 82, change: "+3", trend: "up" },
    { label: "Filler Words", value: 12, change: "-8", trend: "down" },
    { label: "Speaking Pace", value: 155, change: "+2", trend: "up" }
  ];

  return (
    <section className="py-20 bg-muted/30">
      <div className="container mx-auto px-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6 mb-8">
            <div>
              <h2 className="text-3xl font-bold mb-2">Your Speaking Dashboard</h2>
              <p className="text-muted-foreground">
                Track your progress and analyze your speaking performance
              </p>
            </div>
            
            <div className="flex gap-3">
              <Button className="shadow-soft">
                <Mic className="w-4 h-4 mr-2" />
                Record Speech
              </Button>
              <Button variant="outline">
                <Upload className="w-4 h-4 mr-2" />
                Upload Audio
              </Button>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
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
                    {metric.label === "Speaking Pace" && (
                      <span className="text-sm text-muted-foreground">WPM</span>
                    )}
                    {metric.label === "Filler Words" && (
                      <span className="text-sm text-muted-foreground">/min</span>
                    )}
                    {(metric.label === "Clarity Score" || metric.label === "Confidence Level") && (
                      <span className="text-sm text-muted-foreground">%</span>
                    )}
                  </div>
                  {(metric.label === "Clarity Score" || metric.label === "Confidence Level") && (
                    <Progress value={metric.value} className="mt-3" />
                  )}
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Recent Sessions */}
            <div className="lg:col-span-2">
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="w-5 h-5 text-primary" />
                    Recent Sessions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  {recentSessions.map((session) => (
                    <div key={session.id} className="flex items-center justify-between p-4 bg-background/50 rounded-lg hover:bg-background/80 transition-colors">
                      <div className="flex items-center gap-4">
                        <div className="p-2 bg-primary/10 rounded-lg">
                          <Play className="w-4 h-4 text-primary" />
                        </div>
                        <div>
                          <h4 className="font-medium">{session.title}</h4>
                          <div className="flex items-center gap-3 text-sm text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Calendar className="w-3 h-3" />
                              {session.date}
                            </span>
                            <span className="flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {session.duration}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge variant={session.score >= 85 ? "default" : "secondary"}>
                          {session.score}/100
                        </Badge>
                        <Button variant="ghost" size="sm">
                          View
                        </Button>
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>

            {/* Quick Actions & Goals */}
            <div className="space-y-6">
              {/* Quick Actions */}
              <Card className="glass-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="w-5 h-5 text-accent" />
                    Quick Actions
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button variant="outline" className="w-full justify-start">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    View Analytics
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Award className="w-4 h-4 mr-2" />
                    Practice Exercises
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Progress Report
                  </Button>
                </CardContent>
              </Card>

              {/* Weekly Goal */}
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
                    <p className="text-xs text-muted-foreground">
                      Complete 2 more sessions to reach your weekly goal!
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Dashboard;