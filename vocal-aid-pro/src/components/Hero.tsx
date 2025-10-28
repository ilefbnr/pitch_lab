import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Mic, Brain, TrendingUp, PlayCircle } from "lucide-react";
import heroBg from "@/assets/hero-bg.jpg";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-background via-muted/30 to-accent-light/20">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0 z-0">
        <img 
          src={heroBg} 
          alt="Professional presentation background" 
          className="w-full h-full object-cover opacity-10"
        />
        <div className="absolute inset-0 hero-gradient opacity-20"></div>
      </div>
      
      {/* Content */}
      <div className="container mx-auto px-6 py-20 relative z-10">
        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Text Content */}
          <div className="space-y-8">
            <div className="space-y-4">
              <div className="inline-flex items-center gap-2 bg-primary/10 text-primary px-4 py-2 rounded-full text-sm font-medium">
                <Brain className="w-4 h-4" />
                AI-Powered Speech Analysis
              </div>
              
              <h1 className="text-5xl lg:text-6xl font-bold leading-tight">
                Master Your{" "}
                <span className="gradient-text">
                  Public Speaking
                </span>{" "}
                with AI
              </h1>
              
              <p className="text-xl text-muted-foreground leading-relaxed max-w-lg">
                Get research-backed feedback on clarity, confidence, and delivery. 
                Track your progress with advanced speech analysis powered by AI.
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <Button size="lg" className="text-lg px-8 py-6 shadow-medium hover:shadow-strong transition-all duration-300">
                <PlayCircle className="w-5 h-5 mr-2" />
                Start Practicing
              </Button>
              <Button variant="outline" size="lg" className="text-lg px-8 py-6">
                <TrendingUp className="w-5 h-5 mr-2" />
                View Demo
              </Button>
            </div>
            
            {/* Stats */}
            <div className="grid grid-cols-3 gap-6 pt-8 border-t border-border/50">
              <div>
                <div className="text-2xl font-bold text-primary">95%</div>
                <div className="text-sm text-muted-foreground">Accuracy Rate</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-primary">10k+</div>
                <div className="text-sm text-muted-foreground">Speeches Analyzed</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-primary">4.8★</div>
                <div className="text-sm text-muted-foreground">User Rating</div>
              </div>
            </div>
          </div>
          
          {/* Right Column - Feature Cards */}
          <div className="space-y-6">
            <Card className="glass-card p-6 hover:shadow-medium transition-all duration-300">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-primary/10 rounded-lg">
                  <Mic className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg mb-2">Speech Analysis</h3>
                  <p className="text-muted-foreground">
                    Advanced AI analyzes clarity, pace, filler words, and vocal confidence in real-time.
                  </p>
                </div>
              </div>
            </Card>
            
            <Card className="glass-card p-6 hover:shadow-medium transition-all duration-300">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-accent/10 rounded-lg">
                  <Brain className="w-6 h-6 text-accent" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg mb-2">Emotion Detection</h3>
                  <p className="text-muted-foreground">
                    DeepFace technology tracks facial expressions and emotional engagement.
                  </p>
                </div>
              </div>
            </Card>
            
            <Card className="glass-card p-6 hover:shadow-medium transition-all duration-300">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-success/10 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-success" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg mb-2">Progress Tracking</h3>
                  <p className="text-muted-foreground">
                    Visualize improvement over time with detailed analytics and insights.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
      
      {/* Floating Elements */}
      <div className="absolute top-20 left-10 w-20 h-20 bg-primary/10 rounded-full blur-xl animate-pulse"></div>
      <div className="absolute bottom-20 right-10 w-32 h-32 bg-accent/10 rounded-full blur-xl animate-pulse delay-1000"></div>
    </section>
  );
};

export default Hero;