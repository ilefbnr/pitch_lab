import { Button } from "@/components/ui/button";
import { Brain, Menu } from "lucide-react";
import { useState } from "react";
import { useNavigate } from 'react-router-dom';


const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
    const navigate = useNavigate();


  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border/50">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Brain className="w-6 h-6 text-primary" />
            </div>
            <span className="text-xl font-bold gradient-text">Pitch Lab</span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-8">
            <a href="#dashboard" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
              Dashboard
            </a>
            <a href="#practice" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
              Practice
            </a>
            <a href="#analytics" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
              Analytics
            </a>
            <a href="#feedback" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
              Feedback
            </a>
          </div>

          {/* CTA Buttons */}
          <div className="hidden md:flex items-center gap-3">
            <Button variant="outline" size="sm" onClick={() => navigate('/login')}>
              Sign In
            </Button>
            <Button size="sm" className="shadow-soft" onClick={() => navigate('/signup')}>
              Get Started
            </Button>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div className="md:hidden bg-background/95 backdrop-blur-md border-b border-border/50">
          <div className="container mx-auto px-6 py-4 space-y-4">
            <a href="#dashboard" className="block text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
              Dashboard
            </a>
            <a href="#practice" className="block text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
              Practice
            </a>
            <a href="#analytics" className="block text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
              Analytics
            </a>
            <a href="#feedback" className="block text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
              Feedback
            </a>
            <div className="flex gap-3 pt-4 border-t border-border/50">
<Button 
      variant="outline" 
      size="sm" 
      className="flex-1" 
      onClick={() => navigate('/login')}
    >
      Sign In
              </Button>
              <Button size="sm" className="flex-1" onClick={() => navigate('/signup')}>
                Get Started
              </Button>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;