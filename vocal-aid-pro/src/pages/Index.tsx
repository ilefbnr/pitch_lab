import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Dashboard from "@/components/Dashboard";
import FeedbackDisplay from "@/components/FeedbackDisplay";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main>
        <Hero />
        <Dashboard />
        <FeedbackDisplay />
      </main>
    </div>
  );
};

export default Index;
