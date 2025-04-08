import { useState, useEffect } from "react";
import { Brain, Search, MessageSquare } from "lucide-react";

const steps = [
  {
    icon: Search,
    title: "Search for a Book",
    description: "Enter the title of the book you want to explore.",
  },
  {
    icon: Brain,
    title: "AI Analysis",
    description: "The AI analyzes the book and generates a mind map.",
  },
  {
    icon: MessageSquare,
    title: "Explore Insights",
    description:
      "Ask questions and explore relationships, themes, and insights.",
  },
];

export default function HowItWorks() {
  const [activeStep, setActiveStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prevStep) => (prevStep + 1) % steps.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <section className="py-24 bg-gradient-to-b from-indigo-50 to-white">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-500 mb-4">
            How It Works
          </h2>
          <p className="text-lg text-indigo-600/80">
            Discover the power of AI-driven book analysis
          </p>
        </div>
        <div className="flex flex-col md:flex-row justify-center items-center space-y-8 md:space-y-0 md:space-x-8">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`group relative bg-white rounded-xl p-8 
                         shadow-lg transition-all duration-300 
                         hover:shadow-2xl hover:-translate-y-1
                         hover:bg-gradient-to-br hover:from-indigo-50 hover:to-blue-50
                         border border-indigo-100 ${
                           index === activeStep
                             ? "scale-110 shadow-xl"
                             : "scale-100"
                         }`}
            >
              <div
                className="absolute inset-0 bg-gradient-to-r from-indigo-500/5 to-blue-500/5 
                            rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"
              />
              <step.icon
                className={`w-16 h-16 mx-auto mb-6 
                            transition-transform duration-300 
                            group-hover:scale-110 group-hover:rotate-3 ${
                              index === activeStep
                                ? "text-indigo-600"
                                : "text-indigo-400"
                            }`}
              />
              <h3
                className="text-xl font-bold text-gray-900 mb-3 
                           group-hover:text-indigo-700 transition-colors duration-300"
              >
                {step.title}
              </h3>
              <p
                className="text-gray-600 group-hover:text-indigo-900/80 
                          transition-colors duration-300 leading-relaxed"
              >
                {step.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
