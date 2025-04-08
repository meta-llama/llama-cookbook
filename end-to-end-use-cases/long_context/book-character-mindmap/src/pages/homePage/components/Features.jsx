import { BookOpen, MessageSquare, FileText, Users } from "lucide-react";

const features = [
  {
    icon: BookOpen,
    title: "Interactive Mind Maps",
    description:
      "Visualize relationships between characters and plot elements.",
  },
  {
    icon: MessageSquare,
    title: "AI Chatbot",
    description:
      "Ask deep questions about the book and get insightful answers.",
  },
  {
    icon: FileText,
    title: "Book Summaries",
    description: "Get concise overviews of plots and themes.",
  },
  {
    icon: Users,
    title: "Community Contributions",
    description: "Add and refine maps with fellow book lovers.",
  },
];

export default function Features() {
  return (
    <section className="py-24 bg-gradient-to-b from-white to-indigo-50">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-500 mb-4">
            Key Features
          </h2>
          <p className="text-lg text-indigo-600/80">
            Discover the power of AI-driven book analysis
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group relative bg-white rounded-xl p-8 
                         shadow-lg transition-all duration-300 
                         hover:shadow-2xl hover:-translate-y-1
                         hover:bg-gradient-to-br hover:from-indigo-50 hover:to-blue-50
                         border border-indigo-100"
            >
              <div
                className="absolute inset-0 bg-gradient-to-r from-indigo-500/5 to-blue-500/5 
                            rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"
              />
              <feature.icon
                className="w-12 h-12 text-indigo-600 mx-auto mb-6 
                                    transition-transform duration-300 
                                    group-hover:scale-110 group-hover:rotate-3"
              />
              <h3
                className="text-xl font-bold text-gray-900 mb-3 
                           group-hover:text-indigo-700 transition-colors duration-300"
              >
                {feature.title}
              </h3>
              <p
                className="text-gray-600 group-hover:text-indigo-900/80 
                          transition-colors duration-300 leading-relaxed"
              >
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
