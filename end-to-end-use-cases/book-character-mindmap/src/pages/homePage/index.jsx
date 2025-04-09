import Hero from "./components/Hero";
import Features from "./components/Features";
import HowItWorks from "./components/HowItWorks";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-indigo-100 to-white">
      <Hero />
      <Features />
      <HowItWorks />
    </main>
  );
}
