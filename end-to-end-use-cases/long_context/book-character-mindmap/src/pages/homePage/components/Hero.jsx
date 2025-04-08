import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import Lottie from "lottie-react";

export default function Hero() {
  const navigate = useNavigate();
  const [animationData, setAnimationData] = useState(null);

  useEffect(() => {
    const fetchAnimationData = async () => {
      try {
        const response = await fetch(
          "https://lottie.host/909e50db-34ef-47ac-8384-db31f6fc0654/e2xX6qZZ7i.json"
        );
        const data = await response.json();
        setAnimationData(data);
      } catch (error) {
        console.error("Error fetching Lottie animation:", error);
      }
    };

    fetchAnimationData();
  }, []);

  return (
    <section className="relative h-screen flex items-center justify-center overflow-hidden">
      <div className="absolute inset-0 z-0">
        {animationData && (
          <Lottie
            animationData={animationData}
            style={{ width: "100%", height: "100%" }}
          />
        )}
      </div>
      <div className="absolute inset-0 bg-gradient-to-b from-transparent to-indigo-900 opacity-75 z-0"></div>
      <div className="relative z-10 text-center space-y-6 max-w-4xl mx-auto px-4">
        <h1 className="text-5xl md:text-6xl font-bold text-white leading-tight">
          Unravel Stories <br /> One Map at a Time
        </h1>
        <p className="text-xl md:text-2xl text-white">
          Explore character relationships and storylines with AI-powered
          visualizations.
        </p>
        <button
          onClick={() => navigate("/search")}
          className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white font-semibold py-3 px-8 rounded-full text-lg transition-all duration-300 transform hover:scale-105"
        >
          Get Started
        </button>
      </div>
    </section>
  );
}
