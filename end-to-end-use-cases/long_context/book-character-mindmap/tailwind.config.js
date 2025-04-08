/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      keyframes: {
        fadeInUp: {
          "0%": { opacity: 0, transform: "translateY(20px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        fadeOutDown: {
          "0%": { opacity: 1, transform: "translateY(0)" },
          "100%": { opacity: 0, transform: "translateY(-20px)" },
        },
      },
      animation: {
        fadeInUp: "fadeInUp 0.3s ease-out",
        fadeOutDown: "fadeOutDown 0.3s ease-in",
      },
    },
  },
  plugins: [],
};
