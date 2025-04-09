import "../index.css";

export const metadata = {
  title: "BookMind - Unravel Stories, One Map at a Time",
  description:
    "Explore character relationships and storylines with AI-powered visualizations.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <title>{metadata.title}</title>
        <meta name="description" content={metadata.description} />
      </head>
      <body>{children}</body>
    </html>
  );
}
