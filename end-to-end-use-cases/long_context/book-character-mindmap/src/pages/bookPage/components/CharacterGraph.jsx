import { useState, useEffect, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";

export default function CharacterGraph({ graphData }) {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 600, height: 600 });
  const [hoveredLink, setHoveredLink] = useState(null);
  const fgRef = useRef();

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: Math.max(300, containerRef.current.offsetHeight),
        });
      }
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  return (
    <div className="bg-white shadow-lg rounded-lg p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        Character Relationship Graph
      </h2>
      <div ref={containerRef} className="w-full h-[600px]">
        <ForceGraph2D
          ref={fgRef}
          graphData={graphData}
          nodeAutoColorBy="val"
          nodeCanvasObject={(node, ctx, globalScale) => {
            // Always show node label
            const label = node.name;
            const fontSize = 20 / globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            const textWidth = ctx.measureText(label).width;
            const bckgDimensions = [textWidth, fontSize].map(
              (n) => n + fontSize * 0.2
            );

            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = node.color;
            ctx.fillText(label, node.x, node.y);

            node.__bckgDimensions = bckgDimensions;
          }}
          onLinkHover={(link) => setHoveredLink(link ? `${link.source.id}-${link.target.id}` : null)}
          nodePointerAreaPaint={(node, color, ctx) => {
            ctx.fillStyle = color;
            const bckgDimensions = node.__bckgDimensions;
            bckgDimensions &&
              ctx.fillRect(
                node.x - bckgDimensions[0] / 2,
                node.y - bckgDimensions[1] / 2,
                ...bckgDimensions
              );
          }}
          linkCanvasObject={(link, ctx, globalScale) => {
            const start = link.source;
            const end = link.target;
            ctx.beginPath();
            ctx.moveTo(start.x, start.y);
            ctx.lineTo(end.x, end.y);
            ctx.strokeStyle = "#9ca3af";
            ctx.lineWidth = 0.5;
            ctx.stroke();

            // Only draw label if link is hovered
            const linkId = `${link.source.id}-${link.target.id}`;
            if (hoveredLink === linkId && link.label) {
              const textPos = {
                x: start.x + (end.x - start.x) / 2,
                y: start.y + (end.y - start.y) / 2
              };

              const fontSize = 3 + 1/globalScale;
              ctx.font = `${fontSize}px Arial`;

              const textWidth = ctx.measureText(link.label).width;
              const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

              ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
              ctx.fillRect(
                textPos.x - bckgDimensions[0] / 2,
                textPos.y - bckgDimensions[1] / 2,
                ...bckgDimensions
              );

              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = '#666';
              ctx.fillText(link.label, textPos.x, textPos.y);
            }
          }}
          onNodeDragEnd={node => {
            node.fx = node.x;
            node.fy = node.y;
          }}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={3}
          linkWidth={1}
          backgroundColor="#ffffff"
          width={dimensions.width}
          height={dimensions.height}
          onEngineStop={() => fgRef.current.zoomToFit(600)}
          cooldownTicks={100}
        />
      </div>
    </div>
  );
}
