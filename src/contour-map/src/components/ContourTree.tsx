// src/components/ContourTree.tsx
import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import { sampleTree } from "../data/sampleTree";

const width = 600;
const height = 600;

const ContourTree: React.FC = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // 初期化

    // ----------------------------
    // 1️⃣ 階層構造をD3で変換
    // ----------------------------
    const root = d3
      .hierarchy(sampleTree)
      .sum((d: any) => d.value || 0)
      .sort((a, b) => (b.value || 0) - (a.value || 0));

    // ----------------------------
    // 2️⃣ 円充填レイアウト (pack)
    // ----------------------------
    const pack = d3.pack().size([width, height]).padding(10);
    const nodes = pack(root).descendants();

    // ----------------------------
    // 3️⃣ 等高線生成 (contourDensity)
    // ----------------------------
    const contours = d3
      .contourDensity()
      .x((d: any) => d.x)
      .y((d: any) => d.y)
      .size([width, height])
      .bandwidth(30)
      .thresholds(15)(
        nodes.map((n) => ({ x: n.x, y: n.y }))
      );

    // ----------------------------
    // 4️⃣ 等高線の描画
    // ----------------------------
    svg
      .selectAll("path.contour")
      .data(contours)
      .join("path")
      .attr("class", "contour")
      .attr("d", d3.geoPath())
      .attr("fill", (d, i) => d3.interpolateYlGnBu(i / contours.length))
      .attr("opacity", 0.4);

    // ----------------------------
    // 5️⃣ ノード（階層）描画
    // ----------------------------
    svg
      .selectAll("circle.node")
      .data(nodes)
      .join("circle")
      .attr("class", "node")
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y)
      .attr("r", (d) => d.r)
      .attr("fill", (d) => (d.children ? "#69b3a2" : "#ffd166"))
      .attr("stroke", "#333")
      .attr("stroke-width", 0.5)
      .attr("opacity", 0.8);

    // ----------------------------
    // 6️⃣ ノードラベル
    // ----------------------------
    svg
      .selectAll("text.label")
      .data(nodes)
      .join("text")
      .attr("class", "label")
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .attr("dy", "0.3em")
      .attr("text-anchor", "middle")
      .text((d) => d.data.name)
      .style("font-size", "10px")
      .style("fill", "#222");
  }, []);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{ background: "#f8f9fa", borderRadius: "8px" }}
    />
  );
};

export default ContourTree;
