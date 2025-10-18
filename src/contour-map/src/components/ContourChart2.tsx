import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import hullData from "../data/mnist_hull.json";

interface Point {
  id: number;
  x: number;
  y: number;
  cluster: string;
}

interface Cluster {
  id: string;
  points: number[]; // 点IDの配列（凸包）
  child_level: number | null;
}

const ContourChart: React.FC = () => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = 900;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };

    const points: Point[] = hullData.points;
    const clusters: Cluster[] = hullData.clusters;

    // スケール
    const x = d3
      .scaleLinear()
      .domain(d3.extent(points, (d) => d.x) as [number, number])
      .range([margin.left, width - margin.right]);

    const y = d3
      .scaleLinear()
      .domain(d3.extent(points, (d) => d.y) as [number, number])
      .range([height - margin.bottom, margin.top]);

    // カラースケール
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // -------------------------------
    // 凸包の描画
    // -------------------------------
    const hullGroup = svg.append("g").attr("fill", "none").attr("stroke-width", 1);

    const hullPaths = hullGroup
      .selectAll("path")
      .data(clusters)
      .join("path")
      .attr("stroke", "#4a90e2")
      .attr("opacity", 0.4)
      .attr("d", (cluster) => {
        const clusterPoints = cluster.points.map((i) => [x(points[i].x), y(points[i].y)]);
        // Path文字列を作成
        if (clusterPoints.length < 3) return "";
        const lineGenerator = d3.line<number[]>().x(d => d[0]).y(d => d[1]).curve(d3.curveLinearClosed);
        return lineGenerator(clusterPoints);
      })
      // ホバーイベント
      .on("mouseover", function (event, cluster) {
        hullPaths.attr("opacity", 0.1);
        d3.select(this)
          .raise()
          .attr("stroke", "#ff3333")
          .attr("opacity", 1)
          .attr("stroke-width", 2);
      })
      .on("mouseout", function () {
        hullPaths.attr("stroke", "#4a90e2").attr("opacity", 0.4).attr("stroke-width", 1);
      });

    // -------------------------------
    // 点群の描画
    // -------------------------------
    svg
      .append("g")
      .selectAll("circle")
      .data(points)
      .join("circle")
      .attr("cx", (d) => x(d.x))
      .attr("cy", (d) => y(d.y))
      .attr("r", 2)
      .attr("fill", (d) => color(d.cluster))
      .attr("opacity", 0.7);

    // -------------------------------
    // 軸
    // -------------------------------
    svg
      .append("g")
      .attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(10));

    svg
      .append("g")
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(10));
  }, []);

  return (
    <svg
      ref={ref}
      width={900}
      height={600}
      style={{
        display: "block",
        margin: "0 auto",
        border: "1px solid #ccc",
        backgroundColor: "#fff",
      }}
    />
  );
};

export default ContourChart;
