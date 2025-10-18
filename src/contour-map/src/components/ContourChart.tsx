import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import pointsData from "../data/points.json";

interface Point {
  x: number;
  y: number;
  cluster: number;
}

const ContourChart: React.FC = () => {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = 900;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };

    const points: Point[] = pointsData as Point[];

    // スケール
    const x = d3
      .scaleLinear()
      .domain(d3.extent(points, (d) => d.x) as [number, number])
      .range([margin.left, width - margin.right]);

    const y = d3
      .scaleLinear()
      .domain(d3.extent(points, (d) => d.y) as [number, number])
      .range([height - margin.bottom, margin.top]);

    // 等高線を計算
    const contours = d3
      .contourDensity<Point>()
      .x((d) => x(d.x))
      .y((d) => y(d.y))
      .size([width, height])
      .bandwidth(20)
      .thresholds(25)(points);

    // 等高線描画
    const contourGroup = svg
      .append("g")
      .attr("fill", "none")
      .attr("stroke-linejoin", "round");

    const paths = contourGroup
      .selectAll("path")
      .data(contours)
      .join("path")
      .attr("stroke", "#4a90e2")
      .attr("stroke-width", (d, i) => (i % 5 === 0 ? 1.0 : 0.3))
      .attr("opacity", 0.6)
      .attr("d", d3.geoPath())
      // ✅ ホバーイベントを追加
      .on("mouseover", function (event, d) {
        // 全てのパスを薄くする
        paths.attr("opacity", 0.15);

        // 選択されたパスを強調
        d3.select(this)
          .raise() // 一番前に持ってくる
          .attr("stroke", "#ff3333")
          .attr("stroke-width", 2)
          .attr("opacity", 1.0);
      })
      .on("mouseout", function () {
        // スタイルをリセット
        paths
          .attr("stroke", "#4a90e2")
          .attr("stroke-width", (d, i) => (i % 5 === 0 ? 1.0 : 0.3))
          .attr("opacity", 0.6);
      });

    // 散布図
    const color = d3.scaleOrdinal(d3.schemeCategory10);
    svg
      .append("g")
      .selectAll("circle")
      .data(points)
      .join("circle")
      .attr("cx", (d) => x(d.x))
      .attr("cy", (d) => y(d.y))
      .attr("r", 2)
      .attr("fill", (d) => color(String(d.cluster)))
      .attr("opacity", 0.7);

    // 軸
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
