import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

interface Point {
  id: number;
  x: number;
  y: number;
  cluster: string;
}

interface Cluster {
  id: string;
  points: number[];
  cluster_points: number[];
  child_level: any | null;
}

interface HullData {
  level: number;
  points: Point[];
  clusters: Cluster[];
}

const ContourChart: React.FC = () => {
  const ref = useRef<SVGSVGElement | null>(null);
  const [currentData, setCurrentData] = useState<HullData | null>(null);

  // -------------------------------
  // データ取得関数
  // -------------------------------
  const fetchClusterData = async (level: number = 0, parentIndices?: number[]) => {
    try {
      let url = `http://localhost:8000/clusters?level=${level}`;
      if (parentIndices && parentIndices.length > 0) {
        url += `&parent_indices=${parentIndices.join(",")}`;
      }
      
      console.log(`Fetching data from: ${url}`);
      
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data: HullData = await res.json();
      console.log("Received data:", data);
      setCurrentData(data);
    } catch (error) {
      console.error("Error fetching cluster data:", error);
    }
  };

  // 初期データ取得
  useEffect(() => {
    fetchClusterData(0);
  }, []);

  // -------------------------------
  // 描画
  // -------------------------------
  useEffect(() => {
    if (!currentData) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = 900;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };

    const points: Point[] = currentData.points;
    const clusters: Cluster[] = currentData.clusters;

    const x = d3
      .scaleLinear()
      .domain(d3.extent(points, (d) => d.x) as [number, number])
      .range([margin.left, width - margin.right]);

    const y = d3
      .scaleLinear()
      .domain(d3.extent(points, (d) => d.y) as [number, number])
      .range([height - margin.bottom, margin.top]);

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // -------------------------------
    // 凸包描画
    // -------------------------------
    const hullGroup = svg.append("g").attr("fill", "none").attr("stroke-width", 1);

    const hullPaths = hullGroup
      .selectAll("path")
      .data(clusters)
      .join("path")
      .attr("stroke", "#4a90e2")
      .attr("opacity", 0.4)
      .attr("d", (cluster) => {
        // points配列からcluster.pointsに該当する点を検索
        const clusterCoords: [number, number][] = [];
        cluster.points.forEach(pointId => {
          const point = points.find(p => p.id === pointId);
          if (point) {
            clusterCoords.push([x(point.x), y(point.y)]);
          }
        });
        
        if (clusterCoords.length < 3) return "";
        const lineGenerator = d3.line<[number, number]>()
          .x(d => d[0])
          .y(d => d[1])
          .curve(d3.curveLinearClosed);
        return lineGenerator(clusterCoords);
      })
      // ホバー
      .on("mouseover", function () {
        hullPaths.attr("opacity", 0.1);
        d3.select(this).attr("stroke", "#ff3333").attr("opacity", 1).attr("stroke-width", 2).raise();
      })
      .on("mouseout", function () {
        hullPaths.attr("stroke", "#4a90e2").attr("opacity", 0.4).attr("stroke-width", 1);
      })
      // クリックでドリルダウン
      .on("click", function (_, cluster) {
        console.log("Clicked cluster:", cluster);
        if (!cluster.cluster_points || cluster.cluster_points.length < 1) return;
        // 子階層はバックエンドから取得（cluster_pointsを使用）
        fetchClusterData(currentData.level + 1, cluster.cluster_points);
      });

    // -------------------------------
    // 点群描画
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

    // -------------------------------
    // ズーム
    // -------------------------------
    const g = svg.append("g");
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 10])
      .on("zoom", (event) => {
        g.attr("transform", event.transform.toString());
      });
    
    if (ref.current) {
      d3.select(ref.current).call(zoom);
    }
  }, [currentData]);

  // -------------------------------
  // ズームアウト
  // -------------------------------
  const handleZoomOut = () => {
    fetchClusterData(0);
  };

  return (
    <div>
      <svg
        ref={ref}
        width={900}
        height={600}
        style={{ display: "block", margin: "0 auto", border: "1px solid #ccc", backgroundColor: "#fff" }}
      />
      <button onClick={handleZoomOut} style={{ display: "block", margin: "10px auto" }}>
        Zoom Out
      </button>
    </div>
  );
};

export default ContourChart;
