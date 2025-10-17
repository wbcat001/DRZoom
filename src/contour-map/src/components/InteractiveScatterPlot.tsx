import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import axios from 'axios';

// --- 型定義 ---

// App.tsx から渡されるPropsの型
interface PlotProps {
  currentData: EmbeddedData;
  prevCoordinates: number[][] | null;
  currentLevelId: string | null;
}

// 描画データに変換された後のD3データバインディング用オブジェクトの型
interface PointData {
    id: number;
    x: number;
    y: number;
    cluster: number;
    prevX: number;
    prevY: number;
}

// App.tsxのEmbeddedData型を再利用
interface EmbeddedData {
  level_id: string;
  coordinates: number[][]; 
  indices: number[];
  cluster_labels: number[];
  is_representative: boolean;
}

const API_BASE_URL = 'http://localhost:8000/api';
const WIDTH = 800;
const HEIGHT = 600;

const InteractiveScatterPlot: React.FC<PlotProps> = ({ currentData, prevCoordinates, currentLevelId }) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [contourData, setContourData] = useState<any>(null); // TODO: ContourDataの型を定義
  
  // スケールを決定する関数（簡略化）
  const getScales = (coords: number[][]) => {
    const margin = 50;
    const allX = coords.map(c => c[0]);
    const allY = coords.map(c => c[1]);

    const xScale = d3.scaleLinear()
      .domain(d3.extent(allX) as [number, number])
      .range([margin, WIDTH - margin]);

    const yScale = d3.scaleLinear()
      .domain(d3.extent(allY) as [number, number])
      .range([HEIGHT - margin, margin]); // Y軸は反転

    return { xScale, yScale };
  };

  // クラスタの境界線（等高線）データを取得する関数
  const fetchContourData = async (clusterId: number) => {
    try {
      setContourData(null); // 古い等高線をクリア
      const response = await axios.post(`${API_BASE_URL}/contours`, {
        level_id: currentLevelId,
        cluster_id: clusterId,
      });
      setContourData(response.data);
    } catch (error) {
      console.error("Error fetching contour data:", error);
      setContourData(null);
    }
  };

  useEffect(() => {
    if (!currentData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const { coordinates, cluster_labels, indices } = currentData;
    const { xScale, yScale } = getScales(coordinates); 

    // --- 1. データ結合と変換 ---
    const data: PointData[] = coordinates.map((coord, i) => {
        const id = indices[i];
        
        // 前の座標を探す（インデックス i が対応すると仮定）
        const [prevX, prevY] = prevCoordinates && prevCoordinates.length === coordinates.length
            ? prevCoordinates[i] 
            : coord;

        return {
            id,
            x: coord[0],
            y: coord[1],
            cluster: cluster_labels[i],
            prevX: prevX, 
            prevY: prevY,
        };
    });

    // --- 2. D3データバインディング (点の描画とトランジション) ---
    const plotArea = svg.select<SVGGElement>(".plot-area"); // 描画エリアを取得
    
    // データバインディング。キー関数にid（高次元インデックス）を使用
    const circles = plotArea.selectAll<SVGCircleElement, PointData>(".data-point")
      .data(data, d => d.id.toString()); 

    // Exit: 削除される点
    circles.exit()
      .transition().duration(500)
      .attr("r", 0)
      .remove();

    // Enter: 新しく入ってくる点
    circles.enter()
      .append("circle")
      .attr("class", "data-point")
      .attr("r", 3)
      // 初期位置を前の座標、または現在の座標に設定
      .attr("cx", d => xScale(d.prevX)) 
      .attr("cy", d => yScale(d.prevY))
      .style("fill", d => `hsl(${d.cluster * 50 % 360}, 70%, 50%)`)
      .on("mouseover", (event, d) => {
        // 例: ホバーしたクラスタの等高線データをフェッチ
        fetchContourData(d.cluster);
      })
      .merge(circles) // EnterとUpdateの要素を結合
      .transition().duration(1000) // 1秒間の滑らかなアニメーション
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y));

    // --- 3. 等高線（Contour）の描画（ContourDataがある場合） ---
    // ここに d3-contour を使った描画ロジックを実装します。
    if (contourData && contourData.heatmap) {
      // TODO: 等高線描画
    } else {
      // 等高線をクリア
      plotArea.selectAll(".contour-path").remove();
    }

  }, [currentData, prevCoordinates, currentLevelId]); 

  return (
    <svg ref={svgRef} width={WIDTH} height={HEIGHT}>
      {/* 等高線や軸などの要素を追加するためのグループ */}
      <g className="plot-area"></g>
    </svg>
  );
};

export default InteractiveScatterPlot;