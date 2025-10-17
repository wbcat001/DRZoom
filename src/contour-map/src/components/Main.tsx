import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import InteractiveScatterPlot from './InteractiveScatterPlot';
import SemanticZoomControl from './SemanticZoomControl';

// --- 型定義 ---
// バックエンドの /api/zoom から返されるデータの型
interface EmbeddedData {
  level_id: string;
  coordinates: number[][]; // [ [x1, y1], [x2, y2], ... ]
  indices: number[]; // 高次元データのインデックス
  cluster_labels: number[]; // 各点に対応するクラスタID
  is_representative: boolean;
}

const API_BASE_URL = 'http://localhost:8000/api';

function App() {
  // useStateのジェネリクスに型を適用
  const [currentData, setCurrentData] = useState<EmbeddedData | null>(null); 
  const [prevCoordinates, setPrevCoordinates] = useState<number[][] | null>(null); 
  const [currentLevelId, setCurrentLevelId] = useState<string | null>(null); 
  const [zoomLevels, setZoomLevels] = useState<number[]>([]); // レベルを数値の配列として扱う
  const [selectedLevel, setSelectedLevel] = useState<number>(1);

  // ズームレベルのリストを取得
  useEffect(() => {
    axios.get<number[]>(`${API_BASE_URL}/levels`).then(res => {
      setZoomLevels(res.data);
    });
    // 初期 Overview をロード
    fetchData(1, null);
  }, []);

  const fetchData = useCallback(async (level: number, prevId: string | null) => {
    if (currentData) {
        setPrevCoordinates(currentData.coordinates);
    }

    try {
      const response = await axios.post<EmbeddedData>(`${API_BASE_URL}/zoom`, {
        zoom_level: level,
        prev_level_id: prevId,
      });
      
      const newData = response.data;
      
      setCurrentData(newData);
      setCurrentLevelId(newData.level_id);
      setSelectedLevel(level);

    } catch (error) {
      console.error("Error fetching zoom data:", error);
    }
  }, [currentData]); // currentDataを依存配列に含めるのは、前の座標をセットするため

  const handleZoomLevelChange = (newLevel: number) => {
    if (newLevel !== selectedLevel) {
      fetchData(newLevel, currentLevelId);
    }
  };

  return (
    <div>
      <h1>HDBSCAN Semantic Zoom</h1>
      <SemanticZoomControl 
        levels={zoomLevels} 
        selectedLevel={selectedLevel} 
        onChange={handleZoomLevelChange} 
      />
      {currentData ? (
        <InteractiveScatterPlot 
          currentData={currentData} 
          prevCoordinates={prevCoordinates} 
          currentLevelId={currentLevelId}
        />
      ) : (
        <p>Loading data...</p>
      )}
    </div>
  );
}

export default App;