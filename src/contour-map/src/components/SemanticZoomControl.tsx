import React from 'react';

// --- 型定義 ---
interface ControlProps {
    levels: number[];
    selectedLevel: number;
    // イベントハンドラの型を定義: 数値を受け取る関数
    onChange: (newLevel: number) => void;
}

const SemanticZoomControl: React.FC<ControlProps> = ({ levels, selectedLevel, onChange }) => {
  const minLevel = levels.length > 0 ? levels[0] : 1;
  const maxLevel = levels.length > 0 ? levels[levels.length - 1] : 1;

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', margin: '10px 0' }}>
      <h3>Zoom Level (HDBSCAN Hierarchy)</h3>
      <input
        type="range"
        min={minLevel}
        max={maxLevel}
        step="1"
        value={selectedLevel}
        onChange={(e) => onChange(parseInt(e.target.value))}
        style={{ width: '300px' }}
        disabled={levels.length === 0}
      />
      <p>Selected Level: {selectedLevel} / {levels.length}</p>
      <p>Overview First (Low) ↔ Details on Demand (High)</p>
    </div>
  );
};

export default SemanticZoomControl;