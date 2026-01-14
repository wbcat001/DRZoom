import React, { useState } from 'react';
import './NearbyClusterConfig.css';

export interface NearbyClusterParams {
  mode: 'stability' | 'size_ratio';
  minStability: number;
  ratioThreshold: number;
  maxResults: number;
}

interface NearbyClusterConfigProps {
  params: NearbyClusterParams;
  onParamsChange: (params: NearbyClusterParams) => void;
}

const NearbyClusterConfig: React.FC<NearbyClusterConfigProps> = ({
  params,
  onParamsChange,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleModeChange = (mode: 'stability' | 'size_ratio') => {
    onParamsChange({ ...params, mode });
  };

  const handleMinStabilityChange = (value: number) => {
    onParamsChange({ ...params, minStability: value });
  };

  const handleRatioThresholdChange = (value: number) => {
    onParamsChange({ ...params, ratioThreshold: value });
  };

  const handleMaxResultsChange = (value: number) => {
    onParamsChange({ ...params, maxResults: value });
  };

  return (
    <div className="nearby-cluster-config">
      <button
        className="nearby-cluster-toggle"
        onClick={() => setIsOpen(!isOpen)}
        title="Configure nearby cluster detection parameters"
      >
        ⚙️ Nearby Clusters
      </button>

      {isOpen && (
        <div className="nearby-cluster-panel">
          <div className="config-header">
            <h4>Nearby Cluster Detection</h4>
            <button
              className="close-btn"
              onClick={() => setIsOpen(false)}
              title="Close panel"
            >
              ✕
            </button>
          </div>

          <div className="config-section">
            <label>Mode:</label>
            <div className="mode-buttons">
              <button
                className={`mode-btn ${params.mode === 'stability' ? 'active' : ''}`}
                onClick={() => handleModeChange('stability')}
              >
                Stability
              </button>
              <button
                className={`mode-btn ${params.mode === 'size_ratio' ? 'active' : ''}`}
                onClick={() => handleModeChange('size_ratio')}
              >
                Size Ratio
              </button>
            </div>
          </div>

          {params.mode === 'stability' && (
            <div className="config-section">
              <label htmlFor="min-stability">Min Stability: {params.minStability}</label>
              <input
                id="min-stability"
                type="range"
                min="0"
                max="100"
                step="5"
                value={params.minStability}
                onChange={(e) => handleMinStabilityChange(Number(e.target.value))}
                className="slider"
              />
              <input
                type="number"
                min="0"
                max="100"
                value={params.minStability}
                onChange={(e) => handleMinStabilityChange(Number(e.target.value))}
                className="number-input"
              />
            </div>
          )}

          {params.mode === 'size_ratio' && (
            <div className="config-section">
              <label htmlFor="ratio-threshold">
                Size Ratio Threshold: {params.ratioThreshold.toFixed(2)}
              </label>
              <input
                id="ratio-threshold"
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={params.ratioThreshold}
                onChange={(e) => handleRatioThresholdChange(Number(e.target.value))}
                className="slider"
              />
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={params.ratioThreshold.toFixed(2)}
                onChange={(e) => handleRatioThresholdChange(Number(e.target.value))}
                className="number-input"
              />
            </div>
          )}

          <div className="config-section">
            <label htmlFor="max-results">Max Results: {params.maxResults}</label>
            <input
              id="max-results"
              type="range"
              min="10"
              max="500"
              step="10"
              value={params.maxResults}
              onChange={(e) => handleMaxResultsChange(Number(e.target.value))}
              className="slider"
            />
            <input
              type="number"
              min="10"
              max="500"
              value={params.maxResults}
              onChange={(e) => handleMaxResultsChange(Number(e.target.value))}
              className="number-input"
            />
          </div>

          <div className="config-info">
            <p>
              <strong>Mode:</strong>{' '}
              {params.mode === 'stability'
                ? 'Find parent cluster with sufficient stability'
                : 'Find parent cluster with significant size ratio'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default NearbyClusterConfig;
