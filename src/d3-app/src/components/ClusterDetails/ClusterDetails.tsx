import React, { useState } from 'react';
import './ClusterDetails.css';

interface ClusterInfo {
  id: number;
  size: number;
  stability: number;
  color: string;
  description: string;
}

const ClusterDetails: React.FC = () => {
  const [selectedCluster, setSelectedCluster] = useState<number>(0);

  // Sample cluster data
  const clusterData: ClusterInfo[] = [
    {
      id: 0,
      size: 25,
      stability: 0.85,
      color: '#1f77b4',
      description: 'Dense cluster with high cohesion'
    },
    {
      id: 1,
      size: 30,
      stability: 0.92,
      color: '#ff7f0e',
      description: 'Largest cluster, well-separated'
    },
    {
      id: 2,
      size: 18,
      stability: 0.78,
      color: '#2ca02c',
      description: 'Moderate density cluster'
    },
    {
      id: 3,
      size: 22,
      stability: 0.88,
      color: '#d62728',
      description: 'Compact circular cluster'
    },
    {
      id: 4,
      size: 15,
      stability: 0.72,
      color: '#9467bd',
      description: 'Smallest cluster, elongated shape'
    }
  ];

  const currentCluster = clusterData.find(c => c.id === selectedCluster) || clusterData[0];

  return (
    <div className="panel">
      <div className="panel-header">
        Cluster Details
      </div>
      
      <div className="panel-content">
        {/* Cluster selector */}
        <div className="cluster-selector">
          <label htmlFor="cluster-select">Select Cluster:</label>
          <select 
            id="cluster-select"
            className="form-control"
            value={selectedCluster}
            onChange={(e) => setSelectedCluster(parseInt(e.target.value))}
          >
            {clusterData.map(cluster => (
              <option key={cluster.id} value={cluster.id}>
                Cluster {cluster.id} ({cluster.size} points)
              </option>
            ))}
          </select>
        </div>

        {/* Cluster information */}
        <div className="cluster-info">
          <div className="cluster-header">
            <div 
              className="cluster-color-indicator" 
              style={{ backgroundColor: currentCluster.color }}
            ></div>
            <h4>Cluster {currentCluster.id}</h4>
          </div>

          <div className="cluster-stats">
            <div className="stat-item">
              <div className="stat-label">Size:</div>
              <div className="stat-value">{currentCluster.size} points</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-label">Stability:</div>
              <div className="stat-value">
                {currentCluster.stability.toFixed(3)}
                <div className="stability-bar">
                  <div 
                    className="stability-fill"
                    style={{ width: `${currentCluster.stability * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>

            <div className="stat-item">
              <div className="stat-label">Description:</div>
              <div className="stat-value description">{currentCluster.description}</div>
            </div>
          </div>

          {/* Sample points list */}
          <div className="sample-points">
            <h5>Sample Points:</h5>
            <div className="points-list">
              {Array.from({ length: Math.min(currentCluster.size, 8) }, (_, i) => (
                <div key={i} className="point-item">
                  <div 
                    className="point-color" 
                    style={{ backgroundColor: currentCluster.color }}
                  ></div>
                  <span>Point {i + 1}</span>
                  <span className="point-coords">
                    ({(Math.random() * 10).toFixed(2)}, {(Math.random() * 10).toFixed(2)})
                  </span>
                </div>
              ))}
              {currentCluster.size > 8 && (
                <div className="more-points">
                  ... and {currentCluster.size - 8} more points
                </div>
              )}
            </div>
          </div>

          {/* Cluster metrics */}
          <div className="cluster-metrics">
            <h5>Metrics:</h5>
            <div className="metrics-grid">
              <div className="metric-card">
                <div className="metric-name">Density</div>
                <div className="metric-value">{(Math.random() * 0.5 + 0.3).toFixed(3)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-name">Cohesion</div>
                <div className="metric-value">{(Math.random() * 0.3 + 0.6).toFixed(3)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-name">Separation</div>
                <div className="metric-value">{(Math.random() * 0.4 + 0.4).toFixed(3)}</div>
              </div>
              <div className="metric-card">
                <div className="metric-name">Silhouette</div>
                <div className="metric-value">{(Math.random() * 0.6 + 0.2).toFixed(3)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClusterDetails;
