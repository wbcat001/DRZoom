import React from 'react';
import { useAppContext, useSelection } from '../../store/useAppStore.tsx';
import './ClusterDetails.css';

const ClusterDetails: React.FC = () => {
  const { state } = useAppContext();
  const { selection } = useSelection();

  // Get the first selected cluster ID
  const selectedClusterId = Array.from(selection.selectedClusterIds)[0];

  if (!selectedClusterId) {
    return (
      <div className="panel cluster-details-panel">
        <div className="panel-header">Cluster Details</div>
        <div className="panel-content">
          <p className="no-selection">Select a cluster to view details</p>
        </div>
      </div>
    );
  }

  const clusterName = state.clusterNames[selectedClusterId] || `Cluster ${selectedClusterId}`;
  const clusterWords = state.clusterWords[selectedClusterId] || [];
  const clusterMeta = state.clusterMetadata[selectedClusterId];

  return (
    <div className="panel cluster-details-panel">
      <div className="panel-header">Cluster Details</div>
      <div className="panel-content">
        <div className="detail-section">
          <h3 className="cluster-title">{clusterName}</h3>
          <div className="cluster-id">ID: {selectedClusterId}</div>
        </div>

        {clusterWords.length > 0 && (
          <div className="detail-section">
            <h4>Representative Words</h4>
            <div className="word-list">
              {clusterWords.map((word, idx) => (
                <span key={idx} className="word-tag">{word}</span>
              ))}
            </div>
          </div>
        )}

        {clusterMeta && (
          <div className="detail-section">
            <h4>Statistics</h4>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">Size:</span>
                <span className="stat-value">{clusterMeta.z || 0}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Stability:</span>
                <span className="stat-value">{clusterMeta.s?.toFixed(3) || '0.000'}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">Strahler:</span>
                <span className="stat-value">{clusterMeta.h || 0}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ClusterDetails;
