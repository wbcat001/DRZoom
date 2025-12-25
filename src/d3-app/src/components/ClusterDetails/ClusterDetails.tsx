import React from 'react';
import { useAppContext, useSelection } from '../../store/useAppStore.tsx';
import './ClusterDetails.css';

const ClusterDetails: React.FC = () => {
  const { state } = useAppContext();
  const { selection } = useSelection();

  const selectedClusterIds = React.useMemo(
    () => Array.from(selection.selectedClusterIds),
    [selection.selectedClusterIds]
  );

  const clusters = React.useMemo(() => {
    return selectedClusterIds.map((cid) => {
      const name = state.clusterNames[cid] || `Cluster ${cid}`;
      const words = state.clusterWords[cid] || [];
      const meta = state.clusterMetadata[cid];
      return {
        id: cid,
        name,
        words,
        size: meta?.z ?? 0,
        stability: meta?.s ?? 0,
        strahler: meta?.h ?? 0
      };
    });
  }, [selectedClusterIds, state.clusterNames, state.clusterWords, state.clusterMetadata]);

  if (clusters.length === 0) {
    return (
      <div className="panel cluster-details-panel">
        <div className="panel-header">Cluster Details</div>
        <div className="panel-content">
          <p className="no-selection">Select a cluster to view details</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel cluster-details-panel">
      <div className="panel-header">Cluster Details</div>
      <div className="cluster-details-content">
        <div className="summary-row">
          <div className="summary-pill">{clusters.length} selected</div>
        </div>

        <div className="cluster-cards">
          {clusters.map((cluster) => (
            <div key={cluster.id} className="cluster-card">
              <div className="card-header">
                <div className="card-title">{cluster.name}</div>
                <div className="card-sub">ID: {cluster.id}</div>
              </div>

              <div className="card-stats">
                <div className="stat-block">
                  <div className="stat-label">Size</div>
                  <div className="stat-value">{cluster.size}</div>
                </div>
                <div className="stat-block">
                  <div className="stat-label">Stability</div>
                  <div className="stat-value">{cluster.stability?.toFixed(3) ?? '0.000'}</div>
                </div>
                <div className="stat-block">
                  <div className="stat-label">Strahler</div>
                  <div className="stat-value">{cluster.strahler}</div>
                </div>
              </div>

              {cluster.words.length > 0 && (
                <div className="words-container">
                  {cluster.words.slice(0, 8).map((word, idx) => (
                    <span key={idx} className="word-tag">{word}</span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ClusterDetails;
