import React, { useState, useMemo, useEffect } from 'react';
import { useSelection, useData } from '../../store/useAppStore.tsx';
import { formatClusterSize } from '../../utils';
import { apiClient } from '../../api/client';
import type { PointDetail } from '../../types';
import './DetailsPanel.css';

type TabType = 'point-details' | 'selection-stats' | 'dr-selection' | 'cluster-size' | 'system-log';

const DetailsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('selection-stats');
  const { selection } = useSelection();
  const { data } = useData();
  const [pointDetail, setPointDetail] = useState<PointDetail | null>(null);
  const [pointDetailError, setPointDetailError] = useState<string | null>(null);
  const [pointDetailLoading, setPointDetailLoading] = useState(false);

  const tabs: Array<{ id: TabType; label: string }> = [
    { id: 'point-details', label: 'Point Details' },
    { id: 'selection-stats', label: 'Selection Stats' },
    { id: 'dr-selection', label: 'DR Selection' },
    { id: 'cluster-size', label: 'Cluster Size' },
    { id: 'system-log', label: 'System Log' }
  ];

  // Calculate selection statistics
  const selectionStats = useMemo(() => {
    const selectedClusterIds = Array.from(selection.selectedClusterIds);
    const selectedPoints = Array.from(selection.selectedPointIds);

    let totalPointsInSelectedClusters = 0;
    const clusterDetails: Array<{ id: number; size: number; label: string }> = [];

    for (const clusterId of selectedClusterIds) {
      const pointsInCluster = data.points.filter((p) => p.c === clusterId).length;
      totalPointsInSelectedClusters += pointsInCluster;
      const label = data.clusterNames[clusterId] || `Cluster ${clusterId}`;
      clusterDetails.push({ id: clusterId, size: pointsInCluster, label });
    }

    return {
      selectedPointCount: selectedPoints.length,
      selectedClusterCount: selectedClusterIds.length,
      totalPointsInClusters: totalPointsInSelectedClusters,
      clusterDetails,
      coverage:
        data.points.length > 0
          ? ((totalPointsInSelectedClusters / data.points.length) * 100).toFixed(1)
          : '0.0'
    };
  }, [selection.selectedClusterIds, selection.selectedPointIds, data]);

  // Fetch point detail (including neighbors) when selection changes
  useEffect(() => {
    const firstPointId = Array.from(selection.selectedPointIds)[0];
    if (firstPointId === undefined) {
      setPointDetail(null);
      setPointDetailError(null);
      return;
    }

    let cancelled = false;
    const fetchDetail = async () => {
      try {
        setPointDetailLoading(true);
        const response = await apiClient.getPointDetail(firstPointId);
        if (!cancelled) {
          setPointDetail(response.point);
          setPointDetailError(null);
        }
      } catch (error) {
        if (!cancelled) {
          setPointDetail(null);
          setPointDetailError('Failed to load point detail');
          console.error(error);
        }
      } finally {
        if (!cancelled) {
          setPointDetailLoading(false);
        }
      }
    };

    fetchDetail();
    return () => {
      cancelled = true;
    };
  }, [selection.selectedPointIds]);

  // Calculate cluster size distribution
  const clusterDistribution = useMemo(() => {
    const distribution = new Map<number, number>();
    for (const point of data.points) {
      distribution.set(point.c, (distribution.get(point.c) || 0) + 1);
    }
    return Array.from(distribution.entries())
      .map(([clusterId, size]) => ({
        clusterId,
        size,
        label: data.clusterNames[clusterId] || `Cluster ${clusterId}`
      }))
      .sort((a, b) => b.size - a.size)
      .slice(0, 10);
  }, [data.points, data.clusterNames]);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'point-details':
        if (selection.selectedPointIds.size === 0) {
          return (
            <div className="tab-content-area">
              <div className="empty-state">Select points in the DR plot</div>
            </div>
          );
        }
        return (
          <div className="tab-content-area">
            <h5>Point Information</h5>
            {pointDetailLoading && (
              <div className="info-item">Loading point detail...</div>
            )}
            {pointDetailError && (
              <div className="info-item error-text">{pointDetailError}</div>
            )}
            {pointDetail && !pointDetailLoading && !pointDetailError && (
              <>
                <div className="info-item">
                  <strong>Point ID:</strong> <span>{pointDetail.id}</span>
                </div>
                <div className="info-item">
                  <strong>Label:</strong> <span>{pointDetail.label || 'N/A'}</span>
                </div>
                <div className="info-item">
                  <strong>Cluster:</strong> <span>{pointDetail.clusterId}</span>
                </div>
                <div className="info-item">
                  <strong>Coordinates:</strong>
                  <span>
                    ({pointDetail.coordinates.x.toFixed(3)}, {pointDetail.coordinates.y.toFixed(3)})
                  </span>
                </div>

                <h6 style={{ marginTop: '8px' }}>Nearest Neighbors</h6>
                {pointDetail.nearbyPoints && pointDetail.nearbyPoints.length > 0 ? (
                  <div className="neighbor-list">
                    {pointDetail.nearbyPoints.slice(0, 20).map((p) => (
                      <div key={p.i} className="neighbor-item">
                        <span className="neighbor-label">{p.l || `Point ${p.i}`}</span>
                        <span className="neighbor-meta">ID {p.i} · C{p.c}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state small">No neighbor data available</div>
                )}
              </>
            )}
          </div>
        );

      case 'selection-stats':
        return (
          <div className="tab-content-area">
            <h5>Selection Statistics</h5>
            <div className="info-item">
              <strong>Selected Points:</strong> <span>{selectionStats.selectedPointCount}</span>
            </div>
            <div className="info-item">
              <strong>Selected Clusters:</strong> <span>{selectionStats.selectedClusterCount}</span>
            </div>
            <div className="info-item">
              <strong>Points in Clusters:</strong> <span>{selectionStats.totalPointsInClusters}</span>
            </div>
            <div className="info-item">
              <strong>Coverage:</strong> <span>{selectionStats.coverage}%</span>
            </div>
            {selectionStats.clusterDetails.length > 0 && (
              <>
                <h6 style={{ marginTop: '12px' }}>Cluster Details:</h6>
                <div className="cluster-list">
                  {selectionStats.clusterDetails.map((cluster) => (
                    <div key={cluster.id} className="cluster-item">
                      <span className="cluster-name">{cluster.label}</span>
                      <span className="cluster-size">{formatClusterSize(cluster.size)} points</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        );

      case 'dr-selection':
        // Focused view for DR (lasso) selections
        const drSelectedClusters = Array.from(selection.drSelectedClusterIds);
        const drSelectedPoints = Array.from(selection.selectedPointIds);

        const drClusterDetails = drSelectedClusters.map((clusterId) => {
          const size = data.points.filter((p) => p.c === clusterId).length;
          const label = data.clusterNames[clusterId] || `Cluster ${clusterId}`;
          return { id: clusterId, size, label };
        });

        return (
          <div className="tab-content-area">
            <h5>DR (Lasso) Selection</h5>
            <div className="info-item">
              <strong>Selected Points (DR):</strong> <span>{drSelectedPoints.length}</span>
            </div>
            <div className="info-item">
              <strong>DR-Selected Clusters:</strong> <span>{drSelectedClusters.length}</span>
            </div>

            {drClusterDetails.length > 0 && (
              <>
                <h6 style={{ marginTop: '12px' }}>Clusters Intersecting DR Selection</h6>
                <div className="cluster-list">
                  {drClusterDetails.map((cluster) => (
                    <div key={cluster.id} className="cluster-item">
                      <span className="cluster-name">{cluster.label}</span>
                      <span className="cluster-size">{formatClusterSize(cluster.size)} pts</span>
                    </div>
                  ))}
                </div>
              </>
            )}

            {drSelectedPoints.length > 0 && (
              <>
                <h6 style={{ marginTop: '12px' }}>Example Points</h6>
                <div className="neighbor-list">
                  {drSelectedPoints.slice(0, 30).map((pid) => {
                    const p = data.points.find((pp) => pp.i === pid);
                    return (
                      <div key={pid} className="neighbor-item">
                        <span className="neighbor-label">{p?.l || `Point ${pid}`}</span>
                        <span className="neighbor-meta">ID {pid} · C{p?.c}</span>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        );

      case 'cluster-size':
        return (
          <div className="tab-content-area">
            <h5>Top 10 Clusters by Size</h5>
            <div className="cluster-distribution">
              {clusterDistribution.map((cluster, idx) => (
                <div key={cluster.clusterId} className="distribution-item">
                  <div className="bar-container">
                    <div
                      className="bar"
                      style={{
                        width: `${(cluster.size / Math.max(...clusterDistribution.map((c) => c.size))) * 100}%`,
                        backgroundColor: `hsl(${idx * 36}, 70%, 50%)`
                      }}
                    />
                  </div>
                  <div className="label">{cluster.label}</div>
                  <div className="size">{formatClusterSize(cluster.size)}</div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'system-log':
        return (
          <div className="tab-content-area">
            <h5>System Log</h5>
            <div className="log-container">
              <div className="log-entry">System initialized</div>
              <div className="log-entry">Total points loaded: {data.points.length}</div>
              <div className="log-entry">Total clusters: {new Set(data.points.map((p) => p.c)).size}</div>
              <div className="log-entry">Linkage matrix size: {data.linkageMatrix.length}</div>
              <div className="log-entry timestamp">Last updated: {new Date().toLocaleTimeString()}</div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">Detail & Info</div>

      <div className="panel-content">
        {/* Tab navigation */}
        <div className="tab-nav">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {renderTabContent()}
      </div>
    </div>
  );
};

export default DetailsPanel;
