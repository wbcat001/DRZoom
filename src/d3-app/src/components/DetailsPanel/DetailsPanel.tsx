import React, { useState } from 'react';
import type { TabType } from '../../types';
import './DetailsPanel.css';

const DetailsPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('point-details');

  const tabs: Array<{ id: TabType; label: string }> = [
    { id: 'point-details', label: 'Point Details' },
    { id: 'selection-stats', label: 'Selection Stats' },
    { id: 'cluster-size', label: 'Cluster Size' },
    { id: 'system-log', label: 'System Log' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'point-details':
        return (
          <div className="tab-content-area">
            <h5>Point Information</h5>
            <div className="info-item">
              <strong>Point ID:</strong> <span>--</span>
            </div>
            <div className="info-item">
              <strong>Coordinates:</strong> <span>(0.0, 0.0)</span>
            </div>
            <div className="info-item">
              <strong>Cluster:</strong> <span>--</span>
            </div>
            <div className="info-item">
              <strong>Label:</strong> <span>--</span>
            </div>
          </div>
        );
      case 'selection-stats':
        return (
          <div className="tab-content-area">
            <h5>Selection Statistics</h5>
            <div className="info-item">
              <strong>Selected Points:</strong> <span>0</span>
            </div>
            <div className="info-item">
              <strong>Selected Clusters:</strong> <span>0</span>
            </div>
            <div className="info-item">
              <strong>Coverage:</strong> <span>0.0%</span>
            </div>
          </div>
        );
      case 'cluster-size':
        return (
          <div className="tab-content-area">
            <h5>Cluster Information</h5>
            <div className="cluster-list">
              {Array.from({ length: 5 }, (_, i) => (
                <div key={i} className="cluster-item">
                  <div className="cluster-color" style={{ backgroundColor: `hsl(${i * 72}, 70%, 50%)` }}></div>
                  <span>Cluster {i}: {Math.floor(Math.random() * 50) + 10} points</span>
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
              <div className="log-entry">Dataset loaded: iris</div>
              <div className="log-entry">DR method: UMAP</div>
              <div className="log-entry">Clustering complete</div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">
        Detail & Info
      </div>
      
      <div className="panel-content">
        {/* Tab navigation */}
        <div className="tab-nav">
          {tabs.map(tab => (
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
        <div className="tab-content">
          {renderTabContent()}
        </div>
      </div>
    </div>
  );
};

export default DetailsPanel;
