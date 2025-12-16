import React from 'react';
import './MainLayout.css';
import ControlPanel from '../ControlPanel/ControlPanel';
import DRVisualization from '../DRVisualization/DRVisualization';
import Dendrogram from '../Dendrogram/Dendrogram';
import DetailsPanel from '../DetailsPanel/DetailsPanel';
import ClusterHeatmap from '../ClusterHeatmap/ClusterHeatmap';
import ClusterDetails from '../ClusterDetails/ClusterDetails';

const MainLayout: React.FC = () => {
  return (
    <div className="main-container">
      {/* 上段: 4パネルレイアウト */}
      <div className="top-row">
        <div className="control-panel-area">
          <ControlPanel />
        </div>
        <div className="dr-visualization-area">
          <DRVisualization />
        </div>
        <div className="dendrogram-area">
          <Dendrogram />
        </div>
        <div className="details-panel-area">
          <DetailsPanel />
        </div>
      </div>
      
      {/* 下段: 2パネルレイアウト */}
      <div className="bottom-row">
        <div className="cluster-heatmap-area">
          <ClusterHeatmap />
        </div>
        <div className="cluster-details-area">
          <ClusterDetails />
        </div>
      </div>
    </div>
  );
};

export default MainLayout;
