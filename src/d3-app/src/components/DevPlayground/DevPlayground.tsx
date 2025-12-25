import React, { useCallback, useMemo, useState } from 'react';
import { generateMockDataset } from '../../utils/devMockData';
import { useAppContext, useUIState } from '../../store/useAppStore.tsx';
import DRVisualization from '../DRVisualization/DRVisualization';
import Dendrogram from '../Dendrogram/Dendrogram';
import DetailsPanel from '../DetailsPanel/DetailsPanel';
import ClusterHeatmap from '../ClusterHeatmap/ClusterHeatmap';
import ClusterDetails from '../ClusterDetails/ClusterDetails';
import '../Layout/MainLayout.css';
import './DevPlayground.css';

const DATA_PRESETS = [
  { label: '1k points', value: 1_000, clusters: 8 },
  { label: '10k points', value: 10_000, clusters: 10 },
  { label: '100k points', value: 100_000, clusters: 12 }
];

const DevPlayground: React.FC = () => {
  const { state, dispatch } = useAppContext();
  const { setLoading, setError } = useUIState();
  const [lastLoaded, setLastLoaded] = useState<string>('');

  const loadMock = useCallback(async (pointCount: number, clusterCount: number) => {
    try {
      setLoading(true);
      setError(null);

      // Generate synthetic dataset
      const mock = generateMockDataset(pointCount, clusterCount);

      dispatch({
        type: 'SET_DATA',
        payload: {
          points: mock.points,
          linkageMatrix: mock.linkageMatrix,
          clusterMetadata: mock.clusterMetadata,
          clusterNames: mock.clusterNames,
          clusterWords: mock.clusterWords
        }
      });

      dispatch({ type: 'CLEAR_SELECTION' });
      setLastLoaded(`${pointCount.toLocaleString()} pts / ${clusterCount} clusters`);
    } catch (error) {
      console.error(error);
      setError(error instanceof Error ? error.message : 'Mock load failed');
    } finally {
      setLoading(false);
    }
  }, [dispatch, setError, setLoading]);

  const summary = useMemo(() => {
    const clusterCount = Object.keys(state.clusterNames).length;
    return `${state.points.length.toLocaleString()} points · ${clusterCount} clusters`;
  }, [state.clusterNames, state.points.length]);

  return (
    <div className="dev-container">
      <div className="dev-toolbar">
        <div>
          <h3>Dev Playground (Mock Data)</h3>
          <p className="dev-meta">Synthetic data only — backend is not used.</p>
          <div className="dev-buttons">
            {DATA_PRESETS.map((preset) => (
              <button
                key={preset.value}
                className="btn"
                onClick={() => loadMock(preset.value, preset.clusters)}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>
        <div className="dev-status">
          <div className="status-row">Current: {summary}</div>
          {lastLoaded && <div className="status-row">Last loaded: {lastLoaded}</div>}
        </div>
      </div>

      <div className="main-container">
        <div className="top-row">
          <div className="control-panel-area">
            <div className="panel">
              <div className="panel-header">Mock Data Controls</div>
              <div className="panel-content">
                <div className="status-row">Current: {summary}</div>
                <div className="status-row">Last loaded: {lastLoaded || '—'}</div>
                <p className="dev-help">Use the buttons above to reload synthetic datasets at different scales.</p>
              </div>
            </div>
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

        <div className="bottom-row">
          <div className="cluster-heatmap-area">
            <ClusterHeatmap />
          </div>
          <div className="cluster-details-area">
            <ClusterDetails />
          </div>
        </div>
      </div>
    </div>
  );
};

export default DevPlayground;
