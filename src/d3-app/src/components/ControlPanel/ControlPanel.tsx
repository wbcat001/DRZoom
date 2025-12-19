import React, { useState, useEffect } from 'react';
import { useAppContext, useViewConfig, useUIState } from '../../store/useAppStore.tsx';
import { apiClient } from '../../api/client';
import './ControlPanel.css';

interface Dataset {
  id: string;
  name: string;
  description: string;
  drMethods: string[];
}

const ControlPanel: React.FC = () => {
  const { state, dispatch } = useAppContext();
  const { config, setDRMethod, setCurrentMetric, setFilterParams } = useViewConfig();
  const { uiState, setLoading, setError } = useUIState();

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('default');
  const [selectedDRMethod, setSelectedDRMethod] = useState('umap');
  const [umapParams, setUmapParams] = useState({ n_neighbors: 15, min_dist: 0.1 });
  const [tsneParams, setTsneParams] = useState({ perplexity: 30 });
  const [pcaParams, setPcaParams] = useState({ n_components: 2 });

  // Load available datasets on mount
  useEffect(() => {
    let isMounted = true;
    const loadDatasets = async () => {
      try {
        setLoading(true);
        const response = await apiClient.getDatasets();
        if (isMounted) {
          setDatasets(response.datasets as any);
        }
      } catch (error) {
        if (isMounted) {
          setError('Failed to load datasets');
        }
        console.error(error);
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };
    loadDatasets();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleExecute = async () => {
    try {
      setLoading(true);
      
      // Get DR parameters based on selected method
      let drParams: Record<string, any> = {};
      if (selectedDRMethod === 'umap') {
        drParams = umapParams;
      } else if (selectedDRMethod === 'tsne') {
        drParams = tsneParams;
      } else if (selectedDRMethod === 'pca') {
        drParams = pcaParams;
      }

      // Fetch initial data
      const response = await apiClient.getInitialData(
        selectedDataset,
        selectedDRMethod,
        drParams
      );

      if (response.success) {
        // Update app state with fetched data
        dispatch({
          type: 'SET_DATA',
          payload: {
            points: response.data.points,
            linkageMatrix: response.data.zMatrix,
            clusterMetadata: response.data.clusterMeta,
            clusterNames: response.data.clusterNames,
            clusterWords: response.data.clusterWords
          }
        });

        // Update DR method
        setDRMethod(selectedDRMethod as 'umap' | 'tsne' | 'pca');
        
        setError(null);
      } else {
        setError('Failed to fetch data');
      }
    } catch (error) {
      setError(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleMetricChange = (metric: string) => {
    setCurrentMetric(metric as 'kl_divergence' | 'bhattacharyya_coefficient' | 'mahalanobis_distance');
  };

  const handleFilterChange = () => {
    // Update filter params based on current slider values
    setFilterParams(config.filterParams);
  };

  return (
    <div className="panel">
      <div className="panel-header">Control Panel</div>

      <div className="panel-content">
        {/* Dataset Selector */}
        <div className="form-group">
          <label htmlFor="dataset-select">Dataset:</label>
          <select
            id="dataset-select"
            className="form-control"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            disabled={uiState.isLoading}
          >
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name}
              </option>
            ))}
          </select>
        </div>

        {/* DR Method Selector */}
        <div className="form-group">
          <label>DR Method:</label>
          <div className="radio-group">
            {['umap', 'tsne', 'pca'].map((method) => (
              <label key={method} className="radio-label">
                <input
                  type="radio"
                  name="dr-method"
                  value={method}
                  checked={selectedDRMethod === method}
                  onChange={() => setSelectedDRMethod(method)}
                  disabled={uiState.isLoading}
                />
                <span>{method.toUpperCase()}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Dynamic Parameters */}
        <div className="parameter-container">
          {selectedDRMethod === 'umap' && (
            <div>
              <div className="form-group">
                <label>
                  N Neighbors: <span className="param-value">{umapParams.n_neighbors}</span>
                </label>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={umapParams.n_neighbors}
                  onChange={(e) => setUmapParams({ ...umapParams, n_neighbors: parseInt(e.target.value) })}
                  className="slider"
                />
              </div>
              <div className="form-group">
                <label>
                  Min Distance: <span className="param-value">{umapParams.min_dist.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={umapParams.min_dist}
                  onChange={(e) => setUmapParams({ ...umapParams, min_dist: parseFloat(e.target.value) })}
                  className="slider"
                />
              </div>
            </div>
          )}
          {selectedDRMethod === 'tsne' && (
            <div className="form-group">
              <label>
                Perplexity: <span className="param-value">{tsneParams.perplexity}</span>
              </label>
              <input
                type="range"
                min="5"
                max="100"
                value={tsneParams.perplexity}
                onChange={(e) => setTsneParams({ perplexity: parseInt(e.target.value) })}
                className="slider"
              />
            </div>
          )}
          {selectedDRMethod === 'pca' && (
            <div className="form-group">
              <label>
                Components: <span className="param-value">{pcaParams.n_components}</span>
              </label>
              <input
                type="range"
                min="2"
                max="10"
                value={pcaParams.n_components}
                onChange={(e) => setPcaParams({ n_components: parseInt(e.target.value) })}
                className="slider"
              />
            </div>
          )}
        </div>

        {/* Metric Selector */}
        <div className="form-group">
          <label>Similarity Metric:</label>
          <select
            className="form-control"
            value={config.currentMetric}
            onChange={(e) => handleMetricChange(e.target.value)}
          >
            <option value="kl_divergence">KL Divergence</option>
            <option value="bhattacharyya_coefficient">Bhattacharyya Coefficient</option>
            <option value="mahalanobis_distance">Mahalanobis Distance</option>
          </select>
        </div>

        {/* Filter Controls */}
        <div className="filter-container">
          <details>
            <summary>Advanced Filters</summary>
            <div className="form-group">
              <label>
                Stability: [{config.filterParams.stability[0].toFixed(2)}, {config.filterParams.stability[1].toFixed(2)}]
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                className="slider"
                onChange={handleFilterChange}
              />
            </div>
            <div className="form-group">
              <label>
                Strahler: [{config.filterParams.strahler[0].toFixed(0)}, {config.filterParams.strahler[1].toFixed(0)}]
              </label>
              <input
                type="range"
                min="0"
                max="100"
                className="slider"
                onChange={handleFilterChange}
              />
            </div>
          </details>
        </div>

        {/* Status Messages */}
        {uiState.error && (
          <div className="alert alert-danger">{uiState.error}</div>
        )}
        {uiState.isLoading && (
          <div className="alert alert-info">Loading...</div>
        )}

        {/* Execute Button */}
        <button
          className="btn btn-primary execute-btn"
          onClick={handleExecute}
          disabled={uiState.isLoading}
        >
          {uiState.isLoading ? 'Loading...' : 'Load Data'}
        </button>
      </div>
    </div>
  );
};

export default ControlPanel;
