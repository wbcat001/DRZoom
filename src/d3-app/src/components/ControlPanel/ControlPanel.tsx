import React, { useState, useEffect } from 'react';
import { useData, useSelection } from '../../store/useAppStore.tsx';
import { apiClient } from '../../api/client';
import './ControlPanel.css';

interface Dataset {
  id: string;
  name: string;
  description: string;
  drMethods: string[];
}

const ControlPanel: React.FC = () => {
  const { data, setData } = useData();
  const { clearSelection } = useSelection();

  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState('default');
  const [selectedDRMethod, setSelectedDRMethod] = useState('umap');
  const [umapParams, setUmapParams] = useState({ n_neighbors: 15, min_dist: 0.1 });
  const [tsneParams, setTsneParams] = useState({ perplexity: 30 });
  const [pcaParams, setPcaParams] = useState({ n_components: 2 });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load available datasets on mount
  useEffect(() => {
    let isMounted = true;
    const loadDatasets = async () => {
      try {
        setIsLoading(true);
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
          setIsLoading(false);
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
      setIsLoading(true);
      
      // Get DR parameters based on selected method
      let drParams: Record<string, any> = {};
      if (selectedDRMethod === 'umap') {
        drParams = umapParams;
      } else if (selectedDRMethod === 'tsne') {
        drParams = tsneParams;
      } else if (selectedDRMethod === 'pca') {
        drParams = pcaParams;
      }

      // Fetch initial data with noise points filtered out for better performance
      const response = await apiClient.getInitialDataNoNoise(
        selectedDataset,
        selectedDRMethod,
        drParams
      );

      if (response.success) {
        console.log('DEBUG: API Response received:', {
          hasClusterIdMap: !!response.data.clusterIdMap,
          clusterIdMapLength: response.data.clusterIdMap ? Object.keys(response.data.clusterIdMap).length : 0,
          clusterIdMapSample: response.data.clusterIdMap ? Object.entries(response.data.clusterIdMap).slice(0, 5) : [],
          hasClusterNames: !!response.data.clusterNames,
          clusterNamesLength: response.data.clusterNames ? Object.keys(response.data.clusterNames).length : 0,
          clusterNamesSample: response.data.clusterNames ? Object.entries(response.data.clusterNames).slice(0, 5) : [],
          hasClusterWords: !!response.data.clusterWords,
          clusterWordsLength: response.data.clusterWords ? Object.keys(response.data.clusterWords).length : 0
        });
        
        // Update app state with fetched data
        setData(
          response.data.points,
          response.data.zMatrix,
          response.data.clusterMeta,
          response.data.clusterNames,
          response.data.clusterWords,
          response.data.clusterIdMap || {}
        );
        
        setError(null);
      } else {
        setError('Failed to fetch data');
      }
    } catch (error) {
      setError(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFilterChange = () => {
    // Update filter params based on current slider values
    // TODO: implement filter functionality
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
            disabled={isLoading}
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
                  disabled={isLoading}
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

        {/* Status Messages */}
        {error && (
          <div className="alert alert-danger">{error}</div>
        )}
        {isLoading && (
          <div className="alert alert-info">Loading...</div>
        )}

        {/* Execute Button */}
        <div className="button-stack">
          <button
            className="btn btn-primary execute-btn"
            onClick={handleExecute}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : 'Load Data'}
          </button>

          <button
            className="btn clear-btn"
            type="button"
            onClick={() => clearSelection()}
          >
            Clear Selection
          </button>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
