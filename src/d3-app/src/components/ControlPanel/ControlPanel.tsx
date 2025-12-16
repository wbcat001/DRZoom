import React, { useState } from 'react';
import './ControlPanel.css';

interface Dataset {
  label: string;
  value: string;
}

interface DRMethod {
  id: 'UMAP' | 'TSNE' | 'PCA';
  label: string;
}

const ControlPanel: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState('iris');
  const [selectedDRMethod, setSelectedDRMethod] = useState<'UMAP' | 'TSNE' | 'PCA'>('UMAP');
  
  const datasets: Dataset[] = [
    { label: 'Iris', value: 'iris' },
    { label: 'Digits', value: 'digits' },
    { label: 'Wine', value: 'wine' }
  ];

  const drMethods: DRMethod[] = [
    { id: 'UMAP', label: 'UMAP' },
    { id: 'TSNE', label: 'T-SNE' },
    { id: 'PCA', label: 'PCA' }
  ];

  const handleExecute = () => {
    console.log('Executing analysis with:', { selectedDataset, selectedDRMethod });
  };

  return (
    <div className="panel">
      <div className="panel-header">
        Control Panel
      </div>
      
      <div className="panel-content">
        {/* Dataset Selector */}
        <div className="form-group">
          <label htmlFor="dataset-select">Dataset:</label>
          <select 
            id="dataset-select"
            className="form-control"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
          >
            {datasets.map(dataset => (
              <option key={dataset.value} value={dataset.value}>
                {dataset.label}
              </option>
            ))}
          </select>
        </div>

        {/* DR Method Selector */}
        <div className="form-group">
          <label>DR Method:</label>
          <div className="radio-group">
            {drMethods.map(method => (
              <label key={method.id} className="radio-label">
                <input
                  type="radio"
                  name="dr-method"
                  value={method.id}
                  checked={selectedDRMethod === method.id}
                  onChange={() => setSelectedDRMethod(method.id)}
                />
                <span>{method.label}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Dynamic Parameters */}
        <div className="parameter-container">
          {selectedDRMethod === 'UMAP' && (
            <div>
              <div className="form-group">
                <label>N Neighbors: 15</label>
                <input type="range" min="5" max="50" defaultValue="15" className="slider" />
              </div>
              <div className="form-group">
                <label>Min Distance: 0.1</label>
                <input type="range" min="0.01" max="1" step="0.01" defaultValue="0.1" className="slider" />
              </div>
            </div>
          )}
          {selectedDRMethod === 'TSNE' && (
            <div className="form-group">
              <label>Perplexity: 30</label>
              <input type="range" min="5" max="100" defaultValue="30" className="slider" />
            </div>
          )}
          {selectedDRMethod === 'PCA' && (
            <div className="form-group">
              <label>Components: 2</label>
              <input type="range" min="2" max="10" defaultValue="2" className="slider" />
            </div>
          )}
        </div>

        {/* Execute Button */}
        <button 
          className="execute-btn"
          onClick={handleExecute}
        >
          Run Analysis
        </button>
      </div>
    </div>
  );
};

export default ControlPanel;
