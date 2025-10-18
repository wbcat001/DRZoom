import React, { useState, useEffect } from 'react';

const TestConnection: React.FC = () => {
  const [status, setStatus] = useState<string>('Testing...');
  const [data, setData] = useState<any>(null);

  const testConnection = async () => {
    try {
      setStatus('Connecting to backend...');
      
      const response = await fetch('http://localhost:8000/clusters?level=0&n_clusters=3');
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      setData(result);
      setStatus('✅ Connection successful!');
      
    } catch (error) {
      setStatus(`❌ Connection failed: ${error}`);
      console.error('Connection error:', error);
    }
  };

  useEffect(() => {
    testConnection();
  }, []);

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h2>Backend Connection Test</h2>
      <p><strong>Status:</strong> {status}</p>
      
      <button onClick={testConnection} style={{ padding: '10px', margin: '10px 0' }}>
        Retry Connection
      </button>
      
      {data && (
        <div>
          <h3>Response Data:</h3>
          <pre style={{ background: '#f5f5f5', padding: '10px', overflow: 'auto' }}>
            {JSON.stringify(data, null, 2)}
          </pre>
          
          <h4>Summary:</h4>
          <ul>
            <li>Level: {data.level}</li>
            <li>Points: {data.total_points}</li>
            <li>Clusters: {data.total_clusters}</li>
          </ul>
          
          {data.points && data.points.length > 0 && (
            <div>
              <h4>First few points:</h4>
              <pre style={{ background: '#f0f8ff', padding: '10px' }}>
                {JSON.stringify(data.points.slice(0, 3), null, 2)}
              </pre>
            </div>
          )}
          
          {data.clusters && data.clusters.length > 0 && (
            <div>
              <h4>First cluster:</h4>
              <pre style={{ background: '#fff0f0', padding: '10px' }}>
                {JSON.stringify(data.clusters[0], null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TestConnection;