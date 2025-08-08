import { useState, useEffect } from 'react'
import './App.css'
import React from 'react';

import ScatterPlot from './components/scatter-plot';
import type { ScatterPoint } from './types';


const FetchData = async () => {
  // POST: http://localhost:8000/init
  const response = await fetch('http://localhost:8000/init', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({}),
  });
  const data = await response.json();
  return data;
}
function App() {
  const [data, setData] = useState<ScatterPoint[]>([]);
  // Example data
  useEffect(() => {

    // Fetch data from the server
    FetchData().then(fetchedData => {
      setData(fetchedData);
    });

    
  }, []);


  return (
    <>
    <div className="w-full h-screen bg-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="px-4 sm:px-6 lg:px-8 py-4">
          <h2 className="text-2xl font-bold text-gray-900">ScatterPlot Dashboard</h2>
        </div>
      </header>

      {/* Main Content */}
      <main className="overflow-hidden px-4 py-2">
        {/* Dashboard Grid Layout */}
        <div className="grid grid-cols-12 gap-2 h-[calc(100vh-120px)]">
          
          {/* Left Sidebar - Controls */}
          <div className="col-span-2 space-y-6">
            {/* Control Panel */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Controls</h2>
              
              {/* Selection Method */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Selection Method
                </label>
                <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                  <option>Box Select</option>
                  <option>Lasso Select</option>
                  <option>Click Select</option>
                </select>
              </div>

              {/* Buttons */}
              <div className="space-y-2">
                <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
                  Clear Selection
                </button>
                <button className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors">
                  Run Animation
                </button>
                <button className="w-full px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors">
                  Reset
                </button>
              </div>
            </div>

            {/* Statistics Panel */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Statistics</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Total Points:</span>
                  <span className="text-sm font-medium">{data.length}</span>
                </div>
                
              </div>
            </div>
          </div>

          {/* Center - Main Scatter Plot */}
          <div className="col-span-8">
            <div className="bg-white rounded-lg shadow h-full">
              <div className="p-6 border-b">
                <h2 className="text-lg font-semibold text-gray-900">Scatter Plot Visualization</h2>
                <p className="text-sm text-gray-600 mt-1">Click and drag to select regions</p>
              </div>
              <div className="p-6 h-[calc(100%-80px)]">
                <ScatterPlot data={data} />
              </div>
            </div>
          </div>

          {/* Right Sidebar - Analysis */}
          <div className="col-span-2 space-y-6">
            {/* Analysis Charts */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Analysis</h2>
              
              {/* Distance Plot Placeholder */}
              <div className="h-32 bg-gray-100 rounded border-2 border-dashed border-gray-300 flex items-center justify-center mb-4">
                <span className="text-gray-500 text-sm">Distance Plot</span>
              </div>

              {/* CCR Plot Placeholder */}
              <div className="h-32 bg-gray-100 rounded border-2 border-dashed border-gray-300 flex items-center justify-center">
                <span className="text-gray-500 text-sm">CCR Plot</span>
              </div>
            </div>

            {/* Selected Point Details */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Selected Point</h2>
              <div className="h-32 bg-gray-100 rounded border-2 border-dashed border-gray-300 flex items-center justify-center mb-3">
                <span className="text-gray-500 text-sm">Point Details</span>
              </div>
              <div className="text-sm text-gray-600">
                <p>Click on a point to see details</p>
              </div>
            </div>
          </div>
        </div>

        
      </main>
    </div>

    </>
  )
}

export default App
