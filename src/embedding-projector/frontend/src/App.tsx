import { useState, useEffect } from 'react'
import './App.css'
import React from 'react';
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import ScatterPlot from './components/scatter-plot';

import { DashBoard } from './components/basic/DashBoard';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashBoard />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
