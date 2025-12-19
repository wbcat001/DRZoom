import React from 'react';
import { AppProvider } from './store/useAppStore.tsx';
import MainLayout from './components/Layout/MainLayout';
import './App.css';

const App: React.FC = () => {
  return (
    <AppProvider>
      <div className="App">
        <MainLayout />
      </div>
    </AppProvider>
  );
};

export default App;
