import React from 'react';
import { AppProvider } from './store/useAppStore.tsx';
import MainLayout from './components/Layout/MainLayout';
import DevPlayground from './components/DevPlayground/DevPlayground';
import './App.css';

const App: React.FC = () => {
  const useDevPlayground =
    (typeof window !== 'undefined' && window.location.hash.includes('dev-playground')) ||
    (import.meta as any).env?.VITE_DEV_PLAYGROUND === '1';

  return (
    <AppProvider>
      <div className="App">
        {useDevPlayground ? <DevPlayground /> : <MainLayout />}
      </div>
    </AppProvider>
  );
};

export default App;
