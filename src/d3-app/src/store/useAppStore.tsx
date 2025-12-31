/**
 * Global application state management using React Context API
 * Centralized state for all views and interactions
 */

import React, { createContext, useContext, useReducer } from 'react';
import type { ReactNode, Dispatch } from 'react';
import type {
  Point,
  LinkageMatrix,
  ClusterMetaMap,
  DendrogramCoordinates,
  FilterParams,
  SelectionState,
  ZoomState
} from '../types';

/**
 * Define all possible actions
 */
export type AppAction =
  | {
      type: 'SET_DATA';
      payload: {
        points: Point[];
        linkageMatrix: LinkageMatrix;
        clusterMetadata: ClusterMetaMap;
        clusterNames: Record<number, string>;
        clusterWords: Record<number, string[]>;
        clusterIdMap: Record<number, number>;
      };
    }
  | { type: 'SELECT_POINTS'; payload: number[] }
  | { type: 'SELECT_CLUSTERS'; payload: number[] }
  | { type: 'SET_DR_SELECTED_CLUSTERS'; payload: number[] }
  | { type: 'CLEAR_SELECTION' }
  | { type: 'SET_HEATMAP_CLICKED'; payload: number[] }
  | { type: 'SET_DENDROGRAM_HOVERED'; payload: number | null }
  | { type: 'SET_SEARCH_QUERY'; payload: string }
  | { type: 'SET_SEARCH_RESULTS'; payload: number[] }
  | { type: 'SET_DR_METHOD'; payload: 'umap' | 'tsne' | 'pca' }
  | {
      type: 'SET_CURRENT_METRIC';
      payload: 'kl_divergence' | 'bhattacharyya_coefficient' | 'mahalanobis_distance';
    }
  | { type: 'SET_FILTER_PARAMS'; payload: FilterParams }
  | { type: 'UPDATE_DENDROGRAM_COORDS'; payload: DendrogramCoordinates }
  | { type: 'SET_ZOOM_STATE'; payload: ZoomState }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_COLOR_MODE'; payload: 'cluster' | 'distance' };
/**
 * Application state
 */
export interface AppStateValue {
  points: Point[];
  linkageMatrix: LinkageMatrix;
  clusterMetadata: ClusterMetaMap;
  clusterNames: Record<number, string>;
  clusterWords: Record<number, string[]>;
  clusterIdMap: Record<number, number>; // Dendrogram index -> actual cluster ID
  currentDataset: string;
  currentDRMethod: 'umap' | 'tsne' | 'pca';
  currentMetric: 'kl_divergence' | 'bhattacharyya_coefficient' | 'mahalanobis_distance';
  colorMode: 'cluster' | 'distance';
  dendrogramCoords: DendrogramCoordinates | null;
  filterParams: FilterParams;
  selection: SelectionState;
  zoomState: ZoomState;
  isLoading: boolean;
  error: string | null;
  lastUpdated: number;
}

/**
 * Context API type
 */
interface AppContextType {
  state: AppStateValue;
  dispatch: Dispatch<AppAction>;
}

/**
 * Create the context
 */
export const AppContext = createContext<AppContextType | undefined>(undefined);

/**
 * Initialize empty selection state
 */
const initializeSelectionState = (): SelectionState => ({
  selectedPointIds: new Set(),
  selectedClusterIds: new Set(),
  drSelectedClusterIds: new Set(),
  heatmapClickedClusters: new Set(),
  dendrogramHoveredCluster: null,
  searchQuery: '',
  searchResultPointIds: new Set(),
  lastInteractionSource: 'none',
  lastInteractionTime: 0
});

/**
 * Initialize empty zoom state
 */
const initializeZoomState = (): ZoomState => ({
  xAxisRange: null,
  yAxisRange: null,
  scale: 1.0
});

/**
 * Initialize empty filter params
 */
const initializeFilterParams = (): FilterParams => ({
  stability: [0, 1],
  strahler: [0, 100]
});

/**
 * Initial state
 */
const initialState: AppStateValue = {
  points: [],
  linkageMatrix: [],
  clusterMetadata: {},
  clusterNames: {},
  clusterWords: {},  clusterIdMap: {},  currentDataset: '',
  currentDRMethod: 'umap',
  currentMetric: 'kl_divergence',
    colorMode: 'cluster',
  dendrogramCoords: null,
  filterParams: initializeFilterParams(),
  selection: initializeSelectionState(),
  zoomState: initializeZoomState(),
  isLoading: false,
  error: null,
  lastUpdated: 0
};

/**
 * Reducer function
 */
function appReducer(state: AppStateValue, action: AppAction): AppStateValue {
  switch (action.type) {
    case 'SET_DATA':
      return {
        ...state,
        points: action.payload.points,
        linkageMatrix: action.payload.linkageMatrix,
        clusterMetadata: action.payload.clusterMetadata,
        clusterNames: action.payload.clusterNames,
        clusterWords: action.payload.clusterWords,
        clusterIdMap: action.payload.clusterIdMap,
        lastUpdated: Date.now()
      };

    case 'SELECT_POINTS':
      return {
        ...state,
        selection: {
          ...state.selection,
          selectedPointIds: new Set(action.payload),
          lastInteractionSource: 'dr',
          lastInteractionTime: Date.now()
        }
      };

    case 'SELECT_CLUSTERS':
      return {
        ...state,
        selection: {
          ...state.selection,
          selectedClusterIds: new Set(action.payload),
          lastInteractionSource: 'dr',
          lastInteractionTime: Date.now()
        }
      };

    case 'SET_DR_SELECTED_CLUSTERS':
      return {
        ...state,
        selection: {
          ...state.selection,
          drSelectedClusterIds: new Set(action.payload),
          lastInteractionSource: 'dr',
          lastInteractionTime: Date.now()
        }
      };

    case 'CLEAR_SELECTION':
      return {
        ...state,
        selection: initializeSelectionState()
      };

    case 'SET_HEATMAP_CLICKED':
      return {
        ...state,
        selection: {
          ...state.selection,
          heatmapClickedClusters: new Set(action.payload),
          lastInteractionSource: 'heatmap',
          lastInteractionTime: Date.now()
        }
      };

    case 'SET_DENDROGRAM_HOVERED':
      return {
        ...state,
        selection: {
          ...state.selection,
          dendrogramHoveredCluster: action.payload,
          lastInteractionSource: 'dendrogram',
          lastInteractionTime: Date.now()
        }
      };

    case 'SET_SEARCH_QUERY':
      return {
        ...state,
        selection: {
          ...state.selection,
          searchQuery: action.payload,
          lastInteractionSource: 'none',
          lastInteractionTime: Date.now()
        }
      };

    case 'SET_SEARCH_RESULTS':
      return {
        ...state,
        selection: {
          ...state.selection,
          searchResultPointIds: new Set(action.payload),
          lastInteractionSource: 'none',
          lastInteractionTime: Date.now()
        }
      };

    case 'SET_DR_METHOD':
      return {
        ...state,
        currentDRMethod: action.payload
      };

    case 'SET_CURRENT_METRIC':
      return {
        ...state,
        currentMetric: action.payload
      };

    case 'SET_COLOR_MODE':
      return {
        ...state,
        colorMode: action.payload
      };

    case 'SET_FILTER_PARAMS':
      return {
        ...state,
        filterParams: action.payload,
        selection: {
          ...state.selection,
          lastInteractionTime: Date.now()
        }
      };

    case 'UPDATE_DENDROGRAM_COORDS':
      return {
        ...state,
        dendrogramCoords: action.payload,
        lastUpdated: Date.now()
      };

    case 'SET_ZOOM_STATE':
      return {
        ...state,
        zoomState: action.payload
      };

    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload
      };

    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false
      };

    default:
      return state;
  }
}

/**
 * Provider component
 */
export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

/**
 * Custom hook to use the app context
 */
export const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return context;
};

/**
 * Hook to get only the selection state and actions
 */
export const useSelection = () => {
  const { state, dispatch } = useAppContext();
  return {
    selection: state.selection,
    selectPoints: (pointIds: number[]) => dispatch({ type: 'SELECT_POINTS', payload: pointIds }),
    selectClusters: (clusterIds: number[]) => dispatch({ type: 'SELECT_CLUSTERS', payload: clusterIds }),
    setDRSelectedClusters: (clusterIds: number[]) => dispatch({ type: 'SET_DR_SELECTED_CLUSTERS', payload: clusterIds }),
    clearSelection: () => dispatch({ type: 'CLEAR_SELECTION' }),
    setHeatmapClicked: (clusterIds: number[]) => dispatch({ type: 'SET_HEATMAP_CLICKED', payload: clusterIds }),
    setDendrogramHovered: (clusterId: number | null) => dispatch({ type: 'SET_DENDROGRAM_HOVERED', payload: clusterId }),
    setSearchQuery: (query: string) => dispatch({ type: 'SET_SEARCH_QUERY', payload: query }),
    setSearchResults: (pointIds: number[]) => dispatch({ type: 'SET_SEARCH_RESULTS', payload: pointIds })
  };
};

/**
 * Hook to get only the data
 */
export const useData = () => {
  const { state, dispatch } = useAppContext();
  return {
    data: {
      points: state.points,
      linkageMatrix: state.linkageMatrix,
      clusterMetadata: state.clusterMetadata,
      clusterNames: state.clusterNames,
      clusterWords: state.clusterWords,
      clusterIdMap: state.clusterIdMap
    },
    setData: (
      points: Point[],
      linkageMatrix: LinkageMatrix,
      clusterMetadata: ClusterMetaMap,
      clusterNames: Record<number, string>,
      clusterWords: Record<number, string[]>,
      clusterIdMap: Record<number, number>
    ) =>
      dispatch({
        type: 'SET_DATA',
        payload: { points, linkageMatrix, clusterMetadata, clusterNames, clusterWords, clusterIdMap }
      })
  };
};

/**
 * Hook to get only the view configuration
 */
export const useViewConfig = () => {
  const { state, dispatch } = useAppContext();
  return {
    config: {
      currentDataset: state.currentDataset,
      currentDRMethod: state.currentDRMethod,
      currentMetric: state.currentMetric,
      colorMode: state.colorMode,
      filterParams: state.filterParams
    },
    setDRMethod: (method: 'umap' | 'tsne' | 'pca') => dispatch({ type: 'SET_DR_METHOD', payload: method }),
    setCurrentMetric: (metric: 'kl_divergence' | 'bhattacharyya_coefficient' | 'mahalanobis_distance') =>
      dispatch({ type: 'SET_CURRENT_METRIC', payload: metric }),
    setColorMode: (mode: 'cluster' | 'distance') => dispatch({ type: 'SET_COLOR_MODE', payload: mode }),
    setFilterParams: (params: FilterParams) => dispatch({ type: 'SET_FILTER_PARAMS', payload: params }),
    updateDendrogramCoords: (coords: DendrogramCoordinates) =>
      dispatch({ type: 'UPDATE_DENDROGRAM_COORDS', payload: coords }),
    setZoomState: (zoomState: ZoomState) => dispatch({ type: 'SET_ZOOM_STATE', payload: zoomState })
  };
};

/**
 * Hook to get only the UI state and actions
 */
export const useUIState = () => {
  const { state, dispatch } = useAppContext();
  return {
    uiState: {
      isLoading: state.isLoading,
      error: state.error,
      lastUpdated: state.lastUpdated
    },
    setLoading: (isLoading: boolean) => dispatch({ type: 'SET_LOADING', payload: isLoading }),
    setError: (error: string | null) => dispatch({ type: 'SET_ERROR', payload: error })
  };
};
