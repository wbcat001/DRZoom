/**
 * Application state types
 * Defines the structure of the global state managed by Zustand
 */

import type { Point, ClusterMetaMap, LinkageMatrix, DendrogramCoordinates } from './data';

/**
 * Filter parameters for dendrogram
 */
export interface FilterParams {
  stability: [number, number];  // [min, max]
  strahler: [number, number];   // [min, max]
}

/**
 * Zoom state for DR visualization
 */
export interface ZoomState {
  xAxisRange: [number, number] | null;
  yAxisRange: [number, number] | null;
  scale: number;
}

/**
 * Selection state across all views
 */
export interface SelectionState {
  // DR view selections
  selectedPointIds: Set<number>;
  selectedClusterIds: Set<number>;

  // Heatmap selections
  heatmapClickedClusters: Set<number>;

  // Dendrogram selections
  dendrogramHoveredCluster: number | null;

  // Last interaction info
  lastInteractionSource: 'dr' | 'dendrogram' | 'heatmap' | 'none';
  lastInteractionTime: number;
}

/**
 * Main application state
 */
export interface AppState {
  // Data
  points: Point[];
  linkageMatrix: LinkageMatrix;
  clusterMetadata: ClusterMetaMap;
  clusterNames: Record<number, string>;
  clusterWords: Record<number, string[]>;

  // Current view configuration
  currentDataset: string;
  currentDRMethod: 'umap' | 'tsne' | 'pca';
  currentMetric: 'kl_divergence' | 'bhattacharyya_coefficient' | 'mahalanobis_distance';

  // Dendrogram-related state
  dendrogramCoords: DendrogramCoordinates | null;
  filterParams: FilterParams;

  // Selection state
  selection: SelectionState;

  // Zoom state
  zoomState: ZoomState;

  // UI state
  isLoading: boolean;
  error: string | null;
  lastUpdated: number;

  // Actions (will be defined by Zustand)
  // Data actions
  setData: (
    points: Point[],
    linkageMatrix: LinkageMatrix,
    clusterMetadata: ClusterMetaMap,
    clusterNames: Record<number, string>,
    clusterWords: Record<number, string[]>
  ) => void;

  // Selection actions
  selectPoints: (pointIds: number[]) => void;
  selectClusters: (clusterIds: number[]) => void;
  clearSelection: () => void;
  setHeatmapClicked: (clusterIds: number[]) => void;
  setDendrogramHovered: (clusterId: number | null) => void;

  // View configuration actions
  setDRMethod: (method: 'umap' | 'tsne' | 'pca') => void;
  setCurrentMetric: (metric: 'kl_divergence' | 'bhattacharyya_coefficient' | 'mahalanobis_distance') => void;
  setFilterParams: (params: FilterParams) => void;

  // Dendrogram actions
  updateDendrogramCoords: (coords: DendrogramCoordinates) => void;

  // Zoom actions
  setZoomState: (zoomState: ZoomState) => void;

  // UI actions
  setLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
}
