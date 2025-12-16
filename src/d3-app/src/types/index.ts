export interface Point {
  id: number;
  x: number;
  y: number;
  cluster: number;
  label?: string;
}

export interface Cluster {
  id: number;
  size: number;
  color: string;
  stability: number;
}

export interface DendrogramNode {
  id: string;
  x: number;
  y: number;
  children?: DendrogramNode[];
  size?: number;
  highlighted?: boolean;
}

export interface DendrogramLink {
  source: DendrogramNode;
  target: DendrogramNode;
  weight?: number;
}

export interface AppState {
  selectedClusters: number[];
  selectedPoints: number[];
  heatmapClickedClusters: number[];
  dendrogramClickedClusters: number[];
  lastInteractionType: string | null;
  drZoomState: { x: [number, number]; y: [number, number] };
  parameters: {
    drMethod: 'UMAP' | 'TSNE' | 'PCA';
    umapParams: { n_neighbors: number; min_dist: number };
    tsneParams: { perplexity: number };
    pcaParams: { n_components: number };
  };
}

export interface SimilarityData {
  clusters: Cluster[];
  similarities: number[][];
}

export type InteractionMode = 'brush' | 'zoom';
export type TabType = 'point-details' | 'selection-stats' | 'cluster-size' | 'system-log';

export const HIGHLIGHT_COLORS = {
  default: '#4A90E2',
  defaultDimmed: '#B8D4F0',
  drSelection: '#FFA500',
  heatmapClick: '#FF0000',
  heatmapToDr: '#FF1493',
  dendrogramToDr: '#32CD32',
} as const;
