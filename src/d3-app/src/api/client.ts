/**
 * API client for communicating with the backend
 * Handles all HTTP requests to the Python FastAPI backend
 */

import type {
  InitialDataResponse,
  PointToClusterRequest,
  PointToClusterResponse,
  HeatmapRequest,
  HeatmapResponse,
  ClusterDetailResponse,
  PointDetailResponse,
  DatasetsResponse,
  DendrogramFilterRequest,
  DendrogramFilterResponse,
  PointVectorsRequest,
  PointVectorsResponse,
  ZoomRedrawRequest,
  ZoomRedrawResponse
} from '../types';

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000/api';
const ZOOM_API_URL = (import.meta as any).env?.VITE_ZOOM_API_URL || 'http://localhost:8001/api';

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      error: `HTTP ${response.status}`
    }));
    throw new Error(JSON.stringify(error));
  }

  return response.json() as Promise<T>;
}

/**
 * API service object with all endpoints
 */
export const apiClient = {
  /**
   * Get initial data for visualization
   * @param colorMode - Color assignment mode: 'cluster' (default) or 'distance' (similarity-based)
   */
  getInitialData: async (dataset: string, drMethod: string, drParams?: Record<string, any>, colorMode: string = 'cluster') => {
    const queryParams = new URLSearchParams({
      dataset,
      dr_method: drMethod,
      color_mode: colorMode,
      ...(drParams && { dr_params: JSON.stringify(drParams) })
    });
    return fetchAPI<InitialDataResponse>(`/initial_data?${queryParams.toString()}`);
  },

  /**
   * Get initial data with noise points filtered out (cluster_id != -1)
   * Better performance for large datasets
   * @param colorMode - Color assignment mode: 'cluster' (default) or 'distance' (similarity-based)
   */
  getInitialDataNoNoise: async (dataset: string, drMethod: string, drParams?: Record<string, any>, colorMode: string = 'cluster') => {
    const queryParams = new URLSearchParams({
      dataset,
      dr_method: drMethod,
      color_mode: colorMode,
      ...(drParams && { dr_params: JSON.stringify(drParams) })
    });
    return fetchAPI<InitialDataResponse>(`/initial_data_no_noise?${queryParams.toString()}`);
  },

  /**
   * Map selected points to clusters
   */
  pointToCluster: async (request: PointToClusterRequest) => {
    return fetchAPI<PointToClusterResponse>('/point_to_cluster', {
      method: 'POST',
      body: JSON.stringify({
        point_ids: request.pointIds,
        containment_ratio_threshold: request.containmentRatioThreshold ?? 0.1
      })
    });
  },

  /**
   * Get similarity/heatmap data
   */
  getHeatmap: async (request: HeatmapRequest) => {
    const queryParams = new URLSearchParams({
      metric: request.metric,
      ...(request.topN && { top_n: request.topN.toString() }),
      ...(request.clusterIds && { cluster_ids: JSON.stringify(request.clusterIds) })
    });
    return fetchAPI<HeatmapResponse>(`/heatmap?${queryParams.toString()}`);
  },

  /**
   * Get details for a specific cluster
   */
  getClusterDetail: async (clusterId: number) => {
    return fetchAPI<ClusterDetailResponse>(`/cluster/${clusterId}`);
  },

  /**
   * Get details for a specific point
   */
  getPointDetail: async (pointId: number) => {
    return fetchAPI<PointDetailResponse>(`/point/${pointId}`);
  },

  /**
   * Get available datasets
   */
  getDatasets: async () => {
    return fetchAPI<DatasetsResponse>('/datasets');
  },

  /**
   * Filter dendrogram by Strahler and Stability values
   */
  filterDendrogram: async (request: DendrogramFilterRequest) => {
    return fetchAPI<DendrogramFilterResponse>('/dendrogram/filter', {
      method: 'POST',
      body: JSON.stringify({
        strahler_range: request.strahlerRange,
        stability_range: request.stabilityRange
      })
    });
  },

  /**
   * Fetch high-dimensional vectors for selected point IDs
   */
  fetchPointVectors: async (request: PointVectorsRequest) => {
    return fetchAPI<PointVectorsResponse>('/point_vectors', {
      method: 'POST',
      body: JSON.stringify({
        point_ids: request.point_ids,
        dataset: request.dataset ?? 'default'
      })
    });
  },

  /**
   * Zoom/Redraw coordinates using UMAP on dedicated server
   * Calls CPU or GPU UMAP server on port 8001
   */
  zoomRedraw: async (request: ZoomRedrawRequest) => {
    const url = `${ZOOM_API_URL}/zoom/redraw`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: `HTTP ${response.status}`
      }));
      throw new Error(JSON.stringify(error));
    }

    return response.json() as Promise<ZoomRedrawResponse>;
  }
};

/**
 * Type-safe wrapper for API calls with error handling
 */
export async function apiCall<T>(
  fn: () => Promise<T>,
  onError?: (error: Error) => void
): Promise<T | null> {
  try {
    return await fn();
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('API Error:', errorMessage);
    onError?.(error instanceof Error ? error : new Error(errorMessage));
    return null;
  }
}
