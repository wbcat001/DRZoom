/**
 * API request and response types
 * Defines the contract between frontend and backend
 */

import type {
  LinkageMatrix,
  SimilarityData,
  ClusterInfo,
  PointDetail,
  InitialDataPayload
} from './data';

/**
 * GET /api/initial_data?dataset=<dataset_name>&dr_method=<method>
 * Returns initial visualization data
 */
export interface InitialDataRequest {
  dataset: string;
  drMethod: 'umap' | 'tsne' | 'pca';
  drParams?: Record<string, any>;
}

export interface InitialDataResponse {
  success: boolean;
  data: InitialDataPayload;
  timestamp: string;
}

/**
 * POST /api/point_to_cluster
 * Maps selected points to clusters with containment filtering
 */
export interface PointToClusterRequest {
  pointIds: number[];
  containmentRatioThreshold?: number;
}

export interface PointToClusterResponse {
  clusterIds: number[];
  stats: {
    totalSelectedPoints: number;
    uniqueClusters: number;
    details: Record<number, {
      containmentRatio: number;
      pointCount: number;
    }>;
  };
}

/**
 * GET /api/heatmap?metric=<metric>&top_n=<n>&cluster_ids=<ids>
 * Returns similarity matrix for specified clusters
 */
export interface HeatmapRequest {
  metric: 'kl_divergence' | 'bhattacharyya_coefficient' | 'mahalanobis_distance';
  topN?: number;
  clusterIds?: number[];
}

export interface HeatmapResponse {
  data: SimilarityData;
  timestamp: string;
}

/**
 * GET /api/cluster/{id}
 * Returns detailed information about a specific cluster
 */
export interface ClusterDetailResponse {
  cluster: ClusterInfo;
  relatedClusters: Array<{
    id: number;
    label: string;
    similarity: number;
  }>;
  timestamp: string;
}

/**
 * GET /api/point/{id}
 * Returns detailed information about a specific point
 */
export interface PointDetailResponse {
  point: PointDetail;
  timestamp: string;
}

/**
 * GET /api/datasets
 * Returns available datasets
 */
export interface DatasetInfo {
  name: string;
  description: string;
  pointCount: number;
  clusterCount: number;
  drMethods: Array<'umap' | 'tsne' | 'pca'>;
}

export interface DatasetsResponse {
  datasets: DatasetInfo[];
}

/**
 * POST /api/dendrogram/filter
 * Filters dendrogram by Strahler and Stability values
 */
export interface DendrogramFilterRequest {
  strahlerRange?: [number, number];
  stabilityRange?: [number, number];
}

export interface DendrogramFilterResponse {
  filteredLinkageMatrix: LinkageMatrix;
  visibleClusterIds: number[];
}

/**
 * POST /api/point_vectors
 * Fetch high-dimensional vectors for selected point IDs
 */
export interface PointVectorsRequest {
  point_ids: number[];
  dataset?: string;
}

export interface PointVectorsResponse {
  success: boolean;
  vectors?: number[][];
  shape?: [number, number];
  message?: string;
}

/**
 * POST /api/zoom/redraw (CPU/GPU UMAP Server on port 8001)
 * Recalculates 2D coordinates for selected points using UMAP
 */
export interface ZoomRedrawRequest {
  vectors_b64: string;           // Base64-encoded high-dimensional vectors
  initial_embedding_b64?: string; // Base64-encoded initial 2D coordinates (optional, for mental map preservation)
  n_components?: number;          // Number of dimensions (default: 2)
  n_neighbors?: number;           // UMAP n_neighbors (default: 15)
  min_dist?: number;              // UMAP min_dist (default: 0.1)
  metric?: string;                // Distance metric (default: "euclidean")
  n_epochs?: number;              // UMAP training epochs (default: 200)
}

export interface ZoomRedrawResponse {
  status: 'success' | 'error';
  coordinates?: string;  // Base64-encoded (N, 2) array
  shape?: [number, number];
  message?: string;
}

/**
 * Generic error response
 */
export interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
  timestamp: string;
}

/**
 * Type guard for API responses
 */
export function isErrorResponse(response: any): response is ErrorResponse {
  return response && response.success === false && response.error;
}
