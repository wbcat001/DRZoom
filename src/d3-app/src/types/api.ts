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
