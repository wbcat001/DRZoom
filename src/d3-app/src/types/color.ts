/**
 * Color and highlight types
 * Defines the color scheme and highlighting rules
 */

/**
 * Highlight states and their corresponding colors
 * Priority order (highest to lowest):
 * 1. Heatmap click (red)
 * 2. DR selection (orange)
 * 3. DR-derived cluster selection (salmon)
 * 4. Dendrogram selection (lime green)
 * 5. Search result (purple)
 * 6. Default (blue)
 * 7. Dimmed (light blue)
 */
export type HighlightType =
  | 'default'
  | 'default_dimmed'
  | 'dr_selection'
  | 'dr_selected'
  | 'heatmap_click'
  | 'heatmap_to_dr'
  | 'dendrogram_to_dr'
  | 'search_result';

export const HIGHLIGHT_COLORS: Record<HighlightType, string> = {
  default: '#4A90E2',           // Bright blue - default state
  default_dimmed: '#B8D4F0',    // Light blue - background when something else selected
  dr_selection: '#FFA500',      // Orange - DR lasso selection
  dr_selected: '#FF6B6B',       // Salmon - clusters derived from DR point selection
  heatmap_click: '#FF0000',     // Red - heatmap cell click
  heatmap_to_dr: '#FF1493',     // Deep pink - heatmap hover
  dendrogram_to_dr: '#32CD32',  // Lime green - dendrogram hover
  search_result: '#9C27B0'      // Purple - label search result
};

/**
 * Interaction source tracking
 * Identifies which view initiated the selection
 */
export type InteractionSource =
  | 'none'
  | 'dr_selection'
  | 'dendrogram_click'
  | 'dendrogram_hover'
  | 'heatmap_click'
  | 'heatmap_hover';

/**
 * Color determination logic based on selection state
 */
export interface HighlightState {
  isDRSelected: boolean;
  isHeatmapClicked: boolean;
  isDendrogramSelected: boolean;
  isSearchResult: boolean;
  isHovered: boolean;
}

/**
 * Get the appropriate color based on highlight state
 * Follows the priority order: heatmap > dr > dendrogram > search > default
 */
export function getHighlightColor(state: HighlightState): string {
  if (state.isHeatmapClicked) {
    return HIGHLIGHT_COLORS.heatmap_click;
  }
  if (state.isDRSelected) {
    return HIGHLIGHT_COLORS.dr_selection;
  }
  if (state.isDendrogramSelected) {
    return HIGHLIGHT_COLORS.dendrogram_to_dr;
  }
  if (state.isSearchResult) {
    return HIGHLIGHT_COLORS.search_result;
  }
  return HIGHLIGHT_COLORS.default;
}

/**
 * Get the appropriate opacity based on highlight state
 */
export function getHighlightOpacity(state: HighlightState, anySelectionActive: boolean): number {
  if (state.isDRSelected || state.isHeatmapClicked || state.isDendrogramSelected || state.isSearchResult) {
    return 1.0;
  }
  if (anySelectionActive) {
    return 0.2; // Dimmed when something else is selected
  }
  return 1.0;
}

/**
 * SVG style configuration for rendered elements
 */
export interface SVGStyle {
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  opacity?: number;
}

export function getElementStyle(highlight: HighlightState, anySelectionActive: boolean): SVGStyle {
  // Don't override fill color - let cluster colors show through
  // Only use opacity to indicate selection state
  return {
    fill: undefined,  // Always undefined to preserve cluster colors
    stroke: '#333',
    strokeWidth: highlight.isHovered ? 2 : 0.5,
    opacity: getHighlightOpacity(highlight, anySelectionActive)
  };
}
