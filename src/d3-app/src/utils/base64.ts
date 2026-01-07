/**
 * Base64 encoding/decoding utilities for UMAP server communication
 */

/**
 * Encode a Float32Array to Base64 string (NumPy .npy format compatible)
 * @param array - Float32Array to encode
 * @returns Base64-encoded string
 */
export function encodeFloat32ArrayToBase64(array: Float32Array): string {
  // Create a simulated NumPy .npy file format
  // For simplicity, we'll just encode the raw float32 data
  // The server expects numpy.save() format
  
  const buffer = array.buffer;
  const bytes = new Uint8Array(buffer);
  
  // Convert to base64
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

/**
 * Decode Base64 string to Float32Array (NumPy .npy format compatible)
 * @param base64String - Base64-encoded string from server
 * @returns Float32Array
 */
export function decodeBase64ToFloat32Array(base64String: string): Float32Array {
  // Decode base64
  const binary = atob(base64String);
  const bytes = new Uint8Array(binary.length);
  
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  
  // Convert to Float32Array
  return new Float32Array(bytes.buffer);
}

/**
 * Encode 2D array (vectors) to Base64 for UMAP server
 * @param vectors - 2D array of shape (N, D)
 * @returns Base64-encoded string
 */
export function encodeVectorsToBase64(vectors: number[][]): string {
  const n = vectors.length;
  const d = vectors[0]?.length || 0;
  
  // Flatten to 1D array
  const flat = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      flat[i * d + j] = vectors[i][j];
    }
  }
  
  return encodeFloat32ArrayToBase64(flat);
}

/**
 * Decode Base64 string to 2D coordinates array
 * @param base64String - Base64-encoded string from server
 * @param n - Number of points
 * @param d - Number of dimensions (default: 2)
 * @returns 2D array of shape (N, D)
 */
export function decodeBase64ToCoordinates(
  base64String: string,
  n: number,
  d: number = 2
): number[][] {
  const flat = decodeBase64ToFloat32Array(base64String);
  
  // Reshape to 2D array
  const coords: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < d; j++) {
      row.push(flat[i * d + j]);
    }
    coords.push(row);
  }
  
  return coords;
}
