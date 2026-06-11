import { apiRequest } from './client'

export function getSystemHealth() {
  return apiRequest('/api/v1/system/health')
}

export function exploreFileSystem(path = null) {
  const query = path ? `?path=${encodeURIComponent(path)}` : ''
  return apiRequest(`/api/v1/system/explore${query}`)
}

export function searchLocalFile(name) {
  return apiRequest(`/api/v1/system/search?name=${encodeURIComponent(name)}`)
}

export function getOptimizationStreamUrl(modelType, params = {}) {
  const { tileSize = 64, padding = 16, batchSize = 16 } = params
  const baseUrl = 'http://localhost:8000/api/v1/system/optimize/stream'
  return `${baseUrl}?model_type=${modelType}&tile_size=${tileSize}&padding=${padding}&batch_size=${batchSize}`
}

export function getOptimizedModels() {
  return apiRequest('/api/v1/system/models')
}