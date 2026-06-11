import { apiRequest } from './client'

export function processImage(payload) {
  return apiRequest('/api/v1/image/process', {
    method: 'POST',
    body: JSON.stringify(payload)
  })
}

export function processImageBatch(payload) {
  return apiRequest('/api/v1/image/process-batch', {
    method: 'POST',
    body: JSON.stringify(payload)
  })
}