import { apiRequest } from './client'

export function processVideo(payload) {
  return apiRequest('/api/v1/video/process', {
    method: 'POST',
    body: JSON.stringify(payload)
  })
}