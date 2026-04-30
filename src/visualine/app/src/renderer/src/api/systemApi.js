import { apiRequest } from './client'

export function getSystemHealth() {
  return apiRequest('/api/v1/system/health')
}