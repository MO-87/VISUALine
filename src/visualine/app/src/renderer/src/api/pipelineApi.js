import { apiRequest } from './client'

export function listPipelines() {
  return apiRequest('/api/v1/pipelines')
}

export function getPipelineDetail(pipelineId) {
  return apiRequest(`/api/v1/pipelines/${encodeURIComponent(pipelineId)}`)
}