import { apiRequest, buildApiUrl, buildWsUrl } from './client'

export function getJob(jobId) {
  return apiRequest(`/api/v1/jobs/${encodeURIComponent(jobId)}`)
}

export function getJobProgress(jobId) {
  return apiRequest(`/api/v1/jobs/${encodeURIComponent(jobId)}/progress`)
}

export function deleteJob(jobId, deleteFiles = false) {
  return apiRequest(
    `/api/v1/jobs/${encodeURIComponent(jobId)}?delete_files=${deleteFiles}`,
    {
      method: 'DELETE'
    }
  )
}

export function getJobOutputUrl(jobId) {
  return buildApiUrl(`/api/v1/jobs/${encodeURIComponent(jobId)}/output`)
}

export function createJobProgressSocket(jobId, handlers = {}) {
  const ws = new WebSocket(
    buildWsUrl(`/ws/progress/${encodeURIComponent(jobId)}`)
  )

  ws.onopen = () => {
    handlers.onOpen?.()
  }

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      handlers.onMessage?.(data)

      if (
        data.status === 'completed' ||
        data.status === 'failed' ||
        data.status === 'cancelled'
      ) {
        ws.close()
      }
    } catch (error) {
      handlers.onError?.(error)
    }
  }

  ws.onerror = (event) => {
    handlers.onError?.(event)
  }

  ws.onclose = () => {
    handlers.onClose?.()
  }

  return ws
}