const API_BASE_URL = 'http://127.0.0.1:8000'

function buildUrl(path) {
  if (path.startsWith('http')) return path
  return `${API_BASE_URL}${path}`
}

export async function apiRequest(path, options = {}) {
  const url = buildUrl(path)

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    ...options
  })

  const contentType = response.headers.get('content-type') || ''

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`

    if (contentType.includes('application/json')) {
      const errorBody = await response.json().catch(() => null)
      detail = errorBody?.detail || detail
    } else {
      detail = await response.text().catch(() => detail)
    }

    throw new Error(detail)
  }

  if (contentType.includes('application/json')) {
    return response.json()
  }

  return response
}

export function getApiBaseUrl() {
  return API_BASE_URL
}

export function buildApiUrl(path) {
  return buildUrl(path)
}

export function buildWsUrl(path) {
  return buildUrl(path).replace(/^http/, 'ws')
}