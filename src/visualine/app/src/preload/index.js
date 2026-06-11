import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

const visualineAPI = {
  selectMediaFile: () => ipcRenderer.invoke('dialog:select-media-file'),
  selectOutputDir: () => ipcRenderer.invoke('dialog:select-output-dir'),
  openPath: (targetPath) => ipcRenderer.invoke('shell:open-path', targetPath),
  openExternal: (url) => ipcRenderer.invoke('shell:open-external', url),

  createMediaUrl: (filePath) => {
    if (!filePath) return ''

    if (filePath.startsWith('visualine-media://')) {
      return filePath
    }

    return `visualine-media://local?path=${encodeURIComponent(filePath)}`
  },

  backend: {
    baseUrl: 'http://127.0.0.1:8000',
    wsBaseUrl: 'ws://127.0.0.1:8000'
  }
}

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('visualine', visualineAPI)
  } catch (error) {
    console.error(error)
  }
} else {
  window.electron = electronAPI
  window.visualine = visualineAPI
}