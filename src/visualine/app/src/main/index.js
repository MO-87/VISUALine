import { app, shell, BrowserWindow, ipcMain, dialog, protocol, net } from 'electron'
import { join, resolve } from 'path'
import { existsSync } from 'fs'
import { pathToFileURL } from 'url'
import http from 'http'
import { spawn } from 'child_process'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'

// Register this BEFORE app.whenReady().
protocol.registerSchemesAsPrivileged([
  {
    scheme: 'visualine-media',
    privileges: {
      standard: true,
      secure: true,
      stream: true,
      supportFetchAPI: true,
      corsEnabled: true
    }
  }
])

// WSL/Electron sometimes fails Chromium GPU initialization.
// This does NOT disable PyTorch CUDA in the Python backend.
app.disableHardwareAcceleration()
app.commandLine.appendSwitch('disable-gpu')

let pythonServer = null
let backendStartedByElectron = false

const BACKEND_HOST = '127.0.0.1'
const BACKEND_PORT = 8000
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`

function getProjectRoot() {
  // In dev, npm is run from:
  // VISUALine/src/visualine/app
  if (is.dev) {
    return resolve(process.cwd(), '../../..')
  }

  return process.resourcesPath
}

function getDevPythonExecutable(projectRoot) {
  const linuxVenvPython = join(projectRoot, '.venv', 'bin', 'python')
  const winVenvPython = join(projectRoot, '.venv', 'Scripts', 'python.exe')

  if (existsSync(linuxVenvPython)) return linuxVenvPython
  if (existsSync(winVenvPython)) return winVenvPython

  return 'python'
}

function checkBackendOnline(timeoutMs = 700) {
  return new Promise((resolveOnline) => {
    const req = http.get(`${BACKEND_URL}/api/v1/status`, (res) => {
      res.resume()
      resolveOnline(res.statusCode >= 200 && res.statusCode < 500)
    })

    req.on('error', () => resolveOnline(false))

    req.setTimeout(timeoutMs, () => {
      req.destroy()
      resolveOnline(false)
    })
  })
}

function waitForBackend(timeoutMs = 60000) {
  const startedAt = Date.now()

  return new Promise((resolveReady) => {
    const tick = async () => {
      const online = await checkBackendOnline(500)

      if (online) {
        resolveReady(true)
        return
      }

      if (Date.now() - startedAt > timeoutMs) {
        resolveReady(false)
        return
      }

      setTimeout(tick, 500)
    }

    tick()
  })
}

async function startPythonBackendIfNeeded() {
  const alreadyOnline = await checkBackendOnline()

  if (alreadyOnline) {
    console.log('Backend already running. Electron will reuse it.')
    return
  }

  const projectRoot = getProjectRoot()

  if (app.isPackaged) {
    const executableName = process.platform === 'win32' ? 'visualine-api.exe' : 'visualine-api'
    const executablePath = join(process.resourcesPath, 'visualine-api', executableName)

    pythonServer = spawn(executablePath, [], {
      detached: false,
      windowsHide: true
    })
  } else {
    const pythonExecutable = getDevPythonExecutable(projectRoot)

    pythonServer = spawn(
      pythonExecutable,
      ['-m', 'visualine.api.server'],
      {
        cwd: projectRoot,
        detached: false,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1'
        }
      }
    )
  }

  backendStartedByElectron = true

  if (pythonServer) {
    pythonServer.stdout.on('data', (data) => {
      console.log(`Backend: ${data.toString().trim()}`)
    })

    // Uvicorn writes many INFO logs to stderr, so don't label all stderr as fatal.
    pythonServer.stderr.on('data', (data) => {
      console.log(`Backend: ${data.toString().trim()}`)
    })

    pythonServer.on('exit', (code, signal) => {
      console.log(`Backend process exited. code=${code}, signal=${signal}`)
      pythonServer = null
      backendStartedByElectron = false
    })
  }

  console.log('Starting VISUALine AI Suite Backend...')

  const ready = await waitForBackend(60000)

  if (!ready) {
    console.warn('Backend did not become ready within timeout. Renderer will keep retrying.')
  } else {
    console.log('VISUALine backend is ready.')
  }
}

function registerLocalMediaProtocol() {
  protocol.handle('visualine-media', async (request) => {
    try {
      const url = new URL(request.url)
      const rawPath = url.searchParams.get('path')

      if (!rawPath) {
        return new Response('Missing media path', { status: 400 })
      }

      const filePath = rawPath.startsWith('file://')
        ? decodeURIComponent(new URL(rawPath).pathname)
        : rawPath

      return net.fetch(pathToFileURL(filePath).toString())
    } catch (error) {
      console.error('Failed to serve local media:', error)
      return new Response('Failed to serve local media', { status: 500 })
    }
  })
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1180,
    minHeight: 720,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#080d18',
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function registerIpcHandlers() {
  ipcMain.handle('dialog:select-media-file', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select media file',
      properties: ['openFile'],
      filters: [
        {
          name: 'Media Files',
          extensions: [
            'mp4',
            'mov',
            'mkv',
            'avi',
            'webm',
            'gif',
            'png',
            'jpg',
            'jpeg',
            'webp',
            'bmp',
            'tiff'
          ]
        },
        {
          name: 'All Files',
          extensions: ['*']
        }
      ]
    })

    if (result.canceled || !result.filePaths.length) {
      return null
    }

    return result.filePaths[0]
  })

  ipcMain.handle('dialog:select-output-dir', async () => {
    const result = await dialog.showOpenDialog({
      title: 'Select output directory',
      properties: ['openDirectory', 'createDirectory']
    })

    if (result.canceled || !result.filePaths.length) {
      return null
    }

    return result.filePaths[0]
  })

  ipcMain.handle('shell:open-path', async (_event, targetPath) => {
    if (!targetPath) return false
    await shell.openPath(targetPath)
    return true
  })

  ipcMain.handle('shell:open-external', async (_event, url) => {
    if (!url) return false
    await shell.openExternal(url)
    return true
  })
}

function stopPythonBackend() {
  if (!pythonServer || !backendStartedByElectron) {
    return
  }

  try {
    pythonServer.kill('SIGTERM')
  } catch (error) {
    console.warn('Failed to stop backend cleanly:', error)
  }

  pythonServer = null
  backendStartedByElectron = false
}

app.whenReady().then(async () => {
  electronApp.setAppUserModelId('com.visualine.app')

  registerIpcHandlers()
  registerLocalMediaProtocol()

  await startPythonBackendIfNeeded()

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  ipcMain.on('ping', () => console.log('pong'))

  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  stopPythonBackend()

  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopPythonBackend()
})