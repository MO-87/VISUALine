<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

// --- State Variables ---
const systemHealth = ref(null)
const serverStatus = ref('Connecting...')
const currentJob = ref({
  id: null,
  status: 'Idle',
  progress: 0,
  message: ''
})

// --- API Functions ---
const API_BASE = 'http://127.0.0.1:8000/api/v1'
const WS_BASE = 'ws://127.0.0.1:8000/ws'

// 1. Fetch System Health (Hardware Info)
const checkSystemHealth = async () => {
  try {
    const response = await axios.get(`${API_BASE}/system/health`)
    systemHealth.value = response.data.hardware
    serverStatus.value = 'Online 🟢'
  } catch (error) {
    serverStatus.value = 'Offline 🔴 (Is the Python server running?)'
    console.error(error)
  }
}

// 2. Start a Test Job
const startTestJob = async () => {
  try {
    currentJob.value.status = 'Starting...'
    
    // NOTE: For this test, make sure these paths actually exist on your machine!
    // Or, change them to paths of a real short video and real config you have.
    const requestData = {
      input_path: "D:/Graduation Project/VISUALine/data/input/sample 2.mp4", 
      output_path: "D:/Graduation Project/VISUALine/data/output/test_output.mp4",
      pipeline_config_path: "D:/Graduation Project/VISUALine/configs/pipeline_configs/test_grayscale.yaml"
    }

    const response = await axios.post(`${API_BASE}/video/process`, requestData)
    const jobId = response.data.job_id
    currentJob.value.id = jobId
    
    // As soon as we get the ID, connect to the WebSocket!
    connectToWebSocket(jobId)

  } catch (error) {
    currentJob.value.status = 'Failed to start job'
    console.error(error)
  }
}

// 3. Listen to Real-Time Progress via WebSockets
const connectToWebSocket = (jobId) => {
  const socket = new WebSocket(`${WS_BASE}/progress/${jobId}`)

  socket.onopen = () => {
    currentJob.value.status = 'Processing (WebSocket Connected)'
  }

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data)
    
    // Update the UI in real-time
    currentJob.value.progress = data.progress
    currentJob.value.status = data.status
    currentJob.value.message = `Frame ${data.current_frame || 0} of ${data.total_frames || 0}`
    
    if (data.status === 'completed' || data.status === 'failed') {
      socket.close()
    }
  }

  socket.onerror = (error) => {
    console.error("WebSocket Error:", error)
  }
}

// Fetch health immediately when the app loads
onMounted(() => {
  checkSystemHealth()
})
</script>

<template>
  <div class="dashboard">
    <h1>VISUALine AI Suite</h1>
    
    <div class="card">
      <h2>Server Status: {{ serverStatus }}</h2>
      <div v-if="systemHealth">
        <p><strong>Device:</strong> {{ systemHealth.device_name || systemHealth.device.toUpperCase() }}</p>
        <p><strong>VRAM Limit:</strong> {{ systemHealth.vram_limit_gb?.toFixed(2) }} GB</p>
        <p><strong>VRAM Used:</strong> {{ systemHealth.vram_used_gb?.toFixed(2) }} GB</p>
      </div>
      <button @click="checkSystemHealth">Refresh Hardware Stats</button>
    </div>

    <div class="card">
      <h2>Pipeline Tester</h2>
      <button @click="startTestJob" :disabled="currentJob.status.includes('Processing')">
        Run Test Video Pipeline
      </button>

      <div v-if="currentJob.id" class="progress-section">
        <p><strong>Job ID:</strong> {{ currentJob.id }}</p>
        <p><strong>Status:</strong> {{ currentJob.status.toUpperCase() }}</p>
        <p><strong>Details:</strong> {{ currentJob.message }}</p>
        
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" :style="{ width: currentJob.progress + '%' }"></div>
        </div>
        <p>{{ currentJob.progress.toFixed(1) }}%</p>
      </div>
    </div>
  </div>
</template>

<style>
/* Brutally simple styling for the proof-of-concept */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background-color: #1e1e1e;
  color: #ffffff;
  padding: 20px;
}
.dashboard {
  max-width: 600px;
  margin: 0 auto;
}
.card {
  background-color: #2d2d2d;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
  border: 1px solid #404040;
}
button {
  background-color: #4CAF50;
  color: white;
  padding: 10px 15px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-weight: bold;
}
button:disabled {
  background-color: #555;
  cursor: not-allowed;
}
.progress-section {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #404040;
}
.progress-bar-bg {
  width: 100%;
  background-color: #404040;
  border-radius: 10px;
  height: 20px;
  overflow: hidden;
  margin: 10px 0;
}
.progress-bar-fill {
  height: 100%;
  background-color: #4CAF50;
  transition: width 0.3s ease;
}
</style>