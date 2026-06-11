<template>
  <div v-if="open" class="optimization-overlay" @click.self="$emit('close')">
    <div class="optimization-card">
      <header class="opt-header">
        <div class="header-content">
          <h3>Hardware Optimization Hub</h3>
          <p>Compile AI models for maximum performance on your GPU using TensorRT.</p>
        </div>
        <button class="close-button" @click="$emit('close')">✕</button>
      </header>

      <div class="opt-body">
        <section class="opt-config-panel">
          <div class="config-group">
            <label>Select Model to Optimize</label>
            <select v-model="selectedModel" :disabled="isCompiling">
              <option value="span">SPAN x4 (Ultra-Fast)</option>
              <option value="realesr-anime">Real-ESRGAN Anime</option>
              <option value="realesr-x4plus">Real-ESRGAN 4x Plus (Sharp)</option>
            </select>
          </div>

          <div class="config-row">
            <div class="config-group">
              <label>Tile Size</label>
              <input v-model.number="params.tileSize" type="number" step="8" :disabled="isCompiling" />
            </div>
            <div class="config-group">
              <label>Padding</label>
              <input v-model.number="params.padding" type="number" step="4" :disabled="isCompiling" />
            </div>
          </div>

          <div class="config-group">
            <label>Optimal Batch Size</label>
            <input v-model.number="params.batchSize" type="number" :disabled="isCompiling" />
            <small>Recommended: 16-32 for RTX 3050. Higher = faster but more VRAM.</small>
          </div>

          <div class="info-box">
            <div class="info-icon">💡</div>
            <div class="info-text">
              Optimization typically takes <strong>2-5 minutes</strong>. Do not close the app or start other AI jobs while this is running.
            </div>
          </div>

          <button 
            class="run-opt-button" 
            :disabled="isCompiling"
            @click="startOptimization"
          >
            {{ isCompiling ? 'Compiling...' : 'Optimize for my Device' }}
          </button>
        </section>

        <section class="opt-log-panel">
          <div class="log-header">
            <span>Live Compilation Logs</span>
            <span v-if="isCompiling" class="status-pulse">Active</span>
          </div>
          <div ref="logContainer" class="log-display">
            <div v-for="(line, i) in logs" :key="i" class="log-line" :class="getLineClass(line)">
              {{ line }}
            </div>
            <div v-if="logs.length === 0" class="log-placeholder">
              Ready to compile...
            </div>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, watch, nextTick } from 'vue'
import { getOptimizationStreamUrl } from '../../api/systemApi'

const props = defineProps({
  open: Boolean
})

const emit = defineEmits(['close', 'completed'])

const selectedModel = ref('span')
const isCompiling = ref(false)
const logs = ref([])
const logContainer = ref(null)

const params = reactive({
  tileSize: 64,
  padding: 16,
  batchSize: 16
})

let eventSource = null

function getLineClass(line) {
  if (line.includes('[SUCCESS]')) return 'success'
  if (line.includes('[ERROR]')) return 'error'
  if (line.includes('---')) return 'header'
  return ''
}

async function startOptimization() {
  if (isCompiling.value) return
  
  isCompiling.value = true
  logs.value = ['Initializing connection to backend...']
  
  const url = getOptimizationStreamUrl(selectedModel.value, params)
  eventSource = new EventSource(url)
  
  eventSource.onmessage = (event) => {
    const data = event.data
    logs.value.push(data)
    
    if (data.includes('[SUCCESS]')) {
      isCompiling.value = false
      eventSource.close()
      emit('completed')
    }
    
    if (data.includes('[ERROR]')) {
      isCompiling.value = false
      eventSource.close()
    }
    
    scrollToBottom()
  }
  
  eventSource.onerror = (err) => {
    console.error('SSE Error:', err)
    logs.value.push('[ERROR] Connection lost or backend failed.')
    isCompiling.value = false
    eventSource.close()
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
}

watch(() => props.open, (newVal) => {
  if (!newVal && eventSource) {
    // Optionally keep compiling in background or stop?
    // User probably wants it to finish.
  }
})
</script>

<style scoped>
.optimization-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 2500;
  background: rgba(0, 0, 0, 0.75);
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(6px);
}

.optimization-card {
  width: min(1000px, 95vw);
  height: 80vh;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 20px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
}

.opt-header {
  padding: 24px 32px;
  background: linear-gradient(135deg, #1e293b, #0f172a);
  border-bottom: 1px solid #1e293b;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-content h3 {
  margin: 0;
  color: #f8fafc;
  font-size: 22px;
}

.header-content p {
  margin: 4px 0 0;
  color: #94a3b8;
  font-size: 14px;
}

.close-button {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  color: #94a3b8;
  width: 36px;
  height: 36px;
  border-radius: 10px;
  cursor: pointer;
}

.opt-body {
  flex: 1;
  display: grid;
  grid-template-columns: 380px 1fr;
  min-height: 0;
}

.opt-config-panel {
  padding: 32px;
  background: #111827;
  border-right: 1px solid #1e293b;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.config-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.config-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.config-group label {
  color: #f1f5f9;
  font-size: 14px;
  font-weight: 600;
}

.config-group select, .config-group input {
  height: 42px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 10px;
  padding: 0 12px;
  color: #fff;
  font-size: 14px;
}

.config-group small {
  color: #64748b;
  font-size: 12px;
}

.info-box {
  background: rgba(39, 224, 209, 0.05);
  border: 1px solid rgba(39, 224, 209, 0.2);
  border-radius: 12px;
  padding: 16px;
  display: flex;
  gap: 12px;
}

.info-icon { font-size: 20px; }
.info-text { color: #d1d5db; font-size: 13px; line-height: 1.5; }

.run-opt-button {
  margin-top: auto;
  height: 50px;
  background: var(--cyan, #22d3ee);
  color: #0f172a;
  border: none;
  border-radius: 12px;
  font-size: 15px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
}

.run-opt-button:hover:not(:disabled) {
  transform: translateY(-2px);
  filter: brightness(1.1);
  box-shadow: 0 8px 20px rgba(34, 211, 238, 0.3);
}

.run-opt-button:disabled {
  background: #334155;
  color: #94a3b8;
  cursor: wait;
}

.opt-log-panel {
  padding: 32px;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.log-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
  color: #94a3b8;
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.status-pulse {
  color: #22d3ee;
  animation: pulse 2s infinite;
}

.log-display {
  flex: 1;
  background: #020617;
  border: 1px solid #1e293b;
  border-radius: 12px;
  padding: 20px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  line-height: 1.6;
  overflow-y: auto;
  color: #cbd5e1;
}

.log-line { margin-bottom: 4px; }
.log-line.success { color: #4ade80; font-weight: bold; }
.log-line.error { color: #f87171; font-weight: bold; }
.log-line.header { color: #22d3ee; border-bottom: 1px solid #1e293b; padding-bottom: 4px; margin-top: 12px; }

.log-placeholder {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #334155;
  font-style: italic;
}

@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.4; }
  100% { opacity: 1; }
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
