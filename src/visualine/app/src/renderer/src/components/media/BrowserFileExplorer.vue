<template>
  <div v-if="open" class="browser-explorer-overlay" @click.self="$emit('close')">
    <div class="browser-explorer-card">
      <header class="explorer-header">
        <div class="header-left">
          <h3>Local File Browser</h3>
          <p class="current-path">{{ currentPath }}</p>
        </div>
        <button class="close-button" @click="$emit('close')">✕</button>
      </header>

      <div class="explorer-toolbar">
        <button 
          class="nav-button" 
          :disabled="!parentPath"
          @click="navigateTo(parentPath)"
          title="Go to parent folder"
        >
          ↑ Up
        </button>
        <button class="nav-button" @click="navigateTo(null)" title="Go to Project Root">
          ⌂ Project
        </button>
        <button class="nav-button" @click="navigateTo('~')" title="Go to Home directory">
          🏠 Home
        </button>
        <button class="nav-button" @click="navigateTo('~/graduation/visualine_deployed/VISUALine/data/inputs/video')" title="Go to Video Inputs">
          🎬 Videos
        </button>
        
        <div class="flex-spacer"></div>
        
        <div class="search-box">
          <input 
            v-model="manualPath" 
            type="text" 
            placeholder="Paste absolute path here..."
            @keyup.enter="handleManualSubmit"
          />
          <button @click="handleManualSubmit">Go</button>
        </div>
      </div>

      <div v-if="error" class="explorer-error">
        <div class="error-icon">⚠️</div>
        <p>{{ error }}</p>
        <div class="error-actions">
          <button class="retry-button" @click="navigateTo(null)">Project Root</button>
          <button class="retry-button" @click="navigateTo('~')">Home Directory</button>
        </div>
      </div>

      <div v-else-if="loading" class="explorer-loading">
        <div class="spinner"></div>
        <p>Scanning directory...</p>
      </div>

      <div v-else-if="entries.length === 0" class="explorer-empty">
        <div class="empty-icon">📂</div>
        <p>This folder is empty</p>
        <button class="retry-button" @click="navigateTo(parentPath)" v-if="parentPath">Go Back</button>
      </div>

      <div v-else class="explorer-list">
        <div 
          v-for="entry in entries" 
          :key="entry.path"
          class="explorer-item"
          :class="{ 'is-dir': entry.is_dir, 'is-media': isMediaFile(entry.name) }"
          @click="handleEntryClick(entry)"
        >
          <span class="item-icon">{{ entry.is_dir ? '📁' : '📄' }}</span>
          <span class="item-name">{{ entry.name }}</span>
          <span v-if="!entry.is_dir" class="item-size">{{ formatSize(entry.size_bytes) }}</span>
        </div>
      </div>
      
      <footer class="explorer-footer">
        <p>Click a file to select it. Drag and drop also works from your system explorer.</p>
        <p class="hint">Paths starting with ~ are resolved to your home directory.</p>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { exploreFileSystem } from '../../api/systemApi'

const props = defineProps({
  open: Boolean
})

const emit = defineEmits(['close', 'select'])

const entries = ref([])
const currentPath = ref('')
const parentPath = ref(null)
const loading = ref(false)
const error = ref(null)
const manualPath = ref('')

async function navigateTo(path) {
  loading.value = true
  error.value = null
  try {
    const data = await exploreFileSystem(path)
    entries.value = data.entries
    currentPath.value = data.current_path
    parentPath.value = data.parent_path
  } catch (err) {
    console.error('Failed to explore filesystem:', err)
    error.value = err.message || 'Access denied or directory not found.'
  } finally {
    loading.value = false
  }
}

function handleManualSubmit() {
  if (manualPath.value) {
    navigateTo(manualPath.value)
    manualPath.value = ''
  }
}

function isMediaFile(name) {
  const ext = name.toLowerCase().split('.').pop()
  return ['mp4', 'mov', 'mkv', 'avi', 'jpg', 'jpeg', 'png', 'webp'].includes(ext)
}

function handleEntryClick(entry) {
  if (entry.is_dir) {
    navigateTo(entry.path)
  } else {
    emit('select', entry.path)
    emit('close')
  }
}

function formatSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

onMounted(() => {
  navigateTo(null)
})
</script>

<style scoped>
.browser-explorer-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 2000;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(4px);
}

.browser-explorer-card {
  width: min(900px, 95vw);
  height: 85vh;
  background: #0f172a;
  border: 1px solid #1e293b;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

.explorer-header {
  padding: 16px 24px;
  background: #1e293b;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.header-left h3 {
  margin: 0;
  color: #f8fafc;
  font-size: 18px;
  font-weight: 600;
}

.current-path {
  margin: 4px 0 0;
  font-size: 12px;
  color: #94a3b8;
  font-family: 'JetBrains Mono', monospace;
  word-break: break-all;
}

.close-button {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  color: #94a3b8;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  cursor: pointer;
  transition: all 0.2s;
}

.close-button:hover {
  background: #ef4444;
  color: white;
}

.explorer-toolbar {
  padding: 12px 24px;
  background: #111827;
  display: flex;
  align-items: center;
  gap: 8px;
  border-bottom: 1px solid #1e293b;
}

.nav-button {
  height: 34px;
  padding: 0 12px;
  background: #334155;
  border: 1px solid #475569;
  color: #f1f5f9;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.nav-button:hover:not(:disabled) {
  background: #475569;
  border-color: #64748b;
}

.nav-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.flex-spacer { flex: 1; }

.search-box {
  display: flex;
  gap: 8px;
}

.search-box input {
  width: 240px;
  height: 34px;
  background: #0f172a;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 0 12px;
  color: #f8fafc;
  font-size: 13px;
}

.search-box button {
  height: 34px;
  padding: 0 12px;
  background: var(--cyan, #22d3ee);
  color: #0f172a;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
}

.explorer-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.explorer-item {
  padding: 10px 16px;
  display: flex;
  align-items: center;
  gap: 14px;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.explorer-item:hover {
  background: rgba(34, 211, 238, 0.08);
}

.item-icon {
  font-size: 20px;
}

.item-name {
  flex: 1;
  color: #e2e8f0;
  font-size: 14px;
  font-weight: 400;
}

.is-media .item-name {
  color: #22d3ee;
  font-weight: 600;
}

.item-size {
  color: #64748b;
  font-size: 12px;
}

.explorer-loading, .explorer-empty, .explorer-error {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #94a3b8;
  gap: 16px;
}

.error-icon, .empty-icon {
  font-size: 48px;
}

.error-actions {
  display: flex;
  gap: 12px;
}

.retry-button {
  padding: 8px 16px;
  background: #334155;
  border: none;
  border-radius: 8px;
  color: white;
  cursor: pointer;
}

.explorer-footer {
  padding: 12px 24px;
  background: #111827;
  border-top: 1px solid #1e293b;
  font-size: 11px;
  color: #475569;
  display: flex;
  justify-content: space-between;
}

.hint {
  font-style: italic;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(34, 211, 238, 0.1);
  border-top-color: #22d3ee;
  border-radius: 50%;
  animation: spin 1s infinite linear;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 640px) {
  .search-box input {
    width: 140px;
  }
}
</style>
