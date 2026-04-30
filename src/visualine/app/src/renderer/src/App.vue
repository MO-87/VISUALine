<template>
  <AppShell
    :active-section="activeSection"
    :categories="categories"
    :system-health="systemHealth"
    :backend-online="backendOnline"
    :active-jobs-count="activeJobsCount"
    :selected-workflow="selectedWorkflowDetail"
    :media-path="mediaPath"
    @select-section="handleSectionSelect"
    @select-category="activeCategory = $event"
    @refresh-system="loadSystemHealth"
    @open-workflow-picker="openWorkflowPicker"
    @select-file="selectMediaFile"
  >
    <MainWorkspace
      :selected-workflow-detail="selectedWorkflowDetail"
      :media-path="mediaPath"
      :selected-media-kind="selectedMediaKind"
      :original-preview-url="originalPreviewUrl"
      :output-url="outputUrl"
      :control-values="controlValues"
      :current-job="currentJob"
      :is-processing="isProcessing"
      :job-progress="jobProgress"
      :can-run="canRun"
      @select-file="selectMediaFile"
      @update:media-path="mediaPath = $event"
      @update:mediaPath="mediaPath = $event"
      @update-control="updateControl"
      @run="runWorkflow"
      @open-workflow-picker="openWorkflowPicker"
    />

    <WorkflowPickerDrawer
      :open="workflowPickerOpen"
      :workflows="pipelines"
      :selected-workflow-id="selectedPipelineId"
      :loading="loadingPipelines"
      @close="workflowPickerOpen = false"
      @select="selectWorkflow"
    />
  </AppShell>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'

import AppShell from './components/layout/AppShell.vue'
import MainWorkspace from './components/workspace/MainWorkspace.vue'
import WorkflowPickerDrawer from './components/workflows/WorkflowPickerDrawer.vue'

import { listPipelines, getPipelineDetail } from './api/pipelineApi'
import { getSystemHealth } from './api/systemApi'
import { processVideo } from './api/videoApi'
import { processImage } from './api/imageApi'
import { createJobProgressSocket, getJobOutputUrl } from './api/jobsApi'

const activeSection = ref('workflows')
const activeCategory = ref(null)
const workflowPickerOpen = ref(false)

const pipelines = ref([])
const selectedPipelineId = ref(null)
const selectedWorkflowDetail = ref(null)

const loadingPipelines = ref(false)
const systemHealth = ref(null)
const backendOnline = ref(false)

const mediaPath = ref('')
const controlValues = ref({})

const currentJob = ref(null)
const currentSocket = ref(null)
const outputUrl = ref('')

let healthInterval = null

const selectedMediaKind = computed(() => {
  const lower = mediaPath.value.toLowerCase()

  const imageExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']

  if (imageExtensions.some((extension) => lower.endsWith(extension))) {
    return 'image'
  }

  return 'video'
})

const categories = computed(() => {
  const values = pipelines.value
    .map((pipeline) => pipeline.category)
    .filter(Boolean)

  return [...new Set(values)]
})

const activeJobsCount = computed(() => {
  return (
    systemHealth.value?.active_jobs_count ??
    systemHealth.value?.jobs?.active_jobs_count ??
    0
  )
})

const isProcessing = computed(() => {
  const status = String(currentJob.value?.status || '').toLowerCase()
  return ['queued', 'pending', 'processing'].includes(status)
})

const jobProgress = computed(() => {
  const value = Number(currentJob.value?.progress || 0)

  if (Number.isNaN(value)) return 0

  return Math.min(100, Math.max(0, value))
})

const canRun = computed(() => {
  return Boolean(
    selectedWorkflowDetail.value &&
    selectedPipelineId.value &&
    mediaPath.value &&
    !isProcessing.value
  )
})

const originalPreviewUrl = computed(() => {
  if (!mediaPath.value) return ''

  if (window.visualine?.createMediaUrl) {
    return window.visualine.createMediaUrl(mediaPath.value)
  }

  if (mediaPath.value.startsWith('file://')) {
    return mediaPath.value
  }

  const normalizedPath = mediaPath.value.replace(/\\/g, '/')
  return encodeURI(`file://${normalizedPath}`)
})

watch(mediaPath, () => {
  outputUrl.value = ''

  if (!isProcessing.value) {
    currentJob.value = null
  }
})

function handleSectionSelect(section) {
  activeSection.value = section

  if (section === 'workflows') {
    openWorkflowPicker()
  }
}

function openWorkflowPicker() {
  workflowPickerOpen.value = true
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function waitForBackendFromRenderer(maxAttempts = 40, delayMs = 750) {
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      systemHealth.value = await getSystemHealth()
      backendOnline.value = true
      return true
    } catch {
      backendOnline.value = false
      await sleep(delayMs)
    }
  }

  return false
}

async function loadSystemHealth() {
  try {
    systemHealth.value = await getSystemHealth()
    backendOnline.value = true
  } catch (error) {
    backendOnline.value = false
    console.warn('Backend offline:', error)
  }
}

async function loadPipelines() {
  loadingPipelines.value = true

  try {
    const backendReady = await waitForBackendFromRenderer()

    if (!backendReady) {
      console.warn('Backend not ready. Could not load pipelines.')
      pipelines.value = []
      return
    }

    pipelines.value = await listPipelines()

    if (pipelines.value.length && !selectedPipelineId.value) {
      await selectWorkflow(pipelines.value[0])
    }
  } catch (error) {
    console.error('Failed to load pipelines:', error)
    pipelines.value = []
  } finally {
    loadingPipelines.value = false
  }
}

async function selectWorkflow(workflow) {
  if (!workflow?.id) return

  selectedPipelineId.value = workflow.id
  activeSection.value = 'workflows'

  try {
    selectedWorkflowDetail.value = await getPipelineDetail(workflow.id)
    initializeControls(selectedWorkflowDetail.value.controls || [])

    outputUrl.value = ''

    if (!isProcessing.value) {
      currentJob.value = null
    }
  } catch (error) {
    console.error('Failed to load workflow detail:', error)
  }
}

function initializeControls(controls) {
  const values = {}

  for (const control of controls) {
    values[control.key] = control.default
  }

  controlValues.value = values
}

function updateControl(key, value) {
  controlValues.value = {
    ...controlValues.value,
    [key]: value
  }
}

async function selectMediaFile() {
  if (window.visualine?.selectMediaFile) {
    const selected = await window.visualine.selectMediaFile()

    if (selected) {
      mediaPath.value = selected
    }

    return
  }

  const manual = window.prompt('Paste absolute media path:')

  if (manual) {
    mediaPath.value = manual
  }
}

async function runWorkflow() {
  if (!canRun.value) return

  closeCurrentSocket()

  outputUrl.value = ''
  currentJob.value = {
    status: 'queued',
    progress: 0,
    message: 'Queuing job...'
  }

  try {
    const payload = {
      pipeline_id: selectedPipelineId.value,
      input_path: mediaPath.value,
      overrides: controlValues.value
    }

    const response =
      selectedMediaKind.value === 'image'
        ? await processImage(payload)
        : await processVideo(payload)

    currentJob.value = response

    if (response?.job_id) {
      attachJobSocket(response.job_id)
    }
  } catch (error) {
    currentJob.value = {
      status: 'failed',
      progress: 0,
      error_message: error.message || 'Failed to start processing job.'
    }

    await loadSystemHealth()
  }
}

function attachJobSocket(jobId) {
  closeCurrentSocket()

  currentSocket.value = createJobProgressSocket(jobId, {
    onMessage(data) {
      currentJob.value = data

      const status = String(data.status || '').toLowerCase()

      if (status === 'completed') {
        outputUrl.value = getJobOutputUrl(jobId)
        loadSystemHealth()
      }

      if (status === 'failed' || status === 'cancelled') {
        loadSystemHealth()
      }
    },

    onError(error) {
      console.warn('Job websocket error:', error)
    },

    onClose() {
      currentSocket.value = null
    }
  })
}

function closeCurrentSocket() {
  if (!currentSocket.value) return

  try {
    currentSocket.value.close()
  } catch {
    // Ignore close errors.
  }

  currentSocket.value = null
}

onMounted(async () => {
  await waitForBackendFromRenderer()
  await loadPipelines()

  healthInterval = window.setInterval(loadSystemHealth, 5000)
})

onUnmounted(() => {
  if (healthInterval) {
    window.clearInterval(healthInterval)
    healthInterval = null
  }

  closeCurrentSocket()
})
</script>