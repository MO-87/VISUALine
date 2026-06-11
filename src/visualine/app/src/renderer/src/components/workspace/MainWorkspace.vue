<template>
  <section class="main-workspace no-left-panel">
    <main class="workspace-preview-column">
      <MediaPreview
        :original-url="originalPreviewUrl"
        :output-url="outputUrl"
        :is-video="selectedMediaKind === 'video'"
        :processing="isProcessing"
        :progress="jobProgress"
        :message="currentJob?.message"
        :current-frame="currentJob?.current_frame"
        :total-frames="currentJob?.total_frames"
        @request-media="$emit('select-file')"
        @request-workflow="$emit('open-workflow-picker')"
      />
    </main>

    <aside class="workspace-side-column">
      <ParameterPanel
        class="workspace-controls-column"
        :workflow="activeWorkflow"
        :values="controlValues"
        :processing="isProcessing"
        :progress="jobProgress"
        :disabled="!canRun"
        @update-control="forwardControlUpdate"
        @run="$emit('run')"
      />

      <JobProgress
        v-if="currentJob"
        :job="currentJob"
      />

      <OutputActions
        v-if="outputUrl"
        :output-url="outputUrl"
      />
    </aside>
  </section>
</template>

<script setup>
import { computed } from 'vue'

import ParameterPanel from '../controls/ParameterPanel.vue'
import JobProgress from '../jobs/JobProgress.vue'
import OutputActions from '../jobs/OutputActions.vue'
import MediaPreview from '../media/MediaPreview.vue'

const props = defineProps({
  selectedWorkflowDetail: {
    type: Object,
    default: null
  },
  workflow: {
    type: Object,
    default: null
  },
  mediaPath: {
    type: String,
    default: ''
  },
  selectedMediaKind: {
    type: String,
    default: 'video'
  },
  originalPreviewUrl: {
    type: String,
    default: ''
  },
  outputUrl: {
    type: String,
    default: ''
  },
  controlValues: {
    type: Object,
    default: () => ({})
  },
  currentJob: {
    type: Object,
    default: null
  },
  isProcessing: {
    type: Boolean,
    default: false
  },
  jobProgress: {
    type: Number,
    default: 0
  },
  canRun: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits([
  'select-file',
  'update:mediaPath',
  'update-control',
  'run',
  'open-workflow-picker'
])

const activeWorkflow = computed(() => {
  return props.selectedWorkflowDetail || props.workflow
})

function forwardControlUpdate(key, value) {
  emit('update-control', key, value)
}
</script>

<style scoped>
.main-workspace {
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-columns: minmax(620px, 1fr) minmax(320px, 360px);
  gap: 14px;
}

.workspace-preview-column {
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.workspace-side-column {
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  overflow-x: hidden;
  display: grid;
  align-content: start;
  gap: 14px;
}

.workspace-controls-column {
  min-width: 0;
  min-height: 0;
}

.workspace-side-column::-webkit-scrollbar {
  width: 8px;
}

.workspace-side-column::-webkit-scrollbar-thumb {
  background: rgba(142, 155, 179, 0.24);
  border-radius: 999px;
}

.workspace-side-column::-webkit-scrollbar-track {
  background: transparent;
}

@media (max-width: 1120px) {
  .main-workspace {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(420px, 1fr) auto;
    overflow-y: auto;
  }

  .workspace-side-column {
    overflow: visible;
  }
}
</style>