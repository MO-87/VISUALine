<template>
  <header class="top-bar studio-topbar">
    <div class="topbar-left">
      <div class="app-mark">
        <span class="mark-dot" />
        <span class="logo-mark">VISUALine</span>
      </div>

      <div class="workflow-title-block">
        <span class="eyebrow">Current Workflow</span>
        <strong>{{ workflowTitle }}</strong>
      </div>

      <button
        class="topbar-button"
        type="button"
        @click="$emit('open-workflow-picker')"
      >
        Workflows
      </button>

      <button
        class="topbar-button browse"
        type="button"
        @click="$emit('select-file')"
      >
        Browse Media
      </button>

      <div class="media-chip" :class="{ empty: !mediaPath }">
        <span class="media-chip-label">Source</span>
        <strong>{{ mediaLabel }}</strong>
      </div>
    </div>

    <div class="top-actions">
      <StatusBadge
        :label="backendOnline ? 'Backend Online' : 'Backend Offline'"
        :variant="backendOnline ? 'success' : 'danger'"
      />

      <StatusBadge
        v-if="gpuLabel"
        :label="gpuLabel"
        variant="info"
      />

      <StatusBadge
        v-if="activeJobsCount > 0"
        :label="`${activeJobsCount} Active Job${activeJobsCount > 1 ? 's' : ''}`"
        variant="warning"
      />

      <button
        class="icon-button"
        title="Refresh system"
        type="button"
        @click="$emit('refresh-system')"
      >
        ⟳
      </button>
    </div>
  </header>
</template>

<script setup>
import { computed } from 'vue'
import StatusBadge from './StatusBadge.vue'

const props = defineProps({
  systemHealth: {
    type: Object,
    default: null
  },
  backendOnline: {
    type: Boolean,
    default: false
  },
  activeJobsCount: {
    type: Number,
    default: 0
  },
  selectedWorkflow: {
    type: Object,
    default: null
  },
  mediaPath: {
    type: String,
    default: ''
  }
})

defineEmits(['refresh-system', 'open-workflow-picker', 'select-file'])

const workflowTitle = computed(() => {
  return (
    props.selectedWorkflow?.display_name ||
    props.selectedWorkflow?.pipeline_name ||
    'Choose a workflow'
  )
})

const mediaLabel = computed(() => {
  if (!props.mediaPath) return 'No media selected'
  return props.mediaPath.split(/[\\/]/).pop()
})

const gpuLabel = computed(() => {
  const hw = props.systemHealth?.hardware
  if (!hw) return null

  if (hw.device === 'cuda') {
    const used = hw.vram_allocated_gb ?? hw.vram_used_gb
    const total = hw.vram_total_gb

    if (used != null && total != null) {
      return `GPU ${used.toFixed(1)} / ${total.toFixed(1)} GB`
    }

    return hw.device_name || 'GPU Active'
  }

  return hw.device ? hw.device.toUpperCase() : null
})
</script>

<style scoped>
.studio-topbar {
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 0 20px;
  background:
    linear-gradient(90deg, rgba(8, 13, 24, 0.98), rgba(13, 21, 36, 0.94)),
    rgba(9, 15, 28, 0.9);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(18px);
}

.topbar-left {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 14px;
}

.app-mark {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  padding-right: 18px;
  border-right: 1px solid var(--border);
}

.mark-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: var(--cyan);
  box-shadow: 0 0 18px rgba(39, 224, 209, 0.8);
}

.logo-mark {
  color: var(--cyan);
  font-weight: 950;
  letter-spacing: 0.04em;
  font-size: 28px;
  line-height: 1;
}

.workflow-title-block {
  min-width: 0;
  display: grid;
  gap: 3px;
}

.eyebrow,
.media-chip-label {
  color: var(--muted-2);
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.workflow-title-block strong {
  max-width: 360px;
  color: var(--text);
  font-size: 14px;
  line-height: 1.1;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.topbar-button {
  flex: 0 0 auto;
  min-height: 36px;
  padding: 0 13px;
  border-radius: 11px;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.22);
  font-size: 12px;
  font-weight: 900;
  cursor: pointer;
}

.topbar-button:hover {
  background: rgba(39, 224, 209, 0.13);
  border-color: rgba(39, 224, 209, 0.42);
}

.topbar-button.browse {
  color: #ffffff;
  background: linear-gradient(90deg, rgba(39, 224, 209, 0.78), rgba(124, 58, 237, 0.82));
  border-color: rgba(39, 224, 209, 0.26);
}

.media-chip {
  min-width: 170px;
  max-width: 260px;
  display: grid;
  gap: 3px;
  padding: 8px 11px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border);
}

.media-chip.empty strong {
  color: var(--muted);
}

.media-chip strong {
  min-width: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 12px;
  line-height: 1.1;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.top-actions {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 10px;
}

.icon-button {
  width: 34px;
  height: 34px;
  border-radius: 11px;
  display: grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.045);
  color: var(--muted);
  border: 1px solid var(--border);
  cursor: pointer;
}

.icon-button:hover {
  color: var(--cyan);
  border-color: rgba(39, 224, 209, 0.32);
  background: rgba(39, 224, 209, 0.06);
}

@media (max-width: 1220px) {
  .media-chip {
    display: none;
  }

  .workflow-title-block strong {
    max-width: 240px;
  }
}

@media (max-width: 980px) {
  .workflow-title-block {
    display: none;
  }
}
</style>