<template>
  <aside class="parameter-panel control-inspector">
    <header class="inspector-header">
      <div>
        <span class="panel-kicker">Parameters</span>
        <h2>Controls</h2>
      </div>

      <span v-if="workflow" class="control-count">
        {{ controlsCount }}
      </span>
    </header>

    <div v-if="!workflow" class="empty-panel small inspector-empty">
      <div class="empty-icon">⚙</div>
      <strong>No workflow selected</strong>
      <span>Choose a workflow to reveal its controls.</span>
    </div>

    <template v-else>
      <section class="workflow-mini-summary">
        <span class="summary-label">Active workflow</span>
        <strong>{{ workflow.display_name || workflow.pipeline_name || 'VISUALine Workflow' }}</strong>

        <p v-if="workflow.description">
          {{ workflow.description }}
        </p>
      </section>

      <section class="control-section">
        <div class="section-title-row">
          <span class="section-title">Essential</span>
          <span class="section-subtitle">Recommended controls</span>
        </div>

        <div v-if="basicControls.length" class="control-stack">
          <DynamicControl
            v-for="control in basicControls"
            :key="control.key"
            :control="control"
            :model-value="values[control.key]"
            @update:model-value="updateControl(control.key, $event)"
          />
        </div>

        <div v-else class="empty-panel small">
          This workflow has no basic controls.
        </div>
      </section>

      <AdvancedControls
        v-if="advancedControls.length"
        :controls="advancedControls"
        :values="values"
        @update="updateControl"
      />

      <footer class="run-panel">
        <div class="run-summary">
          <span v-if="processing" class="run-state processing">
            Processing
          </span>

          <span v-else-if="disabled" class="run-state muted">
            Waiting for media
          </span>

          <span v-else class="run-state ready">
            Ready
          </span>

          <span class="run-progress-text">
            {{ normalizedProgress }}%
          </span>
        </div>

        <div v-if="processing" class="compact-progress-track">
          <div
            class="compact-progress-fill"
            :style="{ width: `${normalizedProgress}%` }"
          />
        </div>

        <button
          class="run-button"
          :disabled="disabled"
          type="button"
          @click="handleRun"
        >
          <span v-if="processing">Processing...</span>
          <span v-else-if="disabled">Select Media First</span>
          <span v-else>Run Workflow</span>
        </button>
      </footer>
    </template>
  </aside>
</template>

<script setup>
import { computed } from 'vue'
import DynamicControl from './DynamicControl.vue'
import AdvancedControls from './AdvancedControls.vue'

const props = defineProps({
  workflow: {
    type: Object,
    default: null
  },
  values: {
    type: Object,
    default: () => ({})
  },
  processing: {
    type: Boolean,
    default: false
  },
  progress: {
    type: Number,
    default: 0
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['update-control', 'run'])

const controls = computed(() => props.workflow?.controls || [])

const basicControls = computed(() =>
  controls.value.filter((control) => !control.advanced)
)

const advancedControls = computed(() =>
  controls.value.filter((control) => control.advanced)
)

const controlsCount = computed(() => {
  return basicControls.value.length + advancedControls.value.length
})

const normalizedProgress = computed(() => {
  const value = Number(props.progress || 0)

  if (Number.isNaN(value)) return 0

  return Math.round(Math.min(100, Math.max(0, value)))
})

function updateControl(key, value) {
  emit('update-control', key, value)
}

function handleRun() {
  if (props.disabled) return
  emit('run')
}
</script>

<style scoped>
.control-inspector {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.inspector-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}

.inspector-header h2 {
  margin: 4px 0 0;
  font-size: 19px;
  line-height: 1.1;
}

.control-count {
  min-width: 28px;
  height: 28px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.1);
  border: 1px solid rgba(39, 224, 209, 0.24);
  font-size: 12px;
  font-weight: 900;
}

.inspector-empty {
  margin-top: 18px;
  display: grid;
  gap: 8px;
  justify-items: center;
}

.empty-icon {
  width: 42px;
  height: 42px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.22);
}

.workflow-mini-summary {
  margin-top: 16px;
  padding: 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border);
}

.summary-label {
  display: block;
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
}

.workflow-mini-summary strong {
  display: block;
  line-height: 1.2;
}

.workflow-mini-summary p {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

.control-section {
  margin-top: 18px;
}

.section-title-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
}

.section-title {
  color: var(--text);
  font-size: 13px;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.section-subtitle {
  color: var(--muted-2);
  font-size: 11px;
}

.run-panel {
  position: sticky;
  bottom: 0;
  margin-top: auto;
  padding-top: 18px;
  padding-bottom: 2px;
  background: linear-gradient(
    180deg,
    rgba(16, 24, 39, 0),
    rgba(16, 24, 39, 0.96) 30%
  );
}

.run-summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.run-state {
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.run-state.ready {
  color: var(--green);
}

.run-state.processing {
  color: var(--cyan);
}

.run-state.muted {
  color: var(--muted);
}

.run-progress-text {
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
}

.compact-progress-track {
  height: 5px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.08);
  margin-bottom: 12px;
}

.compact-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--cyan), var(--purple));
  transition: width 0.25s ease;
}
</style>