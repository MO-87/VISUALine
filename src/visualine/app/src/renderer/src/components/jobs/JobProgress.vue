<template>
  <section v-if="job" class="job-progress-card" :class="statusClass">
    <header class="job-header">
      <div>
        <span class="panel-kicker">Current Job</span>
        <h3>{{ statusLabel }}</h3>
      </div>

      <span class="job-percent">
        {{ normalizedProgress }}%
      </span>
    </header>

    <div class="job-progress-track">
      <div
        class="job-progress-fill"
        :style="{ width: `${normalizedProgress}%` }"
      />
    </div>

    <div class="job-meta">
      <p class="job-message">
        {{ jobMessage }}
      </p>

      <span
        v-if="hasFrameInfo"
        class="frame-counter"
      >
        Frame {{ job.current_frame }} / {{ job.total_frames }}
      </span>
    </div>

    <pre v-if="isFailed && job.error_message" class="job-error">{{ shortError }}</pre>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  job: {
    type: Object,
    default: null
  }
})

const normalizedProgress = computed(() => {
  const value = Number(props.job?.progress || 0)

  if (Number.isNaN(value)) return 0

  return Math.round(Math.min(100, Math.max(0, value)))
})

const status = computed(() => {
  return String(props.job?.status || 'pending').toLowerCase()
})

const statusLabel = computed(() => {
  return status.value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
})

const statusClass = computed(() => {
  return {
    'is-processing': ['queued', 'pending', 'processing'].includes(status.value),
    'is-completed': status.value === 'completed',
    'is-failed': status.value === 'failed',
    'is-cancelled': status.value === 'cancelled'
  }
})

const isFailed = computed(() => status.value === 'failed')

const jobMessage = computed(() => {
  if (props.job?.message) return props.job.message
  if (props.job?.error_message) return 'The job failed. See details below.'
  return 'Waiting for VISUALine engine updates...'
})

const hasFrameInfo = computed(() => {
  return (
    props.job?.current_frame !== undefined &&
    props.job?.current_frame !== null &&
    props.job?.total_frames !== undefined &&
    props.job?.total_frames !== null
  )
})

const shortError = computed(() => {
  const error = props.job?.error_message || ''

  if (error.length <= 500) return error

  return `${error.slice(0, 500)}...`
})
</script>

<style scoped>
.job-progress-card {
  padding: 16px;
  border-radius: 16px;
  background: rgba(16, 24, 39, 0.92);
  border: 1px solid var(--border);
}

.job-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.job-header h3 {
  margin: 4px 0 0;
  font-size: 17px;
  line-height: 1.2;
}

.job-percent {
  min-width: 48px;
  height: 30px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.1);
  border: 1px solid rgba(39, 224, 209, 0.22);
  font-size: 12px;
  font-weight: 900;
}

.job-progress-track {
  height: 7px;
  margin-top: 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.job-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--cyan), var(--purple));
  transition: width 0.25s ease;
}

.job-meta {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-top: 10px;
}

.job-message {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

.frame-counter {
  flex: 0 0 auto;
  color: var(--muted-2);
  font-size: 11px;
  font-weight: 800;
  white-space: nowrap;
}

.job-error {
  margin: 12px 0 0;
  max-height: 110px;
  overflow: auto;
  padding: 10px;
  border-radius: 10px;
  color: #fecaca;
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.22);
  font-size: 11px;
  line-height: 1.4;
  white-space: pre-wrap;
}

.is-completed .job-percent {
  color: #bbf7d0;
  background: rgba(34, 197, 94, 0.1);
  border-color: rgba(34, 197, 94, 0.22);
}

.is-completed .job-progress-fill {
  background: linear-gradient(90deg, var(--green), var(--cyan));
}

.is-failed .job-percent,
.is-cancelled .job-percent {
  color: #fecaca;
  background: rgba(239, 68, 68, 0.1);
  border-color: rgba(239, 68, 68, 0.24);
}

.is-failed .job-progress-fill,
.is-cancelled .job-progress-fill {
  background: linear-gradient(90deg, var(--red), var(--yellow));
}
</style>