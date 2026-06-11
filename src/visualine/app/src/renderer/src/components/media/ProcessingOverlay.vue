<template>
  <div class="processing-overlay">
    <div class="processing-card">
      <div class="processing-orbit">
        <div class="processing-icon">✦</div>
      </div>

      <span class="processing-kicker">VISUALine Engine</span>

      <h3>Processing Media</h3>

      <p>
        {{ message || 'Running the selected AI workflow locally...' }}
      </p>

      <div class="progress-meta">
        <span v-if="hasFrameInfo">
          Frame {{ currentFrame }} / {{ totalFrames }}
        </span>

        <span v-else>
          Preparing pipeline
        </span>

        <strong>{{ normalizedProgress }}%</strong>
      </div>

      <div class="progress-track">
        <div
          class="progress-fill"
          :style="{ width: `${normalizedProgress}%` }"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  progress: {
    type: Number,
    default: 0
  },
  message: {
    type: String,
    default: ''
  },
  currentFrame: {
    type: Number,
    default: null
  },
  totalFrames: {
    type: Number,
    default: null
  }
})

const normalizedProgress = computed(() => {
  const value = Number(props.progress || 0)

  if (Number.isNaN(value)) return 0

  return Math.round(Math.min(100, Math.max(0, value)))
})

const hasFrameInfo = computed(() => {
  return (
    props.currentFrame !== null &&
    props.currentFrame !== undefined &&
    props.totalFrames !== null &&
    props.totalFrames !== undefined
  )
})
</script>

<style scoped>
.processing-overlay {
  position: absolute;
  inset: 0;
  z-index: 20;
  display: grid;
  place-items: center;
  padding: 28px;
  background:
    radial-gradient(circle at center, rgba(39, 224, 209, 0.08), transparent 45%),
    rgba(8, 13, 24, 0.82);
  backdrop-filter: blur(14px);
}

.processing-card {
  width: min(390px, 92%);
  padding: 28px;
  border-radius: 22px;
  text-align: center;
  background: rgba(16, 24, 39, 0.92);
  border: 1px solid rgba(39, 224, 209, 0.2);
  box-shadow:
    0 24px 80px rgba(0, 0, 0, 0.5),
    0 0 60px rgba(39, 224, 209, 0.08);
}

.processing-orbit {
  width: 74px;
  height: 74px;
  margin: 0 auto 16px;
  border-radius: 24px;
  display: grid;
  place-items: center;
  background:
    radial-gradient(circle, rgba(39, 224, 209, 0.22), transparent 65%),
    rgba(124, 58, 237, 0.16);
  border: 1px solid rgba(39, 224, 209, 0.24);
}

.processing-icon {
  width: 48px;
  height: 48px;
  border-radius: 17px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(8, 13, 24, 0.75);
  border: 1px solid rgba(39, 224, 209, 0.34);
  font-size: 24px;
  box-shadow: 0 0 30px rgba(39, 224, 209, 0.2);
}

.processing-kicker {
  color: var(--cyan);
  font-size: 11px;
  font-weight: 950;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.processing-card h3 {
  margin: 8px 0 0;
  font-size: 22px;
  line-height: 1.15;
}

.processing-card p {
  margin: 10px auto 0;
  max-width: 290px;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.5;
}

.progress-meta {
  margin-top: 22px;
  display: flex;
  justify-content: space-between;
  gap: 14px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
}

.progress-meta strong {
  color: var(--cyan);
}

.progress-track {
  height: 8px;
  margin-top: 9px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.09);
}

.progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--cyan), var(--purple));
  transition: width 0.25s ease;
}
</style>