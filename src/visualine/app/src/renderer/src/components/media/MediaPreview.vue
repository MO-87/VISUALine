<template>
  <section class="preview-panel hero-preview">
    <header class="preview-toolbar">
      <div class="preview-title">
        <span class="panel-kicker">Preview</span>
        <strong>{{ outputUrl ? 'Before / After' : 'Media Canvas' }}</strong>
      </div>

      <div class="preview-toolbar-right">
        <button
          class="preview-mode-button"
          type="button"
          :class="{ active: viewMode === 'split' }"
          @click="viewMode = 'split'"
        >
          Split
        </button>

        <button
          class="preview-mode-button"
          type="button"
          :class="{ active: viewMode === 'processed' }"
          @click="viewMode = 'processed'"
        >
          Result
        </button>
      </div>
    </header>

    <div
      class="preview-body"
      :class="{
        'is-split': viewMode === 'split',
        'is-result-only': viewMode === 'processed'
      }"
    >
      <div
        v-if="viewMode === 'split'"
        class="preview-box"
      >
        <div class="preview-label">Original</div>

        <video
          v-if="originalUrl && isVideo"
          :key="'video-' + originalUrl"
          ref="originalVideoRef"
          class="media-preview"
          :src="originalUrl"
          controls
          preload="auto"
          playsinline
          muted
          @loadedmetadata="warmVideoPreview"
          @loadeddata="warmVideoPreview"
          @error="originalLoadFailed = true"
        />

        <img
          v-else-if="originalUrl"
          :key="'image-' + originalUrl"
          class="media-preview"
          :src="originalUrl"
          alt="Original preview"
          @error="originalLoadFailed = true"
        />

        <div v-if="originalLoadFailed" class="preview-warning">
          Could not preview the original file directly. Processing can still run.
        </div>

        <EmptyPreview
          v-if="!originalUrl"
          title="No media selected"
          description="Use Browse Media in the top bar to select a video, GIF, or image."
        />
      </div>

      <div class="preview-box result-box">
        <div class="preview-label">
          {{ outputUrl ? 'Processed' : 'Output' }}
        </div>

        <video
          v-if="outputUrl && isVideo"
          :key="'video-' + outputUrl"
          class="media-preview"
          :src="outputUrl"
          controls
          preload="auto"
          playsinline
        />

        <img
          v-else-if="outputUrl"
          :key="'image-' + outputUrl"
          class="media-preview"
          :src="outputUrl"
          alt="Processed preview"
        />

        <EmptyPreview
          v-else-if="originalUrl && !processing"
          title="Ready to process"
          description="Tune the parameters on the right, then run the workflow."
        />

        <EmptyPreview
          v-else-if="!processing"
          title="Output will appear here"
          description="Choose a workflow and source media from the top bar."
        />

        <ProcessingOverlay
          v-if="processing"
          :progress="progress"
          :message="message"
          :current-frame="currentFrame"
          :total-frames="totalFrames"
        />
      </div>
    </div>
  </section>
</template>

<script setup>
import { h, ref, watch } from 'vue'
import ProcessingOverlay from './ProcessingOverlay.vue'

const props = defineProps({
  originalUrl: {
    type: String,
    default: ''
  },
  outputUrl: {
    type: String,
    default: ''
  },
  isVideo: {
    type: Boolean,
    default: true
  },
  processing: {
    type: Boolean,
    default: false
  },
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

defineEmits(['request-media', 'request-workflow'])

const viewMode = ref('split')
const originalVideoRef = ref(null)
const originalLoadFailed = ref(false)

watch(
  () => props.originalUrl,
  () => {
    originalLoadFailed.value = false
  }
)

function warmVideoPreview(event) {
  const video = event?.target || originalVideoRef.value
  if (!video) return

  try {
    if (Number.isFinite(video.duration) && video.duration > 0 && video.currentTime < 0.05) {
      video.currentTime = Math.min(0.15, video.duration / 20)
    }
  } catch {
    // Some codecs do not allow seeking before full data load.
  }
}

const EmptyPreview = {
  props: {
    title: {
      type: String,
      required: true
    },
    description: {
      type: String,
      default: ''
    }
  },
  setup(componentProps) {
    return () =>
      h('div', { class: 'preview-empty-state' }, [
        h('div', { class: 'empty-preview-icon' }, '▣'),
        h('strong', componentProps.title),
        h('span', componentProps.description)
      ])
  }
}
</script>

<style scoped>
.hero-preview {
  height: 100%;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  border-radius: 18px;
  background:
    radial-gradient(circle at 30% 20%, rgba(39, 224, 209, 0.06), transparent 35%),
    rgba(16, 24, 39, 0.92);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}

.preview-toolbar {
  height: 58px;
  flex: 0 0 58px;
  padding: 0 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
}

.preview-title {
  display: grid;
  gap: 3px;
}

.preview-title strong {
  font-size: 15px;
  line-height: 1.1;
}

.preview-toolbar-right {
  display: inline-flex;
  padding: 4px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border);
}

.preview-mode-button {
  min-width: 64px;
  height: 30px;
  padding: 0 10px;
  border-radius: 9px;
  color: var(--muted);
  background: transparent;
  cursor: pointer;
  font-size: 12px;
  font-weight: 900;
}

.preview-mode-button.active {
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.1);
}

.preview-body {
  flex: 1;
  min-height: 0;
  display: grid;
  overflow: hidden;
}

.preview-body.is-split {
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
}

.preview-body.is-result-only {
  grid-template-columns: minmax(0, 1fr);
}

.preview-box {
  position: relative;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  display: grid;
  place-items: center;
  background:
    linear-gradient(rgba(255, 255, 255, 0.018), rgba(255, 255, 255, 0.018)),
    rgba(8, 13, 24, 0.38);
}

.preview-body.is-split .preview-box:first-child {
  border-right: 1px solid var(--border);
}

.result-box {
  background:
    radial-gradient(circle at center, rgba(124, 58, 237, 0.07), transparent 45%),
    rgba(8, 13, 24, 0.32);
}

.preview-label {
  position: absolute;
  z-index: 5;
  top: 14px;
  left: 14px;
  min-height: 28px;
  display: inline-flex;
  align-items: center;
  padding: 0 10px;
  border-radius: 999px;
  color: var(--muted);
  background: rgba(0, 0, 0, 0.34);
  border: 1px solid rgba(132, 146, 187, 0.18);
  backdrop-filter: blur(10px);
  font-size: 11px;
  font-weight: 950;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.media-preview {
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #050914;
}

.preview-empty-state {
  width: min(340px, 82%);
  text-align: center;
  display: grid;
  justify-items: center;
  gap: 10px;
  color: var(--muted);
}

.empty-preview-icon {
  width: 66px;
  height: 66px;
  border-radius: 22px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.22);
  font-size: 24px;
}

.preview-empty-state strong {
  color: var(--text);
  font-size: 17px;
  line-height: 1.2;
}

.preview-empty-state span {
  max-width: 280px;
  font-size: 13px;
  line-height: 1.45;
}

.preview-warning {
  position: absolute;
  left: 16px;
  bottom: 16px;
  right: 16px;
  z-index: 8;
  padding: 10px 12px;
  border-radius: 12px;
  color: #fde68a;
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.18);
  font-size: 12px;
  line-height: 1.35;
}
</style>