<template>
  <section class="media-panel source-card">
    <header class="source-header">
      <div>
        <span class="panel-kicker">Source</span>
        <h2>Media Input</h2>
      </div>

      <span
        class="media-kind-pill"
        :class="mediaKind"
      >
        {{ mediaKind }}
      </span>
    </header>

    <button
      class="source-drop"
      type="button"
      @click="$emit('select-file')"
    >
      <div class="source-icon">
        {{ mediaPath ? '✓' : '+' }}
      </div>

      <div class="source-copy">
        <strong>{{ mediaPath ? fileName : 'Select media file' }}</strong>
        <span>
          {{ mediaPath ? 'Ready for local processing.' : 'Video, GIF, image, or drag-and-drop later.' }}
        </span>
      </div>
    </button>

    <div class="media-actions">
      <button
        class="secondary-button select-button"
        type="button"
        @click="$emit('select-file')"
      >
        Browse File
      </button>
    </div>

    <label class="path-field">
      <span>Local path</span>

      <input
        class="studio-input"
        placeholder="/home/user/video.mp4"
        :value="mediaPath"
        @input="$emit('update:mediaPath', $event.target.value)"
      />
    </label>

    <div v-if="mediaPath" class="metadata-list compact-metadata">
      <div class="metadata-row">
        <span>File</span>
        <strong :title="fileName">{{ fileName }}</strong>
      </div>

      <div class="metadata-row">
        <span>Type</span>
        <strong>{{ extension || 'unknown' }}</strong>
      </div>

      <div class="metadata-row">
        <span>Pipeline mode</span>
        <strong>{{ mediaKind }}</strong>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  mediaPath: {
    type: String,
    default: ''
  },
  mediaKind: {
    type: String,
    default: 'video'
  }
})

defineEmits(['select-file', 'update:mediaPath'])

const fileName = computed(() => {
  if (!props.mediaPath) return ''
  return props.mediaPath.split(/[\\/]/).pop()
})

const extension = computed(() => {
  const name = fileName.value
  if (!name.includes('.')) return ''
  return name.split('.').pop()?.toLowerCase()
})
</script>

<style scoped>
.source-card {
  display: grid;
  gap: 14px;
  padding: 16px;
  border-radius: 18px;
  background: rgba(16, 24, 39, 0.92);
  border: 1px solid var(--border);
}

.source-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.source-header h2 {
  margin: 4px 0 0;
  font-size: 17px;
  line-height: 1.15;
}

.media-kind-pill {
  min-height: 26px;
  padding: 0 9px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.2);
  font-size: 11px;
  font-weight: 950;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.media-kind-pill.image {
  color: #f0abfc;
  background: rgba(168, 85, 247, 0.08);
  border-color: rgba(168, 85, 247, 0.22);
}

.source-drop {
  width: 100%;
  min-height: 98px;
  padding: 14px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  gap: 13px;
  text-align: left;
  color: var(--text);
  background:
    linear-gradient(135deg, rgba(39, 224, 209, 0.055), rgba(124, 58, 237, 0.05)),
    rgba(255, 255, 255, 0.025);
  border: 1px dashed rgba(132, 146, 187, 0.28);
  cursor: pointer;
}

.source-drop:hover {
  border-color: rgba(39, 224, 209, 0.45);
  background:
    linear-gradient(135deg, rgba(39, 224, 209, 0.08), rgba(124, 58, 237, 0.07)),
    rgba(255, 255, 255, 0.03);
}

.source-icon {
  flex: 0 0 auto;
  width: 46px;
  height: 46px;
  border-radius: 16px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.09);
  border: 1px solid rgba(39, 224, 209, 0.23);
  font-size: 24px;
  font-weight: 900;
}

.source-copy {
  min-width: 0;
  display: grid;
  gap: 5px;
}

.source-copy strong {
  overflow: hidden;
  color: var(--text);
  font-size: 14px;
  line-height: 1.25;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.source-copy span {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.4;
}

.media-actions {
  display: grid;
}

.select-button {
  min-height: 40px;
}

.path-field {
  display: grid;
  gap: 7px;
}

.path-field span {
  color: var(--muted);
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.path-field .studio-input {
  margin-top: 0;
}

.compact-metadata {
  display: grid;
  gap: 8px;
}

.metadata-row {
  display: grid;
  grid-template-columns: 82px minmax(0, 1fr);
  gap: 10px;
  align-items: center;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.045);
  color: var(--muted);
  font-size: 12px;
}

.metadata-row:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

.metadata-row strong {
  min-width: 0;
  overflow: hidden;
  color: var(--text);
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: right;
}
</style>