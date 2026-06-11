<template>
  <div class="app-shell no-sidebar-shell">
    <TopBar
      :system-health="systemHealth"
      :backend-online="backendOnline"
      :active-jobs-count="activeJobsCount"
      :selected-workflow="selectedWorkflow"
      :media-path="mediaPath"
      @refresh-system="$emit('refresh-system')"
      @open-workflow-picker="$emit('open-workflow-picker')"
      @open-optimization="$emit('open-optimization')"
      @select-file="$emit('select-file')"
    />

    <main class="app-main">
      <section class="app-content">
        <slot />
      </section>
    </main>
  </div>
</template>

<script setup>
import TopBar from './TopBar.vue'

defineProps({
  activeSection: {
    type: String,
    default: 'workflows'
  },
  categories: {
    type: Array,
    default: () => []
  },
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

defineEmits([
  'select-section',
  'select-category',
  'refresh-system',
  'open-workflow-picker',
  'open-optimization',
  'select-file'
])
</script>

<style scoped>
.no-sidebar-shell {
  width: 100vw;
  height: 100vh;
  min-height: 0;
  overflow: hidden;
  display: grid;
  grid-template-rows: 72px minmax(0, 1fr);
  background:
    radial-gradient(circle at top left, rgba(124, 58, 237, 0.12), transparent 35%),
    radial-gradient(circle at top right, rgba(39, 224, 209, 0.08), transparent 30%),
    var(--bg);
}

.app-main {
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.app-content {
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  padding: 14px;
}
</style>
