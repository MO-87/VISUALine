<template>
  <section class="advanced-controls">
    <button
      class="advanced-toggle"
      type="button"
      @click="open = !open"
    >
      <span class="advanced-left">
        <span class="advanced-icon">⚙</span>
        <span>
          <strong>Advanced Engine Options</strong>
          <small>{{ controls.length }} technical controls</small>
        </span>
      </span>

      <span class="advanced-chevron">
        {{ open ? '⌃' : '⌄' }}
      </span>
    </button>

    <Transition name="advanced">
      <div v-if="open" class="control-stack advanced-stack">
        <DynamicControl
          v-for="control in controls"
          :key="control.key"
          :control="control"
          :model-value="values[control.key]"
          @update:model-value="$emit('update', control.key, $event)"
        />
      </div>
    </Transition>
  </section>
</template>

<script setup>
import { ref } from 'vue'
import DynamicControl from './DynamicControl.vue'

defineProps({
  controls: {
    type: Array,
    default: () => []
  },
  values: {
    type: Object,
    default: () => ({})
  }
})

defineEmits(['update'])

const open = ref(false)
</script>

<style scoped>
.advanced-controls {
  margin-top: 18px;
}

.advanced-toggle {
  width: 100%;
  background: rgba(255, 255, 255, 0.04);
  color: var(--text);
  border: 1px solid var(--border);
  padding: 12px;
  border-radius: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  cursor: pointer;
  text-align: left;
}

.advanced-toggle:hover {
  border-color: rgba(39, 224, 209, 0.35);
  background: rgba(39, 224, 209, 0.045);
}

.advanced-left {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.advanced-icon {
  width: 32px;
  height: 32px;
  border-radius: 10px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.18);
}

.advanced-left strong {
  display: block;
  font-size: 13px;
  line-height: 1.2;
}

.advanced-left small {
  display: block;
  margin-top: 2px;
  color: var(--muted);
  font-size: 11px;
}

.advanced-chevron {
  color: var(--muted);
  font-size: 16px;
}

.advanced-stack {
  margin-top: 12px;
}

.advanced-enter-active,
.advanced-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.advanced-enter-from,
.advanced-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>