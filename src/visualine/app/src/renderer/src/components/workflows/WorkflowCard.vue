<template>
  <button
    class="workflow-card"
    :class="{ selected }"
    type="button"
    @click="$emit('select', workflow)"
  >
    <div class="workflow-card-top">
      <div class="workflow-icon">
        {{ icon }}
      </div>

      <div class="workflow-main">
        <div class="workflow-title-row">
          <h3>{{ workflow.display_name }}</h3>

          <span class="speed-pill" :class="workflow.speed">
            {{ workflow.speed }}
          </span>
        </div>

        <p class="workflow-category">
          {{ workflow.category || 'General' }}
        </p>
      </div>
    </div>

    <p class="workflow-description">
      {{ workflow.description || 'AI-powered VISUALine workflow.' }}
    </p>

    <div class="workflow-badges">
      <span
        v-for="type in visibleInputTypes"
        :key="type"
        class="mini-badge"
      >
        {{ type }}
      </span>

      <span v-if="workflow.supports_prompt" class="mini-badge prompt">
        Prompt
      </span>

      <span v-if="workflow.is_hq" class="mini-badge hq">
        HQ
      </span>
    </div>
  </button>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  workflow: {
    type: Object,
    required: true
  },
  selected: {
    type: Boolean,
    default: false
  }
})

defineEmits(['select'])

const visibleInputTypes = computed(() => {
  return (props.workflow.input_types || []).slice(0, 3)
})

const icon = computed(() => {
  const category = String(props.workflow.category || '').toLowerCase()
  const id = String(props.workflow.id || '').toLowerCase()

  if (category.includes('privacy') || id.includes('redaction')) return '◉'
  if (category.includes('restoration') || id.includes('enhancement')) return '✦'
  if (category.includes('social') || id.includes('reframe')) return '▯'
  if (category.includes('motion') || id.includes('slow')) return '≈'
  if (category.includes('editing') || id.includes('blur')) return '◌'

  return '◇'
})
</script>

<style scoped>
.workflow-card {
  width: 100%;
  padding: 15px;
  border-radius: 16px;
  display: grid;
  gap: 12px;
  text-align: left;
  color: var(--text);
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.018)),
    rgba(16, 24, 39, 0.82);
  border: 1px solid var(--border);
  cursor: pointer;
  transition:
    transform 0.18s ease,
    border-color 0.18s ease,
    background 0.18s ease,
    box-shadow 0.18s ease;
}

.workflow-card:hover {
  transform: translateY(-1px);
  border-color: rgba(39, 224, 209, 0.34);
  background:
    linear-gradient(135deg, rgba(39, 224, 209, 0.055), rgba(124, 58, 237, 0.045)),
    rgba(16, 24, 39, 0.9);
}

.workflow-card.selected {
  border-color: rgba(39, 224, 209, 0.62);
  box-shadow:
    0 0 0 1px rgba(39, 224, 209, 0.18),
    0 20px 48px rgba(0, 0, 0, 0.28);
}

.workflow-card-top {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.workflow-icon {
  flex: 0 0 auto;
  width: 42px;
  height: 42px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.2);
  font-size: 20px;
}

.workflow-main {
  min-width: 0;
  flex: 1;
}

.workflow-title-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  gap: 10px;
}

.workflow-card h3 {
  margin: 0;
  color: var(--text);
  font-size: 15px;
  line-height: 1.2;
}

.workflow-category {
  margin: 4px 0 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}

.workflow-description {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.5;
}

.workflow-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.speed-pill {
  flex: 0 0 auto;
  padding: 5px 8px;
  border-radius: 999px;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.055);
  border: 1px solid var(--border);
  font-size: 10px;
  font-weight: 950;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.speed-pill.fast {
  color: #bbf7d0;
  border-color: rgba(34, 197, 94, 0.2);
  background: rgba(34, 197, 94, 0.075);
}

.speed-pill.medium {
  color: #fde68a;
  border-color: rgba(245, 158, 11, 0.2);
  background: rgba(245, 158, 11, 0.075);
}

.speed-pill.heavy {
  color: #f0abfc;
  border-color: rgba(168, 85, 247, 0.24);
  background: rgba(168, 85, 247, 0.08);
}

.mini-badge {
  min-height: 22px;
  display: inline-flex;
  align-items: center;
  padding: 0 7px;
  border-radius: 999px;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border);
  font-size: 10px;
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.mini-badge.prompt,
.mini-badge.hq {
  color: var(--cyan);
  border-color: rgba(39, 224, 209, 0.22);
  background: rgba(39, 224, 209, 0.07);
}
</style>