<template>
  <section class="workflow-header selected-workflow-card">
    <div class="workflow-header-top">
      <span class="panel-kicker">Workflow</span>

      <button
        class="change-mini-button"
        type="button"
        @click="$emit('open-workflow-picker')"
      >
        Change
      </button>
    </div>

    <div v-if="workflow" class="selected-workflow-body">
      <div class="workflow-orb">
        {{ icon }}
      </div>

      <div class="selected-workflow-copy">
        <h1>{{ workflow.display_name }}</h1>

        <p v-if="workflow.description">
          {{ workflow.description }}
        </p>
      </div>

      <div class="workflow-header-badges">
        <span class="mini-badge">{{ workflow.category }}</span>
        <span class="mini-badge" :class="workflow.speed">{{ workflow.speed }}</span>
        <span v-if="workflow.supports_prompt" class="mini-badge prompt">Prompt</span>
        <span v-if="workflow.is_hq" class="mini-badge hq">HQ</span>
      </div>
    </div>

    <div v-else class="no-workflow-state">
      <div class="workflow-orb">+</div>
      <strong>Select a Workflow</strong>
      <span>Choose a VISUALine pipeline to begin.</span>

      <button
        class="primary-soft-button"
        type="button"
        @click="$emit('open-workflow-picker')"
      >
        Choose Workflow
      </button>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  workflow: {
    type: Object,
    default: null
  }
})

defineEmits(['open-workflow-picker'])

const icon = computed(() => {
  const category = String(props.workflow?.category || '').toLowerCase()
  const id = String(props.workflow?.id || '').toLowerCase()

  if (category.includes('privacy') || id.includes('redaction')) return '◉'
  if (category.includes('restoration') || id.includes('enhancement')) return '✦'
  if (category.includes('social') || id.includes('reframe')) return '▯'
  if (category.includes('motion') || id.includes('slow')) return '≈'
  if (category.includes('editing') || id.includes('blur')) return '◌'

  return '◇'
})
</script>

<style scoped>
.selected-workflow-card {
  padding: 16px;
  border-radius: 18px;
  background:
    radial-gradient(circle at top right, rgba(39, 224, 209, 0.055), transparent 45%),
    rgba(16, 24, 39, 0.92);
  border: 1px solid var(--border);
}

.workflow-header-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.change-mini-button {
  min-height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.22);
  font-size: 11px;
  font-weight: 900;
  cursor: pointer;
}

.change-mini-button:hover {
  background: rgba(39, 224, 209, 0.13);
  border-color: rgba(39, 224, 209, 0.38);
}

.selected-workflow-body {
  display: grid;
  gap: 12px;
  margin-top: 12px;
}

.workflow-orb {
  width: 48px;
  height: 48px;
  border-radius: 17px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.22);
  font-size: 22px;
}

.selected-workflow-copy h1 {
  margin: 0;
  color: var(--text);
  font-size: 21px;
  line-height: 1.12;
}

.selected-workflow-copy p {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.48;
}

.workflow-header-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.no-workflow-state {
  margin-top: 14px;
  display: grid;
  justify-items: start;
  gap: 8px;
  color: var(--muted);
}

.no-workflow-state strong {
  color: var(--text);
}

.primary-soft-button {
  min-height: 38px;
  margin-top: 4px;
  padding: 0 12px;
  border-radius: 11px;
  color: white;
  background: linear-gradient(90deg, rgba(39, 224, 209, 0.82), rgba(124, 58, 237, 0.86));
  font-size: 12px;
  font-weight: 900;
  cursor: pointer;
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

.mini-badge.fast {
  color: #bbf7d0;
}

.mini-badge.medium {
  color: #fde68a;
}

.mini-badge.heavy {
  color: #f0abfc;
}

.mini-badge.prompt,
.mini-badge.hq {
  color: var(--cyan);
  border-color: rgba(39, 224, 209, 0.22);
  background: rgba(39, 224, 209, 0.07);
}
</style>