<template>
  <section class="workflow-list">
    <div v-if="loading" class="workflow-empty">
      <div class="empty-orb">⌁</div>
      <strong>Loading workflows...</strong>
      <span>Reading VISUALine pipeline catalog.</span>
    </div>

    <div v-else-if="!filteredWorkflows.length" class="workflow-empty">
      <div class="empty-orb">?</div>
      <strong>No workflows found</strong>
      <span>Try another search or check backend pipeline configs.</span>
    </div>

    <div v-else class="workflow-grid">
      <WorkflowCard
        v-for="workflow in filteredWorkflows"
        :key="workflow.id"
        :workflow="workflow"
        :selected="selectedWorkflowId === workflow.id"
        @select="$emit('select', workflow)"
      />
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import WorkflowCard from './WorkflowCard.vue'

const props = defineProps({
  workflows: {
    type: Array,
    default: () => []
  },
  selectedWorkflowId: {
    type: String,
    default: null
  },
  activeCategory: {
    type: String,
    default: null
  },
  searchQuery: {
    type: String,
    default: ''
  },
  loading: {
    type: Boolean,
    default: false
  }
})

defineEmits(['select'])

const filteredWorkflows = computed(() => {
  const query = props.searchQuery.trim().toLowerCase()
  const activeCat = (props.activeCategory || 'All').toLowerCase()

  return props.workflows.filter((workflow) => {
    const matchesCategory =
      activeCat === 'all' ||
      (workflow.category || '').toLowerCase() === activeCat

    if (!matchesCategory) return false

    if (!query) return true

    const haystack = [
      workflow.id,
      workflow.display_name,
      workflow.pipeline_name,
      workflow.category,
      workflow.description,
      ...(workflow.tags || [])
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()

    return haystack.includes(query)
  })
})
</script>

<style scoped>
.workflow-list {
  min-height: 0;
  overflow: hidden;
}

.workflow-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
  gap: 12px;
}

.workflow-empty {
  min-height: 240px;
  display: grid;
  place-items: center;
  align-content: center;
  gap: 8px;
  text-align: center;
  color: var(--muted);
  border: 1px dashed var(--border);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.025);
}

.workflow-empty strong {
  color: var(--text);
}

.workflow-empty span {
  font-size: 13px;
}

.empty-orb {
  width: 54px;
  height: 54px;
  border-radius: 18px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border: 1px solid rgba(39, 224, 209, 0.22);
  font-size: 22px;
}
</style>