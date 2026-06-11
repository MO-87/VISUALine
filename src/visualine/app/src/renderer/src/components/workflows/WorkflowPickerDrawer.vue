<template>
  <Teleport to="body">
    <Transition name="drawer-fade">
      <div
        v-if="open"
        class="drawer-backdrop"
        @click="$emit('close')"
      >
        <Transition name="drawer-slide">
          <aside
            v-if="open"
            class="workflow-drawer"
            @click.stop
          >
            <header class="drawer-header">
              <div>
                <span class="panel-kicker">Pipeline Library</span>
                <h2>Choose Workflow</h2>
                <p>Select one of VISUALine’s local AI processing workflows.</p>
              </div>

              <button
                class="drawer-close"
                type="button"
                @click="$emit('close')"
              >
                ×
              </button>
            </header>

            <section class="drawer-tools">
              <div class="search-field">
                <span>⌕</span>
                <input
                  v-model="searchQuery"
                  type="text"
                  placeholder="Search workflows..."
                />
              </div>

              <div class="category-chips">
                <button
                  v-for="category in categories"
                  :key="category"
                  class="category-chip"
                  :class="{ active: activeCategory === category }"
                  type="button"
                  @click="activeCategory = category"
                >
                  {{ category }}
                </button>
              </div>
            </section>

            <section class="drawer-body">
              <WorkflowList
                :workflows="workflows"
                :selected-workflow-id="selectedWorkflowId"
                :active-category="activeCategory"
                :search-query="searchQuery"
                :loading="loading"
                @select="handleSelect"
              />
            </section>
          </aside>
        </Transition>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import WorkflowList from './WorkflowList.vue'

const props = defineProps({
  open: {
    type: Boolean,
    default: false
  },
  workflows: {
    type: Array,
    default: () => []
  },
  selectedWorkflowId: {
    type: String,
    default: null
  },
  loading: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['close', 'select'])

const searchQuery = ref('')
const activeCategory = ref('All')

const categories = computed(() => {
  const values = props.workflows
    .map((workflow) => workflow.category)
    .filter(Boolean)

  return ['All', ...new Set(values)]
})

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) {
      searchQuery.value = ''
      activeCategory.value = 'All'
    }
  }
)

function handleSelect(workflow) {
  emit('select', workflow)
  emit('close')
}
</script>

<style scoped>
.drawer-backdrop {
  position: fixed;
  inset: 0;
  z-index: 200;
  display: flex;
  justify-content: flex-end;
  background: rgba(3, 7, 18, 0.62);
  backdrop-filter: blur(10px);
}

.workflow-drawer {
  width: min(760px, 92vw);
  height: 100vh;
  min-height: 0;
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
  background:
    radial-gradient(circle at top left, rgba(39, 224, 209, 0.08), transparent 36%),
    rgba(9, 15, 28, 0.98);
  border-left: 1px solid var(--border);
  box-shadow: -32px 0 90px rgba(0, 0, 0, 0.45);
}

.drawer-header {
  padding: 24px;
  display: flex;
  justify-content: space-between;
  gap: 18px;
  border-bottom: 1px solid var(--border);
}

.drawer-header h2 {
  margin: 6px 0 0;
  font-size: 26px;
  line-height: 1.1;
}

.drawer-header p {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}

.drawer-close {
  flex: 0 0 auto;
  width: 38px;
  height: 38px;
  border-radius: 13px;
  display: grid;
  place-items: center;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border);
  cursor: pointer;
  font-size: 24px;
  line-height: 1;
}

.drawer-close:hover {
  color: var(--text);
  border-color: rgba(39, 224, 209, 0.32);
  background: rgba(39, 224, 209, 0.06);
}

.drawer-tools {
  padding: 16px 24px;
  display: grid;
  gap: 12px;
  border-bottom: 1px solid var(--border);
}

.search-field {
  height: 42px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 13px;
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border);
}

.search-field span {
  color: var(--muted);
}

.search-field input {
  flex: 1;
  min-width: 0;
  border: 0;
  outline: 0;
  color: var(--text);
  background: transparent;
}

.search-field input::placeholder {
  color: var(--muted-2);
}

.category-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.category-chip {
  min-height: 30px;
  padding: 0 10px;
  border-radius: 999px;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border);
  cursor: pointer;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.category-chip:hover,
.category-chip.active {
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.08);
  border-color: rgba(39, 224, 209, 0.28);
}

.drawer-body {
  min-height: 0;
  overflow-y: auto;
  padding: 18px 24px 24px;
}

.drawer-body::-webkit-scrollbar {
  width: 8px;
}

.drawer-body::-webkit-scrollbar-thumb {
  background: rgba(142, 155, 179, 0.28);
  border-radius: 999px;
}

.drawer-fade-enter-active,
.drawer-fade-leave-active {
  transition: opacity 0.18s ease;
}

.drawer-fade-enter-from,
.drawer-fade-leave-to {
  opacity: 0;
}

.drawer-slide-enter-active,
.drawer-slide-leave-active {
  transition: transform 0.22s ease;
}

.drawer-slide-enter-from,
.drawer-slide-leave-to {
  transform: translateX(24px);
}
</style>