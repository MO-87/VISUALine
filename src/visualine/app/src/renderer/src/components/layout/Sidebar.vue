<template>
  <aside class="sidebar studio-sidebar">
    <div class="brand">
      <div class="brand-orb">V</div>
      <div class="brand-copy">
        <div class="brand-title">STUDIO</div>
        <div class="brand-subtitle">Local AI Engine</div>
      </div>
    </div>

    <nav class="nav-section" aria-label="Main navigation">
      <button
        v-for="item in navItems"
        :key="item.key"
        class="nav-item"
        :class="{ active: activeSection === item.key }"
        type="button"
        @click="handleNavClick(item.key)"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span class="nav-label">{{ item.label }}</span>
      </button>
    </nav>

    <div class="sidebar-spacer" />

    <section class="engine-card">
      <span class="engine-dot" />
      <div>
        <strong>Local Mode</strong>
        <small>Private GPU processing</small>
      </div>
    </section>
  </aside>
</template>

<script setup>
defineProps({
  activeSection: {
    type: String,
    default: 'workflows'
  },

  // Kept for compatibility with the previous AppShell API.
  categories: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits([
  'select-section',
  'select-category',
  'open-workflow-picker'
])

const navItems = [
  { key: 'home', label: 'Home', icon: '⌂' },
  { key: 'workflows', label: 'Workflows', icon: '⌘' },
  { key: 'jobs', label: 'Jobs', icon: '☰' },
  { key: 'outputs', label: 'Outputs', icon: '▣' },
  { key: 'settings', label: 'Settings', icon: '⚙' }
]

function handleNavClick(section) {
  emit('select-section', section)

  if (section === 'workflows') {
    emit('open-workflow-picker')
  }
}
</script>

<style scoped>
.studio-sidebar {
  height: 100vh;
  min-height: 0;
  overflow: hidden;
  padding: 20px 14px;
  background:
    radial-gradient(circle at top left, rgba(39, 224, 209, 0.08), transparent 35%),
    rgba(7, 12, 24, 0.98);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 6px 24px;
}

.brand-orb {
  width: 42px;
  height: 42px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.1);
  border: 1px solid rgba(39, 224, 209, 0.28);
  font-weight: 950;
  box-shadow: 0 0 24px rgba(39, 224, 209, 0.12);
}

.brand-copy {
  min-width: 0;
}

.brand-title {
  color: var(--cyan);
  font-size: 18px;
  font-weight: 950;
  letter-spacing: 0.08em;
}

.brand-subtitle {
  color: var(--muted);
  font-size: 12px;
  margin-top: 2px;
  white-space: nowrap;
}

.nav-section {
  display: grid;
  gap: 7px;
}

.nav-item {
  width: 100%;
  min-height: 42px;
  border-radius: 12px;
  background: transparent;
  color: var(--muted);
  padding: 0 12px;
  display: flex;
  gap: 12px;
  align-items: center;
  cursor: pointer;
  text-align: left;
  border: 1px solid transparent;
}

.nav-item:hover {
  color: var(--text);
  background: rgba(255, 255, 255, 0.035);
  border-color: rgba(132, 146, 187, 0.12);
}

.nav-item.active {
  color: var(--cyan);
  background:
    linear-gradient(90deg, rgba(39, 224, 209, 0.17), rgba(39, 224, 209, 0.045));
  border-color: rgba(39, 224, 209, 0.2);
  box-shadow: inset -2px 0 0 var(--cyan);
}

.nav-icon {
  width: 20px;
  flex: 0 0 20px;
  display: grid;
  place-items: center;
  opacity: 0.95;
}

.nav-label {
  font-size: 14px;
  font-weight: 800;
}

.sidebar-spacer {
  flex: 1;
}

.engine-card {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 18px;
  padding: 12px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid var(--border);
}

.engine-dot {
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: var(--green);
  box-shadow: 0 0 12px rgba(34, 197, 94, 0.8);
}

.engine-card strong {
  display: block;
  color: var(--text);
  font-size: 12px;
  line-height: 1.2;
}

.engine-card small {
  display: block;
  color: var(--muted);
  font-size: 11px;
  margin-top: 2px;
}
</style>