<template>
  <span class="status-badge" :class="variant">
    <span class="status-dot" />
    <span class="status-label">{{ label }}</span>
  </span>
</template>

<script setup>
defineProps({
  label: {
    type: String,
    required: true
  },
  variant: {
    type: String,
    default: 'neutral'
  }
})
</script>

<style scoped>
.status-badge {
  min-height: 30px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border);
  padding: 0 11px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  white-space: nowrap;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--muted);
}

.status-label {
  line-height: 1;
}

.status-badge.success {
  color: #bbf7d0;
  border-color: rgba(34, 197, 94, 0.2);
  background: rgba(34, 197, 94, 0.08);
}

.status-badge.success .status-dot {
  background: var(--green);
  box-shadow: 0 0 12px rgba(34, 197, 94, 0.8);
}

.status-badge.danger {
  color: #fecaca;
  border-color: rgba(239, 68, 68, 0.24);
  background: rgba(239, 68, 68, 0.08);
}

.status-badge.danger .status-dot {
  background: var(--red);
  box-shadow: 0 0 12px rgba(239, 68, 68, 0.8);
}

.status-badge.info {
  color: #bae6fd;
  border-color: rgba(39, 224, 209, 0.2);
  background: rgba(39, 224, 209, 0.07);
}

.status-badge.info .status-dot {
  background: var(--cyan);
  box-shadow: 0 0 12px rgba(39, 224, 209, 0.75);
}

.status-badge.warning {
  color: #fde68a;
  border-color: rgba(245, 158, 11, 0.22);
  background: rgba(245, 158, 11, 0.08);
}

.status-badge.warning .status-dot {
  background: var(--yellow);
  box-shadow: 0 0 12px rgba(245, 158, 11, 0.7);
}

@media (max-width: 1080px) {
  .status-badge {
    padding: 0 9px;
  }

  .status-label {
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
}
</style>