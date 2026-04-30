<template>
  <section v-if="outputUrl" class="output-actions-card">
    <div class="output-info">
      <span class="panel-kicker">Result Ready</span>
      <h3>Processed Output</h3>
      <p>Your VISUALine result is ready to preview or download.</p>
    </div>

    <div class="output-actions">
      <a
        class="output-action primary"
        :href="outputUrl"
        target="_blank"
        rel="noreferrer"
      >
        <span class="action-icon">↗</span>
        <span>Open Output</span>
      </a>

      <a
        class="output-action"
        :href="outputUrl"
        download
      >
        <span class="action-icon">↓</span>
        <span>Download</span>
      </a>
    </div>
  </section>
</template>

<script setup>
defineProps({
  outputUrl: {
    type: String,
    default: ''
  }
})
</script>

<style scoped>
.output-actions-card {
  display: grid;
  gap: 14px;
  padding: 16px;
  border-radius: 16px;
  background: rgba(16, 24, 39, 0.92);
  border: 1px solid var(--border);
}

.output-info h3 {
  margin: 4px 0 0;
  font-size: 16px;
  line-height: 1.2;
}

.output-info p {
  margin: 8px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

.output-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.output-action {
  min-height: 42px;
  border-radius: 12px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--text);
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid var(--border);
  text-decoration: none;
  font-size: 13px;
  font-weight: 800;
  transition:
    border-color 0.18s ease,
    background 0.18s ease,
    transform 0.18s ease;
}

.output-action:hover {
  transform: translateY(-1px);
  border-color: rgba(39, 224, 209, 0.38);
  background: rgba(39, 224, 209, 0.055);
}

.output-action.primary {
  color: #ffffff;
  background: linear-gradient(90deg, rgba(39, 224, 209, 0.78), rgba(124, 58, 237, 0.82));
  border-color: rgba(39, 224, 209, 0.34);
}

.action-icon {
  width: 20px;
  height: 20px;
  border-radius: 7px;
  display: grid;
  place-items: center;
  background: rgba(255, 255, 255, 0.12);
}
</style>