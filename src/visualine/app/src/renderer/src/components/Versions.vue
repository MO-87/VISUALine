<script setup>
import { reactive } from 'vue'

const versions = reactive({
  ...(window.electron?.process?.versions || {})
})
</script>

<template>
  <ul class="versions">
    <li class="electron-version">Electron v{{ versions.electron || 'unknown' }}</li>
    <li class="chrome-version">Chromium v{{ versions.chrome || 'unknown' }}</li>
    <li class="node-version">Node v{{ versions.node || 'unknown' }}</li>
  </ul>
</template>
