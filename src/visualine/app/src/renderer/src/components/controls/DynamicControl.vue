<template>
  <div class="control-field" :class="`control-${control.type}`">
    <div class="control-label-row">
      <div class="label-stack">
        <label :for="control.key">{{ control.label }}</label>
        <p v-if="control.description" class="control-description">
          {{ control.description }}
        </p>
      </div>

      <span v-if="displayValue !== null" class="control-value">
        {{ displayValue }}
      </span>
    </div>

    <div class="control-input-wrap">
      <input
        v-if="control.type === 'text'"
        :id="control.key"
        class="studio-input"
        type="text"
        :placeholder="control.default ? String(control.default) : ''"
        :value="textValue"
        @input="updateValue($event.target.value)"
      />

      <input
        v-else-if="control.type === 'number'"
        :id="control.key"
        class="studio-input"
        type="number"
        :min="control.min"
        :max="control.max"
        :step="control.step || 1"
        :value="numberValue"
        @input="updateNumber($event.target.value)"
      />

      <template v-else-if="control.type === 'slider'">
        <input
          :id="control.key"
          class="studio-slider"
          type="range"
          :min="control.min"
          :max="control.max"
          :step="control.step || 1"
          :value="numberValue"
          @input="updateNumber($event.target.value)"
        />

        <div class="slider-meta">
          <span>{{ control.min }}</span>
          <span>{{ control.max }}</span>
        </div>
      </template>

      <select
        v-else-if="control.type === 'select'"
        :id="control.key"
        class="studio-input studio-select"
        :value="String(modelValue ?? control.default ?? '')"
        @change="updateValue(parseSelectValue($event.target.value))"
      >
        <option
          v-for="option in control.options"
          :key="getOptionKey(option)"
          :value="getOptionValue(option)"
        >
          {{ getOptionLabel(option) }}
        </option>
      </select>

      <label
        v-else-if="control.type === 'toggle'"
        class="switch-row"
      >
        <input
          type="checkbox"
          :checked="Boolean(modelValue)"
          @change="updateValue($event.target.checked)"
        />

        <span class="switch-track">
          <span class="switch-thumb" />
        </span>

        <span class="switch-label">
          {{ Boolean(modelValue) ? 'Enabled' : 'Disabled' }}
        </span>
      </label>

      <input
        v-else
        :id="control.key"
        class="studio-input"
        type="text"
        :value="textValue"
        @input="updateValue($event.target.value)"
      />
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  control: {
    type: Object,
    required: true
  },
  modelValue: {
    type: [String, Number, Boolean],
    default: ''
  }
})

const emit = defineEmits(['update:modelValue'])

const fallbackValue = computed(() => {
  if (props.modelValue !== undefined && props.modelValue !== null && props.modelValue !== '') {
    return props.modelValue
  }

  if (props.control.default !== undefined && props.control.default !== null) {
    return props.control.default
  }

  if (props.control.type === 'slider' || props.control.type === 'number') {
    return props.control.min ?? 0
  }

  if (props.control.type === 'toggle') {
    return false
  }

  return ''
})

const textValue = computed(() => String(fallbackValue.value ?? ''))

const numberValue = computed(() => {
  const value = Number(fallbackValue.value)

  if (Number.isNaN(value)) {
    return Number(props.control.min ?? 0)
  }

  return value
})

const displayValue = computed(() => {
  if (props.control.type === 'toggle') return null

  if (props.control.type === 'slider' || props.control.type === 'number') {
    return formatNumber(numberValue.value)
  }

  if (fallbackValue.value === undefined || fallbackValue.value === null || fallbackValue.value === '') {
    return null
  }

  return String(fallbackValue.value)
})

function updateValue(value) {
  emit('update:modelValue', value)
}

function updateNumber(rawValue) {
  let value = Number(rawValue)

  if (Number.isNaN(value)) {
    value = Number(props.control.default ?? props.control.min ?? 0)
  }

  if (props.control.min !== undefined && props.control.min !== null) {
    value = Math.max(Number(props.control.min), value)
  }

  if (props.control.max !== undefined && props.control.max !== null) {
    value = Math.min(Number(props.control.max), value)
  }

  emit('update:modelValue', value)
}

function formatNumber(value) {
  if (!Number.isFinite(Number(value))) return value

  const step = Number(props.control.step ?? 1)

  if (step < 1) {
    return Number(value).toFixed(step < 0.05 ? 2 : 1).replace(/\.0$/, '')
  }

  return String(value)
}

function getOptionLabel(option) {
  if (option && typeof option === 'object') {
    return option.label ?? option.name ?? option.value
  }

  return option
}

function getOptionValue(option) {
  if (option && typeof option === 'object') {
    return String(option.value ?? option.label ?? option.name)
  }

  return String(option)
}

function getOptionKey(option) {
  return getOptionValue(option)
}

function parseSelectValue(value) {
  const original = props.control.options.find((option) => getOptionValue(option) === value)

  if (original && typeof original === 'object') {
    return original.value ?? value
  }

  if (typeof original === 'number') return Number(value)
  if (value === 'true') return true
  if (value === 'false') return false

  const numeric = Number(value)
  if (!Number.isNaN(numeric) && String(numeric) === value) {
    return numeric
  }

  return value
}
</script>

<style scoped>
.control-field {
  padding: 14px;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.025);
  border: 1px solid rgba(132, 146, 187, 0.13);
}

.control-label-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.label-stack {
  min-width: 0;
}

.control-label-row label {
  display: block;
  color: var(--text);
  font-size: 13px;
  font-weight: 900;
  line-height: 1.2;
}

.control-description {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

.control-value {
  flex: 0 0 auto;
  max-width: 96px;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--cyan);
  background: rgba(39, 224, 209, 0.1);
  border: 1px solid rgba(39, 224, 209, 0.18);
  padding: 4px 8px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 900;
}

.control-input-wrap {
  margin-top: 12px;
}

.studio-select {
  cursor: pointer;
}

.slider-meta {
  display: flex;
  justify-content: space-between;
  color: var(--muted-2);
  font-size: 10px;
  margin-top: 4px;
}

.switch-row {
  display: inline-flex;
  align-items: center;
  gap: 10px;
}

.switch-label {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
</style>