<script setup lang="ts">
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'

import { computed, onUnmounted, ref, watch, watchEffect } from 'vue';
import { BoundingBox } from '../core/Detector';
import DetectorWorker from '../core/worker?worker'
import { useDevicesList, useLocalStorage } from '@vueuse/core'

const { videoInputs } = useDevicesList({
  requestPermissions: true,
  constraints: {
    video: true,
    audio: false
  }
})

const worker = new DetectorWorker()
let boundingBoxes: BoundingBox[] = []

const input = useLocalStorage('input', '')

watchEffect(() => {
  input.value ??= videoInputs.value[0]?.deviceId
})

const LISTENERS: Record<string, Function> = {
  progress(_: any, value: number) {
    progress.value = value
  },
  result(_: any, boxes: BoundingBox[]) {
    boundingBoxes = boxes
  },
  ready: () => {
    ready.value = true
    loop()
  }
}

worker.addEventListener('message', (e) => {
  const [event, ...args] = e.data

  if (event in LISTENERS) {
    LISTENERS[event](e, ...args)
  }
})

worker.postMessage(['load', '/yolov7_web_model/model.json'])

const video = document.createElement('video')

video.playsInline = true
video.muted = true
video.autoplay = true
video.play()

const progress = ref(0)
const canvas = ref<HTMLCanvasElement>()
const ctx = computed(() => canvas.value?.getContext('2d'))
const ready = ref(false)

let nextFrame: number
let camera: MediaStream

const BOX_STYLE = {
  box: {
    borderSize: 3,
  },
  label: {
    fontSize: 16,
    fontFamily: 'Arial',
    color: '#fff',
    padding: {
      x: 8,
      y: 4
    }
  }
}

// Loop de renderização do canvas
async function loop() {
  nextFrame = requestAnimationFrame(loop)

  if (!canvas.value || !camera || !ctx.value) {
    return
  }

  canvas.value!.width = video.videoWidth
  canvas.value!.height = video.videoHeight

  const draw = ctx.value;
  const frame = tf.browser.fromPixels(video)
  draw.clearRect(0, 0, canvas.value.width, canvas.value.height)

  worker.postMessage(['detect', {
    data: frame.dataSync(),
    shape: frame.shape
  }])

  frame.dispose()


  draw.drawImage(video, 0, 0)

  for (const box of boundingBoxes) {
    draw.font = `bolder ${BOX_STYLE.label.fontSize}px ${BOX_STYLE.label.fontFamily}`

    draw.textBaseline = 'top'
    draw.textAlign = 'left'
    draw.fillStyle = BOX_STYLE.label.color
    const label = draw.measureText(box.label)

    draw.fillStyle = box.color
    draw.strokeStyle = box.color
    draw.lineWidth = BOX_STYLE.box.borderSize

    draw.strokeRect(
      box.x,
      box.y,
      box.width,
      box.height,
    )

    const labelHeight = BOX_STYLE.label.fontSize + (BOX_STYLE.label.padding.y * 2)

    draw.fillRect(
      box.x - (BOX_STYLE.box.borderSize / 2),
      box.y - labelHeight,
      BOX_STYLE.label.padding.x + label.width + BOX_STYLE.label.padding.x,
      labelHeight,
    )

    draw.fillStyle = BOX_STYLE.label.color
    draw.fillText(
      box.label,
      box.x + BOX_STYLE.label.padding.x,
      box.y - labelHeight + BOX_STYLE.label.padding.y,
    )

  }
}

// Destroi o loop de renderização
onUnmounted(() => {
  cancelAnimationFrame(nextFrame)
  camera.getTracks().forEach((track) => track.stop())
  postMessage(['stop'])
  worker.terminate()
})

// Seleciona a primeira câmera disponível
watch(videoInputs, (devices) => {
  if (devices.length > 0 && !input.value) {
    input.value = devices[0].deviceId
  }
}, { immediate: true })

// Inicia a câmera selecionada
watch(input, async (input) => {
  if (!input) {
    return;
  }

  // Desliga a câmera anterior
  camera?.getTracks().forEach((track) => track.stop())

  // Inicia a nova câmera
  camera = input === 'screen' ? await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: false
  }) : await navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: input
    },
    audio: false
  })

  // Inicia o vídeo
  video.srcObject = camera

  video.play()
}, {
  immediate: true
})
</script>

<template>
  <div class="container mx-auto">
    <div class="flex mb-4">
      <select v-model="input" class="px-4 py-2">
        <option disabled selected value="">Câmera</option>
        <option v-for="device in videoInputs" :key="device.deviceId" :value="device.deviceId">{{ device.label ??
          device.deviceId }}</option>
        <option value="screen">Tela</option>
      </select>
    </div>
    <template v-if="!ready">
      <span>Inicializando...</span>
      <progress indeterminate class="block w-full" min="0" max="1" :value="progress" />
    </template>
    <canvas ref="canvas" class="max-w-full"></canvas>

  </div>
</template>