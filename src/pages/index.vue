<script setup lang="ts">
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'

import { computed, onMounted, onUnmounted, ref, shallowRef, watch } from 'vue';
import { BoundingBox } from '../core/Detector';
import DetectorWorker from '../core/worker?worker'

const worker = new DetectorWorker()

const devices = ref<{
  id: string,
  name: string
}[]>([])

const input = computed({
  get: () => localStorage.getItem('input') ?? '',
  set: (value) => localStorage.setItem('input', value),
});

watch(devices, (devices) => {
  if (devices.length > 0 && !input.value) {
    input.value = devices[0].id
  }
}, { immediate: true })

const LISTENERS: Record<string, Function> = {
  progress(_: any, value: number) {
    progress.value = value
  },
  result(_: any, boxes: BoundingBox[]) {
    boundingBoxes.value = boxes
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

const progress = ref(0)
const canvas = ref<HTMLCanvasElement>()
const ctx = computed(() => canvas.value?.getContext('2d'))
const boundingBoxes = shallowRef<BoundingBox[]>([])
const ready = ref(false)

let nextFrame: number
let camera: MediaStream

// Loop de renderização do canvas
async function loop() {
  nextFrame = requestAnimationFrame(loop)

  if (!canvas.value || !camera || !ctx.value) {
    return
  }

  const draw = ctx.value;
  const frame = tf.browser.fromPixels(video)

  worker.postMessage(['detect', {
    data: frame.dataSync(),
    shape: frame.shape
  }])

  frame.dispose()

  draw.clearRect(0, 0, canvas.value.width, canvas.value.height)

  draw.drawImage(video, 0, 0, canvas.value.width, canvas.value.height)

  for (const box of boundingBoxes.value) {
    draw.fillStyle = box.color
    draw.strokeStyle = box.color
    draw.lineWidth = 3
    draw.strokeRect(
      box.x,
      box.y,
      box.width,
      box.height,
    )

    const label = draw.measureText(box.label)

    draw.fillRect(
      box.x - 2,
      box.y - 20,
      label.width + 10,
      20,
    )

    draw.fillStyle = 'white'
    draw.fillText(
      box.label,
      box.x + 2,
      box.y - 5,
    )
  }
}

// Destroi o loop de renderização
onUnmounted(() => {
  cancelAnimationFrame(nextFrame)
  camera.getTracks().forEach((track) => track.stop())
  postMessage(['stop'])
})

// Lista as câmeras disponíveis
navigator.mediaDevices.enumerateDevices().then(dev => {
  devices.value = dev.filter(d => d.kind === 'videoinput').map(d => ({
    id: d.deviceId,
    name: d.label ?? d.deviceId
  }))
})

// Inicia a câmera selecionada
watch(input, async (input) => {
  if (!input) {
    return;
  }

  // Desliga a câmera anterior
  camera?.getTracks().forEach((track) => track.stop())

  // Inicia a nova câmera
  camera = await navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: input
    }
  })

  // Inicia o vídeo
  video.srcObject = camera

  // Atualiza o tamanho do canvas
  video.addEventListener('loadedmetadata', () => {
    canvas.value!.width = video.videoWidth
    canvas.value!.height = video.videoHeight
    video.play()
  }, { once: true })
}, {
  immediate: true
})
</script>

<template>
  <div class="container mx-auto">
    <div class="flex mb-4">
      <select v-model="input" class="px-4 py-2">
        <option disabled selected>Câmera</option>
        <option v-for="device in devices" :key="device.id" :value="device.id">{{ device.name }}</option>
      </select>
    </div>
    <template v-if="!ready">
      <span>Inicializando...</span>
      <progress indeterminate class="block w-full" min="0" max="1" :value="progress" />
    </template>
    <canvas ref="canvas"></canvas>

  </div>
</template>