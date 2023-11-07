/**
 * Isso é o Worker do Detector,
 * a IA será executada em uma thread separada da interface para que
 * não haja travamentos na interface
 */
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import { Detector } from './Detector'

// Próximo frame a ser processado
let nextTensor: any = null

// Indica se o Worker está pronto para processar frames
let ready = false

const LISTENERS: Record<string, Function> = {
  // Quando a interface pede para carregar o modelo...
  load: async (_: any, model: string) => {
    if (ready) {
      return;
    }

    // Carrega o modelo
    await Detector.loadModel(model, (progress) => {
      globalThis.postMessage(['progress', progress])
    })

    ready = true

    // Informa a interface que o Worker está pronto
    globalThis.postMessage(['ready'])

    // Inicia o loop de processamento
    loop()
  },

  // Quando a interface pedir para detectar objetos em um frame...
  detect: async (_: any, frame: any) => {
    // Salva o frame para ser processado
    nextTensor = frame
  },

  // Quando a interface pedir para parar o Worker...
  stop: async () => {
    ready = false
  }
}

// Recebe as mensagens da interface
globalThis.addEventListener('message', (e) => {
  const [event = '', ...args] = e.data

  if (event in LISTENERS) {
    LISTENERS[event](event, ...args)
  }
})

// Loop de processamento
async function loop() {
  // Enquanto o Worker estiver em execução...
  while (ready) {
    // Se não houver um frame para processar, espera 1 segundo e tenta novamente
    if (!nextTensor) {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      continue;
    }

    // Converte o frame para um Tensor do TensorFlow
    const tensor = tf.tensor(nextTensor.data, nextTensor.shape, 'int32');

    // Libera a memória do frame
    nextTensor = null;

    // Detecta os objetos no frame
    const result = await Detector.detect(tensor as unknown as tf.TensorLike)

    tensor.dispose()

    // Envia o resultado para a interface
    globalThis.postMessage(['result', result])

    // Aguarda 1/30 de segundo para processar o próximo frame
    await new Promise((resolve) => setTimeout(resolve, 1000 / 30))
  }
}