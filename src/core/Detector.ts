import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import { non_max_suppression } from './nonMaxSuppression'
import labels from '../assets/labels.json'

/** LIMIAR de acerto */
const THRESHOLD = 0.8

/**
 * Representa um objeto detectado
 */
export interface BoundingBox {
  /** Posição do objeto na imagem*/
  x: number

  /** Posição do objeto na imagem*/
  y: number

  /** Largura do objeto */
  width: number

  /** Altura do objeto */
  height: number

  /** Nome do objeto */
  label: string

  /** Cor do objeto */
  color: string

  /** Probabilidade de acerto */
  score?: number
}

/**
 * Detector de objetos
 */
export class Detector {
  static yolov7?: tf.GraphModel

  /**
   * Carrega o modelo
   * @param model Modelo a ser carregado
   * @param onProgress Callback de progresso
   */
  static async loadModel(model: string, onProgress: (progress: number) => void) {
    const yolov7 = await tf.loadGraphModel(model, {
      onProgress: (fraction) => {
        onProgress(fraction)
      }
    })

    const dummyInput = tf.ones(yolov7.inputs[0].shape ?? []);
    const warmupResult = await yolov7.executeAsync(dummyInput);

    tf.dispose(warmupResult);
    tf.dispose(dummyInput);

    this.yolov7 = yolov7

    await tf.ready()
  }

  /**
   * Detecta objetos em um frame
   * @param frame Frame a ser processado
   * @param threshold Limiar de acerto
   */
  static async detect(frame: tf.TensorLike, threshold = THRESHOLD
  ): Promise<BoundingBox[]> {
    if (!this.yolov7) {
      throw new Error('Model not loaded')
    }

    const model_dim = [640, 640] as [number, number]

    tf.engine().startScope()

    const input = tf.tidy(() => tf.image
      .resizeBilinear(frame, model_dim)
      .div(255)
      .transpose([2, 0, 1])
      .expandDims(0)
    )

    let result = await this.yolov7.executeAsync(input) as tf.Tensor

    let res = (result.arraySync() as any[])[0]

    const detections = non_max_suppression(res)

    return detections.map(([x, y, width, height, score, classe]) => {
      return {
        color: getColor(classe),
        x: x - width / 2,
        y: y - height / 2,
        height,
        width,
        score,
        label: labels[classe]
      }
    }).filter((detection) => detection.score >= threshold)
  }
}

/**
 * Retorna uma cor para a classe
 * @param classe Número da classe
 * @returns Uma cor em HSL
 */
function getColor(classe: number) {
  const hue = classe * 0.41
  return `hsl(${hue}, 100%, 50%)`
}