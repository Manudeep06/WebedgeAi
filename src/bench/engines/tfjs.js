import * as tf from '@tensorflow/tfjs'
import { activateBackend } from '../backends.js'
import { disposeModel, loadModel, MODELS } from '../models.js'

export const TFJS_BACKENDS = ['webgpu', 'webgl', 'wasm', 'cpu']
export const TFJS_MODELS = MODELS

// Session adapter for TensorFlow.js + @tensorflow-models/mobilenet. All engines expose:
//   load info, timeOnce(), predict(blob) -> Float32Array probabilities, kernelMs(), memory(), dispose()
export async function createTfjsSession({ backend, modelId }, protocol) {
  const act = await activateBackend(backend)
  if (!act.ok) return { unsupported: true }
  const spec = MODELS.find((m) => m.id === modelId)
  const tensorsBefore = tf.memory().numTensors
  const loaded = await loadModel(spec, { modelBase: protocol.modelBase, cache: protocol.cacheModels })
  const model = loaded.model

  let probe = null
  return {
    load: { ms: loaded.loadMs, source: loaded.source, url: loaded.url },
    details: act.details,
    flags: act.flags,
    setLatencyInput(canvas) { probe = canvas },
    // End-to-end: pixel upload, preprocessing, forward pass, read-back (await data() syncs the GPU).
    async timeOnce() {
      const start = performance.now()
      const out = model.infer(probe)
      await out.data()
      const ms = performance.now() - start
      out.dispose()
      return ms
    },
    async kernelMs() {
      try {
        const info = await tf.time(() => { tf.tidy(() => { model.infer(probe) }) })
        return typeof info.kernelMs === 'number' ? info.kernelMs : null
      } catch {
        return null
      }
    },
    async predict(blob) {
      const bitmap = await createImageBitmap(blob)
      const probsT = tf.tidy(() => tf.softmax(model.infer(bitmap)).squeeze())
      const probs = await probsT.data()
      probsT.dispose()
      bitmap.close()
      return probs
    },
    memory() {
      const m = tf.memory()
      return { numBytes: m.numBytes, numTensors: m.numTensors, numBytesInGPU: m.numBytesInGPU ?? null }
    },
    leakedTensors: () => tf.memory().numTensors - tensorsBefore,
    dispose() { disposeModel(model) },
  }
}
