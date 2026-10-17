import * as tf from '@tensorflow/tfjs'
import * as mobilenet from '@tensorflow-models/mobilenet'

// Every variant the @tensorflow-models/mobilenet package ships (there is no V2 at alpha 0.25).
export const MODELS = [
  { id: 'v1-0.25', version: 1, alpha: 0.25 },
  { id: 'v1-0.50', version: 1, alpha: 0.5 },
  { id: 'v1-0.75', version: 1, alpha: 0.75 },
  { id: 'v1-1.00', version: 1, alpha: 1.0 },
  { id: 'v2-0.50', version: 2, alpha: 0.5 },
  { id: 'v2-0.75', version: 2, alpha: 0.75 },
  { id: 'v2-1.00', version: 2, alpha: 1.0 },
]

const cacheKey = (m) => `indexeddb://bench-mobilenet-${m.id}`

async function isCached(m) {
  const models = await tf.io.listModels()
  return cacheKey(m) in models
}

export async function clearModelCache() {
  for (const key of Object.keys(await tf.io.listModels())) {
    if (key.startsWith('indexeddb://bench-mobilenet-')) await tf.io.removeModel(key)
  }
}

// Loads a model and reports where the weights came from, so cold-start cost from the
// network can be separated from cold-start cost out of on-device storage.
//   modelBase: optional self-hosted directory (e.g. "/models") for reproducible weights.
//   cache: persist weights to IndexedDB after a network load, and prefer them afterwards.
export async function loadModel(m, { modelBase = '', cache = true } = {}) {
  let source = 'network'
  let config = { version: m.version, alpha: m.alpha }
  if (cache && (await isCached(m))) {
    source = 'indexeddb'
    config = { version: m.version, modelUrl: cacheKey(m), inputRange: [0, 1] }
  } else if (modelBase) {
    const name = `mobilenet_v${m.version}_${String(Math.round(m.alpha * 100)).padStart(3, '0')}_224`
    config = { version: m.version, modelUrl: `${modelBase.replace(/\/$/, '')}/${name}/model.json`, inputRange: [0, 1] }
  }

  const start = performance.now()
  const model = await mobilenet.load(config)
  const loadMs = performance.now() - start

  if (cache && source === 'network') await model.model.save(cacheKey(m))
  return { model, loadMs, source, url: typeof config.modelUrl === 'string' ? config.modelUrl : 'tfhub (package default)' }
}

export function disposeModel(model) {
  model?.model?.dispose?.()
}
