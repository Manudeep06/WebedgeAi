import { AutoProcessor, CLIPVisionModelWithProjection, env, RawImage } from '@huggingface/transformers'
import ortMjs from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url'
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'
import EMB from './birds-embeddings.json'
import { BIRDS, NOT_BIRD_LABEL } from './birds-labels.js'

env.backends.onnx.wasm.wasmPaths = { mjs: ortMjs, wasm: ortWasm }
env.allowLocalModels = false

const REPO = EMB.model
const LOGIT_SCALE = 100 // CLIP's learned temperature (exp of the stored logit scale, 4.6052)
const PLANS = [
  { device: 'webgpu', dtype: 'q4f16' }, // vision encoder only, 53 MB
  { device: 'webgpu', dtype: 'q4' },
  { device: 'wasm', dtype: 'q8' },
]

const textEmbeddings = (() => {
  const bytes = Uint8Array.from(atob(EMB.embeddings), (c) => c.charCodeAt(0))
  return new Float32Array(bytes.buffer)
})()

let promise
export function loadBirdModel() {
  promise ??= (async () => {
    const processor = await AutoProcessor.from_pretrained(REPO)
    let lastError
    for (const plan of PLANS) {
      if (plan.device === 'webgpu' && !navigator.gpu) continue
      try {
        const model = await CLIPVisionModelWithProjection.from_pretrained(REPO, plan)
        await model(await processor(new RawImage(new Uint8ClampedArray(224 * 224 * 3).fill(128), 224, 224, 3)))
        return { model, processor, plan }
      } catch (err) {
        lastError = err
      }
    }
    throw lastError
  })().catch((err) => {
    promise = undefined
    throw err
  })
  return promise
}

/** Zero-shot bird identification: cosine similarity between the photo and each label's text embedding. */
export async function classifyBird(file, { topk = 3 } = {}) {
  const { model, processor, plan } = await loadBirdModel()
  const start = performance.now()
  const { image_embeds: e } = await model(await processor(await RawImage.fromBlob(file)))
  const img = (e.type === 'float32' ? e : e.to('float32')).data
  let norm = 0
  for (const v of img) norm += v * v
  norm = Math.sqrt(norm)
  const d = EMB.dim
  const n = EMB.birds + EMB.notBird
  const logits = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    let dot = 0
    for (let k = 0; k < d; k++) dot += img[k] * textEmbeddings[i * d + k]
    logits[i] = (LOGIT_SCALE * dot) / norm
  }
  let max = -Infinity
  for (const v of logits) if (v > max) max = v
  let sum = 0
  const p = logits.map((v) => Math.exp(v - max))
  for (const v of p) sum += v
  // the "not a bird" prompts are pooled into a single answer
  const scores = BIRDS.map((name, i) => ({ name, probability: p[i] / sum }))
  let notBird = 0
  for (let i = EMB.birds; i < n; i++) notBird += p[i] / sum
  scores.push({ name: NOT_BIRD_LABEL, probability: notBird })
  scores.sort((a, b) => b.probability - a.probability)
  return { ms: Math.round(performance.now() - start), plan, birds: BIRDS.length, top: scores.slice(0, topk) }
}
