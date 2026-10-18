import { AutoModelForImageClassification, AutoProcessor, env, RawImage } from '@huggingface/transformers'
import ortMjs from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url'
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'

env.backends.onnx.wasm.wasmPaths = { mjs: ortMjs, wasm: ortWasm }
env.allowLocalModels = false

const REPO = 'Xenova/vit-base-patch16-224'
// Tried in order. ViT-B/16 kept full accuracy at every one of these precisions in LOCAL-Bench
// (int8 CNN exports did not, so none are used here); q4f16 on WebGPU was the fastest of them.
const PLANS = [
  { device: 'webgpu', dtype: 'q4f16' }, // needs the shader-f16 WebGPU feature
  { device: 'webgpu', dtype: 'q4' },
  { device: 'wasm', dtype: 'q8' },
]

// ImageNet-1k class indices of the 59 bird species (7–24, 80–100, 127–146).
const BIRD_CLASSES = [
  ...Array.from({ length: 18 }, (_, i) => 7 + i),
  ...Array.from({ length: 21 }, (_, i) => 80 + i),
  ...Array.from({ length: 20 }, (_, i) => 127 + i),
]

let promise
export function loadAccurateModel() {
  promise ??= (async () => {
    const processor = await AutoProcessor.from_pretrained(REPO)
    let lastError
    for (const plan of PLANS) {
      if (plan.device === 'webgpu' && !navigator.gpu) continue
      try {
        const model = await AutoModelForImageClassification.from_pretrained(REPO, plan)
        // A dry run surfaces an unsupported precision now rather than on the user's first photo.
        await model(await processor(new RawImage(new Uint8ClampedArray(224 * 224 * 3).fill(128), 224, 224, 3)))
        return { model, processor, plan, labels: model.config.id2label }
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

function softmax(logits) {
  let max = -Infinity
  for (const v of logits) if (v > max) max = v
  const out = new Float32Array(logits.length)
  let sum = 0
  for (let i = 0; i < logits.length; i++) sum += out[i] = Math.exp(logits[i] - max)
  for (let i = 0; i < out.length; i++) out[i] /= sum
  return out
}

// Draws the photo (optionally mirrored) at most 512 px on its long side; the processor resizes to
// 224 x 224 anyway, so this only saves time on large camera photos.
function toRawImage(bitmap, mirrored) {
  const scale = Math.min(1, 512 / Math.max(bitmap.width, bitmap.height))
  const w = Math.max(1, Math.round(bitmap.width * scale))
  const h = Math.max(1, Math.round(bitmap.height * scale))
  const canvas = new OffscreenCanvas(w, h)
  const ctx = canvas.getContext('2d')
  if (mirrored) ctx.setTransform(-1, 0, 0, 1, w, 0)
  ctx.drawImage(bitmap, 0, 0, w, h)
  return RawImage.fromCanvas(canvas)
}

async function probabilities(model, processor, image) {
  const { logits } = await model(await processor(image))
  const f32 = logits.type === 'float32' ? logits : logits.to('float32')
  return softmax(f32.data)
}

/**
 * Classifies a photo with ViT-B/16. With `mirror`, the photo and its left-right mirror image are
 * both classified and their probabilities averaged (test-time augmentation).
 */
export async function classifyAccurate(file, { mirror = true, topk = 3 } = {}) {
  const { model, processor, plan, labels } = await loadAccurateModel()
  const bitmap = await createImageBitmap(file)
  try {
    const start = performance.now()
    const p = await probabilities(model, processor, toRawImage(bitmap, false))
    if (mirror) {
      const q = await probabilities(model, processor, toRawImage(bitmap, true))
      for (let i = 0; i < p.length; i++) p[i] = (p[i] + q[i]) / 2
    }
    const ms = Math.round(performance.now() - start)
    const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, topk)
    let birdMass = 0
    for (const k of BIRD_CLASSES) birdMass += p[k]
    return { ms, plan, mirror, birdMass, top: order.map((k) => ({ name: labels[k].split(',')[0], probability: p[k] })) }
  } finally {
    bitmap.close()
  }
}
