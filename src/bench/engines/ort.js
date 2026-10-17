import { AutoModelForImageClassification, AutoProcessor, env, RawImage } from '@huggingface/transformers'
import ortMjs from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url'
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'

// Self-host the ONNX Runtime Web binaries instead of the library's CDN default, so the runtime
// is pinned by package-lock.json and runs work offline once cached.
env.backends.onnx.wasm.wasmPaths = { mjs: ortMjs, wasm: ortWasm }
env.allowLocalModels = false

// ONNX Runtime Web execution providers, as exposed by Transformers.js. WebNN reaches the OS ML
// stack (DirectML / Core ML / NNAPI) and, with `webnn-npu`, neural processing units.
export const ORT_DEVICES = ['webgpu', 'wasm', 'webnn-gpu', 'webnn-npu', 'webnn-cpu']

// ImageNet-1k classifiers in sorted-WNID label order, spanning architecture families.
// 4-bit variants only exist where they actually shrink the model: they quantize MatMul weights,
// which CNNs barely have, so the CNN q4 files are fp32-sized and excluded.
export const ORT_MODELS = [
  { id: 'mobilenetv4-s', repo: 'onnx-community/mobilenetv4_conv_small.e2400_r224_in1k', family: 'CNN (2024)', dtypes: ['fp32', 'fp16', 'q8'] },
  { id: 'mobilevit-s', repo: 'Xenova/mobilevit-small', family: 'CNN-Transformer hybrid', dtypes: ['fp32', 'fp16', 'q8', 'q4', 'q4f16'] },
  { id: 'resnet-50', repo: 'Xenova/resnet-50', family: 'CNN (2015)', dtypes: ['fp32', 'fp16', 'q8'] },
  { id: 'vit-b16', repo: 'Xenova/vit-base-patch16-224', family: 'Vision Transformer', dtypes: ['fp32', 'fp16', 'q8', 'q4', 'q4f16'] },
]
export const ORT_DTYPES = ['fp32', 'fp16', 'q8', 'q4', 'q4f16']

const DTYPE_FILE_SUFFIX = { fp32: '', fp16: '_fp16', q8: '_quantized', q4: '_q4', q4f16: '_q4f16' }

async function weightsCached(repo, dtype) {
  try {
    const cache = await caches.open(env.cacheKey)
    const file = `${repo}/resolve/main/onnx/model${DTYPE_FILE_SUFFIX[dtype]}.onnx`
    return (await cache.keys()).some((req) => req.url.includes(file))
  } catch {
    return false
  }
}

export async function clearOrtCache() {
  try { await caches.delete(env.cacheKey) } catch { /* Cache API unavailable */ }
}

function deviceAvailable(device) {
  if (device === 'webgpu') return Boolean(navigator.gpu)
  if (device.startsWith('webnn')) return 'ml' in navigator
  return true
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

export async function createOrtSession({ device, modelId, dtype }, protocol) {
  const spec = ORT_MODELS.find((m) => m.id === modelId)
  if (!deviceAvailable(device)) return { unsupported: true }

  const cached = await weightsCached(spec.repo, dtype)
  const start = performance.now()
  const [model, processor] = await Promise.all([
    AutoModelForImageClassification.from_pretrained(spec.repo, { device, dtype }),
    AutoProcessor.from_pretrained(spec.repo),
  ])
  const loadMs = performance.now() - start

  const wasm = env.backends.onnx.wasm
  const details = {
    runtime: `onnxruntime-web ${env.backends.onnx.versions?.web ?? ''}`.trim(),
    transformersJs: env.version,
    wasmThreads: wasm.numThreads ?? null,
    webgpuPowerPreference: env.backends.onnx.webgpu?.powerPreference ?? null,
    family: spec.family,
  }

  let pixelValues = null
  const run = async (inputs) => {
    const { logits } = await model(inputs)
    const f32 = logits.type === 'float32' ? logits : logits.to('float32')
    return Float32Array.from(f32.data)
  }

  return {
    load: { ms: loadMs, source: cached ? 'cache' : 'network', url: `hf://${spec.repo} (${dtype})` },
    details,
    flags: null,
    async setLatencyInput(canvas) {
      ;({ pixel_values: pixelValues } = await processor(RawImage.fromCanvas(canvas)))
    },
    // Model execution plus output read-back on a fixed, already pre-processed input.
    // Pre-processing is plain JS (not ORT) and is reported separately by preprocessMs().
    async timeOnce() {
      const t0 = performance.now()
      await run({ pixel_values: pixelValues })
      return performance.now() - t0
    },
    async preprocessMs(canvas) {
      const t0 = performance.now()
      await processor(RawImage.fromCanvas(canvas))
      return performance.now() - t0
    },
    kernelMs: async () => null,
    async predict(blob) {
      const inputs = await processor(await RawImage.fromBlob(blob))
      return softmax(await run(inputs))
    },
    memory: () => null,
    leakedTensors: () => null,
    async dispose() { await model.dispose() },
  }
}
