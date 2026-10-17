import { createOrtSession } from './engines/ort.js'
import { createTfjsSession } from './engines/tfjs.js'
import { summarize } from './stats.js'

// Base64 round-trip for Float32 probability vectors (kept compact in the JSON export).
function f32ToBase64(arr) {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)
  let s = ''
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(s)
}

function topK(probs, k) {
  const idx = Array.from(probs.keys())
  idx.sort((a, b) => probs[b] - probs[a])
  return idx.slice(0, k)
}

// A config is { engine, backend, model, dtype }. `backend` is the engine-qualified label used in
// tables (e.g. "webgpu" for TF.js, "ort-webgpu" for ONNX Runtime); `device` is the raw value.
export function configLabel(cfg) {
  return `${cfg.backend} / ${cfg.model}${cfg.dtype && cfg.dtype !== 'fp32' ? `@${cfg.dtype}` : ''}`
}

function openSession(cfg, protocol) {
  if (cfg.engine === 'ort') return createOrtSession({ device: cfg.device, modelId: cfg.model, dtype: cfg.dtype }, protocol)
  return createTfjsSession({ backend: cfg.device, modelId: cfg.model }, protocol)
}

/**
 * Runs one configuration of the protocol:
 *   1. activate backend / execution provider   2. load model (network or on-device cache)
 *   3. cold (first) inference                  4. warm-up runs
 *   5. N timed runs                            6. accuracy pass over the labelled dataset
 */
export async function runConfig(cfg, { latencyInput, dataset, protocol, log }) {
  const record = {
    engine: cfg.engine,
    backend: cfg.backend,
    device: cfg.device,
    model: cfg.model,
    dtype: cfg.dtype,
    rep: cfg.rep,
    order: cfg.order,
    startedAt: new Date().toISOString(),
    status: 'ok',
  }

  let session
  try {
    session = await openSession(cfg, protocol)
    if (session.unsupported) {
      log('  unsupported on this device/browser, skipped')
      return { ...record, status: 'unsupported' }
    }
    record.load = session.load
    record.backendDetails = session.details
    record.tfFlags = session.flags
    log(`  loaded from ${session.load.source} in ${session.load.ms.toFixed(0)} ms`)

    await session.setLatencyInput(latencyInput)
    if (session.preprocessMs) record.preprocessMs = await session.preprocessMs(latencyInput)
    record.coldMs = await session.timeOnce()
    for (let i = 0; i < protocol.warmup; i++) await session.timeOnce()

    const samples = []
    for (let i = 0; i < protocol.runs; i++) samples.push(await session.timeOnce())
    record.latencySamplesMs = samples
    record.latency = summarize(samples, { seed: protocol.seed })
    record.kernelMs = await session.kernelMs()
    record.memory = session.memory()
    log(`  cold ${record.coldMs.toFixed(1)} ms, warm median ${record.latency.median.toFixed(2)} ms`)

    if (dataset?.items.length) {
      // Plain JavaScript is 10-100x slower than every other backend; it serves as the numerical
      // reference, for which the first `referenceImages` images suffice.
      const limit = cfg.engine === 'tfjs' && cfg.device === 'cpu' ? protocol.referenceImages : Infinity
      record.accuracy = await accuracyPass(session, { ...dataset, items: dataset.items.slice(0, limit) }, protocol, log)
    }
  } catch (err) {
    record.status = 'error'
    record.error = String(err?.stack || err)
    log(`  ERROR: ${err?.message || err}`)
  } finally {
    if (session && !session.unsupported) {
      await session.dispose()
      record.leakedTensors = session.leakedTensors()
    }
    record.finishedAt = new Date().toISOString()
  }
  return record
}

async function accuracyPass(session, dataset, protocol, log) {
  let top1 = 0
  let top5 = 0
  const predictions = []
  const referenceProbs = []
  for (let i = 0; i < dataset.items.length; i++) {
    const item = dataset.items[i]
    const probs = await session.predict(item.blob)
    const t5 = topK(probs, 5)
    if (t5[0] === item.label) top1++
    if (t5.includes(item.label)) top5++
    predictions.push({ file: item.file, label: item.label, top5: t5, top5Probs: t5.map((k) => probs[k]) })
    if (i < protocol.referenceImages) referenceProbs.push({ file: item.file, probs: f32ToBase64(Float32Array.from(probs)) })
    if ((i + 1) % 100 === 0) log(`  accuracy ${i + 1}/${dataset.items.length}`)
  }
  const n = dataset.items.length
  log(`  top-1 ${(100 * top1 / n).toFixed(2)}%, top-5 ${(100 * top5 / n).toFixed(2)}% (n=${n})`)
  return { n, top1: top1 / n, top5: top5 / n, predictions, referenceProbs }
}
