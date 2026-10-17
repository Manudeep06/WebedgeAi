import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgpu'
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm'
import wasmPlain from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url'
import wasmSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url'
import wasmThreadedSimd from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url'

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPlain,
  'tfjs-backend-wasm-simd.wasm': wasmSimd,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmThreadedSimd,
})

export const BACKENDS = ['webgpu', 'webgl', 'wasm', 'cpu']

// Switches the active backend and reports the capabilities it actually ended up with,
// since e.g. WASM silently falls back to single-threaded without cross-origin isolation.
export async function activateBackend(name) {
  let ok = false
  try {
    ok = await tf.setBackend(name)
    await tf.ready()
  } catch {
    ok = false
  }
  if (!ok || tf.getBackend() !== name) return { ok: false }

  const details = {}
  if (name === 'wasm') {
    details.simd = await tf.env().getAsync('WASM_HAS_SIMD_SUPPORT')
    details.multithread = await tf.env().getAsync('WASM_HAS_MULTITHREAD_SUPPORT')
  }
  return { ok: true, details, flags: { ...tf.env().getFlags() } }
}
