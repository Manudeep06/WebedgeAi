import * as tf from '@tensorflow/tfjs'
import { version as mobilenetVersion } from '@tensorflow-models/mobilenet'

async function webglInfo() {
  try {
    const gl = document.createElement('canvas').getContext('webgl2') || document.createElement('canvas').getContext('webgl')
    if (!gl) return null
    const ext = gl.getExtension('WEBGL_debug_renderer_info')
    return {
      version: gl.getParameter(gl.VERSION),
      vendor: ext ? gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR),
      renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      disjointTimerQuery: Boolean(gl.getExtension('EXT_disjoint_timer_query_webgl2') || gl.getExtension('EXT_disjoint_timer_query')),
    }
  } catch {
    return null
  }
}

async function webgpuInfo() {
  if (!navigator.gpu) return null
  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return { available: false }
    const info = adapter.info ?? {}
    return {
      available: true,
      vendor: info.vendor ?? '',
      architecture: info.architecture ?? '',
      device: info.device ?? '',
      description: info.description ?? '',
      features: [...adapter.features].sort(),
    }
  } catch (err) {
    return { available: false, error: String(err) }
  }
}

async function uaHints() {
  if (!navigator.userAgentData?.getHighEntropyValues) return null
  try {
    return await navigator.userAgentData.getHighEntropyValues(['platform', 'platformVersion', 'architecture', 'model', 'bitness', 'fullVersionList'])
  } catch {
    return null
  }
}

async function battery() {
  try {
    const b = await navigator.getBattery?.()
    return b ? { level: b.level, charging: b.charging } : null
  } catch {
    return null
  }
}

export async function collectEnvironment() {
  return {
    capturedAt: new Date().toISOString(),
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    userAgent: navigator.userAgent,
    userAgentData: await uaHints(),
    hardwareConcurrency: navigator.hardwareConcurrency ?? null,
    deviceMemoryGB: navigator.deviceMemory ?? null,
    screen: { width: screen.width, height: screen.height, devicePixelRatio: window.devicePixelRatio },
    crossOriginIsolated: window.crossOriginIsolated,
    battery: await battery(),
    webgl: await webglInfo(),
    webgpu: await webgpuInfo(),
    software: { tfjs: tf.version.tfjs, mobilenet: mobilenetVersion },
  }
}
