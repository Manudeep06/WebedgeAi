import { mulberry32 } from './stats.js'

// Loads a labelled evaluation set described by a CSV manifest with a header row:
//   file,label
//   ILSVRC2012_val_00000001.JPEG,65
// `label` is the 0-based ImageNet-1k class index in sorted-WNID order (the order MobileNet emits).
// Image paths are resolved relative to the manifest. Returns null when no manifest is served.
export async function loadDataset(manifestUrl, limit = Infinity) {
  if (!manifestUrl) return null
  const base = new URL(manifestUrl, location.href)
  const res = await fetch(base)
  if (!res.ok || !(res.headers.get('content-type') || '').match(/csv|text\/plain|octet-stream/)) return null

  const rows = (await res.text()).trim().split(/\r?\n/).slice(1).filter(Boolean)
  const items = []
  for (const row of rows.slice(0, limit)) {
    const [file, label] = row.split(',').map((s) => s.trim())
    const img = await fetch(new URL(file, base))
    if (!img.ok) throw new Error(`Dataset image missing: ${file}`)
    items.push({ file, label: Number(label), blob: await img.blob() })
  }
  return { manifestUrl: base.href, items }
}

// Fixed synthetic 224x224 input for latency runs. CNN latency does not depend on image content,
// and a generated input keeps the latency workload identical across every device and site.
export function syntheticInput(seed) {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 224
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(224, 224)
  const rng = mulberry32(seed)
  for (let i = 0; i < img.data.length; i += 4) {
    img.data[i] = rng() * 255
    img.data[i + 1] = rng() * 255
    img.data[i + 2] = rng() * 255
    img.data[i + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return canvas
}
