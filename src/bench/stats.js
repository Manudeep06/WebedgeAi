// Deterministic PRNG so run orders and bootstrap intervals are reproducible from the seed.
export function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function shuffle(items, rng) {
  const out = items.slice()
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

// Linear interpolation between order statistics (Hyndman & Fan type 7, the NumPy default).
export function quantile(sorted, q) {
  if (!sorted.length) return NaN
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.ceil(pos)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
}

export function summarize(samples, { seed = 1, resamples = 2000 } = {}) {
  const n = samples.length
  if (!n) return null
  const sorted = samples.slice().sort((a, b) => a - b)
  const mean = samples.reduce((s, x) => s + x, 0) / n
  const sd = n > 1 ? Math.sqrt(samples.reduce((s, x) => s + (x - mean) ** 2, 0) / (n - 1)) : 0

  // Percentile bootstrap CI for the median.
  const rng = mulberry32(seed)
  const medians = new Float64Array(resamples)
  const buf = new Float64Array(n)
  for (let b = 0; b < resamples; b++) {
    for (let i = 0; i < n; i++) buf[i] = samples[Math.floor(rng() * n)]
    buf.sort()
    medians[b] = quantile(buf, 0.5)
  }
  medians.sort()

  return {
    n,
    mean,
    sd,
    min: sorted[0],
    p05: quantile(sorted, 0.05),
    median: quantile(sorted, 0.5),
    p95: quantile(sorted, 0.95),
    max: sorted[n - 1],
    medianCI95: [quantile(medians, 0.025), quantile(medians, 0.975)],
  }
}
