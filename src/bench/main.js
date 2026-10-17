import { loadDataset, syntheticInput } from './dataset.js'
import { clearOrtCache, ORT_DEVICES, ORT_DTYPES, ORT_MODELS } from './engines/ort.js'
import { TFJS_BACKENDS, TFJS_MODELS } from './engines/tfjs.js'
import { collectEnvironment } from './environment.js'
import { clearModelCache } from './models.js'
import { configLabel, runConfig } from './runner.js'
import { mulberry32, shuffle } from './stats.js'
import { getItem, removeItem, setItem } from './store.js'

const SCHEMA_VERSION = 2
const form = document.getElementById('config')
const logEl = document.getElementById('log')
const statusEl = document.getElementById('status')
const startBtn = document.getElementById('start')
const stopBtn = document.getElementById('stop')
let stopRequested = false

function log(line) {
  logEl.textContent += `${new Date().toLocaleTimeString()}  ${line}\n`
  logEl.scrollTop = logEl.scrollHeight
}

function checkboxes(container, name, values) {
  container.innerHTML = values.map((v) => `<label><input type="checkbox" name="${name}" value="${v}" checked /> ${v}</label>`).join('')
}
checkboxes(document.getElementById('backends'), 'backend', TFJS_BACKENDS)
checkboxes(document.getElementById('models'), 'model', TFJS_MODELS.map((m) => m.id))
checkboxes(document.getElementById('ort-devices'), 'ortDevice', ORT_DEVICES)
checkboxes(document.getElementById('ort-models'), 'ortModel', ORT_MODELS.map((m) => m.id))
checkboxes(document.getElementById('ort-dtypes'), 'ortDtype', ORT_DTYPES)

// URL parameters override form defaults, e.g. bench.html?backend=webgl,wasm&ortModel=&runs=100&auto=1
// (an empty list, like `ortModel=`, deselects that whole group).
const LIST_PARAMS = ['backend', 'model', 'ortDevice', 'ortModel', 'ortDtype']
const params = new URLSearchParams(location.search)
for (const [key, value] of params) {
  if (LIST_PARAMS.includes(key)) {
    const wanted = value.split(',')
    form.querySelectorAll(`input[name=${key}]`).forEach((el) => { el.checked = wanted.includes(el.value) })
  } else if (form.elements[key]) {
    const el = form.elements[key]
    if (el.type === 'checkbox') el.checked = value !== '0' && value !== 'false'
    else el.value = value
  }
}

function readProtocol() {
  const f = new FormData(form)
  const num = (k) => Number(f.get(k))
  return {
    backends: f.getAll('backend'),
    models: f.getAll('model'),
    ortDevices: f.getAll('ortDevice'),
    ortModels: f.getAll('ortModel'),
    ortDtypes: f.getAll('ortDtype'),
    warmup: num('warmup'),
    runs: num('runs'),
    reps: num('reps'),
    cooldownSec: num('cooldown'),
    seed: num('seed'),
    dataset: String(f.get('dataset') || ''),
    datasetLimit: num('datasetLimit'),
    referenceImages: num('referenceImages'),
    modelBase: String(f.get('modelBase') || ''),
    cacheModels: f.get('cacheModels') === 'on',
    isolate: f.get('isolate') === 'on',
    accuracyFirstRepOnly: f.get('accuracyFirstRepOnly') === 'on',
    deviceLabel: String(f.get('deviceLabel') || ''),
    power: String(f.get('power') || ''),
  }
}

// Every repetition visits all configurations in a fresh seeded random order, so thermal
// throttling and caching effects are spread across configurations instead of biasing one.
function buildPlan(protocol) {
  const rng = mulberry32(protocol.seed)
  const configs = [
    ...protocol.backends.flatMap((device) => protocol.models.map((model) => ({ engine: 'tfjs', device, backend: device, model, dtype: 'fp32' }))),
    ...protocol.ortDevices.flatMap((device) => protocol.ortModels.flatMap((model) =>
      ORT_MODELS.find((m) => m.id === model).dtypes.filter((d) => protocol.ortDtypes.includes(d))
        .map((dtype) => ({ engine: 'ort', device, backend: `ort-${device}`, model, dtype })))),
  ]
  const plan = []
  for (let rep = 0; rep < protocol.reps; rep++) {
    for (const c of shuffle(configs, rng)) plan.push({ ...c, rep, order: plan.length })
  }
  return plan
}

let visibilityChanges = 0
document.addEventListener('visibilitychange', () => { visibilityChanges++ })

async function runNext(session) {
  const cfg = session.pending.shift()
  // Persisted before running: if the tab crashes (out of memory on phones is common) the
  // reload finds it here and records a crash instead of retrying the same config forever.
  session.inFlight = cfg
  await setItem('session', session)
  statusEl.textContent = `Running ${session.results.length + 1}/${session.results.length + session.pending.length + 1}: ${configLabel(cfg)} (rep ${cfg.rep + 1})`
  log(`[${cfg.order + 1}] ${configLabel(cfg)}, rep ${cfg.rep + 1}`)

  // Predictions do not change between repetitions, so by default accuracy is measured once.
  const wantAccuracy = cfg.rep === 0 || !session.protocol.accuracyFirstRepOnly
  const dataset = !wantAccuracy ? null : await loadDataset(session.protocol.dataset, session.protocol.datasetLimit || Infinity).catch((err) => {
    log(`  dataset unavailable: ${err.message}`)
    return null
  })
  if (wantAccuracy && !dataset) log('  no dataset manifest found, accuracy pass skipped')

  const visibilityBefore = visibilityChanges
  const result = await runConfig(cfg, {
    latencyInput: syntheticInput(session.protocol.seed),
    dataset,
    protocol: session.protocol,
    log,
  })
  // A backgrounded tab is throttled by the browser, which invalidates timings.
  result.tabWasHidden = visibilityChanges !== visibilityBefore || document.hidden
  if (result.tabWasHidden) log('  WARNING: tab was hidden during this run; flagged as invalid')
  session.results.push(result)
  session.inFlight = null
  await setItem('session', session)
  renderTable(session)
}

async function cooldown(seconds) {
  for (let s = seconds; s > 0 && !stopRequested; s--) {
    statusEl.textContent = `Cooling down (${s}s)`
    await new Promise((r) => setTimeout(r, 1000))
  }
}

async function drive(session) {
  startBtn.disabled = true
  stopBtn.disabled = false
  while (session.pending.length && !stopRequested) {
    if (session.results.length) await cooldown(session.protocol.cooldownSec)
    if (stopRequested) break
    await runNext(session)
    if (session.protocol.isolate && session.pending.length && !stopRequested) {
      // Fresh JS heap and GPU context for the next configuration.
      location.reload()
      return
    }
  }
  session.running = false
  session.finishedAt = new Date().toISOString()
  await setItem('session', session)
  statusEl.textContent = stopRequested ? 'Stopped' : `Done: ${session.results.length} configurations.`
  if (!stopRequested) await upload(session)
  startBtn.disabled = false
  stopBtn.disabled = true
  stopRequested = false
}

async function start() {
  const protocol = readProtocol()
  if (!buildPlan(protocol).length) {
    statusEl.textContent = 'Select at least one backend/execution provider and one model.'
    return
  }
  logEl.textContent = ''
  const session = {
    schemaVersion: SCHEMA_VERSION,
    sessionId: crypto.randomUUID(),
    startedAt: new Date().toISOString(),
    protocol,
    environment: await collectEnvironment(),
    pending: buildPlan(protocol),
    results: [],
    running: true,
  }
  log(`Session ${session.sessionId}: ${session.pending.length} configurations planned`)
  if (!session.environment.crossOriginIsolated) log('Note: page is not cross-origin isolated, so WASM will run single-threaded')
  await setItem('session', session)
  drive(session)
}

function fmt(x, d = 2) {
  return x == null || Number.isNaN(x) ? '' : Number(x).toFixed(d)
}

function renderTable(session) {
  const table = document.getElementById('results')
  if (!session?.results.length) { table.innerHTML = ''; return }
  const head = ['Backend', 'Model', 'Dtype', 'Rep', 'Load (ms)', 'Source', 'Cold (ms)', 'Median (ms)', '95% CI', 'p95 (ms)', 'Kernel (ms)', 'Top-1', 'Top-5', 'Notes']
  const rows = session.results.map((r) => [
    r.backend, r.model, r.dtype ?? 'fp32', r.rep + 1,
    fmt(r.load?.ms, 0), r.load?.source ?? '', fmt(r.coldMs, 1),
    fmt(r.latency?.median), r.latency ? `${fmt(r.latency.medianCI95[0])} to ${fmt(r.latency.medianCI95[1])}` : '',
    fmt(r.latency?.p95), fmt(r.kernelMs),
    r.accuracy ? `${fmt(100 * r.accuracy.top1)}%` : '', r.accuracy ? `${fmt(100 * r.accuracy.top5)}%` : '',
    [r.status !== 'ok' && r.status, r.tabWasHidden && 'tab hidden', r.leakedTensors > 0 && `${r.leakedTensors} leaked tensors`,
      r.backendDetails?.simd != null && `simd=${r.backendDetails.simd} threads=${r.backendDetails.multithread}`,
      r.error && r.error.split('\n')[0].slice(0, 80)].filter(Boolean).join('; '),
  ])
  table.innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody>`
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function filename(session, ext) {
  const label = (session.protocol.deviceLabel || 'device').replace(/[^\w-]+/g, '_')
  return `bench_${label}_${session.startedAt.slice(0, 19).replace(/[:T]/g, '-')}.${ext}`
}

function summaryCsv(session) {
  const cols = ['session_id', 'device_label', 'power', 'engine', 'backend', 'model', 'dtype', 'rep', 'order', 'status', 'tab_hidden', 'wasm_simd', 'wasm_threads',
    'load_ms', 'load_source', 'preprocess_ms', 'cold_ms', 'n', 'median_ms', 'median_ci_lo', 'median_ci_hi', 'mean_ms', 'sd_ms', 'p05_ms', 'p95_ms', 'kernel_ms',
    'top1', 'top5', 'n_images', 'mem_bytes', 'leaked_tensors']
  const esc = (v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v))
  const lines = session.results.map((r) => [
    session.sessionId, session.protocol.deviceLabel, session.protocol.power, r.engine ?? 'tfjs', r.backend, r.model, r.dtype ?? 'fp32', r.rep, r.order, r.status, r.tabWasHidden,
    r.backendDetails?.simd, r.backendDetails?.multithread, r.load?.ms, r.load?.source, r.preprocessMs, r.coldMs, r.latency?.n, r.latency?.median,
    r.latency?.medianCI95[0], r.latency?.medianCI95[1], r.latency?.mean, r.latency?.sd, r.latency?.p05, r.latency?.p95, r.kernelMs,
    r.accuracy?.top1, r.accuracy?.top5, r.accuracy?.n, r.memory?.numBytes, r.leakedTensors,
  ].map(esc).join(','))
  return [cols.join(','), ...lines].join('\n')
}

// Sends the session to the collection server (vite dev/preview). Static hosts have no endpoint,
// in which case the JSON has to be downloaded manually.
async function upload(session) {
  try {
    const res = await fetch('/__results', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(session) })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const { saved } = await res.json()
    log(`Uploaded to collection server: ${saved}`)
    statusEl.textContent += ` Saved on server as ${saved}.`
  } catch {
    log('No collection server reachable; use "Download JSON" to keep the results.')
  }
}

startBtn.addEventListener('click', start)
stopBtn.addEventListener('click', async () => {
  stopRequested = true
  const session = await getItem('session')
  if (session) { session.running = false; await setItem('session', session) }
  statusEl.textContent = 'Stopping after the current configuration...'
})
document.getElementById('download-json').addEventListener('click', async () => {
  const session = await getItem('session')
  if (session) download(filename(session, 'json'), JSON.stringify(session), 'application/json')
})
document.getElementById('upload').addEventListener('click', async () => {
  const session = await getItem('session')
  if (session) upload(session)
})
document.getElementById('download-csv').addEventListener('click', async () => {
  const session = await getItem('session')
  if (session) download(filename(session, 'csv'), summaryCsv(session), 'text/csv')
})
document.getElementById('clear-cache').addEventListener('click', async () => {
  await clearModelCache()
  await clearOrtCache()
  log('Cleared on-device model caches (IndexedDB + Cache API); next loads will come from the network')
})
document.getElementById('reset').addEventListener('click', async () => {
  if (!confirm('Discard all stored results from this browser?')) return
  await removeItem('session')
  renderTable(null)
  logEl.textContent = ''
  statusEl.textContent = 'Idle'
})

// Resume an in-progress isolated session after its reload, or show the last results.
const existing = await getItem('session')
if (existing) {
  renderTable(existing)
  if (existing.inFlight) {
    const cfg = existing.inFlight
    log(`Previous run of ${configLabel(cfg)} did not finish (tab crashed or was closed); recorded as crashed`)
    existing.results.push({ engine: cfg.engine, backend: cfg.backend, device: cfg.device, model: cfg.model, dtype: cfg.dtype,
      rep: cfg.rep, order: cfg.order, status: 'crashed', finishedAt: new Date().toISOString() })
    existing.inFlight = null
    await setItem('session', existing)
    renderTable(existing)
  }
  if (existing.running && existing.pending.length) {
    log(`Resuming session ${existing.sessionId}: ${existing.pending.length} configurations left`)
    drive(existing)
  } else {
    statusEl.textContent = `Last session: ${existing.results.length} configurations`
  }
}
if (params.get('auto') === '1' && !existing?.running) start()
