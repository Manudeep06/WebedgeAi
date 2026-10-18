import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './app.css'

// Only image classification is backed by a real model. Text and audio modes were removed
// because they returned hard-coded outputs, which must never appear in reported results.
const modes = [
  { id: 'image', label: 'Image', note: 'Drop an image to classify', accept: 'image/*' },
]
const colors = ['coral', 'blue', 'gold']
const tickerItems = ['On-device inference', 'Zero upload', 'MobileNet v2', 'TensorFlow.js', 'WebGL backend', 'Private by default']

function NeuralField() {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const pointer = { x: -9999, y: -9999 }
    let nodes = []
    let frame = 0
    let width = 0
    let height = 0

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      width = canvas.clientWidth
      height = canvas.clientHeight
      canvas.width = width * dpr
      canvas.height = height * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const count = Math.round(Math.min(90, (width * height) / 14000))
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * width, y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.35, vy: (Math.random() - 0.5) * 0.35,
      }))
    }
    function draw() {
      ctx.clearRect(0, 0, width, height)
      for (const n of nodes) {
        if (!still) {
          n.x += n.vx
          n.y += n.vy
          if (n.x < 0 || n.x > width) n.vx *= -1
          if (n.y < 0 || n.y > height) n.vy *= -1
        }
        const dx = n.x - pointer.x
        const dy = n.y - pointer.y
        const d = Math.hypot(dx, dy)
        if (d > 0 && d < 140) {
          n.x += (dx / d) * 0.8
          n.y += (dy / d) * 0.8
        }
      }
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i]
          const b = nodes[j]
          const d = Math.hypot(a.x - b.x, a.y - b.y)
          if (d < 130) {
            ctx.strokeStyle = `hsla(${(10 + (a.x / width) * 250) % 360}, 95%, 65%, ${(1 - d / 130) * 0.35})`
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(a.x, a.y)
            ctx.lineTo(b.x, b.y)
            ctx.stroke()
          }
        }
      }
      for (const n of nodes) {
        const near = Math.hypot(n.x - pointer.x, n.y - pointer.y) < 160
        ctx.fillStyle = near ? '#ffffff' : 'rgba(255,255,255,.55)'
        ctx.beginPath()
        ctx.arc(n.x, n.y, near ? 2.4 : 1.4, 0, Math.PI * 2)
        ctx.fill()
      }
      if (!still) frame = requestAnimationFrame(draw)
    }
    function onResize() {
      resize()
      if (still) draw()
    }
    function onMove(event) {
      const rect = canvas.getBoundingClientRect()
      pointer.x = event.clientX - rect.left
      pointer.y = event.clientY - rect.top
    }
    function onLeave() {
      pointer.x = -9999
      pointer.y = -9999
    }

    resize()
    draw()
    window.addEventListener('resize', onResize)
    window.addEventListener('pointermove', onMove)
    document.addEventListener('pointerleave', onLeave)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', onResize)
      window.removeEventListener('pointermove', onMove)
      document.removeEventListener('pointerleave', onLeave)
    }
  }, [])
  return <canvas ref={canvasRef} className="neural-field" aria-hidden="true" />
}

function trackPointer(event) {
  document.documentElement.style.setProperty('--cx', `${event.clientX}px`)
  document.documentElement.style.setProperty('--cy', `${event.clientY}px`)
  const card = event.target.closest('.glow-card')
  if (card) {
    const rect = card.getBoundingClientRect()
    card.style.setProperty('--x', `${event.clientX - rect.left}px`)
    card.style.setProperty('--y', `${event.clientY - rect.top}px`)
  }
}

let modelPromise
function loadImageModel() {
  // Weights are fetched once and cached by the browser; images never leave the tab.
  modelPromise ??= Promise.all([import('@tensorflow/tfjs'), import('@tensorflow-models/mobilenet')])
    .then(async ([tf, mobilenet]) => {
      await tf.ready()
      return { model: await mobilenet.load({ version: 2, alpha: 1.0 }), backend: tf.getBackend() }
    })
    .catch((err) => { modelPromise = undefined; throw err })
  return modelPromise
}

// One "Image" option that picks the model automatically:
//  1. every photo goes to ViT-B/16 (ONNX Runtime Web; 66.0% top-1 end to end on 200 ImageNetV2 photos,
//     against 57.5% for the previous MobileNetV2);
//  2. if ViT puts at least BIRD_ROUTE of its probability on ImageNet's 59 bird classes, the CLIP bird
//     expert (127 names) is asked too;
//  3. CLIP's answer replaces ViT's only when it names a species ImageNet lacks (kingfisher, myna, pigeon…)
//     with at least CLIP_ACCEPT confidence. For ImageNet's own species ViT is more accurate (81.9% vs 62.5%).
// Hidden overrides for measurement: ?model=accurate | birds | fast, ?mirror=1.
const urlParams = new URLSearchParams(location.search)
const ENGINE = ['accurate', 'birds', 'fast'].includes(urlParams.get('model')) ? urlParams.get('model') : 'auto'
const MIRROR = urlParams.get('mirror') === '1'
const LOW_CONFIDENCE = 30
const BIRD_ROUTE = 0.5
const CLIP_ACCEPT = 0.5

let accurateModule
let birdModule
const loadAccurate = () => (accurateModule ??= import('./accurate.js'))
const loadBirds = () => (birdModule ??= Promise.all([import('./birds-clip.js'), import('./birds-labels.js')])
  .then(([clip, labels]) => ({ ...clip, ...labels })))
const capitalise = (s) => s[0].toUpperCase() + s.slice(1)
const toOutputs = (top) => top.map((p, i) => ({ label: capitalise(p.name), confidence: Math.round(p.probability * 100), color: colors[i] }))

async function loadEngine(engine) {
  if (engine === 'fast') {
    const { backend } = await loadImageModel()
    return `${backend.toUpperCase()} ready`
  }
  if (engine === 'birds') {
    const { loadBirdModel } = await loadBirds()
    const { plan } = await loadBirdModel()
    return `CLIP · ${plan.device.toUpperCase()} ready`
  }
  const { loadAccurateModel } = await loadAccurate()
  const { plan } = await loadAccurateModel()
  // Fetch the bird expert in the background, so the first bird photo does not wait for its download.
  if (engine === 'auto') loadBirds().then((m) => m.loadBirdModel()).catch(() => {})
  return `${plan.device.toUpperCase()} ready`
}

async function birdAnswer(file) {
  const { classifyBird, IMAGENET_COVERED, NOT_BIRD_LABEL } = await loadBirds()
  const r = await classifyBird(file)
  const best = r.top[0]
  const usable = best.name !== NOT_BIRD_LABEL && !IMAGENET_COVERED.has(best.name) && best.probability >= CLIP_ACCEPT
  return { r, usable }
}

async function classifyWith(engine, file) {
  if (engine === 'fast') return classifyImage(file)
  if (engine === 'birds') {
    const { r } = await birdAnswer(file)
    return { ms: r.ms, note: `CLIP ViT-B/32 ${r.plan.dtype} / ${r.plan.device} · ${r.birds} bird names`, outputs: toOutputs(r.top) }
  }
  const { classifyAccurate } = await loadAccurate()
  const v = await classifyAccurate(file, { mirror: MIRROR })
  const vit = { ms: v.ms, note: `ViT-B/16 ${v.plan.dtype} / ${v.plan.device}${v.mirror ? ' · mirror-averaged' : ''}`, outputs: toOutputs(v.top) }
  if (engine !== 'auto' || v.birdMass < BIRD_ROUTE) return vit
  try {
    const { r, usable } = await birdAnswer(file)
    if (!usable) return { ...vit, ms: vit.ms + r.ms, note: `${vit.note} · bird expert agreed` }
    return { ms: vit.ms + r.ms, note: `Bird expert: CLIP ViT-B/32 ${r.plan.dtype} / ${r.plan.device} (species not in ImageNet)`, outputs: toOutputs(r.top) }
  } catch {
    return vit // the bird expert is an extra; never fail the main answer because of it
  }
}

async function classifyImage(file) {
  const { model, backend } = await loadImageModel()
  const url = URL.createObjectURL(file)
  try {
    const img = new Image()
    img.src = url
    await img.decode()
    const start = performance.now()
    const predictions = await model.classify(img, 3)
    return {
      ms: Math.round(performance.now() - start),
      note: `MobileNet v2 / ${backend}`,
      outputs: predictions.map((p, i) => {
        const name = p.className.split(',')[0]
        return { label: name[0].toUpperCase() + name.slice(1), confidence: Math.round(p.probability * 100), color: colors[i] }
      }),
    }
  } finally {
    URL.revokeObjectURL(url)
  }
}

function App() {
  const [mode] = useState('image')
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [file, setFile] = useState(null)
  const [text, setText] = useState('')
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const [modelStatus, setModelStatus] = useState('Loading model')
  const [preview, setPreview] = useState('')
  const fileInput = useRef(null)
  const runId = useRef(0)
  const currentMode = modes.find((item) => item.id === mode)
  const hasInput = mode === 'text' ? text.trim().length > 0 || file : file
  const statusTone = /ready/i.test(modelStatus) ? 'ok' : /unavailable/i.test(modelStatus) ? 'bad' : 'loading'

  useEffect(() => {
    let current = true
    setModelStatus('Loading model')
    loadEngine(ENGINE)
      .then((status) => { if (current) setModelStatus(status) })
      .catch(() => { if (current) setModelStatus('Model unavailable') })
    return () => { current = false }
  }, [])

  useEffect(() => {
    if (!file || !file.type.startsWith('image/')) {
      setPreview('')
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function acceptFile(selected) {
    if (!selected) return
    const [type] = currentMode.accept.split(',').at(-1).split('/')
    if (!selected.type.startsWith(type + '/')) {
      setError(`Please choose a ${mode} file.`)
      return
    }
    setError('')
    setFile(selected)
    setResult(null)
  }
  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    acceptFile(event.dataTransfer.files[0])
  }
  function runInference() {
    if (!hasInput) {
      setError(mode === 'text' ? 'Enter some text first.' : `Choose a ${mode} file first.`)
      return
    }
    setError('')
    setRunning(true)
    setResult(null)
    const id = ++runId.current
    classifyWith(ENGINE, file)
      .then((next) => { if (id === runId.current) setResult(next) })
      .catch(() => { if (id === runId.current) setError('Could not run the model. Check your connection or try another image.') })
      .finally(() => { if (id === runId.current) setRunning(false) })
  }
  return (
    <main className="app-shell" onPointerMove={trackPointer}>
      <div className="bg-aurora" aria-hidden="true"><span /><span /><span /></div>
      <div className="bg-grid" aria-hidden="true" />
      <div className="bg-grain" aria-hidden="true" />
      <div className="cursor-glow" aria-hidden="true" />

      <nav className="topbar">
        <a className="brand" href="#top" aria-label="Local home"><span className="brand-mark">L/</span><span>LOCAL</span></a>
        <div className="nav-context"><span className="live-dot" /> Inference lab <span className="slash">/</span> browser runtime</div>
        <div className="nav-right"><span className={`status-pill ${statusTone}`}><i />{modelStatus}</span><span className="avatar">ML</span></div>
      </nav>

      <section className="hero" id="top">
        <NeuralField />
        <div className="hero-copy">
          <p className="eyebrow"><span className="eyebrow-index">01</span> On-device machine learning</p>
          <h1><span className="line">Intelligence,</span><span className="line"><em>kept close.</em></span></h1>
          <p className="hero-lede">A lightweight inference studio for the web. Explore what your browser can understand without sending raw data to a server.</p>
          <div className="hero-actions">
            <a className="cta" href="#workspace"><span>Launch the lab</span><span className="cta-arrow">→</span></a>
            <a className="hero-hint" href="/bench.html">Run the benchmark →</a>
          </div>
          <div className="hero-meta"><span>TensorFlow.js</span><span>MobileNet v2</span><span>Runs in your tab</span></div>
        </div>
        <div className="hero-core" aria-hidden="true">
          <div className="ring ring-1" />
          <div className="ring ring-2" />
          <div className="ring ring-3" />
          <div className="core"><span className="core-dot" /><span className="core-label">LOCAL<br />MODEL</span></div>
          <span className="graphic-label label-a">NO RAW DATA<br />LEAVES DEVICE</span>
          <span className="graphic-label label-b">LOW LATENCY<br />REAL-TIME OUTPUT</span>
        </div>
      </section>

      <div className="marquee" aria-hidden="true">
        <div className="marquee-track">
          {[0, 1].map((copy) => <span key={copy}>{tickerItems.map((item) => <b key={item}>{item}</b>)}</span>)}
        </div>
      </div>

      <section className="workspace-section" id="workspace">
        <div className="section-heading reveal">
          <div><p className="eyebrow"><span className="eyebrow-index">02</span> Try the pipeline</p><h2>Inference <em>workspace</em></h2></div>
          <p className="section-note">Your input is processed in this tab.<br />Nothing is uploaded.</p>
        </div>
        <div className="workspace-grid reveal">
          <div className="input-panel panel glow-card">
            <div className="panel-top"><span className="panel-kicker">01 / Input</span><span className="lock">Private by default</span></div>
            <div className="mode-tabs">
              {modes.map((item) => <button key={item.id} className={mode === item.id ? 'active' : ''}>{item.label}</button>)}
            </div>
            <div className={`drop-zone ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={handleDrop}>
              {preview ? (
                <div className={`preview ${running ? 'scanning' : ''}`}><img src={preview} alt={`Selected: ${file.name}`} /></div>
              ) : (
                <>
                  <div className="drop-icon">{mode === 'image' ? '[ + ]' : mode === 'text' ? 'T_' : ')))'}</div>
                  <strong>{mode === 'image' ? 'Drop an image here' : mode === 'text' ? 'Write or paste text' : 'Choose an audio clip'}</strong>
                </>
              )}
              <span className="drop-note">{file ? `Selected: ${file.name}` : currentMode.note}</span>
              {mode === 'text' && <textarea aria-label="Text input" placeholder="The model stays on this device..." value={text} onChange={(event) => { setText(event.target.value); setResult(null) }} />}
              <input ref={fileInput} type="file" accept={currentMode.accept} hidden onChange={(event) => { acceptFile(event.target.files[0]); event.target.value = '' }} />
              <button className="choose-button" onClick={() => fileInput.current.click()}>{file ? 'Replace file' : 'Choose file'}</button>
              {error && <span className="input-error" role="alert">{error}</span>}
            </div>
            <button className="run-button" onClick={runInference} disabled={running}>{running ? 'Running local model...' : 'Run inference'} <span>{running ? '...' : '→'}</span></button>
          </div>

          <div className="result-panel panel glow-card">
            <div className="panel-top">
              <span className="panel-kicker">02 / Output</span>
              <span className={`result-state ${running ? 'busy' : result ? 'done' : ''}`}><i /> {running ? 'Inferring' : result ? 'Complete' : 'Awaiting input'}</span>
            </div>
            {!result ? (
              <div className="empty-result">
                <div className={`radar ${running ? 'fast' : ''}`} />
                <p>{running ? 'Analyzing...' : 'Your prediction will appear here'}</p>
                <span>{running ? 'Running entirely in this tab' : 'Run the model to see confidence scores'}</span>
              </div>
            ) : (
              <div className="result-content">
                <div className="result-summary">
                  <div className="result-badge">{result.outputs[0].confidence}%</div>
                  <div><span>Top prediction</span><h3>{result.outputs[0].label}</h3></div>
                </div>
                {result.outputs[0].confidence < LOW_CONFIDENCE && (
                  <p className="low-confidence" role="status">Not sure: the model is under {LOW_CONFIDENCE}% confident. Try a sharper, upright photo where the object fills most of the frame.</p>
                )}
                {result.outputs.map((output) => (
                  <div className="score-row" key={output.label}>
                    <span>{output.label}</span>
                    <div className="score-track"><div className={`score-fill ${output.color}`} style={{ width: `${output.confidence}%` }} /></div>
                    <b>{output.confidence}%</b>
                  </div>
                ))}
                <div className="inference-time"><span>Inference time</span><strong>{result.ms === null ? '--' : `${result.ms} ms`}</strong><span className="time-good">{result.note}</span></div>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="signals-section">
        <div className="section-heading reveal">
          <div><p className="eyebrow"><span className="eyebrow-index">03</span> Why local matters</p><h2>Small model.<br /><em>Big difference.</em></h2></div>
        </div>
        <div className="signals-grid reveal">
          <article className="glow-card"><span className="signal-number">01</span><h3>Private by design</h3><p>Images are classified in the tab. Only the model weights are downloaded; your input never is uploaded.</p></article>
          <article className="glow-card"><span className="signal-number">02</span><h3>Responsive by default</h3><p>WebGL acceleration turns a browser tab into a fast, always-available inference surface.</p></article>
          <article className="glow-card"><span className="signal-number">03</span><h3>Measured, not claimed</h3><p>Latency, accuracy and cold-start cost are measured per backend and device in the <a href="/bench.html">benchmark</a>.</p></article>
        </div>
      </section>

      <footer>
        <div className="footer-word" aria-hidden="true">LOCAL</div>
        <div className="footer-row"><span>LOCAL / BROWSER ML STUDY</span><span>Built for the edge <span className="footer-mark">+</span></span><span>v0.1.0</span></div>
      </footer>
    </main>
  )
}
createRoot(document.getElementById('app')).render(<StrictMode><App /></StrictMode>)
