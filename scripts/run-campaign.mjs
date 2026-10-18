// Drives a full benchmark session in an installed Chromium-based browser and waits until the
// session JSON lands in ./results via the collection endpoint of `vite preview`.
//
//   node scripts/run-campaign.mjs --browser chrome --label D1-Chrome --base http://localhost:4317
//
// A fresh, throwaway profile is used for every run, so weight caches start cold and no personal
// browser data is involved. Background throttling is disabled so that an occluded window does not
// invalidate timings (the page still flags runs where the tab became hidden).
import { spawn } from 'node:child_process'
import { existsSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import puppeteer from 'puppeteer-core'

const BROWSERS = {
  chrome: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  edge: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
}

// Accepts "--key value" and "--key=value"; the latter is needed when the value itself starts with
// "--", e.g. --browser-args="--use-angle=swiftshader --enable-unsafe-swiftshader".
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (!a.startsWith('--')) return acc
  const eq = a.indexOf('=')
  if (eq > 0) acc.push([a.slice(2, eq), a.slice(eq + 1)])
  else acc.push([a.slice(2), all[i + 1]?.startsWith('--') ? 'true' : all[i + 1]])
  return acc
}, []))
const browserName = args.browser || 'chrome'
const label = args.label || `D1-${browserName}`
const base = args.base || 'http://localhost:4317'
const extra = args.params || ''
const timeoutHours = Number(args.timeout || 12)

const root = resolve(import.meta.dirname, '..')
const resultsDir = join(root, 'results')
// --profile <dir> reopens an existing profile: the page then resumes its stored session (e.g. after
// the browser exited), recording the configuration that was in flight as crashed.
const profileDir = args.profile ? resolve(args.profile) : join(root, '.bench-profiles', `${browserName}-${Date.now()}`)
mkdirSync(profileDir, { recursive: true })
mkdirSync(resultsDir, { recursive: true })

const stamp = () => new Date().toISOString().slice(11, 19)
const log = (msg) => console.log(`${stamp()} [${label}] ${msg}`)
const resultFiles = () => new Set(readdirSync(resultsDir).filter((f) => f.startsWith(`${label}_`)))
const before = resultFiles()

const browserArgs = [
  '--no-first-run',
  '--no-default-browser-check',
  '--start-maximized',
  '--disable-backgrounding-occluded-windows',
  '--disable-renderer-backgrounding',
  '--disable-background-timer-throttling',
  '--disable-features=CalculateNativeWinOcclusion',
  // Emulated device classes: e.g. --browser-args "--disable-gpu" for a device without a usable GPU.
  ...(args['browser-args'] ? args['browser-args'].split(' ').filter(Boolean) : []),
]

// --affinity <hex mask> restricts the whole browser (all child processes inherit the mask) to the
// given logical processors, emulating a CPU with fewer cores. Windows only: launched via `start /affinity`.
async function launchBrowser() {
  if (!args.affinity) {
    return puppeteer.launch({ executablePath: BROWSERS[browserName], headless: false, userDataDir: profileDir, defaultViewport: null, args: browserArgs })
  }
  const port = 9300 + Math.floor(Math.random() * 500)
  spawn('cmd.exe', ['/c', 'start', '""', '/affinity', args.affinity, BROWSERS[browserName],
    `--user-data-dir=${profileDir}`, `--remote-debugging-port=${port}`, ...browserArgs, 'about:blank'], { detached: true, stdio: 'ignore' }).unref()
  for (let i = 0; i < 60; i++) {
    try {
      return await puppeteer.connect({ browserURL: `http://127.0.0.1:${port}`, defaultViewport: null })
    } catch {
      await new Promise((r) => setTimeout(r, 1000))
    }
  }
  throw new Error('browser did not start with the requested affinity')
}

const browser = await launchBrowser()
const version = await browser.version()
log(`launched ${version}, profile ${profileDir}`)

const [page] = await browser.pages()

// --cores N makes navigator.hardwareConcurrency report N, so that thread pools (TF.js and ONNX Runtime
// WebAssembly) are sized as on a real N-core device; use together with --affinity for a faithful emulation.
if (args.cores) {
  const cdp = await page.createCDPSession()
  await cdp.send('Emulation.setHardwareConcurrencyOverride', { hardwareConcurrency: Number(args.cores) })
  log(`hardwareConcurrency overridden to ${args.cores}`)
}
const url = `${base}/bench.html?auto=1&deviceLabel=${encodeURIComponent(label)}&power=${encodeURIComponent('plugged in')}${extra ? `&${extra}` : ''}`

// A crashed renderer (e.g. out of memory) is reloaded; the page records the in-flight config
// as crashed and resumes the session.
page.on('error', async (err) => {
  log(`renderer crashed: ${err.message}; reloading`)
  try { await page.reload({ waitUntil: 'domcontentloaded' }) } catch (e) { log(`reload failed: ${e.message}`) }
})
await page.goto(url, { waitUntil: 'domcontentloaded' })
log(`started ${url}`)

const deadline = Date.now() + timeoutHours * 3600e3
let lastStatus = ''
let exitCode = 0
for (;;) {
  const fresh = [...resultFiles()].filter((f) => !before.has(f))
  if (fresh.length) {
    const file = join(resultsDir, fresh[0])
    log(`finished: results/${fresh[0]} (${(statSync(file).size / 1e6).toFixed(1)} MB)`)
    break
  }
  if (Date.now() > deadline) {
    log('TIMEOUT waiting for results')
    exitCode = 1
    break
  }
  try {
    const status = await page.$eval('#status', (el) => el.textContent)
    if (status !== lastStatus) { log(status); lastStatus = status }
  } catch { /* page is between reloads */ }
  await new Promise((r) => setTimeout(r, 15000))
}
await browser.close()
if (existsSync(profileDir)) log(`profile kept at ${profileDir} (delete when done)`)
process.exit(exitCode)
