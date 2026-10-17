// Drives a full benchmark session in an installed Chromium-based browser and waits until the
// session JSON lands in ./results via the collection endpoint of `vite preview`.
//
//   node scripts/run-campaign.mjs --browser chrome --label D1-Chrome --base http://localhost:4317
//
// A fresh, throwaway profile is used for every run, so weight caches start cold and no personal
// browser data is involved. Background throttling is disabled so that an occluded window does not
// invalidate timings (the page still flags runs where the tab became hidden).
import { existsSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'
import puppeteer from 'puppeteer-core'

const BROWSERS = {
  chrome: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  edge: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
}

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1]?.startsWith('--') ? 'true' : all[i + 1]])
  return acc
}, []))
const browserName = args.browser || 'chrome'
const label = args.label || `D1-${browserName}`
const base = args.base || 'http://localhost:4317'
const extra = args.params || ''
const timeoutHours = Number(args.timeout || 12)

const root = resolve(import.meta.dirname, '..')
const resultsDir = join(root, 'results')
const profileDir = join(root, '.bench-profiles', `${browserName}-${Date.now()}`)
mkdirSync(profileDir, { recursive: true })
mkdirSync(resultsDir, { recursive: true })

const stamp = () => new Date().toISOString().slice(11, 19)
const log = (msg) => console.log(`${stamp()} [${label}] ${msg}`)
const resultFiles = () => new Set(readdirSync(resultsDir).filter((f) => f.startsWith(`${label}_`)))
const before = resultFiles()

const browser = await puppeteer.launch({
  executablePath: BROWSERS[browserName],
  headless: false,
  userDataDir: profileDir,
  defaultViewport: null,
  args: [
    '--no-first-run',
    '--no-default-browser-check',
    '--start-maximized',
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
    '--disable-background-timer-throttling',
    '--disable-features=CalculateNativeWinOcclusion',
  ],
})
const version = await browser.version()
log(`launched ${version}, profile ${profileDir}`)

const [page] = await browser.pages()
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
