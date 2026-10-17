import { mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { defineConfig } from 'vite'

// Cross-origin isolation enables SharedArrayBuffer, which the multi-threaded WASM backend needs.
// `credentialless` keeps cross-origin fonts and model weights loadable without CORP headers.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

// Collection endpoint for multi-device campaigns: devices that finish a benchmark POST their
// session JSON here and it is written to ./results. Only exists on `vite` / `vite preview`.
const MAX_BYTES = 256 * 2 ** 20
function resultsCollector() {
  const handler = (req, res, next) => {
    if (req.url !== '/__results' || req.method !== 'POST') return next()
    const chunks = []
    let size = 0
    req.on('data', (c) => {
      size += c.length
      if (size > MAX_BYTES) req.destroy()
      else chunks.push(c)
    })
    req.on('end', async () => {
      try {
        const session = JSON.parse(Buffer.concat(chunks).toString('utf8'))
        const label = String(session.protocol?.deviceLabel || 'device').replace(/[^\w-]+/g, '_').slice(0, 60)
        const id = String(session.sessionId || Date.now()).replace(/[^\w-]+/g, '')
        const dir = resolve(import.meta.dirname, 'results')
        await mkdir(dir, { recursive: true })
        const file = `${label}_${id}.json`
        await writeFile(resolve(dir, file), JSON.stringify(session))
        res.writeHead(200, { 'Content-Type': 'application/json' }).end(JSON.stringify({ saved: `results/${file}` }))
      } catch (err) {
        res.writeHead(400).end(String(err))
      }
    })
  }
  return {
    name: 'results-collector',
    configureServer: (server) => { server.middlewares.use(handler) },
    configurePreviewServer: (server) => { server.middlewares.use(handler) },
  }
}

export default defineConfig({
  plugins: [resultsCollector()],
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  build: {
    // The threaded WASM backend stringifies its Emscripten factory into a worker; minification
    // renames the factory's closure variables (e.g. `_scriptDir`) and the worker then crashes.
    minify: false,
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        bench: resolve(import.meta.dirname, 'bench.html'),
      },
    },
  },
})
