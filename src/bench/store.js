// Tiny IndexedDB key-value store. The benchmark session lives here (not in memory) so that
// "isolated" mode can reload the page between configurations without losing results.
const DB = 'web-edge-bench'
const STORE = 'kv'

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

async function tx(mode, fn) {
  const db = await open()
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode)
    const req = fn(t.objectStore(STORE))
    t.oncomplete = () => { db.close(); resolve(req?.result) }
    t.onerror = () => { db.close(); reject(t.error) }
  })
}

export const getItem = (key) => tx('readonly', (s) => s.get(key))
export const setItem = (key, value) => tx('readwrite', (s) => s.put(value, key))
export const removeItem = (key) => tx('readwrite', (s) => s.delete(key))
