// Precomputes CLIP text embeddings for the bird labels, so the browser only needs CLIP's image encoder
// (53 MB instead of an extra 72 MB text encoder). Writes src/birds-embeddings.json.
//   node scripts/precompute-bird-embeddings.mjs
import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { AutoTokenizer, CLIPTextModelWithProjection } from '@huggingface/transformers'
import { BIRD_PROMPT, BIRDS, NOT_BIRD_PROMPTS } from '../src/birds-labels.js'

const REPO = 'Xenova/clip-vit-base-patch32'
const prompts = [...BIRDS.map(BIRD_PROMPT), ...NOT_BIRD_PROMPTS]
const tokenizer = await AutoTokenizer.from_pretrained(REPO)
const model = await CLIPTextModelWithProjection.from_pretrained(REPO, { dtype: 'fp32' })
const { text_embeds: e } = await model(tokenizer(prompts, { padding: true, truncation: true }))
const [n, d] = e.dims
const out = new Float32Array(n * d)
for (let i = 0; i < n; i++) {
  let norm = 0
  for (let k = 0; k < d; k++) norm += e.data[i * d + k] ** 2
  norm = Math.sqrt(norm)
  for (let k = 0; k < d; k++) out[i * d + k] = e.data[i * d + k] / norm
}
const json = {
  model: REPO,
  dim: d,
  birds: BIRDS.length,
  notBird: NOT_BIRD_PROMPTS.length,
  embeddings: Buffer.from(out.buffer).toString('base64'),
}
writeFileSync(resolve(import.meta.dirname, '../src/birds-embeddings.json'), JSON.stringify(json))
console.log(`wrote ${n} embeddings of dimension ${d}`)
