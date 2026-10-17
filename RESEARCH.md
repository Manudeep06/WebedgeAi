# Web Edge AI: research plan and benchmark protocol

This repository contains (1) a demo of on-device image classification in the browser and
(2) a measurement instrument (`bench.html`) for an empirical study of **web-based edge AI**:
deep-learning inference executed inside the browser on the user's own device.

## 1. Honest status

The software is now a sound *instrument*. **It is not yet a paper.** A Q1 journal will judge
the study: breadth of devices, rigour of the protocol, and the insight extracted from the data.
Prior work already benchmarks in-browser DL (see §7), so the novelty must come from what is
new since then: WebGPU, WASM SIMD + threads, cache-backed cold start, numerical fidelity of
GPU backends across vendors, and a modern device fleet including phones.

Removed from the app: text and audio modes that returned **hard-coded fake predictions**.
Never report results from code paths that are not backed by a real model.

## 2. Candidate research questions

| RQ | Question | Measured by |
|----|----------|-------------|
| RQ1 | How do WebGPU, WebGL, WASM (SIMD/threads) and plain JS compare for CNN inference across heterogeneous edge devices and browsers? | warm latency (median, p95, hierarchical-bootstrap 95% CI); speed-up vs WASM; Friedman + Holm-corrected Wilcoxon across devices |
| RQ2 | What does cold start cost a web edge-AI application, and how much does on-device weight caching recover? | load time (network vs IndexedDB), first-inference time (shader compilation / JIT) |
| RQ3 | Do browser backends compute the *same* model? | top-1 agreement, top-5 overlap, max abs probability difference and KL vs the CPU reference on the same device; top-1/top-5 accuracy on an ImageNet subset |
| RQ4 | Which model/backend choices are Pareto-optimal on each device class? | accuracy-vs-latency Pareto fronts, memory footprint |
| RQ5 | Does quantization (fp16 / int8 / int4) pay off in the browser, and does it interact with the backend? | latency, download size, top-1 delta and prediction agreement vs fp32 on the same backend |
| RQ6 | How do two web ML runtimes (TensorFlow.js vs ONNX Runtime Web) and architecture families (CNN vs Vision Transformer) differ on the same hardware? | per-runtime latency, cold start, fidelity; WebNN (OS ML stack / NPU) where available |

RQ3 is the least studied and most likely to give a distinctive finding. GPU backends use
reduced precision (WebGL may use fp16 textures on mobile GPUs; WebGPU may use `shader-f16`).

## 3. Protocol implemented in `bench.html`

Per configuration *(backend, model)*:
1. Activate backend; record actual capabilities (WASM SIMD/threads, TF.js flags).
2. Load weights; record source (`network` / `indexeddb`) and time.
3. Cold inference (first call; includes shader compilation).
4. `W` warm-up inferences (default 10), then `N` timed inferences (default 50) on a fixed,
   seeded synthetic 224×224 input. Each timing is end-to-end (pixel upload → read-back,
   `await data()` forces GPU completion). Kernel time via `tf.time()` where available.
5. Accuracy pass over the labelled set: top-1/top-5, top-5 indices per image, and full
   probability vectors for the first `K` images (fidelity analysis).
6. Memory (`tf.memory()`), tensor-leak check.

Across configurations:
- `R` repetitions (default 5), each a fresh **seeded random order** of all configurations
  (spreads thermal throttling and ordering effects).
- **Isolated mode** (default): page reload between configurations, giving a fresh JS heap and GPU context.
- Cool-down between configurations (default 10 s).
- Runs where the tab was hidden (browser throttling) are flagged and excluded.
- Environment captured automatically: UA + client hints, cores, memory, GPU adapter
  (WebGPU `adapter.info`, WebGL unmasked renderer), cross-origin isolation, battery state,
  library versions. Record power state and a device label manually in the form.

Two runtimes, one protocol:

| Runtime | Backends / execution providers | Models | Precision |
|---|---|---|---|
| TensorFlow.js 4.22.0 | `webgpu`, `webgl`, `wasm` (SIMD + threads), `cpu` | MobileNet V1 (α 0.25–1.0), V2 (α 0.50–1.0) | fp32 |
| ONNX Runtime Web (via Transformers.js 4.3.0) | `webgpu`, `wasm`, `webnn-gpu`, `webnn-npu`, `webnn-cpu` | MobileNetV4-S (2024 CNN), MobileViT-S (hybrid), ResNet-50 (CNN), ViT-B/16 (transformer) | fp32, fp16, q8; plus q4 / q4f16 for the transformer models |

Timing note: TF.js latency is end-to-end (pixels → probabilities). ONNX latency is model
execution only on pre-processed input; JS pre-processing is recorded separately
(`preprocessMs`). Add it before comparing runtimes head-to-head.

Crashes (e.g. a phone tab killed for memory) are detected on reload and recorded as
`crashed`, so the campaign continues. Report crash rates: they are a finding in themselves.

## 4. Running a campaign

```bash
npm install
npm run build
npx vite preview --host        # serves the app + the /__results collection endpoint
```

Open `http://localhost:4173/bench.html` on each device, set *Device label* and *Power*,
press **Start**. On completion the JSON is written to `results/` on the serving machine
(otherwise use *Download JSON*). URL parameters can preset everything, e.g.
`bench.html?backend=webgpu,wasm&reps=10&deviceLabel=Pixel8-Chrome&auto=1`.

**Phones/tablets need HTTPS.** WebGPU and cross-origin isolation only work in a secure
context (HTTPS or `localhost`). Either deploy `dist/` to an HTTPS host that sends the COOP/COEP
headers from `vite.config.js` (results then come back via *Download JSON*), or use
USB port forwarding (Chrome `chrome://inspect` → port forwarding) so the phone sees `localhost`.

**Accuracy dataset** (not redistributable; build locally from ImageNet-1k val):

```bash
python -m venv .venv && .venv/Scripts/pip install -r analysis/requirements.txt
.venv/Scripts/python analysis/prepare_imagenet_subset.py --val-dir <path-to-val> --n 1000 --seed 20261017
```

Report the seed and the printed manifest SHA-256.

**Reproducible weights:** by default weights come from TF Hub (now redirected to Kaggle),
which can change or disappear. For the paper, place the model folders under `public/models/`
and set *Self-hosted model dir* to `/models`; archive them with the artefact.

**Analysis:**

```bash
.venv/Scripts/python analysis/analyze.py results/*.json --out analysis/out
```

Produces `latency_summary.csv`, `speedup.csv`, `fidelity.csv`, `cross_device_*.{csv,json}`,
`exclusions.csv`, LaTeX tables and PDF figures.

## 5. What a Q1-level study still needs (your work)

1. **Device fleet**: at least 10–15 devices across tiers: low/mid/high-end Android, iPhone/iPad
   (Safari: WebGPU since Safari 26), Windows/macOS/Linux laptops with Intel/AMD/NVIDIA/Apple GPUs,
   ideally a Chromebook or single-board computer. At least two browsers (Chrome/Edge, Firefox, Safari).
2. **Repetitions**: `reps ≥ 5` per device; run plugged in and on battery for a subset.
3. **Energy** (strongly recommended for an edge-AI venue): external USB power meter for phones,
   or Intel Power Gadget / `powermetrics` (macOS) / Android `batterystats`. Report J per inference.
4. **Broader workloads**: done for vision (CNN + transformer, fp32 to int4, two runtimes).
   Optionally add one language model (e.g. a small text classifier) to claim generality
   beyond vision; new models go in `src/bench/engines/ort.js` (`ORT_MODELS`).
5. **Comparison to native**: the same models in TFLite on the same phones gives the
   "web vs native gap", which reviewers will ask about.
6. **Threats to validity** section: browser version drift, thermal state, background
   processes, GPU driver differences, timer resolution (cross-origin isolation gives 5 µs
   resolution, otherwise coarser), synthetic latency input vs real images.
7. **Artefact**: archive code, weights, raw JSON and analysis on Zenodo (DOI) and cite it;
   many journals now value or require a reproducibility package.

## 6. Suggested paper structure

1. Introduction: web edge AI motivation (privacy, zero install, offline), contributions
2. Background: TF.js backends, WebGPU, WASM SIMD/threads, cross-origin isolation
3. Related work
4. Methodology: devices, models, protocol (§3), statistics (`analysis/analyze.py` docstring)
5. Results: RQ1–RQ4
6. Discussion: guidelines for developers ("which backend/model for which device class")
7. Threats to validity
8. Conclusion + artefact availability

## 7. Starting points for related work

- D. Smilkov et al., "TensorFlow.js: Machine Learning for the Web and Beyond," *SysML/MLSys* 2019.
- Y. Ma et al., "Moving Deep Learning into Web Browser: How Far Can We Go?," *WWW* 2019.

Do a systematic search (Scopus / Web of Science / Google Scholar: "in-browser deep learning",
"WebGPU inference", "web edge AI", "on-device inference benchmark") and cite recent 2023–2026
work. Your contribution must be positioned against it.

**Venue fit**: check the *current* quartile in SJR/JCR before submitting (quartiles shift
yearly). Candidates to evaluate: journals on IoT / edge computing, software systems and
empirical software engineering, and future-generation computing systems.
