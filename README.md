# WebedgeAi: LOCAL-Bench

A measurement instrument for **web edge AI**: deep-learning inference running entirely inside the
browser on the user's own device. It benchmarks two runtimes, several execution backends, eleven
ImageNet classifiers and up to five numerical precisions, under a controlled and reproducible protocol.

| Runtime | Backends | Models | Precision |
|---|---|---|---|
| TensorFlow.js 4.22 | WebGPU, WebGL, WebAssembly (SIMD + threads), plain JS | MobileNet V1 / V2 | fp32 |
| ONNX Runtime Web (Transformers.js 4.3) | WebGPU, WebAssembly, WebNN | MobileNetV4-S, MobileViT-S, ResNet-50, ViT-B/16 | fp32, fp16, int8, int4 |

It measures warm latency (with hierarchical-bootstrap confidence intervals), time to first prediction
(network vs on-device cache, first call), top-1/top-5 accuracy and numerical fidelity against a float32
reference. Validity guards: seeded randomized repetitions, page reload per configuration, cool-down,
and detection of hidden-tab (throttled) and crashed runs.

## Quick start

```bash
npm install
npm run bench:serve          # builds and serves the app + the /__results collection endpoint
```

Open `http://localhost:4173/bench.html`, set a device label, and press **Start**. The finished session
is written to `results/`. To run a whole session automatically in an installed Chrome or Edge:

```bash
node scripts/run-campaign.mjs --browser chrome --label MyDevice-Chrome --base http://localhost:4173
```

## Accuracy dataset

Images are not redistributed. Build the 1,000-image subset (seed 20261017) from ImageNetV2
(matched-frequency) or a local ImageNet validation copy:

```bash
python -m venv .venv && .venv/Scripts/pip install -r analysis/requirements.txt
.venv/Scripts/python analysis/prepare_imagenet_subset.py --val-dir <path> --n 1000 --seed 20261017
```

## Analysis

```bash
.venv/Scripts/python analysis/analyze.py results/*.json --out analysis/out
```

Produces latency, speed-up, fidelity and statistical-test tables (CSV and LaTeX) and figures.
See [RESEARCH.md](RESEARCH.md) for the research questions, the full protocol and threats to validity.

## Layout

- `src/bench/`: benchmark page (protocol controller, engines, statistics, storage)
- `src/main.jsx`: on-device image-classification demo
- `scripts/run-campaign.mjs`: browser automation for unattended sessions
- `analysis/`: dataset preparation and statistical analysis
