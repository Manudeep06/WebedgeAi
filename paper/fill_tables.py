"""Turn campaign session files into the manuscript's result tables (JSON) and figures (PNG).

  python paper/fill_tables.py results/D1-*.json --out paper/results_d1.json

All numbers come from analysis/analyze.py, so the paper and the analysis outputs cannot diverge.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import analyze as A  # noqa: E402

TFJS_NAMES = {"v1-0.25": "MobileNetV1-0.25", "v1-0.50": "MobileNetV1-0.50", "v1-0.75": "MobileNetV1-0.75", "v1-1.00": "MobileNetV1-1.0",
              "v2-0.50": "MobileNetV2-0.50", "v2-0.75": "MobileNetV2-0.75", "v2-1.00": "MobileNetV2-1.0"}
ORT_NAMES = {"mobilenetv4-s": "MobileNetV4-S", "mobilevit-s": "MobileViT-S", "resnet-50": "ResNet-50", "vit-b16": "ViT-B/16"}
NAMES = {**TFJS_NAMES, **ORT_NAMES}
DTYPE_NAMES = {"fp32": "fp32", "fp16": "fp16", "q8": "int8", "q4": "int4", "q4f16": "int4/fp16"}
SIZES_MB = {("mobilenetv4-s", "fp32"): 15.1, ("mobilenetv4-s", "fp16"): 7.6, ("mobilenetv4-s", "q8"): 3.9,
            ("mobilevit-s", "fp32"): 22.6, ("mobilevit-s", "fp16"): 11.6, ("mobilevit-s", "q8"): 6.3, ("mobilevit-s", "q4"): 12.8, ("mobilevit-s", "q4f16"): 7.3,
            ("resnet-50", "fp32"): 102.2, ("resnet-50", "fp16"): 51.1, ("resnet-50", "q8"): 25.8,
            ("vit-b16", "fp32"): 346.5, ("vit-b16", "fp16"): 173.5, ("vit-b16", "q8"): 88.3, ("vit-b16", "q4"): 59.8, ("vit-b16", "q4f16"): 51.2}
BACKEND_NAMES = {"webgpu": "WebGPU", "webgl": "WebGL", "wasm": "WASM", "cpu": "Plain JS", "ort-webgpu": "WebGPU", "ort-wasm": "WASM"}
BROWSER_SHORT = lambda b: b.replace("Google Chrome", "Chrome").replace("Microsoft Edge", "Edge")  # noqa: E731


def fmt(x, d=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:,.{d}f}"


def ms(x):
    """Latency with sensible precision."""
    if x is None or np.isnan(x):
        return "–"
    return fmt(x, 2 if x < 10 else 1 if x < 1000 else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="paper/results_d1.json")
    ap.add_argument("--figdir", default="paper")
    args = ap.parse_args()
    figdir = Path(args.figdir)

    sessions = A.load_sessions(args.files)
    runs, samples, excl = A.tidy(sessions)
    lat = A.latency_table(samples, runs)
    speed = A.speedup_table(samples, "wasm")
    tests, pairs = A.cross_device_tests(lat)
    fid = A.fidelity_table(sessions)
    if speed.empty:
        speed = pd.DataFrame(columns=["device", "browser", "model", "backend", "baseline", "speedup", "ci_lo", "ci_hi"])
    if fid.empty:
        fid = pd.DataFrame(columns=["device", "browser", "engine", "model", "dtype", "backend", "top1_agreement", "max_abs_prob_diff"])
    for df in (lat, speed, fid):
        if "browser" in df:
            df["browser"] = df["browser"].map(BROWSER_SHORT)
    runs["browser"] = runs["browser"].map(BROWSER_SHORT)
    browsers = sorted(lat.browser.unique())
    out = {"browsers": browsers, "devices": sorted(lat.device.unique())}

    # ---------------- run accounting ----------------
    total = sum(len(s["results"]) for s in sessions)
    out["runs"] = {"total": total, "valid": int(len(runs)),
                   "excluded": excl.groupby("reason").size().to_dict() if not excl.empty else {}}
    env = sessions[0]["environment"]
    out["env"] = {"webgpu": env.get("webgpu", {}).get("architecture"), "renderer": (env.get("webgl") or {}).get("renderer"),
                  "cores": env.get("hardwareConcurrency"), "coi": env.get("crossOriginIsolated")}
    out["wasm_threads"] = sorted({str(x) for x in runs.loc[runs.backend == "wasm", "wasm_threads"].dropna().unique()})

    # ---------------- RQ1: speed-up over WASM (fp32 models) ----------------
    fp32 = speed[~speed.model.str.contains("@")]
    rq1 = []
    for (browser, model), g in fp32.groupby(["browser", "model"]):
        base = lat[(lat.browser == browser) & (lat.model == model) & lat.backend.isin(["wasm", "ort-wasm"])]
        row = {"browser": browser, "model": NAMES.get(model, model), "wasm_ms": ms(base.median_ms.iloc[0]) if len(base) else "–"}
        for b in ("webgpu", "webgl", "cpu"):
            r = g[g.backend.isin([b, f"ort-{b}"])]
            row[b] = f"{r.speedup.iloc[0]:.2f}× ({r.ci_lo.iloc[0]:.2f}–{r.ci_hi.iloc[0]:.2f})" if len(r) else "–"
        rq1.append(row)
    order = list(NAMES.values())
    rq1.sort(key=lambda r: (r["browser"], order.index(r["model"]) if r["model"] in order else 99))
    out["rq1"] = rq1
    out["rq1_tests"] = {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in v.items()} for k, v in tests.items()}
    out["rq1_pairs"] = pairs.round(6).to_dict("records") if not pairs.empty else []
    wins = {}
    for (browser, model), g in lat[~lat.model.str.contains("@")].groupby(["browser", "model"]):
        best = g.loc[g.median_ms.idxmin(), "backend"]
        wins.setdefault(browser, {}).setdefault(BACKEND_NAMES[best], []).append(NAMES.get(model, model))
    out["rq1_fastest"] = wins

    # ---------------- RQ2: time to first prediction (fp32) ----------------
    rq2 = []
    r32 = runs[runs.dtype == "fp32"]
    for (browser, model, backend), g in r32.groupby(["browser", "base_model", "backend"]):
        net = g.loc[g.load_source == "network", "load_ms"]
        cache = g.loc[g.load_source.isin(["indexeddb", "cache"]), "load_ms"]
        warm = lat[(lat.browser == browser) & (lat.model == model) & (lat.backend == backend)].median_ms
        cold = g.cold_ms.median()
        rq2.append({"browser": browser, "model": NAMES.get(model, model), "backend": BACKEND_NAMES[backend],
                    "load_net_ms": fmt(net.median(), 0) if len(net) else "–", "load_cache_ms": fmt(cache.median(), 0) if len(cache) else "–",
                    "cold_ms": fmt(cold, 0), "warm_ms": ms(warm.iloc[0]) if len(warm) else "–",
                    "rho": f"{cold / warm.iloc[0]:.1f}" if len(warm) else "–",
                    "_cold": float(cold), "_warm": float(warm.iloc[0]) if len(warm) else None})
    rq2.sort(key=lambda r: (r["browser"], order.index(r["model"]) if r["model"] in order else 99, r["backend"]))
    out["rq2"] = rq2

    # ---------------- RQ3: accuracy and fidelity ----------------
    acc = runs.dropna(subset=["top1"]).groupby(["browser", "model", "backend"]).agg(top1=("top1", "first"), top5=("top5", "first"), n=("n_images", "first")).reset_index()
    rq3 = []
    for (browser, model), g in acc[~acc.model.str.contains("@")].groupby(["browser", "model"]):
        ref = g[g.backend.isin(["wasm", "ort-wasm"])]
        rec = {"browser": browser, "model": NAMES.get(model, model)}
        rec["top1"] = fmt(100 * ref.top1.iloc[0], 1) if len(ref) else "–"
        rec["top5"] = fmt(100 * ref.top5.iloc[0], 1) if len(ref) else "–"
        f = fid[(fid.browser == browser) & (fid.model == model)]
        for b in ("webgpu", "webgl", "wasm"):
            fb = f[f.backend.isin([b, f"ort-{b}"])]
            rec[f"{b}_agree"] = fmt(100 * fb.top1_agreement.iloc[0], 1) if len(fb) else "–"
            rec[f"{b}_dmax"] = f"{fb.max_abs_prob_diff.iloc[0]:.1e}" if len(fb) else "–"
        rq3.append(rec)
    rq3.sort(key=lambda r: (r["browser"], order.index(r["model"]) if r["model"] in order else 99))
    out["rq3"] = rq3
    fp32fid = fid[fid.dtype == "fp32"]
    out["rq3_summary"] = {"max_dmax_fp32": float(fp32fid.max_abs_prob_diff.max()) if len(fp32fid) else None,
                          "min_agree_fp32": float(fp32fid.top1_agreement.min()) if len(fp32fid) else None,
                          "n_images": int(acc.n.max()) if len(acc) else 0}

    # ---------------- RQ5: quantization (ONNX models) ----------------
    rq5 = []
    olat = lat[lat.engine == "ort"].copy()
    olat["base"] = olat.model.str.split("@").str[0]
    olat["dtype"] = olat.model.apply(lambda m: m.split("@")[1] if "@" in m else "fp32")
    for (browser, base, backend), g in olat.groupby(["browser", "base", "backend"]):
        f32 = g[g.dtype == "fp32"].median_ms
        f32_acc = acc[(acc.browser == browser) & (acc.model == base) & (acc.backend == backend)].top1
        for _, r in g.iterrows():
            a = acc[(acc.browser == browser) & (acc.model == r.model) & (acc.backend == backend)].top1
            fr = fid[(fid.browser == browser) & (fid.model == r.model) & (fid.backend == backend)]
            rq5.append({"browser": browser, "model": NAMES[base], "backend": BACKEND_NAMES[backend], "dtype": DTYPE_NAMES[r["dtype"]],
                        "size_mb": fmt(SIZES_MB.get((base, r["dtype"])), 1), "median_ms": ms(r.median_ms),
                        "vs_fp32": f"{f32.iloc[0] / r.median_ms:.2f}×" if len(f32) else "–",
                        "top1": fmt(100 * a.iloc[0], 1) if len(a) else "–",
                        "dtop1": (f"{100 * (a.iloc[0] - f32_acc.iloc[0]):+.1f}" if len(a) and len(f32_acc) and r["dtype"] != "fp32" else "–"),
                        "agree": fmt(100 * fr.top1_agreement.iloc[0], 1) if len(fr) else ("100.0" if r["dtype"] == "fp32" and backend == "ort-wasm" else "–"),
                        "_ratio": float(f32.iloc[0] / r.median_ms) if len(f32) else None})
    dorder = list(DTYPE_NAMES.values())
    rq5.sort(key=lambda r: (r["browser"], list(ORT_NAMES.values()).index(r["model"]), r["backend"], dorder.index(r["dtype"])))
    out["rq5"] = rq5

    # ---------------- RQ6: pre-processing share for ONNX ----------------
    pre = runs[(runs.engine == "ort") & (runs.dtype == "fp32")].groupby(["browser", "base_model"]).preprocess_ms.median()
    out["rq6_preprocess_ms"] = {f"{b}|{NAMES[m]}": float(v) for (b, m), v in pre.items()}

    # ---------------- figures ----------------
    colors = {"WebGPU": "#d9822b", "WebGL": "#3f8f5a", "WASM": "#2f6fb0", "Plain JS": "#7b8794"}
    # Figure: warm latency per fp32 model and backend, one panel per browser
    fig, axes = plt.subplots(1, len(browsers), figsize=(7.4, 3.4), sharey=True, squeeze=False)
    for ax, browser in zip(axes[0], browsers):
        g = lat[(lat.browser == browser) & ~lat.model.str.contains("@")]
        models = [m for m in NAMES if m in set(g.model)]
        bk = ["WASM", "WebGPU", "WebGL", "Plain JS"]
        w = 0.8 / len(bk)
        for k, b in enumerate(bk):
            for i, m in enumerate(models):
                r = g[(g.model == m) & (g.backend.map(BACKEND_NAMES) == b)]
                if len(r):
                    ax.bar(i + k * w, r.median_ms.iloc[0], width=w, color=colors[b], label=b if i == 0 else None)
        ax.set_xticks(np.arange(len(models)) + w * 1.5, [NAMES[m] for m in models], rotation=60, ha="right", fontsize=6.5)
        ax.set_yscale("log")
        ax.set_title(browser, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("Warm median latency (ms, log)")
    handles, labels = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncols=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figdir / "fig4_rq1_latency.png", dpi=220)
    plt.close(fig)

    # Figure: accuracy-latency Pareto plot (first browser), all models and precisions
    browser = browsers[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    pts = []
    for _, r in lat[lat.browser == browser].iterrows():
        a = acc[(acc.browser == browser) & (acc.model == r.model) & (acc.backend == r.backend)].top1
        if len(a) and r.backend != "cpu":
            pts.append((r.median_ms, 100 * a.iloc[0], BACKEND_NAMES[r.backend], r.model))
    for b in colors:
        p = [x for x in pts if x[2] == b]
        if p:
            ax.scatter([x[0] for x in p], [x[1] for x in p], s=16, color=colors[b], label=b, alpha=0.85)
    pts.sort()
    front, best = [], -1
    for x in pts:
        if x[1] > best:
            front.append(x)
            best = x[1]
    ax.plot([x[0] for x in front], [x[1] for x in front], color="#1f2933", lw=0.9, ls="--", label="Pareto front")
    for x in front:
        base, _, dt = x[3].partition("@")
        ax.annotate(f"{NAMES.get(base, base)}{' ' + DTYPE_NAMES[dt] if dt else ''} ({x[2]})", (x[0], x[1]), fontsize=5.8, xytext=(3, -8), textcoords="offset points")
    out["pareto"] = [{"model": NAMES.get(x[3].split("@")[0], x[3]) + (f" {DTYPE_NAMES[x[3].split('@')[1]]}" if "@" in x[3] else ""), "backend": x[2], "ms": ms(x[0]), "top1": fmt(x[1], 1)} for x in front]
    ax.set_xscale("log")
    ax.set_xlabel("Warm median latency (ms, log)")
    ax.set_ylabel("Top-1 accuracy on ImageNetV2 subset (%)")
    ax.set_title(f"Accuracy vs latency on D1 ({browser})", fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "fig5_pareto.png", dpi=220)
    plt.close(fig)

    Path(args.out).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("browsers", "runs", "env", "wasm_threads", "rq1_tests", "rq1_fastest", "rq3_summary")}, indent=1, default=str))


if __name__ == "__main__":
    main()
