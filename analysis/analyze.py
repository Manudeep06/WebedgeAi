"""Aggregate benchmark JSON exports (one per device/browser session) into paper tables and figures.

Usage:
  python analysis/analyze.py results/*.json --out analysis/out [--reference cpu] [--baseline wasm]

Statistical approach
  * Unit of replication within a session is the repetition (each repetition visits every config
    in a fresh random order). Latency CIs use a two-level (hierarchical) bootstrap: resample
    repetitions, then samples within each repetition, so the interval reflects run-to-run drift
    and not only within-run jitter.
  * Cross-device claims treat the device/browser as the unit: a Friedman test across backends
    with (device, model) blocks, followed by pairwise Wilcoxon signed-rank tests, Holm-corrected.
  * Numerical fidelity compares each backend's output probabilities with a reference backend
    (default: cpu, plain IEEE-754 float32 JavaScript) on the same device.
Runs flagged as hidden-tab, errored or unsupported are excluded and counted in exclusions.csv.
"""
import argparse
import base64
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Columns such as memory are empty for engines that do not report them; empty medians are expected.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

BACKEND_ORDER = ["webgpu", "webgl", "wasm", "cpu", "ort-webgpu", "ort-webnn-npu", "ort-webnn-gpu", "ort-webnn-cpu", "ort-wasm"]
# Numerical reference per engine: plain float32 execution of the same model at fp32.
REFERENCE_BACKEND = {"tfjs": "cpu", "ort": "ort-wasm"}


def model_label(r):
    """Model plus precision, e.g. "vit-b16@q8"; fp32 keeps the bare name."""
    dtype = r.get("dtype") or "fp32"
    return r["model"] if dtype == "fp32" else f"{r['model']}@{dtype}"
RNG_SEED = 20261017


def browser_name(env):
    brands = (env.get("userAgentData") or {}).get("fullVersionList") or []
    for b in brands:
        if b["brand"] in ("Google Chrome", "Microsoft Edge", "Opera", "Samsung Internet", "Chromium"):
            return f"{b['brand']} {b['version'].split('.')[0]}"
    ua = env.get("userAgent", "")
    for key, name in (("Firefox/", "Firefox"), ("Version/", "Safari")):
        if key in ua:
            return f"{name} {ua.split(key)[1].split('.')[0]}"
    return "unknown"


def load_sessions(paths):
    sessions = []
    for p in paths:
        s = json.loads(Path(p).read_text(encoding="utf-8"))
        env = s["environment"]
        s["_device"] = s["protocol"].get("deviceLabel") or (env.get("webgl") or {}).get("renderer", "unknown")
        s["_browser"] = browser_name(env)
        s["_file"] = str(p)
        sessions.append(s)
    return sessions


def tidy(sessions):
    runs, samples, exclusions = [], [], []
    for s in sessions:
        for r in s["results"]:
            key = dict(session=s["sessionId"], device=s["_device"], browser=s["_browser"], power=s["protocol"].get("power", ""),
                       engine=r.get("engine", "tfjs"), backend=r["backend"], model=model_label(r), base_model=r["model"],
                       dtype=r.get("dtype") or "fp32", rep=r["rep"], order=r["order"])
            reason = None if r["status"] == "ok" else r["status"]
            if reason is None and r.get("tabWasHidden"):
                reason = "tab hidden"
            if reason:
                exclusions.append({**key, "reason": reason, "error": (r.get("error") or "")[:200]})
                continue
            acc = r.get("accuracy") or {}
            det = r.get("backendDetails") or {}
            runs.append({**key, "load_ms": r["load"]["ms"], "load_source": r["load"]["source"], "cold_ms": r["coldMs"],
                         "preprocess_ms": r.get("preprocessMs"),
                         "median_ms": r["latency"]["median"], "kernel_ms": r.get("kernelMs"),
                         "top1": acc.get("top1"), "top5": acc.get("top5"), "n_images": acc.get("n"),
                         "mem_mb": (r.get("memory") or {}).get("numBytes", np.nan) / 2**20, "wasm_simd": det.get("simd"), "wasm_threads": det.get("multithread")})
            samples += [{**key, "i": i, "ms": ms} for i, ms in enumerate(r["latencySamplesMs"])]
    return pd.DataFrame(runs), pd.DataFrame(samples), pd.DataFrame(exclusions)


def hierarchical_bootstrap(groups, stat=np.median, B=2000, rng=None):
    """groups: list of 1-D arrays (one per repetition). Returns (point, lo, hi)."""
    rng = rng or np.random.default_rng(RNG_SEED)
    point = stat(np.concatenate(groups))
    boots = np.empty(B)
    for b in range(B):
        picked = rng.integers(len(groups), size=len(groups))
        boots[b] = stat(np.concatenate([rng.choice(groups[k], size=len(groups[k])) for k in picked]))
    return point, *np.percentile(boots, [2.5, 97.5])


def latency_table(samples, runs):
    rows = []
    for key, g in samples.groupby(["engine", "device", "browser", "backend", "model"]):
        groups = [x["ms"].to_numpy() for _, x in g.groupby("rep")]
        med, lo, hi = hierarchical_bootstrap(groups)
        r = runs[(runs[["engine", "device", "browser", "backend", "model"]] == key).all(axis=1)]
        rows.append(dict(zip(["engine", "device", "browser", "backend", "model"], key), reps=len(groups), median_ms=med, ci_lo=lo, ci_hi=hi,
                         p95_ms=np.percentile(g["ms"], 95), cold_ms=r["cold_ms"].median(),
                         load_network_ms=r.loc[r.load_source == "network", "load_ms"].median(),
                         load_cache_ms=r.loc[r.load_source.isin(["indexeddb", "cache"]), "load_ms"].median(),
                         preprocess_ms=r["preprocess_ms"].median(),
                         top1=r["top1"].mean(), top5=r["top5"].mean(), mem_mb=r["mem_mb"].median()))
    return pd.DataFrame(rows)


def speedup_table(samples, baseline, B=2000):
    """Ratio of median latency vs the baseline backend, per device/browser/model, with a
    hierarchical-bootstrap CI (both arms resampled independently)."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for key, g in samples.groupby(["device", "browser", "model"]):
        base_name = baseline if g.engine.iloc[0] == "tfjs" else f"ort-{baseline}"
        if base_name not in set(g.backend):
            continue
        base = [x["ms"].to_numpy() for _, x in g[g.backend == base_name].groupby("rep")]
        for backend in sorted(set(g.backend) - {base_name}):
            arm = [x["ms"].to_numpy() for _, x in g[g.backend == backend].groupby("rep")]

            def draw(groups):
                picked = rng.integers(len(groups), size=len(groups))
                return np.median(np.concatenate([rng.choice(groups[k], size=len(groups[k])) for k in picked]))

            ratios = np.array([draw(base) / draw(arm) for _ in range(B)])
            point = np.median(np.concatenate(base)) / np.median(np.concatenate(arm))
            rows.append(dict(zip(["device", "browser", "model"], key), backend=backend, baseline=base_name,
                             speedup=point, ci_lo=np.percentile(ratios, 2.5), ci_hi=np.percentile(ratios, 97.5)))
    return pd.DataFrame(rows)


def cross_device_tests(lat):
    """Per engine: Friedman across backends with (device, browser, model) blocks; pairwise Wilcoxon + Holm."""
    summaries, tables = {}, []
    for engine, g in lat.groupby("engine"):
        summary, pairs = _cross_device_tests(g)
        summaries[engine] = summary
        if not pairs.empty:
            tables.append(pairs.assign(engine=engine))
    return summaries, (pd.concat(tables) if tables else pd.DataFrame())


def _cross_device_tests(lat):
    wide = lat.pivot_table(index=["device", "browser", "model"], columns="backend", values="median_ms").dropna()
    backends = [b for b in BACKEND_ORDER if b in wide.columns]
    out = {"blocks": len(wide), "backends": backends}
    if len(wide) < 3 or len(backends) < 2:
        out["note"] = "Need >=3 complete blocks and >=2 backends; collect more devices."
        return out, pd.DataFrame()
    if len(backends) >= 3:
        fr = stats.friedmanchisquare(*[wide[b] for b in backends])
        out.update(friedman_chi2=fr.statistic, friedman_p=fr.pvalue)
    else:
        out["note"] = "Two backends: paired Wilcoxon signed-rank test only (Friedman needs three or more)."
    pairs = []
    for a, b in itertools.combinations(backends, 2):
        diff = np.log(wide[a]) - np.log(wide[b])
        w = stats.wilcoxon(diff)
        pairs.append(dict(a=a, b=b, n=len(diff), median_ratio=float(np.exp(np.median(diff))), W=w.statistic, p=w.pvalue))
    pairs = pd.DataFrame(pairs).sort_values("p").reset_index(drop=True)
    m = len(pairs)
    pairs["p_holm"] = np.minimum(1, np.maximum.accumulate(pairs["p"] * (m - np.arange(m))))
    return out, pairs


def decode_probs(b64):
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32)


def fidelity_table(sessions, reference_override=None):
    """Per session: every run vs the fp32 model on its engine's reference backend, same device.
    Captures both backend numerics (same precision, different backend) and quantization
    effects (lower precision vs fp32)."""
    rows = []
    for s in sessions:
        by = {}
        for r in s["results"]:
            if r["status"] == "ok" and r.get("accuracy"):
                by.setdefault((model_label(r), r["backend"]), r)  # first valid repetition
        for (label, backend), r in by.items():
            engine = r.get("engine", "tfjs")
            ref_backend = reference_override or REFERENCE_BACKEND[engine]
            ref = by.get((r["model"], ref_backend))
            if not ref or ref is r:
                continue
            p_ref = {p["file"]: p for p in ref["accuracy"]["predictions"]}
            preds = [p for p in r["accuracy"]["predictions"] if p["file"] in p_ref]
            top1_agree = np.mean([p["top5"][0] == p_ref[p["file"]]["top5"][0] for p in preds])
            top5_overlap = np.mean([len(set(p["top5"]) & set(p_ref[p["file"]]["top5"])) / 5 for p in preds])
            v_ref = {v["file"]: decode_probs(v["probs"]) for v in ref["accuracy"]["referenceProbs"]}
            diffs, kls = [], []
            for v in r["accuracy"]["referenceProbs"]:
                if v["file"] in v_ref:
                    p, q = v_ref[v["file"]].astype(np.float64), decode_probs(v["probs"]).astype(np.float64)
                    diffs.append(np.abs(p - q).max())
                    p, q = p / p.sum(), q / q.sum()  # float32 softmax sums drift from 1, giving spurious negative KL
                    kls.append(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))
            rows.append(dict(device=s["_device"], browser=s["_browser"], engine=engine, model=label, dtype=r.get("dtype") or "fp32",
                             backend=backend, reference=f"{ref_backend} fp32", n_images=len(preds),
                             top1_agreement=top1_agree, top5_overlap=top5_overlap,
                             max_abs_prob_diff=np.max(diffs) if diffs else np.nan, mean_kl=np.mean(kls) if kls else np.nan,
                             top1_acc=r["accuracy"]["top1"], top1_acc_ref=ref["accuracy"]["top1"],
                             top1_acc_delta=r["accuracy"]["top1"] - ref["accuracy"]["top1"]))
    return pd.DataFrame(rows)


def plot_latency(lat, out):
    for (device, browser), g in lat.groupby(["device", "browser"]):
        models = sorted(g.model.unique())
        fig, ax = plt.subplots(figsize=(7, 3.6))
        width = 0.8 / max(1, g.backend.nunique())
        for k, backend in enumerate([b for b in BACKEND_ORDER if b in set(g.backend)]):
            gb = g[g.backend == backend].set_index("model").reindex(models)
            x = np.arange(len(models)) + k * width
            ax.errorbar(x, gb.median_ms, yerr=[gb.median_ms - gb.ci_lo, gb.ci_hi - gb.median_ms], fmt="o", capsize=2, label=backend)
        ax.set_xticks(np.arange(len(models)) + width * (g.backend.nunique() - 1) / 2, models, rotation=30)
        ax.set_yscale("log")
        ax.set_ylabel("Warm latency, median (ms, log)")
        ax.set_title(f"{device} / {browser}", fontsize=10)
        ax.legend(fontsize=8, ncols=4)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"latency_{slug(device)}_{slug(browser)}.pdf")
        plt.close(fig)


def plot_pareto(lat, out):
    g = lat.dropna(subset=["top1"])
    if g.empty:
        return
    for (device, browser), d in g.groupby(["device", "browser"]):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for backend, b in d.groupby("backend"):
            ax.scatter(b.median_ms, 100 * b.top1, label=backend, s=18)
            for _, r in b.iterrows():
                ax.annotate(r.model, (r.median_ms, 100 * r.top1), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Warm latency, median (ms, log)")
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.set_title(f"Accuracy vs latency: {device} / {browser}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"pareto_{slug(device)}_{slug(browser)}.pdf")
        plt.close(fig)


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--reference", default=None, help="override the numerical reference backend (default: cpu for TF.js, ort-wasm for ONNX Runtime)")
    ap.add_argument("--baseline", default="wasm", help="backend used as speed-up baseline")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(args.files)
    runs, samples, excl = tidy(sessions)
    excl.to_csv(out / "exclusions.csv", index=False)
    print(f"{len(sessions)} sessions, {len(runs)} valid runs, {len(excl)} excluded")
    if runs.empty:
        return

    lat = latency_table(samples, runs)
    lat.to_csv(out / "latency_summary.csv", index=False)
    speed = speedup_table(samples, args.baseline)
    speed.to_csv(out / "speedup.csv", index=False)
    summary, pairs = cross_device_tests(lat)
    pairs.to_csv(out / "cross_device_wilcoxon.csv", index=False)
    (out / "cross_device_friedman.json").write_text(json.dumps(summary, indent=2, default=float))
    fid = fidelity_table(sessions, args.reference)
    fid.to_csv(out / "fidelity.csv", index=False)

    cols = ["device", "browser", "backend", "model", "median_ms", "ci_lo", "ci_hi", "cold_ms", "load_network_ms", "load_cache_ms", "preprocess_ms", "top1"]
    lat[cols].sort_values(["device", "browser", "model", "backend"]).to_latex(out / "latency_table.tex", index=False, float_format="%.2f")
    if not fid.empty:
        fid.to_latex(out / "fidelity_table.tex", index=False, float_format="%.4g")

    plot_latency(lat, out)
    plot_pareto(lat, out)
    print(lat[cols].round(2).to_string(index=False))
    print(json.dumps(summary, indent=2, default=float))
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
