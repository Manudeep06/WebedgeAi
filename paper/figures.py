"""Figures for the revised manuscript. Pilot numbers are copied from the valid (non-hidden-tab)
runs in the two pilot sessions' latency_summary.csv files."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
INK, MUTED, A, B, C = "#1f2933", "#52606d", "#2f6fb0", "#d9822b", "#3f8f5a"


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fs=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", color=INK):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color=color, lw=1.1, ls=ls))


# Figure 1: architecture of the LOCAL-Bench instrument
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")
ax.add_patch(FancyBboxPatch((0.2, 0.25), 7.0, 5.7, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#f5f7fa", ec=MUTED, lw=1, ls="--"))
ax.text(0.45, 5.62, "Browser tab on the device under test (cross-origin isolated)", fontsize=8.5, color=MUTED, weight="bold")
box(ax, 0.5, 4.45, 2.0, 0.85, "Protocol controller\nseeded plan, isolation,\ncool-down, resume", fc="#e8f0fa")
box(ax, 2.75, 4.45, 2.0, 0.85, "Environment probe\nUA, GPU adapter,\ncores, memory, power", fc="#e8f0fa")
box(ax, 5.0, 4.45, 2.0, 0.85, "Session store\n(IndexedDB)\nresults + in-flight", fc="#e8f0fa")
box(ax, 0.5, 2.75, 3.1, 1.2, "TensorFlow.js 4.22\nWebGPU | WebGL | WASM | CPU\nMobileNet V1 / V2", fc="#fdf0e3")
box(ax, 3.85, 2.75, 3.15, 1.2, "ONNX Runtime Web\nWebGPU | WASM | WebNN\nMobileNetV4, MobileViT,\nResNet-50, ViT-B/16; fp32 to int4", fc="#fdf0e3")
box(ax, 0.5, 0.6, 3.1, 1.55, "Measurement loop\nload (network | cache)\ncold call, warm-up,\nN timed runs, accuracy pass", fc="#e9f5ec")
box(ax, 3.85, 0.6, 3.15, 1.55, "Validity guards\nhidden-tab flag, crash record,\nleak check, randomised order,\npage reload per configuration", fc="#e9f5ec")
box(ax, 7.75, 3.85, 2.05, 1.2, "Model hosts\nTF Hub / Hugging Face\n(weights only)", fc="#ffffff")
box(ax, 7.75, 1.95, 2.05, 1.2, "Collection server\nPOST /__results\n(session JSON)", fc="#ffffff")
box(ax, 7.75, 0.3, 2.05, 1.05, "Offline analysis\nbootstrap, Friedman,\nWilcoxon-Holm", fc="#ffffff")
for x in (2.05, 5.4):
    arrow(ax, x, 4.45, x, 3.95)
arrow(ax, 2.05, 2.75, 2.05, 2.15)
arrow(ax, 5.4, 2.75, 5.4, 2.15)
arrow(ax, 7.75, 4.45, 7.0, 3.6, ls="--", color=MUTED)
ax.text(7.05, 4.25, "weights", fontsize=7.5, color=MUTED)
arrow(ax, 7.0, 1.4, 7.75, 2.4)
arrow(ax, 8.77, 1.95, 8.77, 1.35)
fig.tight_layout()
fig.savefig("fig1_architecture.png", dpi=220)
plt.close(fig)

# Figure 2: measurement protocol for one configuration, inside the randomised plan
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis("off")
steps = [("Activate\nbackend", "#e8f0fa"), ("Load\nweights", "#e8f0fa"), ("Cold\ncall", "#fdf0e3"),
         ("Warm-up\nW = 10", "#fdf0e3"), ("Timed\nN = 50", "#fdf0e3"), ("Accuracy\npass", "#e9f5ec"), ("Dispose,\nreload", "#e9f5ec")]
w, gap, y = 1.22, 0.17, 1.55
for i, (t, fc) in enumerate(steps):
    x = 0.15 + i * (w + gap)
    box(ax, x, y, w, 0.95, t, fc=fc, fs=7.8)
    if i:
        arrow(ax, x - gap, y + 0.47, x, y + 0.47)
ax.text(5, 0.95, "Repeated for every (runtime, backend, model, precision) configuration; each of R = 5 repetitions visits all\n"
        "configurations in a fresh seeded random order, with a page reload and a 10 s cool-down between configurations.",
        ha="center", va="center", fontsize=7.8, color=MUTED)
ax.text(5, 2.85, "One configuration", ha="center", fontsize=8.5, weight="bold", color=INK)
fig.tight_layout()
fig.savefig("fig2_protocol.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# Figure 3: pilot latencies (valid runs only), log scale, with 95% bootstrap CIs
rows = [
    ("TF.js  MobileNetV1-0.25", [("WASM", 4.76, 3.65, 6.19), ("WebGPU", 16.56, 16.02, 17.25), ("WebGL", 18.04, 16.78, 21.03), ("CPU (JS)", 243.12, 184.49, 258.17)]),
    ("TF.js  MobileNetV2-0.50", [("WASM", 13.53, 10.10, 16.04), ("WebGL", 31.47, 18.37, 41.40), ("WebGPU", 32.02, 17.11, 33.50), ("CPU (JS)", 354.49, 346.51, 365.56)]),
    ("ORT  MobileNetV4-S", [("WebGPU fp32", 22.08, 21.23, 22.39), ("WASM int8", 40.55, 35.82, 46.24), ("WebGPU int8", 253.90, 238.64, 254.82)]),
    ("ORT  MobileViT-S", [("WebGPU q4f16", 62.69, 59.83, 65.50), ("WebGPU fp32", 89.06, 88.06, 91.05), ("WASM fp32", 124.75, 118.31, 133.71), ("WASM int8", 136.47, 127.39, 144.57)]),
]
colors = {"WASM": A, "WebGPU": B, "WebGL": C, "CPU": MUTED}
fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.0), sharey=True)
for ax, (title, items) in zip(axes, rows):
    for i, (lab, med, lo, hi) in enumerate(items):
        key = next(k for k in colors if lab.startswith(k))
        ax.bar(i, med, color=colors[key], alpha=0.85, width=0.7)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], color=INK, capsize=2, lw=0.8)
        ax.text(i, hi * 1.12, f"{med:.0f}" if med >= 10 else f"{med:.1f}", ha="center", fontsize=6.8)
    ax.set_xticks(range(len(items)), [l for l, *_ in items], rotation=55, ha="right", fontsize=6.8)
    ax.set_title(title, fontsize=7.6)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Warm median latency (ms, log)")
fig.tight_layout()
fig.savefig("fig3_pilot.png", dpi=220)
plt.close(fig)
print("figures written")
