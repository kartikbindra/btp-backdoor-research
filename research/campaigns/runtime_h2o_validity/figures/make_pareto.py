"""Stealth-fix sweep: the stealth/firing trade-off figure.

Two panels:
  (a) Pareto trade-off  -- full-cache leak C0 (stealth) vs runtime-H2O marker rate (trigger),
      with the empty "stealthy + firing" target box.
  (b) Firing and selectivity collapse  -- C0, runtime-H2O and random-eviction rate vs the
      leak-hinge strength delta; H2O loses its edge over random exactly as C0 is suppressed.

All points: Qwen2.5-1.5B, seed 42, n=139 dev prompts (retargeted c0_leak_hinge, code commit 8bb7c5f).
Outputs paper/figures/stealth_firing_pareto.{pdf,png}. Run with the repo venv python.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---- solid palette (no gradients, no purple) ----
BLUE = "#2563eb"   # runtime-H2O firing
TEAL = "#0891b2"   # random eviction
GRAY = "#6b7280"   # full-cache leak C0 / dominated runs
RED = "#dc2626"    # unreached target

# ---- data (rate, with k/n in comments) ----
# hinge-delta runs: (label, delta, C0, h2o, random)
HINGE = [
    ("$\\delta$=0", 0, 0.266, 0.317, 0.187),   # 37/139, 44/139, 26/139  (Campaign 008, no term)
    ("$\\delta$=2", 2, 0.0144, 0.1727, 0.2086),  # 2/139, 24/139, 29/139
    ("$\\delta$=3", 3, 0.0144, 0.1007, 0.1007),  # 2/139, 14/139, 14/139
    ("$\\delta$=4", 4, 0.0072, 0.0935, 0.1007),  # 1/139, 13/139, 14/139
    ("$\\delta$=5", 5, 0.0000, 0.0935, 0.0719),  # 0/139, 13/139, 10/139
]
# KL runs (different knob, shown only in panel a): (label, C0, h2o)
KL = [
    ("$\\kappa$=0.5", 0.194, 0.396),  # 27/139, 55/139
    ("$\\kappa$=4",   0.252, 0.374),  # 35/139, 52/139
]

PASS_C0 = 0.05    # stealth bar
PASS_FIRE = 0.25  # firing needed for gap >= 0.20 when C0 <= 0.05

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "legend.frameon": False, "figure.dpi": 150,
})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.3))

# ===== Panel (a): Pareto trade-off =====
axA.set_xlim(-0.02, 0.30)
axA.set_ylim(0.0, 0.45)

# target box: C0 <= 0.05 AND firing >= 0.25 -> top-left, empty
axA.add_patch(Rectangle((-0.02, PASS_FIRE), 0.02 + PASS_C0, 0.45 - PASS_FIRE,
                        facecolor=RED, alpha=0.10, hatch="//", edgecolor=RED, linewidth=1.0, zorder=1))
axA.text(0.015, 0.355, "target:\nstealthy + firing\n(no run reaches it)",
         color=RED, fontsize=8.5, ha="center", va="center", zorder=3)
axA.axvline(PASS_C0, ls="--", color=GRAY, lw=1.0, zorder=2)
axA.text(PASS_C0 + 0.004, 0.02, "stealth bar\n$C_0=0.05$", color=GRAY, fontsize=8, va="bottom")

# achievable frontier (non-dominated: delta5 -> delta2 -> kappa0.5)
fx = [0.0000, 0.0144, 0.194]
fy = [0.0935, 0.1727, 0.396]
axA.plot(fx, fy, "-", color=BLUE, lw=1.2, alpha=0.5, zorder=2)
axA.text(0.085, 0.235, "achievable\nfrontier", color=BLUE, fontsize=8, rotation=38, alpha=0.8)

# hinge-delta points
for label, d, c0, h2o, _ in HINGE:
    axA.scatter([c0], [h2o], s=58, color=BLUE, zorder=5, edgecolor="white", linewidth=0.6)
off = {0: (8, 5), 2: (10, 2), 3: (11, -10), 4: (-2, -17), 5: (-20, -5)}
for label, d, c0, h2o, _ in HINGE:
    dx, dy = off[d]
    axA.annotate(label, (c0, h2o), textcoords="offset points", xytext=(dx, dy),
                 fontsize=8.5, color=BLUE,
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5) if d in (4, 5) else None)
# KL points (different knob)
for label, c0, h2o in KL:
    axA.scatter([c0], [h2o], s=52, marker="s", color=GRAY, zorder=5, edgecolor="white", linewidth=0.6)
    axA.annotate(label, (c0, h2o), textcoords="offset points", xytext=(8, -3), fontsize=8.5, color=GRAY)

axA.scatter([], [], s=58, color=BLUE, label="leak-hinge runs ($\\delta$)")
axA.scatter([], [], s=52, marker="s", color=GRAY, label="KL runs ($\\kappa$)")
axA.legend(loc="center right", fontsize=8.5)
axA.set_xlabel("Full-cache leak $C_0$  (lower = stealthier)")
axA.set_ylabel("Runtime-H2O marker rate  (trigger strength)")
axA.set_title("(a) Stealth–firing trade-off  (seed 42, $n$=139)")

# ===== Panel (b): firing & selectivity collapse vs delta =====
ds = [d for _, d, _, _, _ in HINGE]
c0s = [c0 for _, _, c0, _, _ in HINGE]
h2os = [h2o for _, _, _, h2o, _ in HINGE]
rnds = [r for _, _, _, _, r in HINGE]

# stealthy band (C0 <= 0.05 holds for delta >= ~1.5)
axB.axvspan(1.5, 5.3, color=GRAY, alpha=0.07, zorder=0)
axB.text(3.4, 0.345, "stealthy regime ($C_0\\leq0.05$)", color=GRAY, fontsize=8, ha="center")

axB.plot(ds, h2os, "-o", color=BLUE, lw=1.8, ms=5, label="runtime-H2O (trigger)")
axB.plot(ds, rnds, "-s", color=TEAL, lw=1.8, ms=5, label="random eviction")
axB.plot(ds, c0s, "-^", color=GRAY, lw=1.8, ms=5, label="full-cache leak $C_0$")

axB.annotate("H2O loses its edge\nover random\n(not selective)",
             xy=(2.0, 0.19), xytext=(3.1, 0.255), fontsize=8, color="black", ha="left",
             arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
axB.annotate("H2O $>$ random\n(+0.13, selective)", xy=(0.0, 0.317), xytext=(0.15, 0.40),
             fontsize=8, color=BLUE, ha="left",
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))

axB.set_xlim(-0.3, 5.3)
axB.set_ylim(0.0, 0.45)
axB.set_xticks(ds)
axB.set_xlabel("Leak-hinge strength $\\delta$  (0 = no stealth term)")
axB.set_ylabel("marker rate")
axB.set_title("(b) Firing and selectivity collapse with the leak")
axB.legend(loc="upper right", fontsize=8.5)

fig.tight_layout()
out = Path(__file__).resolve().parents[4] / "paper" / "figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(out / f"stealth_firing_pareto.{ext}", bbox_inches="tight",
                dpi=300 if ext == "png" else None)
    print("wrote", out / f"stealth_firing_pareto.{ext}")
