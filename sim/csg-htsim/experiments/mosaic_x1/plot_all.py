#!/usr/bin/env python3
"""Render plot.png for every experiment folder under experiments/mosaic_x1.

Style follows the dataviz method: entity-stable colors from the validated
reference palette (MOSAIC=blue, FAST=aqua, ECMP=neutral gray, bound=dashed
ink), thin marks, recessive grid, direct value labels (text ink, never series
color), one axis per chart, legend whenever >=2 series. Static PNG: the CSV
next to each plot is the table view.
Run with: /home/weihaoni/miniconda3/envs/myenv/bin/python3 plot_all.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C = {"mosaic": "#2a78d6", "fast": "#1baf7a", "ecmp": "#9a9a92",
     "bound": "#444444",
     "v1": "#2a78d6", "v2": "#4a3aa7", "v3": "#eb6834", "v4": "#e34948",
     "v5": "#eda100", "v6": "#1baf7a"}
INK = "#1a1a19"
MUTED = "#6b6a61"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5df",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def rows(folder):
    with open(os.path.join(HERE, folder, "results.csv")) as f:
        return list(csv.DictReader(f))


def save(fig, folder):
    out = os.path.join(HERE, folder, "plot.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def label_bars(ax, bars, fmt="{:.0f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, color=INK)


def grouped_bars(ax, groups, series, width=0.22):
    """series: list of (name, color, values). groups: x tick labels."""
    n = len(series)
    xs = range(len(groups))
    for i, (name, color, vals) in enumerate(series):
        pos = [x + (i - (n - 1) / 2) * (width + 0.02) for x in xs]
        bars = ax.bar(pos, vals, width=width, color=color, label=name,
                      zorder=3)
        label_bars(ax, bars)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(groups)


# 01 crossover -------------------------------------------------------------
r = rows("01_crossover")
nv = [int(x["nv_gbps"]) for x in r]
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(nv, [float(x["ecmp_us"]) for x in r], color=C["ecmp"], lw=2,
        marker="o", ms=4, label="ECMP")
ax.plot(nv, [float(x["fast_guard_us"]) for x in r], color=C["fast"], lw=2,
        marker="o", ms=4, label="FAST (guard)")
ax.plot(nv, [float(x["mosaic_us"]) for x in r], color=C["mosaic"], lw=2,
        marker="o", ms=4, label="MOSAIC")
ax.plot(nv, [float(x["lower_bound_us"]) for x in r], color=C["bound"],
        lw=1.4, ls="--", label="cut lower bound")
ax.set_xscale("log")
ax.set_xticks(nv)
ax.set_xticklabels([str(v) for v in nv], rotation=45, fontsize=7.5)
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_yscale("log")
ax.set_xlabel("scale-up bandwidth nv (Gbps, NIC fixed at 180)")
ax.set_ylabel("completion (µs)")
ax.set_title("Crossover: non-uniform A2A on flex_rail (hot-dst zipf 1.0)")
ax.annotate("crossover ≈ NV:NIC 3.5:1", xy=(650, 5800), fontsize=8,
            color=MUTED)
ax.legend()
save(fig, "01_crossover")

# 02 ablation ---------------------------------------------------------------
r = rows("02_ablation")
variants = ["full_prior_adapt", "prior_static", "w1_pure_prior",
            "uniform_adapt", "uniform_static", "rx_heuristic_static"]
names = {"full_prior_adapt": "prior+adapt (default)",
         "prior_static": "prior+static", "w1_pure_prior": "W=1 pure prior",
         "uniform_adapt": "uniform+adapt", "uniform_static": "uniform+static",
         "rx_heuristic_static": "naive RX static"}
colors = [C["mosaic"], C["v3"], C["v2"], C["v5"], C["v4"], C["v6"]]
nvs = [2400, 180, 90]
fig, ax = plt.subplots(figsize=(7.2, 4))
series = []
for v, col in zip(variants, colors):
    vals = []
    for n in nvs:
        m = [float(x["completion_us"]) for x in r
             if x["variant"] == v and int(x["nv_gbps"]) == n]
        vals.append(m[0] / 1000 if m else 0)
    series.append((names[v], col, vals))
grouped_bars(ax, [f"nv={n}" for n in nvs], series, width=0.13)
ax.set_ylabel("completion (ms)")
ax.set_title("Ablation: prior × feedback × corridor mix (lower is better)")
ax.legend(ncol=2, fontsize=7.5)
save(fig, "02_ablation")

# 03 fairness ---------------------------------------------------------------
r = rows("03_fairness_fast")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
pts = [("fast_noguard", 2400), ("fast_guard", 2400), ("fast_guard_rr", 2400),
       ("fast_sc4MB", 2400),
       ("fast_guard", 90), ("fast_guard_rr", 90), ("fast_sc32MB", 90)]
labels = ["no-guard\n2400", "guard\n2400", "round-robin\n2400",
          "scaling 4MB\n2400", "guard\n90", "round-robin\n90",
          "scaling 32MB\n90"]
vals = []
for v, n in pts:
    m = [float(x["completion_us"]) for x in r
         if x["variant"] == v and int(x["nv_gbps"]) == n]
    vals.append(m[0] / 1000)
bars = ax.bar(range(len(vals)), vals, width=0.5, color=C["fast"], zorder=3)
label_bars(ax, bars, "{:.1f}")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("completion (ms)")
ax.set_title("FAST fairness variants: weak-nv collapse is mapping-insensitive")
save(fig, "03_fairness_fast")

# 04 robustness -------------------------------------------------------------
r = rows("04_robustness")
works = ["hotdst_seed42", "hotdst_seed43", "hotdst_seed44", "hotdst_a0.5",
         "hotdst_a1.5", "persrc_seed42"]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=False)
for ax, n in zip(axes, [2400, 360, 90]):
    speed = []
    for w in works:
        m = [x for x in r if x["workload"] == w and int(x["nv_gbps"]) == n]
        speed.append(float(m[0]["fast_us"]) / float(m[0]["mosaic_us"]))
    bars = ax.bar(range(len(works)), speed, width=0.55, color=C["mosaic"],
                  zorder=3)
    label_bars(ax, bars, "{:.2f}")
    ax.axhline(1.0, color=C["bound"], lw=1, ls="--")
    ax.set_xticks(range(len(works)))
    ax.set_xticklabels([w.replace("hotdst_", "").replace("persrc_seed42",
                        "persrc") for w in works], rotation=45, fontsize=7)
    ax.set_title(f"nv={n}", fontsize=9)
axes[0].set_ylabel("MOSAIC speedup vs FAST (×)")
fig.suptitle("Robustness: seeds × skew × workload family", fontsize=10)
save(fig, "04_robustness")

# 05 drift ------------------------------------------------------------------
r = rows("05_drift")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
groups = [f"nv={x['nv_gbps']}" for x in r]
series = [
    ("fresh prior + static", C["v3"],
     [float(x["fresh_static_us"]) / 1000 for x in r]),
    ("stale prior + static", C["v4"],
     [float(x["stale_static_us"]) / 1000 for x in r]),
    ("fresh prior + adapt", C["mosaic"],
     [float(x["fresh_adapt_us"]) / 1000 for x in r]),
    ("stale prior + adapt", C["v2"],
     [float(x["stale_adapt_us"]) / 1000 for x in r]),
]
grouped_bars(ax, groups, series, width=0.18)
ax.set_ylabel("completion (ms)")
ax.set_title("Drift (persrc): feedback makes stale priors harmless")
ax.legend(fontsize=7.5)
save(fig, "05_drift")


def three_scheds(folder, title, group_key, unit=1000.0, keep=None):
    r = rows(folder)
    if keep:
        r = [x for x in r if keep(x)]
    groups = []
    for x in r:
        g = f"{x[group_key]}@nv{x['nv']}" if group_key != "nv" \
            else f"nv={x['nv']}"
        if g not in groups:
            groups.append(g)
    fig, ax = plt.subplots(figsize=(max(6.4, 1.3 * len(groups)), 3.8))
    series = []
    for sched, name, col in [("static_ecmp", "ECMP", C["ecmp"]),
                             ("fast_schedule", "FAST (guard)", C["fast"]),
                             ("mosaic_schedule", "MOSAIC", C["mosaic"])]:
        vals = []
        for g in groups:
            m = [float(x["completion_us"]) for x in r
                 if x["scheduler"] == sched
                 and (f"{x[group_key]}@nv{x['nv']}" == g
                      or f"nv={x['nv']}" == g)]
            vals.append(m[0] / unit if m else 0)
        series.append((name, col, vals))
    grouped_bars(ax, groups, series)
    ax.set_xticklabels(groups, rotation=15, fontsize=7.5)
    ax.set_ylabel("completion (ms)")
    ax.set_title(title, fontsize=10)
    ax.legend()
    save(fig, folder)


three_scheds("06_uniform", "Uniform A2A sanity (2MB/pair)", "nv")
three_scheds("07_scale_n64",
             "Scale: 64 GPUs, 4GB buffer (1GB runs: ECMP & FAST@nv90 overflow-FAILED, MOSAIC all passed)",
             "nv", keep=lambda x: x["label"] == "n64_buf4g")
three_scheds("09_multihot", "Multi-hotspot (K equally-hot dsts, 8× weight)",
             "label")
three_scheds("10_msg_size",
             "Collective size 0.25x/4x (8MB from 4GB-buffer rerun; at 1GB ECMP/FAST FAILED)",
             "label", keep=lambda x: x["label"] in ("msg512k", "msg8m_buf4g"))
three_scheds("11_taper", "Spine taper sensitivity at nv=360", "label")

# 08 parameter sensitivity ---------------------------------------------------
r = rows("08_param_sensitivity")
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
for ax, (pref, xlabel) in zip(axes, [("W", "waves W"), ("G", "gamma"),
                                     ("T", "threshold")]):
    pts = [(x["label"][1:], float(x["completion_us"]) / 1000) for x in r
           if x["label"].startswith(pref)]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C["mosaic"],
            lw=2, marker="o", ms=5)
    ax.set_xlabel(xlabel)
    ax.set_title(f"nv=360, vary {xlabel}", fontsize=9)
axes[0].set_ylabel("completion (ms)")
fig.suptitle("MOSAIC parameter sensitivity (defaults: W=6, γ=0.7, thr=0.75)",
             fontsize=10)
save(fig, "08_param_sensitivity")

# 12 lambda traces -----------------------------------------------------------
def parse_trace(path):
    waves, lam = [], {"direct": [], "tx": [], "rx": [], "sp3": []}
    for line in open(path):
        t = line.split()
        waves.append(int(t[4]))
        for name in lam:
            i = t.index(name)
            lam[name].append(float(t[i + 3]))
    return waves, lam


fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
cc = {"direct": "#2a78d6", "tx": "#eda100", "rx": "#1baf7a",
      "sp3": "#4a3aa7"}
for ax, nv in zip(axes, [2400, 90]):
    waves, lam = parse_trace(os.path.join(
        HERE, "12_lambda_trace", f"x1_trace_hot26_nv{nv}.txt"))
    for name, ys in lam.items():
        ax.plot(waves, ys, color=cc[name], lw=2, marker="o", ms=4,
                label=name.upper())
    ax.set_xlabel("wave")
    ax.set_title(f"hot GPU 26, nv={nv}", fontsize=9)
axes[0].set_ylabel("corridor ratio λ")
axes[1].legend(fontsize=8)
fig.suptitle("λ convergence: NV-pipe corridors win at 2400, NIC-pipe at 90",
             fontsize=10)
save(fig, "12_lambda_trace")
print("ALL PLOTS DONE")


# ---- v5 expansion experiments (13-21); each section skips if absent ------
def _has(folder):
    return os.path.exists(os.path.join(HERE, folder, "results.csv"))

SCHEME_COLOR = {"static_ecmp": C["ecmp"], "ecmp": C["ecmp"],
                "teccl": C["v2"], "taccl": C["v3"], "syccl": C["v5"],
                "fast": C["fast"], "fast_schedule": C["fast"],
                "mosaic": C["mosaic"], "mosaic_schedule": C["mosaic"]}
SCHEME_NAME = {"static_ecmp": "ECMP", "ecmp": "ECMP", "teccl": "TE-CCL",
               "taccl": "TACCL", "syccl": "SyCCL", "fast": "FAST",
               "fast_schedule": "FAST", "mosaic": "MOSAIC",
               "mosaic_schedule": "MOSAIC"}

if _has("13_full_baseline"):
    r = rows("13_full_baseline")
    nvs = sorted({int(x["nv_gbps"]) for x in r}, reverse=True)
    schemes = ["ecmp", "teccl", "taccl", "syccl", "fast", "mosaic"]
    fig, ax = plt.subplots(figsize=(7.4, 4))
    series = []
    for s in schemes:
        vals = []
        for n in nvs:
            m = [x for x in r if int(x["nv_gbps"]) == n]
            v = m[0].get(s + "_us", "") if m else ""
            vals.append(float(v) / 1000 if v not in ("", "SKIPPED") else 0)
        series.append((SCHEME_NAME[s], SCHEME_COLOR[s], vals))
    grouped_bars(ax, ["NV:NIC %s" % ("13.3:1" if n == 2400 else
                 ("2:1" if n == 360 else "0.5:1")) for n in nvs],
                 series, width=0.13)
    for i, n in enumerate(nvs):
        b = [float(x["bound_us"]) for x in r if int(x["nv_gbps"]) == n]
        if b:
            ax.hlines(b[0] / 1000, i - 0.45, i + 0.45, color=C["bound"],
                      ls="--", lw=1.2)
    ax.set_ylabel("completion (ms)")
    ax.set_title("Full baseline, non-uniform A2A (zipf1.0/seed42); "
                 "dashes = cut bound")
    ax.legend(ncol=3, fontsize=7.5)
    save(fig, "13_full_baseline")

if _has("14_uniform_congestion"):
    r = [x for x in rows("14_uniform_congestion") if x["ecn"] == "0"]
    tapers = [180, 90, 60]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True)
    for ax, nv in zip(axes, [2400, 360]):
        series = []
        for s in ["static_ecmp", "fast_schedule", "mosaic_schedule"]:
            vals = [float(x["completion_us"]) / 1000 for t in tapers
                    for x in r if x["scheme"] == s and int(x["nv_gbps"]) == nv
                    and int(x["taper_gbps"]) == t]
            series.append((SCHEME_NAME[s], SCHEME_COLOR[s], vals))
        grouped_bars(ax, ["taper %dG" % t for t in tapers], series)
        ax.axhline(2.133, color=C["bound"], ls="--", lw=1.2)
        ax.set_title("nv=%d" % nv, fontsize=9)
    axes[0].set_ylabel("completion (ms)")
    fig.suptitle("Uniform A2A under spine taper (dash = cut bound 2.13ms)",
                 fontsize=10)
    save(fig, "14_uniform_congestion")

if _has("15_skew_sweep"):
    r = rows("15_skew_sweep")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax, nv in zip(axes, [2400, 90]):
        for s in ["ecmp", "teccl", "taccl", "syccl", "fast", "mosaic"]:
            pts = sorted([(float(x["alpha"]), float(x["completion_us"]) / 1e3)
                          for x in r if x["scheme"] == s
                          and int(x["nv_gbps"]) == nv
                          and x["status"] == "PASSED"])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker="o", ms=4, lw=1.8, color=SCHEME_COLOR[s],
                        label=SCHEME_NAME[s])
        bl = sorted([(float(x["alpha"]), float(x["bound_us"]) / 1e3)
                     for x in r if x["scheme"] == "mosaic"
                     and int(x["nv_gbps"]) == nv])
        ax.plot([p[0] for p in bl], [p[1] for p in bl], ls="--", lw=1.2,
                color=C["bound"], label="cut bound")
        ax.set_yscale("log")
        ax.set_xlabel("zipf skew α")
        ax.set_title("nv=%d" % nv, fontsize=9)
    axes[0].set_ylabel("completion (ms)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Skew sweep (hot-dst zipf, seed42)", fontsize=10)
    save(fig, "15_skew_sweep")

if _has("16_corridor_count"):
    r = [x for x in rows("16_corridor_count")
         if x["workload"] == "zipf1.0_seed42"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for nv, col in [(2400, C["v1"]), (360, C["v3"]), (90, C["v5"])]:
        pts = sorted([(int(x["corridors"]), float(x["completion_us"]) / 1e3)
                      for x in r if int(x["nv_gbps"]) == nv])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                lw=1.8, color=col, label="nv=%d" % nv)
    ax.set_yscale("log")
    ax.set_xlabel("corridor count C (1=DIRECT, 2=+RX, 3=+TX, 4=+SPREAD3, "
                  "5/6=+spread lanes)")
    ax.set_ylabel("completion (ms)")
    ax.set_title("Corridor-count ablation (zipf1.0/seed42)")
    ax.legend()
    save(fig, "16_corridor_count")

if _has("17_threshold"):
    r = rows("17_threshold")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for nv, col in [(2400, C["v1"]), (360, C["v3"]), (90, C["v5"])]:
        pts = sorted([(float(x["threshold"]), float(x["completion_us"]) / 1e3)
                      for x in r if int(x["nv_gbps"]) == nv])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4,
                lw=1.8, color=col, label="nv=%d" % nv)
    ax.set_yscale("log")
    ax.set_xlabel("wave delivery threshold")
    ax.set_ylabel("completion (ms)")
    ax.set_title("Threshold sensitivity (default 0.75; band ≤ 6%)")
    ax.legend()
    save(fig, "17_threshold")

if _has("18_synthesis_speed"):
    r = rows("18_synthesis_speed")
    fig, ax = plt.subplots(figsize=(6.8, 4))
    for s, col, mk in [("mosaic", C["mosaic"], "o"), ("fast", C["fast"], "o"),
                       ("teccl", C["v2"], "s"), ("taccl", C["v3"], "s"),
                       ("syccl", C["v5"], "s")]:
        pts = sorted([(int(x["n"]), float(x["time_us"]))
                      for x in r if x["scheme"] == s and x["time_us"]
                      and x["status"] in ("OK", "reused")])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                    ms=5, lw=1.8, color=col, label=SCHEME_NAME[s])
        to = [(int(x["n"])) for x in r if x["scheme"] == s
              and x["status"] == "TIMEOUT"]
        for n in to:
            ax.annotate("timeout", xy=(n, 1.8e9), color=col, fontsize=7,
                        ha="center")
            ax.scatter([n], [1.8e9], marker="x", color=col)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("schedule synthesis time (µs, log)")
    ax.set_title("Schedule synthesis time vs scale (×= >30min timeout)")
    ax.legend(fontsize=7.5)
    save(fig, "18_synthesis_speed")

if _has("19_degraded_nic"):
    r = rows("19_degraded_nic")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=False)
    for ax, nv in zip(axes, [2400, 360]):
        groups, series = [], []
        cases = [("26", "2"), ("26", "4"), ("7", "2"), ("7", "4")]
        for s in ["static_ecmp", "fast_schedule", "mosaic_schedule"]:
            vals = []
            for g, f in cases:
                m = [float(x["completion_us"]) / 1e3 for x in r
                     if x["scheme"] == s and int(x["nv_gbps"]) == nv
                     and x["degrade_gpu"] == g and x["factor"] == f]
                vals.append(m[0] if m else 0)
            series.append((SCHEME_NAME[s], SCHEME_COLOR[s], vals))
        groups = ["hot/2x", "hot/4x", "cold/2x", "cold/4x"]
        grouped_bars(ax, groups, series)
        ax.set_yscale("log")
        ax.set_title("nv=%d" % nv, fontsize=9)
    axes[0].set_ylabel("completion (ms)")
    fig.suptitle("Unannounced NIC degradation (hot=dst26, cold=gpu7)",
                 fontsize=10)
    save(fig, "19_degraded_nic")

if _has("20_host_shape"):
    r = rows("20_host_shape")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, (h, g) in zip(axes, [("8", "4"), ("2", "16")]):
        series = []
        for s in ["static_ecmp", "fast_schedule", "mosaic_schedule"]:
            vals = [float(x["completion_us"]) / 1e3 for nv in [2400, 360, 90]
                    for x in r if x["scheme"] == s and x["hosts"] == h
                    and int(x["nv_gbps"]) == nv]
            series.append((SCHEME_NAME[s], SCHEME_COLOR[s], vals))
        grouped_bars(ax, ["nv=%d" % nv for nv in [2400, 360, 90]], series)
        ax.set_yscale("log")
        ax.set_title("%s hosts × %s GPUs" % (h, g), fontsize=9)
    axes[0].set_ylabel("completion (ms)")
    fig.suptitle("Host-shape generality (same 32-GPU demand matrix)",
                 fontsize=10)
    save(fig, "20_host_shape")

if _has("21_design_iter"):
    r = rows("21_design_iter")
    fig, ax = plt.subplots(figsize=(6.8, 4))
    nvs = sorted({int(x["nv_gbps"]) for x in r}, reverse=True)
    for key, col, name in [("fast_us", C["fast"], "FAST"),
                           ("mosaic_c4_us", C["v2"], "MOSAIC C=4 (v4 default)"),
                           ("mosaic_c6_us", C["mosaic"], "MOSAIC C=6 (v5 default)")]:
        ax.plot(nvs, [float(x[key]) / 1e3 for n in nvs for x in r
                      if int(x["nv_gbps"]) == n], marker="o", ms=4, lw=1.8,
                color=col, label=name)
    ax.plot(nvs, [float(x["bound_us"]) / 1e3 for n in nvs for x in r
                  if int(x["nv_gbps"]) == n], ls="--", lw=1.2,
            color=C["bound"], label="cut bound")
    ax.set_xscale("log")
    ax.set_xticks(nvs)
    ax.set_xticklabels([str(v) for v in nvs], rotation=45, fontsize=7.5)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yscale("log")
    ax.set_xlabel("scale-up bandwidth nv (Gbps)")
    ax.set_ylabel("completion (ms)")
    ax.set_title("Design iteration: C=6 ≥ FAST at every swept ratio")
    ax.legend(fontsize=8)
    save(fig, "21_design_iter")

if _has("22_x2_pod"):
    r = [x for x in rows("22_x2_pod") if "[grid]" in x["extra"]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, nv in zip(axes, [2400, 360, 90]):
        series = []
        for s in ["static_ecmp", "fast_schedule", "mosaic_schedule"]:
            vals = [float(x["completion_us"]) / 1e3 for t in [1440, 720, 360]
                    for x in r if x["scheme"] == s and int(x["nv_gbps"]) == nv
                    and int(x["leaf_uplink_total"]) == t]
            series.append((SCHEME_NAME[s], SCHEME_COLOR[s], vals))
        grouped_bars(ax, ["1:1", "2:1", "4:1"], series)
        for i, t in enumerate([1440, 720, 360]):
            b = [float(x["bound_us"]) for x in r if int(x["nv_gbps"]) == nv
                 and int(x["leaf_uplink_total"]) == t and x["bound_us"]]
            if b:
                ax.hlines(b[0] / 1e3, i - 0.4, i + 0.4, color=C["bound"],
                          ls="--", lw=1.2)
        ax.set_title("nv=%d" % nv, fontsize=9)
        ax.set_xlabel("spine oversubscription")
    axes[0].set_ylabel("completion (ms)")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("X2: pod wiring + 4 spines (corridors regenerated: "
                 "DIRECT + RX-lanes); dash = cut bound", fontsize=10)
    save(fig, "22_x2_pod")

if _has("23_x3_rail2p"):
    r = rows("23_x3_rail2p")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    for ax, nv in zip(axes, [2400, 360, 90]):
        series = []
        for s, tag in [("static_ecmp", ""), ("fast_schedule", ""),
                       ("mosaic_schedule", "c4"), ("mosaic_schedule", "c6")]:
            vals = [float(x["completion_us"]) / 1e3 for pf in [1, 2, 4]
                    for x in r if x["scheme"] == s and int(x["nv_gbps"]) == nv
                    and int(x["planeb_factor"]) == pf and x["variant"] == tag]
            name = SCHEME_NAME[s] + (" C=6" if tag == "c6" else "")
            col = C["v2"] if tag == "c6" else SCHEME_COLOR[s]
            series.append((name, col, vals))
        grouped_bars(ax, ["sym", "B/2", "B/4"], series, width=0.17)
        ax.set_yscale("log")
        ax.set_title("nv=%d" % nv, fontsize=9)
        ax.set_xlabel("plane-B degradation")
    axes[0].set_ylabel("completion (ms)")
    axes[0].legend(fontsize=7)
    fig.suptitle("X3: dual-plane rail (2x90G NICs; pin=(s+d)%2); "
                 "parity-blind relays pin to plane A", fontsize=10)
    save(fig, "23_x3_rail2p")

if _has("24_x4_threepipe"):
    r = rows("24_x4_threepipe")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, nv in zip(axes, [2400, 360, 90]):
        pfs = [1, 2, 4]
        def pick(kn):
            out = []
            for pf in pfs:
                m = [float(x["completion_us"]) / 1e3 for x in r
                     if int(x["nv_gbps"]) == nv and int(x["planeb_factor"]) == pf
                     and x["known_prior"] == kn]
                m2 = [float(x["completion_us"]) / 1e3 for x in r
                      if int(x["nv_gbps"]) == nv and int(x["planeb_factor"]) == pf]
                out.append(m[0] if m else (m2[0] if m2 else None))
            return out
        old = [float(x["old_family_us"]) / 1e3 for pf in pfs for x in r
               if int(x["nv_gbps"]) == nv and int(x["planeb_factor"]) == pf][::2]
        ax.plot(pfs, old, marker="o", ms=4, lw=1.8, color=C["v3"],
                label="rail family (23)")
        ax.plot(pfs, pick("1"), marker="o", ms=4, lw=1.8, color=C["mosaic"],
                label="3-pipe unknown")
        kn = [ [float(x["completion_us"])/1e3 for x in r
                if int(x["nv_gbps"])==nv and int(x["planeb_factor"])==pf
                and x["known_prior"]==str(pf) and pf>1] for pf in pfs]
        ax.plot([p for p,v in zip(pfs,kn) if v], [v[0] for v in kn if v],
                marker="s", ms=4, lw=1.4, ls=":", color=C["v2"],
                label="3-pipe known")
        f = [float(x["fast_us"]) / 1e3 for pf in pfs for x in r
             if int(x["nv_gbps"]) == nv and int(x["planeb_factor"]) == pf][::2]
        ax.plot(pfs, f, marker="o", ms=4, lw=1.8, color=C["fast"], label="FAST")
        ax.set_xticks(pfs)
        ax.set_xticklabels(["sym", "B/2", "B/4"])
        ax.set_title("nv=%d" % nv, fontsize=9)
    axes[0].set_ylabel("completion (ms)")
    axes[0].legend(fontsize=7)
    fig.suptitle("X4: 3-pipe corridor family (DIRECT split per plane) "
                 "fixes the parity blind spot", fontsize=10)
    save(fig, "24_x4_threepipe")

if _has("25_x2full_spine"):
    r = rows("25_x2full_spine")
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    series = []
    for name, key, col in [("ECMP", "static_ecmp", C["ecmp"]),
                           ("FAST", "fast_schedule", C["fast"]),
                           ("MOSAIC implicit", "mosaic_implicit", C["v3"]),
                           ("MOSAIC explicit", "mosaic_explicit", C["mosaic"])]:
        vals = []
        for S in ["3", "4", "5"]:
            m = [float(x["completion_us"]) / 1e3 for x in r
                 if x["spines"] == S and x["scheme"] == key]
            vals.append(m[0] if m else 0)
        series.append((name, col, vals))
    grouped_bars(ax, ["S=3 (coprime)", "S=4 (divides)", "S=5 (coprime)"],
                 series, width=0.18)
    ax.set_ylabel("completion (ms)")
    ax.set_title("X2-full: explicit per-spine lanes win iff G % S != 0")
    ax.legend(fontsize=7.5)
    save(fig, "25_x2full_spine")

if _has("28_wave_ramp"):
    r = rows("28_wave_ramp")
    main = [x for x in r if x["section"] == "mainline"
            and x["demand"] == "zipf1.0_seed42"]
    nvs = [90, 180, 360, 420, 480, 600, 720, 2400]

    def line(variant):
        out = []
        for nv in nvs:
            m = [float(x["completion_us"]) for x in main
                 if int(x["nv_gbps"]) == nv and x["variant"] == variant]
            out.append(m[0] if m else None)
        return out

    bound = line("bound")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axes[0]
    xs = range(len(nvs))
    for name, variant, col, ls in [
            ("base (v5)", "base(v5)", C["ecmp"], "-"),
            ("MOSAIC-P (26)", "MOSAIC-P", C["v3"], "-"),
            ("coldE1 (28)", "coldE1", C["v2"], "-"),
            ("MOSAIC-R (28)", "coldE1_P(MOSAIC-R)", C["mosaic"], "-")]:
        vals = line(variant)
        ratio = [v / b if v else None for v, b in zip(vals, bound)]
        ax.plot(xs, ratio, marker="o", ms=4, lw=1.8, color=col, ls=ls,
                label=name)
    ax.axhline(1.0, color=C["bound"], lw=1.0, ls="--", label="cut bound")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(n) for n in nvs], fontsize=8)
    ax.set_xlabel("NV Gbps")
    ax.set_ylabel("completion / bound")
    ax.set_title("mainline: cold-restrict ramp closes the gap", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    gate = [x for x in r if x["section"] == "gate"]
    gnvs = [420, 480, 600, 720]
    forced, auto, basev = [], [], []
    for nv in gnvs:
        f = [float(x["completion_us"]) for x in gate
             if int(x["nv_gbps"]) == nv and x["variant"] == "coldE1_force"]
        a = [float(x["completion_us"]) for x in gate
             if int(x["nv_gbps"]) == nv and x["variant"] == "coldE1_auto"]
        b = [float(x["completion_us"]) for x in main
             if int(x["nv_gbps"]) == nv and x["variant"] == "base(v5)"]
        forced.append(f[0] / 1e3)
        auto.append(a[0] / 1e3)
        basev.append(b[0] / 1e3)
    grouped_bars(ax, ["nv%d" % n for n in gnvs],
                 [("base", C["ecmp"], basev),
                  ("coldE1 forced", C["v4"], forced),
                  ("coldE1 auto-gated", C["mosaic"], auto)], width=0.22)
    ax.set_ylabel("completion (ms)")
    ax.set_title("spine gate: forced cap mints a new cut at nv>=480",
                 fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle("28: corridor ramp (cold-restrict, dual-gated) + "
                 "fixed-size waves exploration", fontsize=10)
    save(fig, "28_wave_ramp")

# 29 flex_cube ------------------------------------------------------------
if _has("29_flex_cube"):
    r = rows("29_flex_cube")

    def get(section, profile, rot, variant):
        for x in r:
            if (x["section"] == section and x["profile"] == profile
                    and x["rot"] == rot and x["variant"] == variant
                    and x["completion_us"]):
                return float(x["completion_us"]), float(x["bound_us"])
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    profiles = ["50x16", "100x16", "250x16", "mix", "400x16"]
    series_spec = [
        ("FAST (guard)", C["fast"], [("profile", p, "0", "fast") if p != "mix"
                                     else ("online", p, "0", "fast")
                                     for p in profiles]),
        ("MOSAIC C6 W6", C["mosaic"], [("profile", p, "0", "mosaic") if p != "mix"
                                       else ("online", p, "0", "mosaic_c6")
                                       for p in profiles]),
        ("MOSAIC C6 W16", C["v3"], [("explore", p, "0", "w16")
                                    for p in profiles]),
        ("MOSAIC-R", C["v2"], [("profile", p, "0", "mosaic_r") if p != "mix"
                               else ("online", p, "0", "mosaic_r")
                               for p in profiles]),
    ]
    xs = range(len(profiles))
    for name, col, keys in series_spec:
        vals = []
        for k in keys:
            us, b = get(*k)
            vals.append(us / b if us else None)
        ax.plot(xs, vals, marker="o", ms=4, lw=1.8, color=col, label=name)
    ax.axhline(1.0, color=C["bound"], lw=1.0, ls="--", label="cut bound")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(profiles, fontsize=8)
    ax.set_xlabel("intra-host cube link speed profile")
    ax.set_ylabel("completion / bound")
    ax.set_title("flex_cube: MOSAIC wins the NV-scarce/heterogeneous side;"
                 " W16 is the transferable lever", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    schemes = [("ECMP", "online", "ecmp", C["ecmp"]),
               ("TE-CCL", "solver", "teccl(OPTIMAL,fill)", C["v5"]),
               ("TACCL", "solver", "taccl(OPTIMAL,fill)", C["v4"]),
               ("FAST", "online", "fast", C["fast"]),
               ("MOSAIC W6", "online", "mosaic_c6", C["mosaic"]),
               ("MOSAIC W16", "explore", "w16", C["v3"])]
    names, vals, cols = [], [], []
    bound = None
    for nm, sec, var, col in schemes:
        us, b = get(sec, "mix", "0", var)
        names.append(nm)
        vals.append(us / 1e3)
        cols.append(col)
        bound = b / 1e3
    bars = ax.bar(names, vals, color=cols, zorder=3)
    label_bars(ax, bars)
    ax.axhline(bound, color=C["bound"], lw=1.0, ls="--", label="cut bound")
    ax.set_ylabel("completion (ms)")
    ax.set_title("mainline 400/200/100G mix, zipf1.0/seed42", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7)
    fig.suptitle("29: DGX-shaped cube-mesh intra-host (heterogeneous NVLink)"
                 " -- M2 boundary tested", fontsize=10)
    save(fig, "29_flex_cube")

# 30 mosaic_lp ------------------------------------------------------------
if _has("30_mosaic_lp"):
    r = rows("30_mosaic_lp")

    def gv(cfg, var):
        for x in r:
            if x["config"] == cfg and x["variant"] == var:
                return float(x["completion_us"]), float(x["bound_us"])
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    cfgs = ["50x16", "100x16", "250x16", "mix_rot0", "400x16"]
    sota_best = {"50x16": "sota_r", "100x16": "sota_w16",
                 "250x16": "sota_w6", "mix_rot0": "sota_w16",
                 "400x16": "sota_w12"}
    lp_best = {"50x16": "r_lploc", "100x16": "w16_lploc",
               "250x16": "w6_lploc", "mix_rot0": "w12_lploc",
               "400x16": "w12_lploc"}
    xs = range(len(cfgs))
    for name, col, pick in [("29-SOTA (no LP)", C["mosaic"], sota_best),
                            ("MOSAIC-LP", C["v3"], lp_best)]:
        vals = []
        for cfg in cfgs:
            us, b = gv(cfg, pick[cfg])
            vals.append(us / b if us else None)
        ax.plot(xs, vals, marker="o", ms=4, lw=1.8, color=col, label=name)
    fvals = []
    for cfg in cfgs:
        us, b = gv(cfg, "fast_ref")
        fvals.append(us / b if us else None)
    ax.plot(xs, fvals, marker="s", ms=4, lw=1.4, color=C["fast"],
            label="FAST (ref)")
    ax.axhline(1.0, color=C["bound"], lw=1.0, ls="--", label="cut bound")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cfgs, fontsize=8)
    ax.set_xlabel("intra-host speed profile")
    ax.set_ylabel("completion / bound")
    ax.set_title("MOSAIC-LP: uniform win on heterogeneous/scarce, "
                 "neutral on homogeneous-fast", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    cfgs2 = ["mix_rot0", "mix_rot4", "mix_rot8", "mix_rot12",
             "zipf1.0_seed43", "zipf1.5_seed42", "uniform2mb"]
    base_v = {"mix_rot0": "sota_w16", "mix_rot4": "sota_w16",
              "mix_rot8": "sota_w16", "mix_rot12": "sota_w16",
              "zipf1.0_seed43": "sota_w16", "zipf1.5_seed42": "sota_w16",
              "uniform2mb": "sota_w6"}
    lp_v = {"mix_rot0": "w12_lploc", "mix_rot4": "w16_lploc",
            "mix_rot8": "w16_lp", "mix_rot12": "w16_lp",
            "zipf1.0_seed43": "w16_lploc", "zipf1.5_seed42": "w16_lploc",
            "uniform2mb": "w6_lploc"}
    deltas = []
    for cfg in cfgs2:
        b_us, _ = gv(cfg, base_v[cfg])
        l_us, _ = gv(cfg, lp_v[cfg])
        deltas.append((l_us / b_us - 1.0) * 100.0)
    bars = ax.bar(range(len(cfgs2)), deltas, color=C["v3"], zorder=3)
    label_bars(ax, bars, fmt="{:.1f}")
    ax.axhline(0.0, color=C["bound"], lw=1.0)
    ax.set_xticks(range(len(cfgs2)))
    ax.set_xticklabels(cfgs2, fontsize=7, rotation=20)
    ax.set_ylabel("LP vs 29-SOTA (%)")
    ax.set_title("hot-dst placement / seeds / alpha / uniform: "
                 "LP never loses", fontsize=9)
    fig.suptitle("30: MOSAIC-LP -- fixed topology => 35ms intra-host LP; "
                 "unknown workload => runtime lambda", fontsize=10)
    save(fig, "30_mosaic_lp")

# 31 mosaic_lp_v2 ---------------------------------------------------------
if _has("31_mosaic_lp_v2"):
    r = rows("31_mosaic_lp_v2")

    def gv31(cfg, var):
        for x in r:
            if x["config"] == cfg and x["variant"] == var:
                return float(x["completion_us"]), float(x["bound_us"])
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    steps = [("hand lanes (29)", "diag_base_w6"),
             ("LP vertex (1 rail/pair)", "jd_k1_vertex"),
             ("+diversity K=4", "jd_k4"),
             ("+diversity K=6", "jd_k6"),
             ("hot-column LP", "jh_hotcol")]
    names = [n for n, _ in steps]
    vals = [gv31("full300", v)[0] / 1e3 for _, v in steps]
    bars = ax.bar(range(len(steps)), vals, color=C["v3"], zorder=3)
    label_bars(ax, bars, fmt="{:.2f}")
    f_us, b = gv31("full300", "fast_ref")
    ax.axhline(f_us / 1e3, color=C["fast"], lw=1.4, ls="-",
               label="FAST 3.66ms")
    ax.axhline(b / 1e3, color=C["bound"], lw=1.0, ls="--",
               label="cut bound")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(names, fontsize=6.5, rotation=15)
    ax.set_ylabel("completion (ms)")
    ax.set_title("full300 (homogeneous crossbar): landing spread is THE"
                 " gap; LP iterations close and pass FAST", fontsize=8.5)
    ax.legend(fontsize=7)

    ax = axes[1]
    cfgs = ["50x16", "100x16", "250x16", "mix", "400x16", "full300",
            "full50"]
    best31 = {"50x16": "stack_jh_r", "100x16": "stack_jh_r",
              "250x16": "jh_hotcol", "mix": "stack_jh_w12",
              "400x16": "jd_k6", "full300": "jh_hotcol",
              "full50": "jh_hotcol"}
    xs = range(len(cfgs))
    mos, fst = [], []
    for cfg in cfgs:
        us, b = gv31(cfg, best31[cfg])
        mos.append(us / b)
        fu, _ = gv31(cfg, "fast_ref")
        fst.append(fu / b if fu else None)
    ax.plot(xs, fst, marker="s", ms=4, lw=1.6, color=C["fast"],
            label="FAST (guard)")
    ax.plot(xs, mos, marker="o", ms=4, lw=1.8, color=C["v3"],
            label="MOSAIC-LP v2 (one config family)")
    ax.axhline(1.0, color=C["bound"], lw=1.0, ls="--", label="cut bound")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cfgs, fontsize=8)
    ax.set_ylabel("completion / bound")
    ax.set_title("across profiles: >= FAST everywhere except 250x16 "
                 "(M1 residue)", fontsize=8.5)
    ax.legend(fontsize=7)
    fig.suptitle("31: MOSAIC-LP v2 -- landing jdist from a hot-column LP"
                 " (no homogeneity detection anywhere)", fontsize=10)
    save(fig, "31_mosaic_lp_v2")

if _has("32_paradox_pacing"):
    import csv as _csv

    def rows32():
        out = []
        with open(os.path.join(HERE, "32_paradox_pacing",
                               "results.csv")) as f:
            for x in _csv.DictReader(f):
                out.append(x)
        return out

    r32 = rows32()

    def g32(sect, label):
        for x in r32:
            if x["section"] == sect and x["label"] == label:
                return float(x["completion_us"])
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.2))

    # (a) P1: taper paradox -- base vs solo vs R vs pace
    ax = axes[0]
    tapers = [45, 90, 180, 360, 720, 1440]
    for name, key, color in [
            ("base C6", "zipf t{} base", C["mosaic"]),
            ("solo hot-dst", "solo t{}", C["v2"]),
            ("MOSAIC-R", "zipf t{} R", C["v5"]),
            ("+cold_pace(F6)", None, C["fast"])]:
        vals = []
        for t in tapers:
            if key is None:
                v = g32("mech", "taper{} pace1".format(t))
            else:
                v = g32("p1rail", key.format(t))
            vals.append(v / 1e3 if v else float("nan"))
        ax.plot(range(len(tapers)), vals, "o-", color=color, label=name,
                lw=1.6, ms=4)
    ax.axhline(7.552, color=C["bound"], lw=1.0, ls="--",
               label="ingress bound")
    ax.set_xticks(range(len(tapers)))
    ax.set_xticklabels([str(t) for t in tapers])
    ax.set_xlabel("taper (Gbps)")
    ax.set_ylabel("completion (ms)")
    ax.set_title("P1: wider fabric feeds slack flows, not the critical\n"
                 "path; solo-hot is monotone, F6 restores monotonicity",
                 fontsize=8.5)
    ax.legend(fontsize=7)

    # (b) P3: relay degradation granularity
    ax = axes[1]
    cases = [("clean", ("p23", "ref c6"), ("p23", "ref fast")),
             ("1/3 relays\n(g2 x4)", ("p23b", "relay2 c6"),
              ("p23b", "relay2 fast")),
             ("2/3 relays", ("p23b", "relay2_10 c6"),
              ("p23b", "relay2_10 fast")),
             ("3/3 relays", ("p23b", "relay2_10_18 c6"),
              ("p23b", "relay2_10_18 fast")),
             ("1/3 + F7\nhost-lambda", ("mech", "g2 f4 d1 hostl"),
              ("p23b", "relay2 fast"))]
    xs = range(len(cases))
    mvals = [g32(*c[1]) / 1e3 for c in cases]
    fvals = [g32(*c[2]) / 1e3 for c in cases]
    w = 0.38
    b1 = ax.bar([x - w / 2 for x in xs], mvals, w, color=C["mosaic"],
                label="MOSAIC", zorder=3)
    b2 = ax.bar([x + w / 2 for x in xs], fvals, w, color=C["fast"],
                label="FAST", zorder=3)
    label_bars(ax, b1, fmt="{:.1f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cases], fontsize=7)
    ax.set_ylabel("completion (ms)")
    ax.set_title("P3: per-dst lambda absorbs class-wide faults (3/3)\n"
                 "but not in-class ones (1/3); F7 fixes the granularity",
                 fontsize=8.5)
    ax.legend(fontsize=7)

    # (c) F6 across nv (rail mainline)
    ax = axes[2]
    nvs = [90, 180, 360, 420, 480, 600, 720, 2400]
    base = [16276.9, 12672.7, 8418.42, 7542.22, 6589.8, 5610.43,
            4960.22, 3525.76]  # 21 组归档 C6
    for name, key, color in [("MOSAIC-R (28)", "nv{} R", C["v5"]),
                             ("F6 pace1", "nv{} pace1", C["mosaic"]),
                             ("R+F6", "nv{} Rpace1", C["fast"])]:
        vals = []
        for nv in nvs:
            v = g32("mech", key.format(nv))
            vals.append(v / 1e3 if v else float("nan"))
        ax.plot(range(len(nvs)), vals, "o-", color=color, label=name,
                lw=1.6, ms=4)
    ax.plot(range(len(nvs)), [b / 1e3 for b in base], "s--",
            color=C["ecmp"], label="C6 base (21)", lw=1.2, ms=3)
    ax.set_xticks(range(len(nvs)))
    ax.set_xticklabels([str(n) for n in nvs], fontsize=7)
    ax.set_xlabel("nv (Gbps)")
    ax.set_ylabel("completion (ms)")
    ax.set_yscale("log")
    ax.set_title("F6 cold pacing across the rail mainline", fontsize=8.5)
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "32_paradox_pacing", "plot.png"),
                dpi=150)
    plt.close(fig)
    print("32_paradox_pacing/plot.png")


if _has("33_fast_sync"):
    # FAST runtime stage sync: none (legacy async) vs credit (faithful
    # NVSHMEM-kernel gates) vs barrier (analytical-model global barrier),
    # plus the largest-freq-first stage-order fix.
    data = rows("33_fast_sync")
    zipf = [r for r in data if r["workload"] == "zipf"]
    zipf.sort(key=lambda r: int(r["nv_gbps"]))
    nvs = [r["nv_gbps"] for r in zipf]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    # (a) zipf mainline: completion across nv per sync mode
    ax = axes[0]
    for name, key, color in [("none (async)", "none_desc_us", C["fast"]),
                             ("credit", "credit_desc_us", C["v2"]),
                             ("barrier", "barrier_desc_us", C["v4"])]:
        vals = [float(r[key]) / 1e3 for r in zipf]
        ax.plot(range(len(nvs)), vals, "o-", color=color, label=name,
                lw=1.6, ms=4)
    ax.plot(range(len(nvs)), [float(r["bound_us"]) / 1e3 for r in zipf],
            "--", color=C["bound"], label="cut bound", lw=1.0)
    ax.set_xticks(range(len(nvs)))
    ax.set_xticklabels(nvs, fontsize=7)
    ax.set_xlabel("nv (Gbps)")
    ax.set_ylabel("completion (ms)")
    ax.set_yscale("log")
    ax.set_title("FAST zipf1.0: runtime sync cost across nv", fontsize=8.5)
    ax.legend(fontsize=7)

    # (b) sync overhead percentage
    ax = axes[1]
    for name, key, color in [("credit", "credit_vs_none_pct", C["v2"]),
                             ("barrier", "barrier_vs_none_pct", C["v4"])]:
        vals = [float(r[key]) for r in zipf]
        ax.plot(range(len(nvs)), vals, "o-", color=color, label=name,
                lw=1.6, ms=4)
    uni = {r["nv_gbps"]: r for r in data if r["workload"] == "uniform"}
    ax.scatter([nvs.index(nv) for nv in uni if nv in nvs],
               [float(uni[nv]["credit_vs_none_pct"]) for nv in uni
                if nv in nvs],
               marker="s", color=C["v2"], s=18, label="credit (uniform)")
    ax.set_xticks(range(len(nvs)))
    ax.set_xticklabels(nvs, fontsize=7)
    ax.set_xlabel("nv (Gbps)")
    ax.set_ylabel("overhead vs none (%)")
    ax.set_title("sync overhead: barrier ~12-14%, credit 2-4%\n"
                 "(except zipf@2400: +11.5%)", fontsize=8.5)
    ax.legend(fontsize=7)

    # (c) stage-order fix at nv360: sync modes benefit, async unchanged
    ax = axes[2]
    n360 = [r for r in zipf if r["nv_gbps"] == "360"][0]
    cases = [("none", "none_plan_us", "none_desc_us"),
             ("credit", "credit_plan_us", "credit_desc_us"),
             ("barrier", "barrier_plan_us", "barrier_desc_us")]
    xs = range(len(cases))
    w = 0.38
    b1 = ax.bar([x - w / 2 for x in xs],
                [float(n360[c[1]]) / 1e3 for c in cases], w,
                color=C["ecmp"], label="plan order (legacy)", zorder=3)
    b2 = ax.bar([x + w / 2 for x in xs],
                [float(n360[c[2]]) / 1e3 for c in cases], w,
                color=C["fast"], label="desc order (official)", zorder=3)
    label_bars(ax, b1, fmt="{:.1f}")
    label_bars(ax, b2, fmt="{:.1f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cases], fontsize=7)
    ax.set_ylabel("completion (ms)")
    ax.set_title("nv360: largest-freq-first stage order\n"
                 "helps sync modes, async bit-identical", fontsize=8.5)
    ax.legend(fontsize=7)

    fig.suptitle("33: FAST execution-time stage synchronization "
                 "(none/credit/barrier)", fontsize=9.5, y=1.04)
    save(fig, "33_fast_sync")

if _has("41_ledger"):
    # 41: MOSAIC-L resource-ledger core vs the lambda-feedback scheme
    data = rows("41_ledger")
    def v41(suite, cell, scheme):
        for row in data:
            if (row["suite"] == suite and row["cell"] == cell
                    and row["scheme"] == scheme):
                return float(row["completion_us"])
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    ax = axes[0]
    nvs = [2400, 720, 600, 480, 420, 360, 180, 90]
    u = [v41("mainline_unified", "nv%d" % n, "mosaic_U_sota38")
         for n in nvs]
    lu = [v41("mainline_unified", "nv%d" % n,
              "mosaic_LU(ledger+PC+PACE)") for n in nvs]
    bd = [3327, 4532, 5229, 6181, 6798, 7556, 11330, 15105]
    xs = range(len(nvs))
    w = 0.35
    ax.bar([x - w / 2 for x in xs], [a / 1e3 for a in u], w,
           color=C["fast"], label="MOSAIC-U (38, lambda feedback)",
           zorder=3)
    ax.bar([x + w / 2 for x in xs], [a / 1e3 for a in lu], w,
           color=C["mosaic"], label="MOSAIC-LU (41, resource ledger)",
           zorder=3)
    for x, b in enumerate(bd):
        ax.plot([x - 0.45, x + 0.45], [b / 1e3] * 2, color="gray",
                ls="--", lw=1, zorder=4)
    for x, (a, b) in enumerate(zip(u, lu)):
        if a / b >= 1.005:
            ax.text(x + w / 2, b / 1e3, "%.1f%%" % ((a / b - 1) * 100),
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(n) for n in nvs], fontsize=7)
    ax.set_xlabel("nv_gbps")
    ax.set_ylabel("completion (ms)")
    ax.set_title("mainline: ledger core beats the SOTA stack at 7/8\n"
                 "(dashes = cut bound; nv90 -5.0% honest regression)",
                 fontsize=8)
    ax.legend(fontsize=6.5)
    # (b) the exogenous axis: unannounced NIC degrade
    ax = axes[1]
    names = ["hot x2", "hot x4", "cold x2", "cold x4"]
    c4 = [4669.21, 9253.09, 4540.18, 9007.39]
    led = [v41("degrade", "hotx2@2400", "mosaic_ledger"),
           v41("degrade", "hotx4@2400", "mosaic_ledger"),
           v41("degrade", "coldx2@2400", "mosaic_ledger"),
           v41("degrade", "coldx4@2400", "mosaic_ledger")]
    fast = [6864, 13301, 4703, 9356]
    xs2 = range(4)
    w2 = 0.28
    ax.bar([x - w2 for x in xs2], [a / 1e3 for a in fast], w2,
           color=C["ecmp"], label="FAST (static plan)", zorder=3)
    ax.bar(list(xs2), [a / 1e3 for a in c4], w2, color=C["fast"],
           label="MOSAIC lambda-feedback (19)", zorder=3)
    ax.bar([x + w2 for x in xs2], [a / 1e3 for a in led], w2,
           color=C["mosaic"], label="MOSAIC-L (per-NIC detector)",
           zorder=3)
    for x, (a, b) in enumerate(zip(c4, led)):
        ax.text(x + w2, b / 1e3, "%.2fx" % (a / b), ha="center",
                va="bottom", fontsize=6.5)
    ax.set_xticks(list(xs2))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("completion (ms)")
    ax.set_title("unannounced NIC degrade @2400 (scheduler not told):\n"
                 "per-resource detection+reroute lands +4.3% off the "
                 "degraded cut", fontsize=8)
    ax.legend(fontsize=6.5)
    fig.suptitle("41: MOSAIC-L -- bookkeeping replaces learning"
                 " (foundation-audit verdict)", fontsize=9.5, y=1.04)
    save(fig, "41_ledger")
