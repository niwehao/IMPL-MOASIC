#!/usr/bin/env python3
"""37 组图:左 = EPS incast1 min-buffer 阈值(wave 定律);右 = OCS
m-sink incast 三象限 @R=1000(vs LB 倍数)。数据从 results.csv 读。"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- left: min buffer to PASS, incast1
bars = [("MOSAIC W=6", 512, "#2b8cbe"), ("MOSAIC W=12", 256, "#74a9cf"),
        ("MOSAIC W=24", 128, "#bdc9e1"), ("FAST", 2048, "#e34a33"),
        ("ECMP", 2048, "#fdbb84")]
names = [b[0] for b in bars]
vals = [b[1] for b in bars]
cols = [b[2] for b in bars]
ax1.bar(names, vals, color=cols)
for i, v in enumerate(vals):
    ax1.text(i, v * 1.05, "%dMB" % v, ha="center", fontsize=9)
ax1.axhline(1024, color="gray", ls="--", lw=0.8)
ax1.text(4.4, 1060, "default 1GB", fontsize=8, color="gray", ha="right")
ax1.set_yscale("log", base=2)
ax1.set_ylabel("min fabric buffer to PASS (MB, log2)")
ax1.set_title("EPS incast1: buffer = C/W (MOSAIC) vs ~0.9C (blast)")
ax1.tick_params(axis="x", rotation=20)

# ---- right: OCS xLB @R=1000 per m
ms = ["incast1", "incast2", "incast4", "incast8"]
schemes = [("ron_best", "RoN oracle", "#e34a33"),
           ("harvest", "Harvest static", "#31a354"),
           ("mosaic_o", "MOSAIC-O", "#2b8cbe")]
lb = {r["run"].replace("ocs_", "").replace("_LB", ""): float(r["completion_us"])
      for r in rows if r["run"].endswith("_LB")}
x = range(len(ms))
w = 0.26
for k, (key, label, col) in enumerate(schemes):
    ys = []
    for dem in ms:
        v = [r for r in rows
             if r["run"] == "ocs_%s_R1000_%s" % (dem, key)]
        ys.append(float(v[0]["completion_us"]) / lb[dem] if v else 0)
    ax2.bar([i + (k - 1) * w for i in x], ys, w, label=label, color=col)
ax2.axhline(1.0, color="black", lw=0.8)
ax2.set_xticks(list(x))
ax2.set_xticklabels(["m=1", "m=2", "m=4", "m=8"])
ax2.set_ylabel("completion / max-line LB")
ax2.set_title("OCS m-sink incast @R=1ms (lower=better)")
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "plot.png"), dpi=130)
print("wrote plot.png")
