#!/usr/bin/env python3
"""37 组图:incast1 的 min-buffer 阈值(wave 定律)。数据从 results.csv 读。"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))

fig, ax1 = plt.subplots(1, 1, figsize=(5.6, 4.2))

# min buffer to PASS, incast1
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
ax1.set_title("incast1: buffer = C/W (MOSAIC) vs ~0.9C (blast)")
ax1.tick_params(axis="x", rotation=20)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "plot.png"), dpi=130)
print("wrote plot.png")
