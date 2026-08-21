#!/usr/bin/env python3
"""38 组图:MOSAIC-U vs 历史特调 SOTA 的逐 cell 差值(%)。"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))
rows = [r for r in rows if r["delta_pct"] not in ("0",)]
cells = [r["cell"] for r in rows]
deltas = [float(r["delta_pct"].replace("+", "")) for r in rows]
colors = ["#2b8cbe" if d < 0 else "#e34a33" for d in deltas]

fig, ax = plt.subplots(figsize=(11, 5))
ax.barh(cells, deltas, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("MOSAIC-U vs prior SOTA (%)  [blue = new SOTA]")
ax.set_title("38_unified: one scheme, all regimes (ties omitted)")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plot.png"), dpi=130)
print("wrote plot.png")
