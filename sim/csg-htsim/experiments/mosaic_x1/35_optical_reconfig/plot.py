#!/usr/bin/env python3
"""35_optical: completion time vs reconfiguration delay R, per demand."""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))

demands = ["uniform", "zipf10", "zipf20", "persrc"]
titles = {"uniform": "uniform 2MB/pair", "zipf10": "hot-dst zipf a=1.0",
          "zipf20": "hot-dst zipf a=2.0", "persrc": "persrc a=1.0"}
schemes = [("ron_auto", "o-", "RoN (model d)"),
           ("ron_best", "s--", "RoN (oracle d)"),
           ("harvest", "^-", "Harvest theta-opt"),
           ("mosaic_o", "d-", "MOSAIC-O (online)")]

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)
for ax, dem in zip(axes, demands):
    lb = None
    for r in rows:
        if r["demand"] == dem and r["scheme"] == "lower_bound":
            lb = float(r["completion_us"])
    for key, style, label in schemes:
        xs, ys = [], []
        for r in rows:
            if r["demand"] != dem or r["scheme"] != key:
                continue
            try:
                xs.append(float(r["R_us"]))
                ys.append(float(r["completion_us"]))
            except ValueError:
                continue
        if xs:
            ax.plot(xs, ys, style, label=label, ms=5)
    if lb:
        ax.axhline(lb, color="gray", ls=":", label="max-line cut LB")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("reconfiguration delay R (us)")
    ax.set_title(titles[dem])
    ax.grid(alpha=0.3)
axes[0].set_ylabel("completion time (us)")
axes[0].legend(fontsize=8)
fig.suptitle("n=32, k=1 photonic scale-up domain, 800G/500ns, "
             "store-and-forward chunks (pay-initial R accounting)")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plot.png"), dpi=130)
print("wrote plot.png")
