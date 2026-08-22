#!/usr/bin/env python3
"""Assemble 37_switch_centric results.csv from the raw batch logs."""
import csv
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
EPS_LOG = os.path.join(ROOT, "cache/nonuniform/x37_eps.log")
PROBES = [os.path.join(ROOT, "cache/nonuniform/x37_probe_buf.txt"),
          os.path.join(ROOT, "cache/nonuniform/x37_probe_wave.txt")]

LB = {
    "p1_incast1": 161229.0, "p1_incast2": 83215.0,
    "p1_incast4": 41607.3, "p1_incast8": 20803.3,
    "p1b_incast1": 161229.0, "p1b_incast2": 83215.0,
    "p1b_incast4": 41607.3, "p1b_incast8": 20803.3,
    "p2_uni_up800": 5200.9, "p2_uni_up400": 8053.1,
    "p2_uni_up200": 16106.1, "p2_uni_up100": 32212.3,
    "p2_zipf_up800": 40781.3, "p2_zipf_up400": 40781.3,
    "p2_zipf_up200": 40781.3, "p2_zipf_up100": 47911.1,
    "p3_permcollide": 41607.5, "p3_permbal": 10401.9,
    "p4_mixed": 80614.5,
    "probe_incast1": 161229.0,
}

rows = []


def lb_for(tag):
    for k in sorted(LB, key=len, reverse=True):
        if tag.startswith(k):
            return LB[k]
    return None


def add(side, tag, status, comp, drops, note=""):
    lb = lb_for(tag)
    x = ""
    if lb and comp and status == "PASSED":
        x = "%.3f" % (float(comp) / lb)
    rows.append([side, tag, status, comp, "%.1f" % lb if lb else "", x,
                 drops, note])


pat = re.compile(r"^(\S+) validation (PASSED|FAILED) rounds 1 "
                 r"completion_us ([0-9.e+]+).*?queue_drops (\d+)")
for line in open(EPS_LOG):
    m = pat.match(line)
    if m:
        add("eps", m.group(1), m.group(2), m.group(3), m.group(4))
    elif " TIMEOUT" in line:
        add("eps", line.split()[0], "TIMEOUT", "", "", ">20min policy")

for pf in PROBES:
    if not os.path.exists(pf):
        continue
    for line in open(pf):
        m = re.match(r"^(\S+) validation (PASSED|FAILED) rounds 1 "
                     r"completion_us ([0-9.e+]+) queue_drops (\d+)", line)
        if m:
            add("eps", m.group(1), m.group(2), m.group(3), m.group(4))

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["side", "run", "status", "completion_us", "lb_us", "x_lb",
                "queue_drops", "notes"])
    w.writerows(rows)
print("wrote %s (%d rows)" % (out, len(rows)))
