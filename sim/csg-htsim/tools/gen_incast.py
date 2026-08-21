#!/usr/bin/env python3
"""Switch-centric (37 组) demand generators for the clos topology.

Modes (all keep the uniform-contract per-src budget T = (n-1)*base bytes,
granularity 1024B, seedless/deterministic):

  incast M    : M aggregator dsts {0..M-1}. Every src splits its whole
                budget equally over the aggregators it can reach (a src
                that is itself an aggregator sends to the other M-1; for
                M=1 the sole aggregator sends nothing -- pure incast, the
                receiver is a sink). Per-pair bytes floored to 1024.
  perm KIND   : single permutation, T bytes per pair (KIND=collide: all
                pairs pinned to one spine, s+d = n/2+1 const; KIND=
                balanced: brute-matched so (s+d) % spines is uniform).
  mixed       : src 0 sends uniformly (T/(n-1) each); every other src
                sends T/2 to dst 0 (incast half) + T/2 uniform over the
                remaining n-2 dsts (background half).

Emits the standard demand-file format (node_count / demand s d bytes with
max_pair_bytes + padded_pow2_bytes header).
"""
import argparse
import sys

GRAN = 1024


def floor_gran(x):
    return (int(x) // GRAN) * GRAN


def gen_incast(n, budget, m):
    # Strided aggregator placement (one per n/m block) so the endpoint
    # ingress cut, not the aggregators' shared leaf, is what binds --
    # isolates the pure incast phenomenon from "hot rack" placement.
    hot = [i * (n // m) for i in range(m)]
    d = [[0] * n for _ in range(n)]
    for s in range(n):
        targets = [t for t in hot if t != s]
        if not targets:
            continue
        per = floor_gran(budget / len(targets))
        for t in targets:
            d[s][t] = per
    return d


def gen_perm(n, budget, kind, spines):
    if kind == "collide":
        # s + d == n/2 + 1 (odd const => no fixed point), all pairs on
        # spine (n/2+1) % spines.
        c = n // 2 + 1
        perm = [(c - s) % n for s in range(n)]
    else:
        # Greedy: for each s pick an unused d with (s+d)%spines == the
        # least-filled spine class, no fixed point.
        used = set()
        counts = [0] * spines
        perm = [None] * n
        for s in range(n):
            best = None
            for spine in sorted(range(spines), key=lambda k: counts[k]):
                for dd in range(n):
                    if dd != s and dd not in used \
                            and (s + dd) % spines == spine:
                        best = (spine, dd)
                        break
                if best:
                    break
            if best is None:
                raise RuntimeError("balanced perm construction failed")
            counts[best[0]] += 1
            used.add(best[1])
            perm[s] = best[1]
        assert sorted(perm) == list(range(n))
    d = [[0] * n for _ in range(n)]
    for s in range(n):
        d[s][perm[s]] = budget
    return d


def gen_mixed(n, budget):
    d = [[0] * n for _ in range(n)]
    per_uni_full = floor_gran(budget / (n - 1))
    for t in range(1, n):
        d[0][t] = per_uni_full
    half = budget // 2
    per_bg = floor_gran(half / (n - 2))
    for s in range(1, n):
        d[s][0] = half
        for t in range(1, n):
            if t != s:
                d[s][t] = per_bg
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["incast", "perm", "mixed"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--nodes", type=int, default=32)
    ap.add_argument("--base", type=int, default=2097152,
                    help="uniform per-pair base; budget=(n-1)*base")
    ap.add_argument("--spines", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = args.nodes
    budget = (n - 1) * args.base
    if args.mode == "incast":
        m = int(args.arg)
        d = gen_incast(n, budget, m)
        desc = "incast m=%d" % m
    elif args.mode == "perm":
        d = gen_perm(n, budget, args.arg, args.spines)
        desc = "perm %s spines=%d" % (args.arg, args.spines)
    else:
        d = gen_mixed(n, budget)
        desc = "mixed incast+uniform"

    max_pair = max(max(row) for row in d)
    padded = 1
    while padded < max_pair:
        padded *= 2
    with open(args.out, "w") as f:
        f.write("# switch-centric demand: %s budget=%dB granularity=%dB\n"
                % (desc, budget, GRAN))
        f.write("# max_pair_bytes %d padded_pow2_bytes %d\n"
                % (max_pair, padded))
        f.write("node_count %d\n" % n)
        for s in range(n):
            for t in range(n):
                if d[s][t] > 0:
                    f.write("demand %d %d %d\n" % (s, t, d[s][t]))
    tot = sum(sum(row) for row in d)
    sys.stderr.write("%s: total %dB max_pair %dB padded %dB -> %s\n"
                     % (desc, tot, max_pair, padded, args.out))


if __name__ == "__main__":
    main()
