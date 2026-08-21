#!/usr/bin/env python3
"""Generate a seeded non-uniform (MoE-like) alltoallv demand matrix.

Model: hot-destination Zipf. A seeded permutation ranks the N nodes; node
with rank r has weight 1/(r+1)^alpha. Every source splits the SAME total
egress volume as the uniform contract -- (N-1) * message_bytes -- across
destinations proportionally to those weights, rounded to `granularity`
(default 1KB = packet size) with the residual folded into the source's
hottest destination so per-source totals stay exact. Hot destinations thus
receive far more than average (receiver incast/straggler skew, the case FAST
targets), while total volume matches the uniform experiment bit-for-bit.

Output demand file format (consumed by htsim -demand_file and the fill tool):
    node_count N
    demand <src> <dst> <bytes>

Also prints max pair bytes and the power-of-two padded plan size that
uniform-only solvers (TE-CCL/TACCL/SyCCL) must be run with.
"""

import argparse
import random


def next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def build_matrix(nodes, message_bytes, alpha, seed, granularity,
                 mode="hotdst"):
    rng = random.Random(seed)
    perm = list(range(nodes))
    rng.shuffle(perm)
    rank = {node: r for r, node in enumerate(perm)}
    if mode == "khot":
        # K equally-hot destinations (weight boost 8x), rest flat: the
        # multi-hotspot shape reviewers ask about; alpha reused as K.
        k = max(1, int(alpha))
        weight = {d: (8.0 if rank[d] < k else 1.0) for d in range(nodes)}
    else:
        weight = {d: 1.0 / ((rank[d] + 1) ** alpha) for d in range(nodes)}

    total_per_src = (nodes - 1) * message_bytes
    matrix = [[0] * nodes for _ in range(nodes)]
    for s in range(nodes):
        if mode == "persrc":
            # Independent destination ranking per source: per-destination
            # inflow STRUCTURE varies across destinations (unlike hotdst,
            # where every dst has the same same:cross ratio), so
            # per-destination corridor priors genuinely differ and
            # distribution drift actually moves them.
            sperm = list(range(nodes))
            rng.shuffle(sperm)
            srank = {node: r for r, node in enumerate(sperm)}
            weight = {d: 1.0 / ((srank[d] + 1) ** alpha)
                      for d in range(nodes)}
        wsum = sum(weight[d] for d in range(nodes) if d != s)
        assigned = 0
        hottest = None
        for d in range(nodes):
            if d == s:
                continue
            raw = total_per_src * weight[d] / wsum
            b = max(granularity, int(raw // granularity) * granularity)
            matrix[s][d] = b
            assigned += b
            if hottest is None or weight[d] > weight[hottest]:
                hottest = d
        # Fold rounding residual into the hottest destination; keep >= 1 granule.
        residual = total_per_src - assigned
        matrix[s][hottest] = max(granularity, matrix[s][hottest] + residual)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--message-bytes", type=int, default=2097152,
                        help="uniform per-pair bytes whose total volume is preserved")
    parser.add_argument("--alpha", type=float, default=1.0, help="Zipf exponent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--granularity", type=int, default=1024)
    parser.add_argument("--mode", choices=["hotdst", "persrc", "khot"],
                        default="hotdst")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    matrix = build_matrix(args.nodes, args.message_bytes, args.alpha,
                          args.seed, args.granularity, args.mode)

    max_pair = max(matrix[s][d] for s in range(args.nodes)
                   for d in range(args.nodes) if s != d)
    min_pair = min(matrix[s][d] for s in range(args.nodes)
                   for d in range(args.nodes) if s != d)
    padded = next_pow2(max_pair)
    in_totals = [sum(matrix[s][d] for s in range(args.nodes) if s != d)
                 for d in range(args.nodes)]

    with open(args.output, "w") as out:
        out.write("# nonuniform A2A demand: hot-dst zipf alpha=%g seed=%d "
                  "base=%dB granularity=%dB\n"
                  % (args.alpha, args.seed, args.message_bytes,
                     args.granularity))
        out.write("# max_pair_bytes %d padded_pow2_bytes %d\n"
                  % (max_pair, padded))
        out.write("node_count %d\n" % args.nodes)
        for s in range(args.nodes):
            for d in range(args.nodes):
                if s != d and matrix[s][d] > 0:
                    out.write("demand %d %d %d\n" % (s, d, matrix[s][d]))

    total = sum(sum(row) for row in matrix)
    print("nodes %d total_bytes %d max_pair_bytes %d min_pair_bytes %d "
          "padded_pow2_bytes %d incast_max %d incast_min %d"
          % (args.nodes, total, max_pair, min_pair, padded,
             max(in_totals), min(in_totals)))


if __name__ == "__main__":
    main()
