#!/usr/bin/env python3
"""Reconfigure-or-Not (arXiv:2602.10468) strategy generator, degree-1.

Mathematical model
------------------
A degree-1 circuit configuration is a permutation; the connected ones used
by the paper are the Cayley graphs Cay(Z_n, {s}) : u -> u+s (mod n).
Let g = gcd(s, n). Cay(Z_n, {s}) decomposes into g disjoint cycles of
length n/g; destination offset o (= d-s mod n) is reachable iff g | o,
with hop count

    h_s(o) = (o/g) * (s/g)^{-1}  (mod n/g),   h in [1, n/g-1]

(the discrete log is closed-form because Z_n is cyclic).

A strategy with d topologies <s_1..s_d> assigns every offset o in
{1..n-1} to one topology; round (s_j, o) sends all n flows u -> u+o along
the h_s_j(o)-hop path. On a shift-s cycle each round is contention-free
and its store-and-forward makespan (verified against the simulator) is

    t(o) = h * ceil(m / c) * T_c   with T_c = 8c/BW  (chunk c, pair m)
         ~ h * m / BW  for m >> c, plus (h-1) pipeline-fill chunks.

Exact per-round makespan used in the cost model (n flows, nc chunks,
each link carries h flows):  t = (h*nc) * T_c + (h-1)*delta, where the
(h-1) chain-fill term is absorbed by pipelining across the h*nc link
slots (matches the simulator to the chunk).

Total cost:  C(d) = sum_o t(o) + d*R    (paper bookkeeping: the initial
configuration also pays R).

Sequence generation = greedy contracting (paper Sec 4.3): starting from
the base ring s=1, repeatedly direct-connect the currently most expensive
round (shift s := argmax_o h(o)), reassigning every offset to its
cheapest topology. The first contraction picks s = n-1 (the paper's
"prioritize the (n-1)-shift").

Lower bound (paper Theorem 1, k=1, d topologies):

    LB(d) = d*R + T_m * ( d*q(q+1)/2 + u*(q+1) ),
    q = floor((n-1)/d), u = (n-1) mod d,  T_m = m*8/BW.

Usage:
  ron_gen.py --nodes 8 --strategy auto --reconf-us 10 \
             --bytes-per-pair 8388608 --output plan.txt
  --strategy {auto,static,everyround,d<INT>}  --lb  (print bound table)
"""

import argparse
import math
import sys


def hop_count(s, o, n):
    """Hops to reach offset o on Cay(Z_n, {s}); None if unreachable."""
    g = math.gcd(s, n)
    if o % g != 0:
        return None
    m = n // g
    if m == 1:
        return None
    so = (s // g) % m
    inv = pow(so, -1, m)
    h = ((o // g) * inv) % m
    return None if h == 0 else h


def greedy_strategy(n, d):
    """d topologies via greedy contracting. Returns (shifts, assign)
    where assign[o] = (topo_index, hops)."""
    shifts = [1]
    assign = {o: (0, o) for o in range(1, n)}   # base ring: h(o) = o
    while len(shifts) < d:
        o_star = max(assign, key=lambda o: assign[o][1])
        s_new = o_star
        if s_new in shifts:
            break                                # nothing left to contract
        shifts.append(s_new)
        j = len(shifts) - 1
        for o in assign:
            h = hop_count(s_new, o, n)
            if h is not None and h < assign[o][1]:
                assign[o] = (j, h)
    return shifts, assign


def strategy_cost(n, assign, num_topos, m_bytes, gbps, reconf_us,
                  chunk_bytes, delay_ns):
    tc = chunk_bytes * 8.0 / gbps / 1e3          # us per chunk per hop
    nc = math.ceil(m_bytes / chunk_bytes)
    trans = 0.0
    for o, (_, h) in assign.items():
        trans += h * nc * tc + h * delay_ns / 1e3
    return trans + num_topos * reconf_us


def theorem1_lb(n, d, m_bytes, gbps, reconf_us):
    tm = m_bytes * 8.0 / gbps / 1e3              # us to send one flow 1 hop
    q = (n - 1) // d
    u = (n - 1) % d
    return d * reconf_us + tm * (d * q * (q + 1) / 2.0 + u * (q + 1))


def emit_plan(out, n, shifts, assign, m_bytes, args, demand=None):
    print("nodes %d degree 1" % n, file=out)
    print("gbps %g delay_ns %g reconf_us %g chunk_bytes %d"
          % (args.gbps, args.delay_ns, args.reconf_us, args.chunk_bytes),
          file=out)
    print("pay_initial", file=out)
    for j, s in enumerate(shifts):
        offsets = sorted([o for o in assign if assign[o][0] == j],
                         key=lambda o: assign[o][1])
        if not offsets:
            continue
        for o in offsets:                        # one round per offset
            h = assign[o][1]
            print("phase  # topo shift %d, offset %d, %d hops" % (s, o, h),
                  file=out)
            for u in range(n):
                print("edge %d %d" % (u, (u + s) % n), file=out)
            for u in range(n):
                path = [(u + t * s) % n for t in range(h + 1)]
                assert path[-1] == (u + o) % n
                b = m_bytes if demand is None else demand[u][(u + o) % n]
                if b <= 0:
                    continue
                print("xfer %d %d %d via %s"
                      % (u, (u + o) % n, b, " ".join(map(str, path))),
                      file=out)


def read_demand(path, n):
    dem = [[0] * n for _ in range(n)]
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4 and parts[0] == "demand":
                s, d, b = int(parts[1]), int(parts[2]), int(parts[3])
                if s < n and d < n:
                    dem[s][d] = b
    return dem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--strategy", default="auto",
                    help="auto | static | everyround | d<INT>")
    ap.add_argument("--bytes-per-pair", type=int, default=8388608)
    ap.add_argument("--gbps", type=float, default=800.0)
    ap.add_argument("--delay-ns", type=float, default=500.0)
    ap.add_argument("--reconf-us", type=float, default=10.0)
    ap.add_argument("--chunk-bytes", type=int, default=4194304)
    ap.add_argument("--demand-file", default=None)
    ap.add_argument("--output", default="-")
    ap.add_argument("--lb", action="store_true",
                    help="print the Theorem-1 bound table and exit")
    args = ap.parse_args()
    n = args.nodes
    m = args.bytes_per_pair

    if args.lb:
        best = None
        for d in range(1, n):
            lb = theorem1_lb(n, d, m, args.gbps, args.reconf_us)
            best = lb if best is None else min(best, lb)
            print("d=%d LB=%.2f us" % (d, lb))
        print("LB* = %.2f us" % best)
        return

    if args.strategy == "static":
        d_list = [1]
    elif args.strategy == "everyround":
        d_list = [n - 1]
    elif args.strategy.startswith("d"):
        d_list = [int(args.strategy[1:])]
    else:                                        # auto: scan d
        d_list = list(range(1, n))

    best = None
    for d in d_list:
        shifts, assign = greedy_strategy(n, d)
        cost = strategy_cost(n, assign, len(shifts), m, args.gbps,
                             args.reconf_us, args.chunk_bytes,
                             args.delay_ns)
        if best is None or cost < best[0]:
            best = (cost, shifts, assign)
    cost, shifts, assign = best
    lb = min(theorem1_lb(n, d, m, args.gbps, args.reconf_us)
             for d in range(1, n))
    sys.stderr.write(
        "ron_gen n=%d chose d=%d shifts=%s model_cost=%.2fus "
        "LB*=%.2fus ratio=%.2f\n"
        % (n, len(shifts), shifts, cost, lb, cost / lb if lb > 0 else 0))

    demand = read_demand(args.demand_file, n) if args.demand_file else None
    out = sys.stdout if args.output == "-" else open(args.output, "w")
    emit_plan(out, n, shifts, assign, m, args, demand)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
