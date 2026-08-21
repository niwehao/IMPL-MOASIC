#!/usr/bin/env python3
"""Harvest (arXiv:2602.09188) topology-synthesis reproduction.

Two parts:

(1) --collective rd : the paper's Recursive-Doubling specialization,
    optimizer-free. Steps s = 0..log2(n)-1; step s exchanges m bytes
    bidirectionally between u and u +/- 2^s (mod n). For an interval
    [a,b] of steps run without reconfiguration, the optimal topology is
    the step-a pattern (paper Obs. 4): directed cycles u -> u+2^a and
    u -> u-2^a (degree 2). Step t in the interval then routes offset 2^t
    in 2^(t-a) hops, every circuit carries 2^(t-a) flows, so with C =
    per-circuit bandwidth the per-step makespan in our fabric physics is

        DCT(t | a) = alpha + 2^(t-a) * (8m/C) + 2^(t-a) * delta.

    Interval-partition DP (Harvest's DP with the per-interval closed
    form; the reconfiguration count is implicit in the partition):

        F(a) = min_{b >= a} [ R + sum_{t=a}^{b} DCT(t|a) + F(b+1) ],

    O(log^2 n) states -- the paper's polylog claim. Baselines: static
    ring (one interval [0, s-1] anchored at a=0, i.e. +/-1 ring) and
    BvN-per-step (every step its own interval).

(2) --collective a2a --demand-file F : theta-optimal single topology for
    a stationary demand sequence (the paper's MISOCP subproblem; with W
    identical steps the interval DP degenerates to ONE topology, so this
    IS the Harvest plan for stationary nonuniform A2A). Gurobi MILP:

        max theta
        s.t. sum_v x_uv <= k, sum_v x_vu <= k          (transceivers)
             flow conservation per destination d:
               sum_u f^d(u,v) - sum_w f^d(v,w) = -theta*D(v,d)   v != d
               sum_u f^d(u,d) = theta * sum_s D(s,d)
             sum_d f^d(u,v) <= C * x_uv                (circuit capacity)

    followed by flow decomposition into paths -> plan transfers (split
    paths allowed, matching the maximum-concurrent-flow semantics).

Outputs a plan file for htsim_optical.
"""

import argparse
import math
import sys
from collections import defaultdict


def rd_dct_us(t, a, m_bytes, gbps, delay_ns, alpha_us):
    hops = 2 ** (t - a)
    ser = 8.0 * m_bytes / gbps / 1e3          # us for m bytes on 1 circuit
    return alpha_us + hops * ser + hops * delay_ns / 1e3


def rd_dp(steps, m_bytes, gbps, delay_ns, alpha_us, reconf_us):
    """Interval-partition DP. Returns (cost, list of (a, b) intervals)."""
    INF = float("inf")
    F = [INF] * (steps + 1)
    nxt = [None] * (steps + 1)
    F[steps] = 0.0
    for a in range(steps - 1, -1, -1):
        for b in range(a, steps):
            c = reconf_us + sum(
                rd_dct_us(t, a, m_bytes, gbps, delay_ns, alpha_us)
                for t in range(a, b + 1)) + F[b + 1]
            if c < F[a]:
                F[a] = c
                nxt[a] = b
    ivs, a = [], 0
    while a < steps:
        ivs.append((a, nxt[a]))
        a = nxt[a] + 1
    return F[0], ivs


def rd_emit(out, n, ivs, m_bytes, args):
    print("nodes %d degree 2" % n, file=out)
    print("gbps %g delay_ns %g reconf_us %g chunk_bytes %d"
          % (args.gbps, args.delay_ns, args.reconf_us, args.chunk_bytes),
          file=out)
    print("pay_initial", file=out)
    if args.alpha_us > 0:
        print("phase_alpha_us %g" % args.alpha_us, file=out)
    for (a, b) in ivs:
        base = 2 ** a
        for t in range(a, b + 1):
            hops = 2 ** (t - a)
            print("phase  # interval [%d,%d] topo +/-%d, step %d (%d hops)"
                  % (a, b, base, t, hops), file=out)
            for u in range(n):
                print("edge %d %d" % (u, (u + base) % n), file=out)
                print("edge %d %d" % (u, (u - base) % n), file=out)
            half = (2 * hops * base) % n == 0   # offset n/2: fwd == bwd
            for u in range(n):
                fwd = [(u + i * base) % n for i in range(hops + 1)]
                bwd = [(u - i * base) % n for i in range(hops + 1)]
                print("xfer %d %d %d via %s" % (u, fwd[-1], m_bytes,
                      " ".join(map(str, fwd))), file=out)
                if not half:
                    print("xfer %d %d %d via %s" % (u, bwd[-1], m_bytes,
                          " ".join(map(str, bwd))), file=out)


def read_demand(path, n):
    dem = [[0] * n for _ in range(n)]
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4 and p[0] == "demand":
                s, d, b = int(p[1]), int(p[2]), int(p[3])
                dem[s][d] = b
    return dem


def a2a_theta_opt(n, k, dem, gbps, time_limit):
    import gurobipy as gp
    from gurobipy import GRB
    mdl = gp.Model("harvest_theta")
    mdl.Params.OutputFlag = 0
    mdl.Params.TimeLimit = time_limit
    arcs = [(u, v) for u in range(n) for v in range(n) if u != v]
    x = mdl.addVars(arcs, vtype=GRB.BINARY, name="x")
    f = mdl.addVars([(d, u, v) for d in range(n) for (u, v) in arcs],
                    lb=0.0, name="f")
    th = mdl.addVar(lb=0.0, name="theta")
    for u in range(n):
        mdl.addConstr(gp.quicksum(x[u, v] for v in range(n) if v != u) <= k)
        mdl.addConstr(gp.quicksum(x[v, u] for v in range(n) if v != u) <= k)
    C = 1.0                                    # normalized circuit capacity
    tot = sum(map(sum, dem))
    scale = 1.0 / max(1, tot)                  # demands normalized
    for d in range(n):
        col = sum(dem[s][d] for s in range(n))
        for v in range(n):
            inflow = gp.quicksum(f[d, u, v] for u in range(n) if u != v)
            outflow = gp.quicksum(f[d, v, w] for w in range(n) if w != v)
            if v == d:
                mdl.addConstr(inflow - outflow == th * col * scale)
            else:
                mdl.addConstr(outflow - inflow == th * dem[v][d] * scale)
    for (u, v) in arcs:
        mdl.addConstr(gp.quicksum(f[d, u, v] for d in range(n)) <= C * x[u, v])
    mdl.setObjective(th, GRB.MAXIMIZE)
    mdl.optimize()
    if mdl.SolCount == 0:
        return None
    edges = [(u, v) for (u, v) in arcs if x[u, v].X > 0.5]
    live = set(edges)
    # keep only flow on BUILT arcs and above solver feasibility noise
    flow = {(d, u, v): f[d, u, v].X for d in range(n) for (u, v) in arcs
            if f[d, u, v].X > 1e-6 and (u, v) in live}
    return th.X, edges, flow, scale


def bfs_edges(n, edges, s, d):
    adj = defaultdict(list)
    for (u, v) in edges:
        adj[u].append(v)
    prev = {s: None}
    q = [s]
    while q:
        u = q.pop(0)
        if u == d:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    if d not in prev:
        return None
    path = []
    v = d
    while v is not None:
        path.append(v)
        v = prev[v]
    return list(reversed(path))


def decompose_paths(n, dem, flow, scale, theta, edges=None):
    """Per (s,d): decompose destination-aggregated flow into s->d paths.
    Standard successive shortest-path peeling on the positive-flow DAG."""
    plans = defaultdict(list)                  # (s,d) -> [(bytes, path)]
    for d in range(n):
        g = defaultdict(dict)
        for (dd, u, v), val in flow.items():
            if dd == d:
                g[u][v] = val
        for s in range(n):
            need = dem[s][d]
            if need <= 0 or s == d:
                continue
            remaining = float(need)
            guard = 0
            while remaining > 1e-6 and guard < 4 * n:
                guard += 1
                # BFS a positive-capacity path s -> d
                prev = {s: None}
                q = [s]
                while q:
                    u = q.pop(0)
                    if u == d:
                        break
                    for v, c in g[u].items():
                        if c > 1e-12 and v not in prev:
                            prev[v] = u
                            q.append(v)
                if d not in prev:
                    break
                path = []
                v = d
                while v is not None:
                    path.append(v)
                    v = prev[v]
                path.reverse()
                cap = min(g[path[i]][path[i + 1]]
                          for i in range(len(path) - 1))
                # flow units are theta * bytes * scale for the whole
                # matrix; convert capacity back to bytes of THIS pair
                pair_flow = theta * need * scale
                take_units = min(cap, pair_flow)
                take_bytes = remaining if abs(take_units - pair_flow) < 1e-12 \
                    else remaining * (take_units / pair_flow)
                take_bytes = min(take_bytes, remaining)
                for i in range(len(path) - 1):
                    g[path[i]][path[i + 1]] -= take_units
                plans[(s, d)].append((int(round(take_bytes)), path))
                remaining -= take_bytes
            if remaining > 1e-6:
                # residual rides the first found path (rounding slack);
                # under drift a pair may have no prior flow at all --
                # fall back to a BFS route on the built topology.
                if plans[(s, d)]:
                    b, pth = plans[(s, d)][0]
                    plans[(s, d)][0] = (b + int(round(remaining)), pth)
                elif edges is not None:
                    pth = bfs_edges(n, edges, s, d)
                    if pth is None:
                        return None            # genuinely disconnected
                    plans[(s, d)].append((int(round(remaining)), pth))
                else:
                    return None
    return plans


def a2a_emit(out, n, k, edges, plans, args):
    print("nodes %d degree %d" % (n, k), file=out)
    print("gbps %g delay_ns %g reconf_us %g chunk_bytes %d"
          % (args.gbps, args.delay_ns, args.reconf_us, args.chunk_bytes),
          file=out)
    print("pay_initial", file=out)
    print("phase  # theta-optimal static topology", file=out)
    for (u, v) in edges:
        print("edge %d %d" % (u, v), file=out)
    for (s, d), lst in sorted(plans.items()):
        for (b, path) in lst:
            if b > 0:
                print("xfer %d %d %d via %s"
                      % (s, d, b, " ".join(map(str, path))), file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collective", required=True,
                    help="rd | a2a")
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--strategy", default="harvest",
                    help="rd: harvest | static | bvn")
    ap.add_argument("--msg-bytes", type=int, default=262144)
    ap.add_argument("--demand-file", default=None)
    ap.add_argument("--actual-file", default=None,
                    help="drift mode: solve theta on --demand-file (the "
                         "stale prior) but ship --actual-file bytes")
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--gbps", type=float, default=800.0)
    ap.add_argument("--delay-ns", type=float, default=500.0)
    ap.add_argument("--alpha-us", type=float, default=0.5)
    ap.add_argument("--reconf-us", type=float, default=10.0)
    ap.add_argument("--chunk-bytes", type=int, default=262144)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--output", default="-")
    args = ap.parse_args()
    n = args.nodes
    out = sys.stdout if args.output == "-" else open(args.output, "w")

    if args.collective == "rd":
        assert n & (n - 1) == 0, "rd needs power-of-two n"
        steps = int(math.log2(n))
        if args.strategy == "harvest":
            cost, ivs = rd_dp(steps, args.msg_bytes, args.gbps,
                              args.delay_ns, args.alpha_us, args.reconf_us)
        elif args.strategy == "static":
            ivs = [(0, steps - 1)]
            cost = args.reconf_us + sum(
                rd_dct_us(t, 0, args.msg_bytes, args.gbps, args.delay_ns,
                          args.alpha_us) for t in range(steps))
        else:                                   # bvn: reconfig every step
            ivs = [(t, t) for t in range(steps)]
            cost = sum(args.reconf_us + rd_dct_us(t, t, args.msg_bytes,
                       args.gbps, args.delay_ns, args.alpha_us)
                       for t in range(steps))
        sys.stderr.write("harvest_rd n=%d strategy=%s intervals=%s "
                         "model_cost=%.2fus\n"
                         % (n, args.strategy, ivs, cost))
        rd_emit(out, n, ivs, args.msg_bytes, args)
    elif args.collective == "a2a":
        dem = read_demand(args.demand_file, n)
        res = a2a_theta_opt(n, args.degree, dem, args.gbps, args.time_limit)
        if res is None:
            sys.stderr.write("harvest_a2a SKIPPED (no solution in limit)\n")
            sys.exit(3)
        theta, edges, flow, scale = res
        if args.actual_file:
            dem = read_demand(args.actual_file, n)   # ship actual bytes
        plans = decompose_paths(n, dem, flow, scale, theta, edges)
        if plans is None:
            sys.stderr.write("harvest_a2a decomposition failed\n")
            sys.exit(3)
        # Concurrent-flow property: every pair finishes simultaneously at
        # T = 8*tot / (theta * gbps)  (scale = 1/tot cancels).
        tot = sum(map(sum, dem))
        model_us = 8.0 * tot / (theta * args.gbps * 1e3) if theta > 0 else -1
        sys.stderr.write("harvest_a2a n=%d k=%d theta=%.6f edges=%d "
                         "model_time_us=%.2f\n"
                         % (n, args.degree, theta, len(edges), model_us))
        a2a_emit(out, n, args.degree, edges, plans, args)
    else:
        sys.exit(2)
    if out is not sys.stdout:
        out.close()


if __name__ == "__main__":
    main()
