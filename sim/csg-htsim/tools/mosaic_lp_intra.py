#!/usr/bin/env python3
"""MOSAIC-LP: workload-free intra-host min-congestion LP (per host type).

Fixed topology => LP; unknown workload => runtime lambda adjustment.
For every local destination d, solve ONE small multicommodity-flow LP on
the intra-host graph: a unit commodity from every other vertex j to d,
all simultaneously active (the workload-free uniform prior), minimizing
the max link utilization t (+ eps * total flow to suppress detours).
The per-commodity flow is then decomposed into <=3-link paths with
weights; at run time MOSAIC splits a relay corridor's intra segment
across these paths (fixed ratios), while lambda keeps adjusting the
corridor mix per dst from measured feedback.

Cost argument: the LP never sees the demand matrix, so it runs once per
host TYPE at topology-acquisition time. 8 GPUs => 8 LPs of ~230 vars
(milliseconds), vs minutes for a whole-fabric solver LP; identical hosts
share one table, so the fabric level never appears in any LP.

Output (one table per topology):
  # mosaic_lp_intra v1 gpus 8 t_opt <max over d> solve_ms <total>
  path <d> <j> <weight> <v0> <v1> [... vd]    # vertices j..d inclusive
"""
import argparse
import time

import numpy as np
from scipy.optimize import linprog

CUBE_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6), (4, 7),
    (5, 6), (5, 7),
    (6, 7),
]


def local_matrix(gpus, spec, rot, full_gbps):
    local = [[0.0] * gpus for _ in range(gpus)]
    if full_gbps and full_gbps > 0.0:
        for a in range(gpus):
            for b in range(gpus):
                if a != b:
                    local[a][b] = full_gbps
        return local
    assert gpus == 8, "cube-mesh mode requires gpus_per_host = 8"
    tiers = []
    for item in spec.split(","):
        gbps, count = item.split("x")
        tiers.extend([float(gbps)] * int(count))
    assert len(tiers) == len(CUBE_EDGES)
    for e, (a, b) in enumerate(CUBE_EDGES):
        s = tiers[(e + rot) % len(tiers)]
        local[a][b] = s
        local[b][a] = s
    return local


def solve_dst(local, d, eps=1e-4):
    """Min-congestion LP for all unit commodities j -> d. Returns
    {j: [(weight, path)]} and the optimal congestion t."""
    g = len(local)
    edges = [(u, v) for u in range(g) for v in range(g)
             if local[u][v] > 0.0]
    ei = {e: i for i, e in enumerate(edges)}
    coms = [j for j in range(g) if j != d]
    nf = len(coms) * len(edges)
    nv = nf + 1  # + t

    def fv(ci, e):
        return ci * len(edges) + ei[e]

    c = np.full(nv, eps)
    c[nf] = 1.0
    a_eq, b_eq = [], []
    for ci, j in enumerate(coms):
        for v in range(g):
            if v == d:
                continue
            row = np.zeros(nv)
            for (u, w) in edges:
                if w == v:
                    row[fv(ci, (u, w))] += 1.0
                if u == v:
                    row[fv(ci, (u, w))] -= 1.0
            a_eq.append(row)
            b_eq.append(-1.0 if v == j else 0.0)
    a_ub, b_ub = [], []
    for e in edges:
        row = np.zeros(nv)
        for ci in range(len(coms)):
            row[fv(ci, e)] = 1.0 / local[e[0]][e[1]]
        row[nf] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)
    res = linprog(c, A_ub=np.array(a_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(a_eq), b_eq=np.array(b_eq),
                  bounds=[(0, None)] * nv, method="highs")
    assert res.success, res.message
    out = {}
    for ci, j in enumerate(coms):
        flow = {e: res.x[fv(ci, e)] for e in edges
                if res.x[fv(ci, e)] > 1e-9}
        out[j] = decompose(flow, j, d)
    return out, res.x[nf]


def decompose(flow, src, dst, max_links=3):
    """Greedy path decomposition of a unit flow; drops >max_links paths
    (cube diameter is 2, so nothing real is lost) and renormalizes."""
    paths = []
    flow = dict(flow)
    guard = 64
    while guard > 0:
        guard -= 1
        # walk highest-flow outgoing edges src -> dst
        path = [src]
        seen = {src}
        while path[-1] != dst:
            cands = [(e, f) for e, f in flow.items()
                     if e[0] == path[-1] and f > 1e-9 and e[1] not in seen]
            if not cands:
                path = None
                break
            e, _ = max(cands, key=lambda x: x[1])
            path.append(e[1])
            seen.add(e[1])
        if path is None:
            break
        w = min(flow[(path[i], path[i + 1])] for i in range(len(path) - 1))
        for i in range(len(path) - 1):
            flow[(path[i], path[i + 1])] -= w
        if len(path) - 1 <= max_links:
            paths.append((w, path))
        if sum(f for f in flow.values()) < 1e-6:
            break
    total = sum(w for w, _ in paths)
    if total <= 0:
        raise RuntimeError("empty decomposition %d->%d" % (src, dst))
    # merge identical paths, normalize, drop dust below 2%
    merged = {}
    for w, p in paths:
        merged[tuple(p)] = merged.get(tuple(p), 0.0) + w / total
    keep = {p: w for p, w in merged.items() if w >= 0.02}
    norm = sum(keep.values())
    return sorted([(w / norm, list(p)) for p, w in keep.items()],
                  reverse=True)


def solve_jdist(local, nic, nv_star=None, div_k=4):
    """Two-host relay LP: unit commodity per (r, rd) pair, all
    simultaneous, forced through a TRANSIT rail j not in {r, rd}
    (that is the spread corridor's job; DIRECT/RX/TX stay outside and
    are lambda's business). Graph: source-host intra + rail edges
    S_j->T_j at nic + dst-host intra. nv_star models an NVSwitch host
    (star through a virtual hub at nv per pipe) instead of point-to-
    point cube links -- ONE formulation covers both wirings.
    Returns jdist[r][rd] = [(weight, j)] (the rail-edge marginal)."""
    g = len(local)
    # vertices: 0..g-1 = S, g..2g-1 = T (+ hubs 2g, 2g+1 for star)
    edges, cap = [], []

    def add(u, v, c):
        edges.append((u, v))
        cap.append(c)

    if nv_star is not None:
        hs, ht = 2 * g, 2 * g + 1
        for j in range(g):
            add(j, hs, nv_star)
            add(hs, j, nv_star)
            add(g + j, ht, nv_star)
            add(ht, g + j, nv_star)
        nverts = 2 * g + 2
    else:
        for u in range(g):
            for v in range(g):
                if local[u][v] > 0.0:
                    add(u, v, local[u][v])
                    add(g + u, g + v, local[u][v])
        nverts = 2 * g
    rail0 = len(edges)
    for j in range(g):
        add(j, g + j, nic)
    ei = {e: i for i, e in enumerate(edges)}
    coms = [(r, rd) for r in range(g) for rd in range(g)]
    ne, nc = len(edges), len(coms)
    nf = nc * ne
    c_obj = np.full(nf + 1, 1e-4)
    c_obj[nf] = 1.0
    a_eq, b_eq = [], []
    bounds = [(0, None)] * (nf + 1)
    # Diversity floor: no commodity may put more than 1/div_k of its
    # unit on ONE transit rail. Without it the vertex optimum assigns a
    # SINGLE rail per (r, rd) -- balanced under the uniform prior but a
    # per-pair hotspot the moment the workload is skewed (measured:
    # full300 4906 vs 4754 base). With it, every pair spreads over
    # >= div_k rails regardless of which pairs turn out hot: skew
    # robustness by construction, still workload-free.
    per_rail_cap = 1.0 / div_k
    for ci, (r, rd) in enumerate(coms):
        # ban the r and rd rail edges: transit must be a third rail
        bounds[ci * ne + rail0 + r] = (0, 0)
        if rd != r:
            bounds[ci * ne + rail0 + rd] = (0, 0)
        for j in range(g):
            if bounds[ci * ne + rail0 + j] != (0, 0):
                bounds[ci * ne + rail0 + j] = (0, per_rail_cap)
        for v in range(nverts):
            if v == g + rd:
                continue
            row = np.zeros(nf + 1)
            for k, (u, w) in enumerate(edges):
                if w == v:
                    row[ci * ne + k] += 1.0
                if u == v:
                    row[ci * ne + k] -= 1.0
            a_eq.append(row)
            b_eq.append(-1.0 if v == r else 0.0)
    a_ub, b_ub = [], []
    for k in range(ne):
        row = np.zeros(nf + 1)
        for ci in range(nc):
            row[ci * ne + k] = 1.0 / cap[k]
        row[nf] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)
    res = linprog(c_obj, A_ub=np.array(a_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(a_eq), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    assert res.success, res.message
    jdist = [[[] for _ in range(g)] for _ in range(g)]
    for ci, (r, rd) in enumerate(coms):
        for j in range(g):
            w = res.x[ci * ne + rail0 + j]
            if w > 0.02:
                jdist[r][rd].append((w, j))
        tot = sum(w for w, _ in jdist[r][rd])
        jdist[r][rd] = sorted([(w / tot, j) for w, j in jdist[r][rd]],
                              reverse=True)
    return jdist, res.x[nf]


def solve_jdist_hot(local, nic, div_k=4, nv_star=None, eps=1e-3):
    """Hot-column jdist: for each dst rail rd, assume EVERY source rail
    r sends one unit into rd simultaneously (the concentrated-ingress
    scenario the spread corridor exists for -- still workload-free: it
    is a structural worst case, not a measured demand). Variables:
    x[r][j] = landing split (j not in {r, rd}, x <= 1/div_k), plus an
    aggregated multi-source flow from the landings to rd on the dst
    intra graph. Minimize max(NIC util, intra util) + eps * intra flow
    (locality: neighbors of rd win ties). Prices BOTH the landing NICs
    and the dst-side links, so a weak-B_in rd gets capacity-aware
    weights while a homogeneous host gets the uniform Latin spread."""
    g = len(local)
    jdist = [[[] for _ in range(g)] for _ in range(g)]
    for rd in range(g):
        if nv_star is not None:
            # star: all landings one hop from rd; only NICs price.
            for r in range(g):
                js = [j for j in range(g) if j != r and j != rd]
                w = 1.0 / len(js)
                jdist[r][rd] = [(w, j) for j in js]
            continue
        edges = [(u, v) for u in range(g) for v in range(g)
                 if local[u][v] > 0.0]
        ei = {e: i for i, e in enumerate(edges)}
        ne = len(edges)
        # vars: x[r][j] (g*g) + f_e (ne) + t
        nx = g * g
        nv = nx + ne + 1

        def xv(r, j):
            return r * g + j

        c = np.zeros(nv)
        c[nx + ne:] = 1.0
        for k in range(ne):
            c[nx + k] = eps
        bounds = []
        for r in range(g):
            for j in range(g):
                if r == rd or j == r or j == rd:
                    bounds.append((0, 0))
                else:
                    bounds.append((0, 1.0 / div_k))
        bounds += [(0, None)] * ne + [(0, None)]
        a_eq, b_eq = [], []
        for r in range(g):
            if r == rd:
                continue
            row = np.zeros(nv)
            for j in range(g):
                row[xv(r, j)] = 1.0
            a_eq.append(row)
            b_eq.append(1.0)
        # flow conservation at every vertex v != rd:
        # inflow(f) + landings(v) = outflow(f)
        for v in range(g):
            if v == rd:
                continue
            row = np.zeros(nv)
            for e, (u, w) in enumerate(edges):
                if w == v:
                    row[nx + e] += 1.0
                if u == v:
                    row[nx + e] -= 1.0
            for r in range(g):
                row[xv(r, v)] += 1.0
            a_eq.append(row)
            b_eq.append(0.0)
        a_ub, b_ub = [], []
        for j in range(g):  # NIC util
            row = np.zeros(nv)
            for r in range(g):
                row[xv(r, j)] = 1.0 / nic
            row[nv - 1] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)
        for e, (u, w) in enumerate(edges):  # intra util
            row = np.zeros(nv)
            row[nx + e] = 1.0 / local[u][w]
            row[nv - 1] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)
        res = linprog(c, A_ub=np.array(a_ub), b_ub=np.array(b_ub),
                      A_eq=np.array(a_eq), b_eq=np.array(b_eq),
                      bounds=bounds, method="highs")
        assert res.success, res.message
        for r in range(g):
            if r == rd:
                continue
            ws = [(res.x[xv(r, j)], j) for j in range(g)
                  if res.x[xv(r, j)] > 0.02]
            tot = sum(w for w, _ in ws)
            jdist[r][rd] = sorted([(w / tot, j) for w, j in ws],
                                  reverse=True)
        # same-rail pairs (r == rd) never spread into rd's own rail;
        # give them the aggregate landing distribution.
        agg = {}
        for r in range(g):
            for w, j in jdist[r][rd]:
                agg[j] = agg.get(j, 0.0) + w
        tot = sum(agg.values())
        jdist[rd][rd] = sorted([(w / tot, j) for j, w in agg.items()],
                               reverse=True)
    return jdist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus-per-host", type=int, default=8)
    ap.add_argument("--speeds", default="400x6,200x6,100x4")
    ap.add_argument("--rot", type=int, default=0)
    ap.add_argument("--full-gbps", type=float, default=None)
    ap.add_argument("--mode", choices=["paths", "jdist", "jdist_hot"],
                    default="paths")
    ap.add_argument("--div-k", type=int, default=4,
                    help="per-pair diversity floor: >= div_k transit rails")
    ap.add_argument("--nic-gbps", type=float, default=180.0)
    ap.add_argument("--nv-gbps", type=float, default=None,
                    help="NVSwitch star host (flex_rail) instead of cube")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.mode == "jdist_hot":
        t0 = time.time()
        if args.nv_gbps:
            local = [[0.0] * args.gpus_per_host
                     for _ in range(args.gpus_per_host)]
            jdist = solve_jdist_hot(local, args.nic_gbps,
                                    div_k=args.div_k,
                                    nv_star=args.nv_gbps)
        else:
            local = local_matrix(args.gpus_per_host, args.speeds,
                                 args.rot, args.full_gbps)
            jdist = solve_jdist_hot(local, args.nic_gbps,
                                    div_k=args.div_k)
        ms = (time.time() - t0) * 1e3
        with open(args.output, "w") as f:
            f.write("# mosaic_lp_jdist_hot v1 gpus %d solve_ms %.1f\n"
                    % (args.gpus_per_host, ms))
            for r in range(args.gpus_per_host):
                for rd in range(args.gpus_per_host):
                    for w, j in jdist[r][rd]:
                        f.write("jdist %d %d %d %.6f\n" % (r, rd, j, w))
        print("solve_ms %.1f -> %s" % (ms, args.output))
        return
    if args.mode == "jdist":
        t0 = time.time()
        if args.nv_gbps:
            local = [[0.0] * args.gpus_per_host
                     for _ in range(args.gpus_per_host)]
            jdist, t = solve_jdist(local, args.nic_gbps,
                                   nv_star=args.nv_gbps,
                                   div_k=args.div_k)
        else:
            local = local_matrix(args.gpus_per_host, args.speeds,
                                 args.rot, args.full_gbps)
            jdist, t = solve_jdist(local, args.nic_gbps,
                                   div_k=args.div_k)
        ms = (time.time() - t0) * 1e3
        with open(args.output, "w") as f:
            f.write("# mosaic_lp_jdist v1 gpus %d t_opt %.6f "
                    "solve_ms %.1f\n" % (args.gpus_per_host, t, ms))
            for r in range(args.gpus_per_host):
                for rd in range(args.gpus_per_host):
                    for w, j in jdist[r][rd]:
                        f.write("jdist %d %d %d %.6f\n" % (r, rd, j, w))
        print("t_opt %.4f solve_ms %.1f -> %s" % (t, ms, args.output))
        return
    local = local_matrix(args.gpus_per_host, args.speeds, args.rot,
                         args.full_gbps)
    t0 = time.time()
    tables, t_max = {}, 0.0
    for d in range(args.gpus_per_host):
        tables[d], t = solve_dst(local, d)
        t_max = max(t_max, t)
    ms = (time.time() - t0) * 1e3
    with open(args.output, "w") as f:
        f.write("# mosaic_lp_intra v1 gpus %d t_opt %.6f solve_ms %.1f\n"
                % (args.gpus_per_host, t_max, ms))
        for d in sorted(tables):
            for j in sorted(tables[d]):
                for w, p in tables[d][j]:
                    f.write("path %d %d %.6f %s\n"
                            % (d, j, w, " ".join(map(str, p))))
    print("t_opt %.4f solve_ms %.1f -> %s" % (t_max, ms, args.output))


if __name__ == "__main__":
    main()
