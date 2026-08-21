#!/usr/bin/env python3
"""Cut lower-bound family for flex_rail alltoallv (report §5).

B1(d): per-GPU two-pipe INGRESS waterfilling cut. Destination d has an NV
       pipe (beta_v) and a NIC pipe (beta_n); same-server inflow can only
       use NV, cross-server inflow is divisible across both:
           t(d) = max(same_d / bv, (same_d + cross_d) / (bv + bn))
B2(n): per-server NIC ingress cut: all cross-server bytes into server n
       must cross its G NICs:  t(n) = cross_in(n) / (G * bn)
B3(s): per-GPU two-pipe EGRESS cut (symmetric to B1 with outflow).

Bound = max over all of B1/B2/B3, reported in us with the binding term.
Optionally a single degraded GPU (its NIC at nic/degrade_factor) is folded
into all three terms. Matches cache/nonuniform/x1_bounds.txt.

flex_cube mode (--cube-speeds / --cube-full-gbps, matching
topo/flex_cube_graph.h): the per-GPU scale-up pipe becomes the SUM of the
GPU's incident cube-link capacities B_in(d) (B1/B3 use it in place of nv),
and two subset-cut families are added per host over local GPU sets S:
  B5out(S): (same-host demand S->S~ + all cross-host demand out of S)
            / (cube links S->S~ + |S| * nic)
  B5in(S):  symmetric ingress form.
These are exact directed graph cuts (for |S|=1 they refine B1/B3).
"""
import argparse

# NDv2 hybrid cube-mesh edge list -- MUST match
# topo/ndv2_graph.h NDV2_32_CUBE_EDGES order (used for tier assignment).
CUBE_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6), (4, 7),
    (5, 6), (5, 7),
    (6, 7),
]


def cube_local_matrix(gpus, spec, rot, full_gbps):
    """Replicates flex_cube_local_matrix (flex_cube_graph.h)."""
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
    assert len(tiers) == len(CUBE_EDGES), \
        "tier counts must sum to %d" % len(CUBE_EDGES)
    for e, (a, b) in enumerate(CUBE_EDGES):
        s = tiers[(e + rot) % len(tiers)]
        local[a][b] = s
        local[b][a] = s
    return local


def cube_bounds(demand, gpus, local, nic):
    n = len(demand)
    b_in = [sum(local[g]) for g in range(gpus)]

    best = (0.0, "none", -1)
    for d in range(n):
        bd = b_in[d % gpus]
        same = sum(demand[s][d] for s in range(n)
                   if s != d and s // gpus == d // gpus)
        cross = sum(demand[s][d] for s in range(n)
                    if s // gpus != d // gpus)
        t = max(us(same, bd), us(same + cross, bd + nic))
        if t > best[0]:
            best = (t, "B1", d)
        same_o = sum(demand[d][t2] for t2 in range(n)
                     if t2 != d and t2 // gpus == d // gpus)
        cross_o = sum(demand[d][t2] for t2 in range(n)
                      if t2 // gpus != d // gpus)
        t3 = max(us(same_o, bd), us(same_o + cross_o, bd + nic))
        if t3 > best[0]:
            best = (t3, "B3", d)
    servers = n // gpus
    for srv in range(servers):
        members = range(srv * gpus, (srv + 1) * gpus)
        cross_in = sum(demand[s][d] for d in members for s in range(n)
                       if s // gpus != srv)
        t = us(cross_in, gpus * nic)
        if t > best[0]:
            best = (t, "B2", srv)
        cross_out = sum(demand[s][d] for s in members for d in range(n)
                        if d // gpus != srv)
        t2 = us(cross_out, gpus * nic)
        if t2 > best[0]:
            best = (t2, "B2out", srv)
        # B5: directed subset cuts inside the host.
        for mask in range(1, (1 << gpus) - 1):
            S = [g for g in range(gpus) if mask & (1 << g)]
            Sc = [g for g in range(gpus) if not mask & (1 << g)]
            cut = sum(local[a][b] for a in S for b in Sc)
            cap = cut + len(S) * nic
            out_bytes = 0
            in_bytes = 0
            for a in S:
                ga = srv * gpus + a
                for b in Sc:
                    gb = srv * gpus + b
                    out_bytes += demand[ga][gb]
                    in_bytes += demand[gb][ga]
                for other in range(n):
                    if other // gpus != srv:
                        out_bytes += demand[ga][other]
                        in_bytes += demand[other][ga]
            t5 = us(out_bytes, cap)
            if t5 > best[0]:
                best = (t5, "B5out", srv * gpus + mask)
            t5i = us(in_bytes, cap)
            if t5i > best[0]:
                best = (t5i, "B5in", srv * gpus + mask)
    return best


def load_demand(path):
    n = 0
    entries = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "node_count":
                n = int(parts[1])
            elif parts[0] == "demand":
                entries.append((int(parts[1]), int(parts[2]),
                                int(parts[3])))
    assert n > 0, "demand file missing node_count"
    rows = [[0] * n for _ in range(n)]
    for s, d, b in entries:
        rows[s][d] = b
    return rows


def us(bytes_, gbps):
    return bytes_ * 8.0 / (gbps * 1000.0)


def bounds(demand, gpus, nv, nic, degrade_gpu=None, degrade_factor=1.0):
    n = len(demand)
    servers = n // gpus

    def nic_of(g):
        if degrade_gpu is not None and g == degrade_gpu:
            return nic / degrade_factor
        return nic

    best = (0.0, "none", -1)
    for d in range(n):
        same = sum(demand[s][d] for s in range(n)
                   if s != d and s // gpus == d // gpus)
        cross = sum(demand[s][d] for s in range(n)
                    if s // gpus != d // gpus)
        t = max(us(same, nv), us(same + cross, nv + nic_of(d)))
        if t > best[0]:
            best = (t, "B1", d)
        same_o = sum(demand[d][t2] for t2 in range(n)
                     if t2 != d and t2 // gpus == d // gpus)
        cross_o = sum(demand[d][t2] for t2 in range(n)
                      if t2 // gpus != d // gpus)
        t3 = max(us(same_o, nv), us(same_o + cross_o, nv + nic_of(d)))
        if t3 > best[0]:
            best = (t3, "B3", d)
    for srv in range(servers):
        members = range(srv * gpus, (srv + 1) * gpus)
        cross_in = sum(demand[s][d] for d in members for s in range(n)
                       if s // gpus != srv)
        nic_sum = sum(nic_of(g) for g in members)
        t = us(cross_in, nic_sum)
        if t > best[0]:
            best = (t, "B2", srv)
        cross_out = sum(demand[s][d] for s in members for d in range(n)
                        if d // gpus != srv)
        t2 = us(cross_out, nic_sum)
        if t2 > best[0]:
            best = (t2, "B2out", srv)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demand", required=True)
    ap.add_argument("--gpus-per-host", type=int, default=8)
    ap.add_argument("--nv-gbps", type=float, default=None,
                    help="required unless a --cube-* mode is used")
    ap.add_argument("--nic-gbps", type=float, default=180.0)
    ap.add_argument("--leaf-uplink-total", type=float, default=None,
                    help="pod wiring: per-host leaf uplink total Gbps "
                         "(adds B4 = host cross in/out over this)")
    ap.add_argument("--degrade-gpu", type=int, default=None)
    ap.add_argument("--degrade-factor", type=float, default=1.0)
    ap.add_argument("--cube-speeds", default=None,
                    help="flex_cube tier spec, e.g. 400x6,200x6,100x4 "
                         "(switches to cube bounds; --nv-gbps ignored)")
    ap.add_argument("--cube-rot", type=int, default=0)
    ap.add_argument("--cube-full-gbps", type=float, default=None,
                    help="flex_cube full-crossbar degenerate mode")
    args = ap.parse_args()
    demand = load_demand(args.demand)
    if args.cube_speeds or args.cube_full_gbps:
        local = cube_local_matrix(args.gpus_per_host,
                                  args.cube_speeds or "",
                                  args.cube_rot, args.cube_full_gbps)
        t, kind, who = cube_bounds(demand, args.gpus_per_host, local,
                                   args.nic_gbps)
    else:
        t, kind, who = bounds(demand, args.gpus_per_host, args.nv_gbps,
                              args.nic_gbps, args.degrade_gpu,
                              args.degrade_factor)
    if args.leaf_uplink_total:
        n = len(demand)
        g = args.gpus_per_host
        for srv in range(n // g):
            m = range(srv * g, (srv + 1) * g)
            cin = sum(demand[s2][d] for d in m for s2 in range(n)
                      if s2 // g != srv)
            cout = sum(demand[s2][d] for s2 in m for d in range(n)
                       if d // g != srv)
            t4 = us(max(cin, cout), args.leaf_uplink_total)
            if t4 > t:
                t, kind, who = t4, "B4", srv
    print("bound_us %.0f term %s index %d" % (t, kind, who))


if __name__ == "__main__":
    main()
