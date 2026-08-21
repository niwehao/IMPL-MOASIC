#!/usr/bin/env python3
"""Cut lower bounds for a demand file on the two-layer clos topology.

Cuts (store-and-forward on switch-as-wire; every byte crosses each cut
at most/least once, so max over cuts is a completion-time LB):
  host egress   : max_s sum_d D[s][d] / host_rate
  host ingress  : max_d sum_s D[s][d] / host_rate
  leaf up  (agg): max_l sum_{s in l, d not in l} D / (uplink*spines)
  leaf up (pin) : max_{l,k} bytes routed on leaf_up[l][k] under the
                  (s+d)%spines pinning / uplink   <- the real routed cut
  spine down(pin): symmetric for spine_down[k][l]
"""
import argparse


def load(path, n):
    d = [[0] * n for _ in range(n)]
    for line in open(path):
        p = line.split()
        if len(p) == 4 and p[0] == "demand":
            d[int(p[1])][int(p[2])] = int(p[3])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demand")
    ap.add_argument("--nodes", type=int, default=32)
    ap.add_argument("--leafs", type=int, default=4)
    ap.add_argument("--spines", type=int, default=1)
    ap.add_argument("--host-gbps", type=float, default=100.0)
    ap.add_argument("--uplink-gbps", type=float, default=0.0,
                    help="per leaf->spine link; 0 = auto 2:1")
    args = ap.parse_args()

    n, L, S = args.nodes, args.leafs, args.spines
    hpl = (n + L - 1) // L
    up = args.uplink_gbps
    if up == 0:
        up = args.host_gbps * hpl / (2.0 * S)
    D = load(args.demand, n)
    hr = args.host_gbps / 8.0 * 1e3   # bytes/us
    ur = up / 8.0 * 1e3

    eg = max(sum(D[s]) for s in range(n)) / hr
    ing = max(sum(D[s][d] for s in range(n)) for d in range(n)) / hr

    leaf_up = [[0] * S for _ in range(L)]
    spine_dn = [[0] * L for _ in range(S)]
    agg_up = [0] * L
    for s in range(n):
        ls = min(s // hpl, L - 1)
        for d in range(n):
            if D[s][d] == 0 or s == d:
                continue
            ld = min(d // hpl, L - 1)
            if ls != ld:
                k = (s + d) % S
                leaf_up[ls][k] += D[s][d]
                spine_dn[k][ld] += D[s][d]
                agg_up[ls] += D[s][d]
    cuts = {
        "host_egress": eg,
        "host_ingress": ing,
        "leaf_up_agg": max(agg_up) / (ur * S),
        "leaf_up_pinned": max(max(r) for r in leaf_up) / ur,
        "spine_down_pinned": max(max(r) for r in spine_dn) / ur,
    }
    for k, v in cuts.items():
        print("%-18s %12.1f us" % (k, v))
    print("%-18s %12.1f us  (binding: %s)" % (
        "LB", max(cuts.values()), max(cuts, key=cuts.get)))


if __name__ == "__main__":
    main()
