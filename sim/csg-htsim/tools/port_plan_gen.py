#!/usr/bin/env python3
# port_plan_gen.py -- funnel-skeleton construction for PARTIAL (per-port)
# reconfiguration, emitted as a standard phased plan; the -async executor
# of htsim_optical re-bills it under port-level semantics.
#
# Construction (the "composition of matchings" view: a persistent hot
# skeleton whose holding time = the hot column's share, plus d rotating
# relay configurations for the cold complement; ports transition locally):
#   P0 (skeleton + chains): non-hot nodes form c chains, each ending in a
#      gateway; gateway g_i holds the only ingress circuit of the hot Rx
#      (sequentially: hot Rx pays c R's total). Chains funnel hot-column
#      bytes to their gateway and also carry intra-chain cold pairs at
#      forward distance <= L (span). hot's own row rides as chain head.
#   P1..Pd (rotation): the residual cold matrix goes to d demand-aware
#      Cayley shift topologies (RoN-style greedy contraction on h*maxent),
#      explicit shift paths.
# Under -async each port exits P0 exactly when its chain duty drains,
# pays ONE port-level R per subsequent topology it participates in, and
# never waits for the funnel stragglers.
import argparse, math, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', type=int, default=32)
    ap.add_argument('--demand-file', required=True)
    ap.add_argument('--reconf-us', type=float, default=1000.0)
    ap.add_argument('--chunk-bytes', type=int, default=4194304)
    ap.add_argument('--gbps', type=float, default=800.0)
    ap.add_argument('--delay-ns', type=float, default=500.0)
    ap.add_argument('--gateways', type=int, default=1, help='c: hot Rx sessions')
    ap.add_argument('--cold-span', type=int, default=2, help='L: max chain distance for cold-on-P0')
    ap.add_argument('--shifts', type=int, default=2, help='d: rotation topology count')
    ap.add_argument('--order', default='aschot', choices=['aschot', 'nn'],
                    help='chain order: ascending hot entry, or cold-NN')
    ap.add_argument('--mode', default='chain',
                    choices=['chain', 'skeleton', 'rot'],
                    help='chain = funnel chains; skeleton = persistent '
                         '(g->hot) circuit + relay via the rotation '
                         'topologies; rot = plain rotation over --gens '
                         'with per-pair min-hop assignment (per-port '
                         'billing optimum shape)')
    ap.add_argument('--gens', default='',
                    help='rot mode: comma list of generators, default 1,n-1')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    n = a.nodes

    D = [[0] * n for _ in range(n)]
    for line in open(a.demand_file):
        t = line.split()
        if len(t) == 4 and t[0] == 'demand':
            s, d, b = int(t[1]), int(t[2]), int(t[3])
            if s < n and d < n:
                D[s][d] = b
    hot = max(range(n), key=lambda j: sum(D[i][j] for i in range(n)))

    if a.mode == 'skeleton':
        gen_skeleton(a, n, D, hot)
        return
    if a.mode == 'rot':
        gen_rot(a, n, D)
        return

    others = [u for u in range(n) if u != hot]
    if a.order == 'aschot':
        others.sort(key=lambda u: (D[u][hot], u))
    else:  # nn on cold affinity, still gateway = max hot entry last
        rest = sorted(others, key=lambda u: (D[u][hot], u))
        cur = rest.pop(0)
        nn = [cur]
        while rest:
            nxt = max(rest, key=lambda x: (D[cur][x], -x))
            rest.remove(nxt)
            nn.append(nxt)
            cur = nxt
        others = nn

    # c chains: contiguous segments; each gateway = its segment's last
    # node (max hot entry sits in the LAST segment's tail by aschot).
    c = max(1, min(a.gateways, len(others)))
    seg = []
    base = 0
    for i in range(c):
        ln = len(others) // c + (1 if i < len(others) % c else 0)
        seg.append(others[base:base + ln])
        base += ln
    gateways = [s[-1] for s in seg]

    served = [[False] * n for _ in range(n)]
    phases = []  # list of (edges, xfers); xfer = (s, d, bytes, path)

    # ---- P0: chains + hot row head + gateway skeleton ----------------
    chain_edges = []
    pos_chain = {}
    for si, s in enumerate(seg):
        for i, u in enumerate(s):
            pos_chain[u] = (si, i)
        for i in range(len(s) - 1):
            chain_edges.append((s[i], s[i + 1]))
    # hot's Tx heads chain 0
    chain_edges.append((hot, seg[0][0]))

    def chain_path(u, w):
        si, i = pos_chain[u]
        sj, j = pos_chain[w]
        if si != sj or j <= i:
            return None
        return seg[si][i:j + 1]

    p0_xfers = []                  # phase 0 (chains + first gateway)
    funnel_xf = [[] for _ in range(c)]   # per gateway session phase
    # funnel: every hot entry rides its chain to its gateway, then g->hot
    for si, s in enumerate(seg):
        for i, u in enumerate(s):
            if D[u][hot] > 0:
                path = s[i:] + [hot]
                funnel_xf[si].append((u, hot, D[u][hot], path))
                served[u][hot] = True
    # hot's own row: forward along chain 0 within span
    for j, w in enumerate(seg[0]):
        if j + 1 <= a.cold_span and D[hot][w] > 0:
            p0_xfers.append((hot, w, D[hot][w], [hot] + seg[0][:j + 1]))
            served[hot][w] = True
    # intra-chain cold pairs within span
    for u in others:
        for w in others:
            if u == w or D[u][w] <= 0 or served[u][w]:
                continue
            p = chain_path(u, w)
            if p is not None and len(p) - 1 <= a.cold_span:
                p0_xfers.append((u, w, D[u][w], p))
                served[u][w] = True
    # one phase per gateway session (chains persist; only hot Rx changes).
    # Each segment's funnel xfers sit in ITS gateway's phase so the
    # (g_i -> hot) duty lands on the session that owns the circuit.
    for si in range(c):
        edges = list(chain_edges) + [(gateways[si], hot)]
        xf = (p0_xfers if si == 0 else []) + funnel_xf[si]
        phases.append((edges, xf))

    # ---- P1..Pd: demand-aware shift rotation for the residual --------
    res_off = [dict() for _ in range(n)]  # offset -> {u: bytes}
    for u in range(n):
        for w in range(n):
            if u != w and D[u][w] > 0 and not served[u][w]:
                res_off[(w - u) % n][u] = D[u][w]

    def hops(s, o):
        g = math.gcd(s, n)
        if o % g:
            return None
        m = n // g
        so = (s // g) % m
        x = so
        for k in range(1, m):
            if x == (o // g) % m:
                return k
            x = (x + so) % m
        return None

    live = [o for o in range(1, n) if res_off[o]]
    maxent = {o: max(res_off[o].values()) for o in live}
    gens = [1]
    h = {o: hops(1, o) for o in live}
    while len(gens) < a.shifts:
        cand = [o for o in live if h[o] and h[o] > 1 and o not in gens]
        if not cand:
            break
        o_star = max(cand, key=lambda o: h[o] * maxent[o])
        gens.append(o_star)
        for o in live:
            hh = hops(o_star, o)
            if hh and hh < h[o]:
                h[o] = hh
    topo_of = {}
    for o in live:
        best = min(((hops(s, o), j) for j, s in enumerate(gens)
                    if hops(s, o)), default=None)
        topo_of[o] = best[1]
        h[o] = best[0]

    for j, s0 in enumerate(gens):
        edges = [(u, (u + s0) % n) for u in range(n)]
        xf = []
        for o in sorted(live, key=lambda o: h[o]):
            if topo_of[o] != j:
                continue
            for u, b in sorted(res_off[o].items()):
                path = [(u + s0 * t) % n for t in range(h[o] + 1)]
                assert path[-1] == (u + o) % n
                xf.append((u, (u + o) % n, b, path))
        if xf:
            phases.append((edges, xf))

    tot = sum(b for _, x in phases for (_, _, b, _) in x)
    want = sum(D[u][w] for u in range(n) for w in range(n) if u != w)
    if tot != want:
        sys.stderr.write('BUG: plan bytes %d != demand %d\n' % (tot, want))
        sys.exit(1)

    with open(a.output, 'w') as f:
        f.write('nodes %d degree 1\n' % n)
        f.write('gbps %g delay_ns %g reconf_us %g chunk_bytes %d\n'
                % (a.gbps, a.delay_ns, a.reconf_us, a.chunk_bytes))
        f.write('pay_initial\n')
        for edges, xf in phases:
            f.write('phase\n')
            for (u, v) in edges:
                f.write('edge %d %d\n' % (u, v))
            for (s, d, b, p) in xf:
                f.write('xfer %d %d %d via %s\n'
                        % (s, d, b, ' '.join(map(str, p))))
    sys.stderr.write('port_plan: hot=%d c=%d L=%d d=%d gens=%s phases=%d\n'
                     % (hot, c, a.cold_span, len(gens), gens, len(phases)))

def gen_rot(a, n, D):
    # Plain rotation over an explicit generator set, per-pair min-hop
    # assignment (ties -> earlier generator). Under per-port billing the
    # cost is  d*R + that port's byte-hop work; the bidirectional ring
    # {1, n-1} minimizes the MAX port work on skewed demand (the hot
    # column's two natural ingress edges make the c=2 skeleton embedded,
    # no pinning or path avoidance needed).
    import math as _m

    def hops(s, o):
        g2 = _m.gcd(s, n)
        if o % g2:
            return None
        m = n // g2
        so = (s // g2) % m
        x = so
        for k in range(1, m):
            if x == (o // g2) % m:
                return k
            x = (x + so) % m
        return None

    gens = ([int(x) for x in a.gens.split(',') if x]
            if a.gens else [1, n - 1])
    for o in range(1, n):
        assert any(hops(s, o) for s in gens), 'gens do not cover offsets'
    # The hot Rx is the cut: cold transits THROUGH the hot node consume
    # the cut port's serial time, so a path whose interior contains hot
    # is only taken when no hot-avoiding candidate exists.
    hot = max(range(n), key=lambda j2: sum(D[i][j2] for i in range(n)))
    phases = []
    xf = [[] for _ in gens]
    for u in range(n):
        for w in range(n):
            if u == w or D[u][w] <= 0:
                continue
            o = (w - u) % n
            best = None
            best_dirty = None
            for j, s0 in enumerate(gens):
                hh = hops(s0, o)
                if not hh:
                    continue
                p = [(u + s0 * t) % n for t in range(hh + 1)]
                if hot in p[1:-1]:
                    if best_dirty is None or hh < best_dirty[0]:
                        best_dirty = (hh, j, p)
                elif best is None or hh < best[0]:
                    best = (hh, j, p)
            if best is None:
                best = best_dirty
            hh, j, path = best
            xf[j].append((u, w, D[u][w], path))
    # Hot-column transfers first: injection order sets the tie-break on
    # shared circuits, so the hot Rx (the cut) is fed with priority and
    # never starves behind cold traffic that has slack anyway.
    hot = max(range(n), key=lambda j2: sum(D[i][j2] for i in range(n)))
    for j in range(len(gens)):
        xf[j].sort(key=lambda x: (x[1] != hot, -x[2]))
    for j, s0 in enumerate(gens):
        edges = [(u, (u + s0) % n) for u in range(n)]
        phases.append((edges, xf[j]))

    with open(a.output, 'w') as f:
        f.write('nodes %d degree 1\n' % n)
        f.write('gbps %g delay_ns %g reconf_us %g chunk_bytes %d\n'
                % (a.gbps, a.delay_ns, a.reconf_us, a.chunk_bytes))
        f.write('pay_initial\n')
        for edges, xfr in phases:
            f.write('phase\n')
            for (u, v) in edges:
                f.write('edge %d %d\n' % (u, v))
            for (s, d, b, p) in xfr:
                f.write('xfer %d %d %d via %s\n'
                        % (s, d, b, ' '.join(map(str, p))))
    sys.stderr.write('rot: gens=%s\n' % gens)


def gen_skeleton(a, n, D, hot):
    # Persistent-skeleton construction ("composition of matchings"):
    #   - ONE circuit (g -> hot) is held for the entire run (g = source
    #     of the largest hot entry); the hot Rx pays exactly one R --
    #     this removes the per-topology R that binds RoN-async (its hot
    #     ingress partner changes with every rotation).
    #   - every other hot-column byte is RELAYED: (u,hot) becomes
    #     ordinary (u,g) traffic riding the d rotating Cayley topologies,
    #     then hops (g,hot) on the skeleton (multi-topology paths; the
    #     async executor lets a chunk wait for a later session).
    #   - rotation paths must not TRANSIT g (its Tx is pinned); pairs
    #     with no g-avoiding path fall back to direct cleanup matchings.
    #   - g's own cold row cannot ride the rotations (Tx pinned), so it
    #     drains after the skeleton over a dedicated relay chain.
    import math as _m

    def hops(s, o):
        g2 = _m.gcd(s, n)
        if o % g2:
            return None
        m = n // g2
        so = (s // g2) % m
        x = so
        for k in range(1, m):
            if x == (o // g2) % m:
                return k
            x = (x + so) % m
        return None

    # Gateways ADJACENT to hot (id space): the pinned block {g1, hot}
    # (g1's Tx, hot's Rx cannot relay) is contiguous, so every pair has
    # a ring direction avoiding it. With --gateways 2 the hot Rx runs
    # TWO sequential skeleton sessions (2R): g1 = hot-1 collects the
    # forward side, g2 = hot+1 the backward side -- relay distance drops
    # from ~n/2 (worst) to the ring distance to the block (~n/4 mean).
    # g2's Tx is only pinned in the FINAL phase, so g2 still rides the
    # rotations for its own cold row.
    g = (hot - 1) % n
    g2 = (hot + 1) % n
    c = 2 if a.gateways >= 2 else 1

    # rotation matrix: cold pairs only (relay legs get fixed ring paths
    # below; g1's cold row drains over the tail chain after skeleton 1)
    R2 = [[0] * n for _ in range(n)]
    for u in range(n):
        for w in range(n):
            if u == w or D[u][w] <= 0 or w == hot:
                continue
            if u == g:
                pass                          # g1's cold row -> tail chain
            else:
                R2[u][w] += D[u][w]

    res_off = [dict() for _ in range(n)]
    for u in range(n):
        for w in range(n):
            if u != w and R2[u][w] > 0:
                res_off[(w - u) % n][u] = R2[u][w]
    live = [o for o in range(1, n) if res_off[o]]

    def path_of(u, w, s0, hh):
        return [(u + s0 * t) % n for t in range(hh + 1)]

    def clean_h(u, w, s0):
        # hops of the g/hot-avoiding path of (u,w) under s0; None if dirty
        o = (w - u) % n
        hh = hops(s0, o)
        if not hh:
            return None
        p = path_of(u, w, s0, hh)
        if g in p[1:-1] or hot in p[1:-1]:
            return None
        return hh

    # Base generators {1, n-1}: with the pinned block {g=hot-1, hot}
    # contiguous, one ring direction is always clean. Extra generators
    # (d > 2) picked greedily by CLEAN byte-hop reduction.
    gens = [1, n - 1][:max(1, a.shifts)]
    def best_h(u, w, cur):
        bh = None
        for s0 in cur:
            hh = clean_h(u, w, s0)
            if hh and (bh is None or hh < bh):
                bh = hh
        return bh
    while len(gens) < a.shifts:
        best_gain, best_s = 0, None
        for s0 in range(2, n - 1):
            if s0 in gens:
                continue
            gain = 0
            for o in live:
                for u, b in res_off[o].items():
                    w = (u + o) % n
                    bh = best_h(u, w, gens)
                    nh = clean_h(u, w, s0)
                    if nh and (bh is None or nh < bh):
                        gain += b * ((bh if bh else n) - nh)
            if gain > best_gain:
                best_gain, best_s = gain, s0
        if best_s is None:
            break
        gens.append(best_s)

    # per-pair topology choice for COLD pairs: shortest block-avoiding path
    rot_xf = [[] for _ in range(len(gens))]     # per phase
    leftovers = []
    for o in live:
        for u, b in sorted(res_off[o].items()):
            w = (u + o) % n
            best = None
            for j, s0 in enumerate(gens):
                hh = hops(s0, o)
                if not hh:
                    continue
                p = path_of(u, w, s0, hh)
                if g in p[1:-1] or hot in p[1:-1]:
                    continue
                if best is None or hh < best[0]:
                    best = (hh, j, p)
            if best is None:
                leftovers.append((u, w, b))
            else:
                rot_xf[best[1]].append((u, w, b, best[2]))

    # skeleton xfers: fixed ring relay leg + gateway hop. Side split at
    # c=2 by ring distance to the block; the relay leg rides gen 1
    # (forward, phase 0) or gen n-1 (backward, phase 1).
    skel_xf = [[] for _ in range(len(gens))]
    direct_skel = []
    skel2_bytes = 0
    for u in range(n):
        if u == hot or D[u][hot] <= 0:
            continue
        if u == g:
            direct_skel.append((g, hot, D[u][hot], [g, hot]))
            continue
        if c >= 2 and u == g2:
            direct_skel.append((g2, hot, D[u][hot], [g2, hot]))
            skel2_bytes += D[u][hot]
            continue
        dfwd = (g - u) % n
        dbwd = (u - g2) % n
        if c >= 2 and dbwd < dfwd:
            p = [(u - t) % n for t in range(dbwd + 1)]   # u -> g2 backward
            assert p[-1] == g2 and len(gens) >= 2
            skel_xf[1].append((u, hot, D[u][hot], p + [hot]))
            skel2_bytes += D[u][hot]
        else:
            p = [(u + t) % n for t in range(dfwd + 1)]   # u -> g1 forward
            assert p[-1] == g
            skel_xf[0].append((u, hot, D[u][hot], p + [hot]))

    # cleanup matchings for leftovers (port-disjoint per phase)
    cleanup = []
    rest = list(leftovers)
    while rest:
        used_tx, used_rx, ph, keep = set(), set(), [], []
        for (u, w, b) in rest:
            if u not in used_tx and w not in used_rx:
                used_tx.add(u)
                used_rx.add(w)
                ph.append((u, w, b, [u, w]))
            else:
                keep.append((u, w, b))
        cleanup.append(ph)
        rest = keep

    # g's cold row: relay chain after the skeleton drains
    tail = sorted(((D[g][w], w) for w in range(n)
                   if w != g and w != hot and D[g][w] > 0), reverse=True)
    chain = [w for (_, w) in tail]
    tail_xf = []
    for i, w in enumerate(chain):
        tail_xf.append((g, w, D[g][w], [g] + chain[:i + 1]))

    phases = []
    for j, s0 in enumerate(gens):
        edges = [(u, (u + s0) % n) for u in range(n)]
        edges.append((g, hot))               # skeleton 1 spans rotations
        phases.append((edges, rot_xf[j] + skel_xf[j]
                       + (direct_skel if j == 0 else [])))
    for ph in cleanup:
        edges = [(u, w) for (u, w, _, _) in ph]
        phases.append((edges, ph))
    final_edges = []
    if tail_xf:
        final_edges.append((g, chain[0]))
        for i in range(len(chain) - 1):
            final_edges.append((chain[i], chain[i + 1]))
    if c >= 2:
        final_edges.append((g2, hot))        # skeleton 2 (hot Rx pays 2nd R)
    if final_edges:
        phases.append((final_edges, tail_xf))

    tot = sum(b for _, x in phases for (_, _, b, _) in x)
    want = sum(D[u][w] for u in range(n) for w in range(n) if u != w)
    if tot != want:
        sys.stderr.write('BUG: skeleton plan bytes %d != %d\n' % (tot, want))
        sys.exit(1)
    with open(a.output, 'w') as f:
        f.write('nodes %d degree 1\n' % n)
        f.write('gbps %g delay_ns %g reconf_us %g chunk_bytes %d\n'
                % (a.gbps, a.delay_ns, a.reconf_us, a.chunk_bytes))
        f.write('pay_initial\n')
        for edges, xf in phases:
            f.write('phase\n')
            for (u, v) in edges:
                f.write('edge %d %d\n' % (u, v))
            for (s, d, b, p) in xf:
                f.write('xfer %d %d %d via %s\n'
                        % (s, d, b, ' '.join(map(str, p))))
    sys.stderr.write('skeleton: hot=%d g=%d gens=%s leftovers=%d '
                     'cleanup_phases=%d tail=%d\n'
                     % (hot, g, gens, len(leftovers), len(cleanup),
                        len(chain)))

if __name__ == '__main__':
    main()

